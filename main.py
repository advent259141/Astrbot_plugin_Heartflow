import json
import time
import datetime
import asyncio
from typing import Dict, Optional
from dataclasses import dataclass

import astrbot.api.star as star
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api import logger


@dataclass
class JudgeResult:
    """判断结果数据类"""
    relevance: float = 0.0
    willingness: float = 0.0
    social: float = 0.0
    timing: float = 0.0
    continuity: float = 0.0  # 新增：与上次回复的连贯性
    reasoning: str = ""
    should_reply: bool = False
    confidence: float = 0.0
    overall_score: float = 0.0
    related_messages: list = None

    def __post_init__(self):
        if self.related_messages is None:
            self.related_messages = []


@dataclass
class ChatState:
    """群聊状态数据类"""
    energy: float = 1.0
    last_reply_time: float = 0.0
    last_reset_date: str = ""
    total_messages: int = 0
    total_replies: int = 0

@dataclass
class UserWaitState:
    """用户等待状态数据类"""
    user_id: str
    start_time: float
    accumulated_messages: list  # 累积的消息列表
    timer_task: Optional[asyncio.Task] = None
    waiting_since: float = 0.0  # 开始等待的时间戳



class HeartflowPlugin(star.Star):

    def __init__(self, context: star.Context, config):
        super().__init__(context)
        self.config = config

        # 判断模型配置
        self.judge_provider_name = self.config.get("judge_provider_name", "")

        # 心流参数配置
        self.reply_threshold = self.config.get("reply_threshold", 0.6)
        self.energy_decay_rate = self.config.get("energy_decay_rate", 0.1)
        self.energy_recovery_rate = self.config.get("energy_recovery_rate", 0.02)
        self.context_messages_count = self.config.get("context_messages_count", 5)
        self.whitelist_enabled = self.config.get("whitelist_enabled", False)
        self.chat_whitelist = self.config.get("chat_whitelist", [])

        # 静默等待配置
        self.silent_wait_duration = 6  # 静默等待时长，单位：秒

        # 群聊状态管理
        self.chat_states: Dict[str, ChatState] = {}
        
        # 用户等待状态管理：{chat_id}_{user_id}: UserWaitState
        self.user_wait_states: Dict[str, UserWaitState] = {}
        
        # 系统提示词缓存：{conversation_id: {"original": str, "summarized": str, "persona_id": str}}
        self.system_prompt_cache: Dict[str, Dict[str, str]] = {}

        # 判断配置
        self.judge_include_reasoning = self.config.get("judge_include_reasoning", True)
        self.judge_max_retries = max(0, self.config.get("judge_max_retries", 3))  # 确保最小为0
        
        # 判断权重配置
        self.weights = {
            "relevance": self.config.get("judge_relevance", 0.25),
            "willingness": self.config.get("judge_willingness", 0.2),
            "social": self.config.get("judge_social", 0.2),
            "timing": self.config.get("judge_timing", 0.15),
            "continuity": self.config.get("judge_continuity", 0.2)
        }
        # 检查权重和
        weight_sum = sum(self.weights.values())
        if abs(weight_sum - 1.0) > 1e-6:
            logger.warning(f"判断权重和不为1，当前和为{weight_sum}")
            # 进行归一化处理
            self.weights = {k: v / weight_sum for k, v in self.weights.items()}
            logger.info(f"判断权重和已归一化，当前配置为: {self.weights}")

        logger.info("心流插件已初始化")

    async def _get_or_create_summarized_system_prompt(self, event: AstrMessageEvent, original_prompt: str) -> str:
        """获取或创建精简版系统提示词"""
        try:
            # 获取当前会话ID
            curr_cid = await self.context.conversation_manager.get_curr_conversation_id(event.unified_msg_origin)
            if not curr_cid:
                return original_prompt
            
            # 获取当前人格ID作为缓存键的一部分
            conversation = await self.context.conversation_manager.get_conversation(event.unified_msg_origin, curr_cid)
            persona_id = conversation.persona_id if conversation else "default"
            
            # 构建缓存键
            cache_key = f"{curr_cid}_{persona_id}"
            
            # 检查缓存
            if cache_key in self.system_prompt_cache:
                cached = self.system_prompt_cache[cache_key]
                # 如果原始提示词没有变化，返回缓存的总结
                if cached.get("original") == original_prompt:
                    logger.debug(f"使用缓存的精简系统提示词: {cache_key}")
                    return cached.get("summarized", original_prompt)
            
            # 如果没有缓存或原始提示词发生变化，进行总结
            if not original_prompt or len(original_prompt.strip()) < 50:
                # 如果原始提示词太短，直接返回
                return original_prompt
            
            summarized_prompt = await self._summarize_system_prompt(original_prompt)
            
            # 更新缓存
            self.system_prompt_cache[cache_key] = {
                "original": original_prompt,
                "summarized": summarized_prompt,
                "persona_id": persona_id
            }
            
            logger.info(f"创建新的精简系统提示词: {cache_key} | 原长度:{len(original_prompt)} -> 新长度:{len(summarized_prompt)}")
            return summarized_prompt
            
        except Exception as e:
            logger.error(f"获取精简系统提示词失败: {e}")
            return original_prompt
    
    async def _summarize_system_prompt(self, original_prompt: str) -> str:
        """使用小模型对系统提示词进行总结"""
        try:
            if not self.judge_provider_name:
                return original_prompt
            
            judge_provider = self.context.get_provider_by_id(self.judge_provider_name)
            if not judge_provider:
                return original_prompt
            
            summarize_prompt = f"""请将以下机器人角色设定总结为简洁的核心要点，保留关键的性格特征、行为方式和角色定位。
总结后的内容应该在100-200字以内，突出最重要的角色特点。

原始角色设定：
{original_prompt}

请以JSON格式回复：
{{
    "summarized_persona": "精简后的角色设定，保留核心特征和行为方式"
}}

**重要：你的回复必须是完整的JSON对象，不要包含任何其他内容！**"""

            llm_response = await judge_provider.text_chat(
                prompt=summarize_prompt,
                contexts=[]  # 不需要上下文
            )

            content = llm_response.completion_text.strip()
            
            # 尝试提取JSON
            try:
                if content.startswith("```json"):
                    content = content.replace("```json", "").replace("```", "").strip()
                elif content.startswith("```"):
                    content = content.replace("```", "").strip()

                result_data = json.loads(content)
                summarized = result_data.get("summarized_persona", "")
                
                if summarized and len(summarized.strip()) > 10:
                    return summarized.strip()
                else:
                    logger.warning("小模型返回的总结内容为空或过短")
                    return original_prompt
                    
            except json.JSONDecodeError:
                logger.error(f"小模型总结系统提示词返回非有效JSON: {content}")
                return original_prompt
                
        except Exception as e:
            logger.error(f"总结系统提示词异常: {e}")
            return original_prompt

    async def judge_with_tiny_model(self, event: AstrMessageEvent) -> JudgeResult:
        """使用小模型进行智能判断"""

        if not self.judge_provider_name:
            logger.warning("小参数判断模型提供商名称未配置，跳过心流判断")
            return JudgeResult(should_reply=False, reasoning="提供商未配置")

        # 获取指定的 provider
        try:
            judge_provider = self.context.get_provider_by_id(self.judge_provider_name)
            if not judge_provider:
                logger.warning(f"未找到提供商: {self.judge_provider_name}")
                return JudgeResult(should_reply=False, reasoning=f"提供商不存在: {self.judge_provider_name}")
        except Exception as e:
            logger.error(f"获取提供商失败: {e}")
            return JudgeResult(should_reply=False, reasoning=f"获取提供商失败: {str(e)}")

        # 获取群聊状态
        chat_state = self._get_chat_state(event.unified_msg_origin)

        # 获取当前对话的人格系统提示词，让模型了解大参数LLM的角色设定
        original_persona_prompt = await self._get_persona_system_prompt(event)
        logger.debug(f"小参数模型获取原始人格提示词: {'有' if original_persona_prompt else '无'} | 长度: {len(original_persona_prompt) if original_persona_prompt else 0}")
        
        # 获取或创建精简版系统提示词
        persona_system_prompt = await self._get_or_create_summarized_system_prompt(event, original_persona_prompt)
        logger.debug(f"小参数模型使用精简人格提示词: {'有' if persona_system_prompt else '无'} | 长度: {len(persona_system_prompt) if persona_system_prompt else 0}")

        # 构建判断上下文
        chat_context = await self._build_chat_context(event)
        recent_messages = await self._get_recent_messages(event)
        last_bot_reply = await self._get_last_bot_reply(event)  # 新增：获取上次bot回复

        reasoning_part = ""
        if self.judge_include_reasoning:
            reasoning_part = ',\n    "reasoning": "详细分析原因，说明为什么应该或不应该回复，需要结合机器人角色特点进行分析，特别说明与上次回复的关联性"'

        judge_prompt = f"""
你是群聊机器人的决策系统，需要判断是否应该主动回复以下消息。

## 机器人角色设定
{persona_system_prompt if persona_system_prompt else "默认角色：智能助手"}

## 当前群聊情况
- 群聊ID: {event.unified_msg_origin}
- 我的精力水平: {chat_state.energy:.1f}/1.0
- 上次发言: {self._get_minutes_since_last_reply(event.unified_msg_origin)}分钟前

## 群聊基本信息
{chat_context}

## 最近{self.context_messages_count}条对话历史
{recent_messages}

## 上次机器人回复
{last_bot_reply if last_bot_reply else "暂无上次回复记录"}

## 待判断消息
发送者: {event.get_sender_name()}
内容: {event.message_str}
时间: {datetime.datetime.now().strftime('%H:%M:%S')}

## 评估要求
请从以下5个维度评估（0-10分），**重要提醒：基于上述机器人角色设定来判断是否适合回复**：

1. **内容相关度**(0-10)：消息是否有趣、有价值、适合我回复
   - 考虑消息的质量、话题性、是否需要回应
   - 识别并过滤垃圾消息、无意义内容
   - **结合机器人角色特点，判断是否符合角色定位**

2. **回复意愿**(0-10)：基于当前状态，我回复此消息的意愿
   - 考虑当前精力水平和心情状态
   - 考虑今日回复频率控制
   - **基于机器人角色设定，判断是否应该主动参与此话题**

3. **社交适宜性**(0-10)：在当前群聊氛围下回复是否合适
   - 考虑群聊活跃度和讨论氛围
   - **考虑机器人角色在群中的定位和表现方式**

4. **时机恰当性**(0-10)：回复时机是否恰当
   - 考虑距离上次回复的时间间隔
   - 考虑消息的紧急性和时效性

5. **对话连贯性**(0-10)：当前消息与上次机器人回复的关联程度
   - 如果当前消息是对上次回复的回应或延续，应给高分
   - 如果当前消息与上次回复完全无关，给中等分数
   - 如果没有上次回复记录，给默认分数5分

**回复阈值**: {self.reply_threshold} (综合评分达到此分数才回复)

**重要！！！请严格按照以下JSON格式回复，不要添加任何其他内容：**

请以JSON格式回复：
{{
    "relevance": 分数,
    "willingness": 分数,
    "social": 分数,
    "timing": 分数,
    "continuity": 分数{reasoning_part}
}}

**注意：你的回复必须是完整的JSON对象，不要包含任何解释性文字或其他内容！**
"""

        try:
            # 使用 provider 调用模型，传入最近的对话历史作为上下文
            recent_contexts = await self._get_recent_contexts(event)

            # 构建完整的判断提示词，将系统提示直接整合到prompt中
            complete_judge_prompt = "你是一个专业的群聊回复决策系统，能够准确判断消息价值和回复时机。"
            if persona_system_prompt:
                complete_judge_prompt += f"\n\n你正在为以下角色的机器人做决策：\n{persona_system_prompt}"
            complete_judge_prompt += "\n\n**重要提醒：你必须严格按照JSON格式返回结果，不要包含任何其他内容！请不要进行对话，只返回JSON！**\n\n"
            complete_judge_prompt += judge_prompt

            # 重试机制：使用配置的重试次数
            max_retries = self.judge_max_retries + 1  # 配置的次数+原始尝试=总尝试次数
            
            # 如果配置的重试次数为0，只尝试一次
            if self.judge_max_retries == 0:
                max_retries = 1
            
            for attempt in range(max_retries):
                try:
                    logger.debug(f"小参数模型判断尝试 {attempt + 1}/{max_retries}")
                    
                    llm_response = await judge_provider.text_chat(
                        prompt=complete_judge_prompt,
                        contexts=recent_contexts  # 传入最近的对话历史
                    )

                    content = llm_response.completion_text.strip()
                    logger.debug(f"小参数模型原始返回内容: {content[:200]}...")

                    # 尝试提取JSON
                    if content.startswith("```json"):
                        content = content.replace("```json", "").replace("```", "").strip()
                    elif content.startswith("```"):
                        content = content.replace("```", "").strip()

                    judge_data = json.loads(content)

                    # 直接从JSON根对象获取分数
                    relevance = judge_data.get("relevance", 0)
                    willingness = judge_data.get("willingness", 0)
                    social = judge_data.get("social", 0)
                    timing = judge_data.get("timing", 0)
                    continuity = judge_data.get("continuity", 0)
                    
                    # 计算综合评分
                    overall_score = (
                        relevance * self.weights["relevance"] +
                        willingness * self.weights["willingness"] +
                        social * self.weights["social"] +
                        timing * self.weights["timing"] +
                        continuity * self.weights["continuity"]
                    ) / 10.0

                    # 根据综合评分判断是否应该回复
                    should_reply = overall_score >= self.reply_threshold

                    logger.debug(f"小参数模型判断成功，综合评分: {overall_score:.3f}, 是否回复: {should_reply}")

                    return JudgeResult(
                        relevance=relevance,
                        willingness=willingness,
                        social=social,
                        timing=timing,
                        continuity=continuity,
                        reasoning=judge_data.get("reasoning", "") if self.judge_include_reasoning else "",
                        should_reply=should_reply,
                        confidence=overall_score,  # 使用综合评分作为置信度
                        overall_score=overall_score,
                        related_messages=[]  # 不再使用关联消息功能
                    )
                    
                except json.JSONDecodeError as e:
                    logger.warning(f"小参数模型返回JSON解析失败 (尝试 {attempt + 1}/{max_retries}): {str(e)}")
                    logger.warning(f"无法解析的内容: {content[:500]}...")
                    
                    if attempt == max_retries - 1:
                        # 最后一次尝试失败，返回失败结果
                        logger.error(f"小参数模型重试{self.judge_max_retries}次后仍然返回无效JSON，放弃处理")
                        return JudgeResult(should_reply=False, reasoning=f"JSON解析失败，重试{self.judge_max_retries}次")
                    else:
                        # 还有重试机会，添加更强的提示
                        complete_judge_prompt = complete_judge_prompt.replace(
                            "**重要提醒：你必须严格按照JSON格式返回结果，不要包含任何其他内容！请不要进行对话，只返回JSON！**",
                            f"**重要提醒：你必须严格按照JSON格式返回结果，不要包含任何其他内容！请不要进行对话，只返回JSON！这是第{attempt + 2}次尝试，请确保返回有效的JSON格式！**"
                        )
                        continue

        except Exception as e:
            logger.error(f"小参数模型判断异常: {e}")
            return JudgeResult(should_reply=False, reasoning=f"异常: {str(e)}")

    @filter.event_message_type(filter.EventMessageType.GROUP_MESSAGE, priority=1000)
    async def on_group_message(self, event: AstrMessageEvent):
        """群聊消息处理入口"""

        # 检查基本条件
        if not self._should_process_message(event):
            return

        try:
            # 小参数模型判断是否需要回复
            judge_result = await self.judge_with_tiny_model(event)

            if judge_result.should_reply:
                logger.info(f"🔥 心流触发主动回复 | {event.unified_msg_origin[:20]}... | 评分:{judge_result.overall_score:.2f}")

                # 进入静默等待状态，而不是立即回复
                await self._enter_silent_wait(event, judge_result)
                
                return
            else:
                # 记录被动状态
                logger.debug(f"心流判断不通过 | {event.unified_msg_origin[:20]}... | 评分:{judge_result.overall_score:.2f} | 原因: {judge_result.reasoning[:30]}...")
                self._update_passive_state(event, judge_result)

        except Exception as e:
            logger.error(f"心流插件处理消息异常: {e}")
            import traceback
            logger.error(traceback.format_exc())

    def _should_process_message(self, event: AstrMessageEvent) -> bool:
        """检查是否应该处理这条消息"""

        # 检查插件是否启用
        if not self.config.get("enable_heartflow", False):
            return False

        # 跳过已经被其他插件或系统标记为唤醒的消息
        if event.is_at_or_wake_command:
            logger.debug(f"跳过已被标记为唤醒的消息: {event.message_str}")
            return False

        # 检查白名单
        if self.whitelist_enabled:
            if not self.chat_whitelist:
                logger.debug(f"白名单为空，跳过处理: {event.unified_msg_origin}")
                return False

            if event.unified_msg_origin not in self.chat_whitelist:
                logger.debug(f"群聊不在白名单中，跳过处理: {event.unified_msg_origin}")
                return False

        # 跳过机器人自己的消息
        if event.get_sender_id() == event.get_self_id():
            return False

        # 跳过空消息
        if not event.message_str or not event.message_str.strip():
            return False

        return True

    def _get_chat_state(self, chat_id: str) -> ChatState:
        """获取群聊状态"""
        if chat_id not in self.chat_states:
            self.chat_states[chat_id] = ChatState()

        # 检查日期重置
        today = datetime.date.today().isoformat()
        state = self.chat_states[chat_id]

        if state.last_reset_date != today:
            state.last_reset_date = today
            # 每日重置时恢复一些精力
            state.energy = min(1.0, state.energy + 0.2)

        return state

    def _get_minutes_since_last_reply(self, chat_id: str) -> int:
        """获取距离上次回复的分钟数"""
        chat_state = self._get_chat_state(chat_id)

        if chat_state.last_reply_time == 0:
            return 999  # 从未回复过

        return int((time.time() - chat_state.last_reply_time) / 60)

    async def _get_recent_contexts(self, event: AstrMessageEvent) -> list:
        """获取最近的对话上下文（用于传递给小参数模型）
        
        注意：此方法会过滤掉函数调用相关内容，只保留纯文本消息，
        以避免小参数模型因不支持函数调用而报错。
        """
        try:
            curr_cid = await self.context.conversation_manager.get_curr_conversation_id(event.unified_msg_origin)
            if not curr_cid:
                return []

            conversation = await self.context.conversation_manager.get_conversation(event.unified_msg_origin, curr_cid)
            if not conversation or not conversation.history:
                return []

            context = json.loads(conversation.history)

            # 获取最近的 context_messages_count 条消息
            recent_context = context[-self.context_messages_count:] if len(context) > self.context_messages_count else context

            # 过滤掉函数调用相关内容，避免小参数模型报错
            filtered_context = []
            for msg in recent_context:
                # 只保留纯文本的用户和助手消息
                role = msg.get("role", "")
                content = msg.get("content", "")
                
                if role in ["user", "assistant"] and content and isinstance(content, str):
                    # 创建一个干净的消息副本，只包含文本内容
                    clean_msg = {
                        "role": role,
                        "content": content
                    }
                    filtered_context.append(clean_msg)

            return filtered_context

        except Exception as e:
            logger.debug(f"获取对话上下文失败: {e}")
            return []

    async def _build_chat_context(self, event: AstrMessageEvent) -> str:
        """构建群聊上下文"""
        chat_state = self._get_chat_state(event.unified_msg_origin)

        context_info = f"""最近活跃度: {'高' if chat_state.total_messages > 100 else '中' if chat_state.total_messages > 20 else '低'}
历史回复率: {(chat_state.total_replies / max(1, chat_state.total_messages) * 100):.1f}%
当前时间: {datetime.datetime.now().strftime('%H:%M')}"""
        return context_info

    async def _get_recent_messages(self, event: AstrMessageEvent) -> str:
        """获取最近的消息历史（用于小参数模型判断）"""
        try:
            curr_cid = await self.context.conversation_manager.get_curr_conversation_id(event.unified_msg_origin)
            if not curr_cid:
                return "暂无对话历史"

            conversation = await self.context.conversation_manager.get_conversation(event.unified_msg_origin, curr_cid)
            if not conversation or not conversation.history:
                return "暂无对话历史"

            context = json.loads(conversation.history)

            # 获取最近的 context_messages_count 条消息
            recent_context = context[-self.context_messages_count:] if len(context) > self.context_messages_count else context

            # 直接返回原始的对话历史，让小参数模型自己判断
            messages_text = []
            for msg in recent_context:
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
                if role in ["user", "assistant"]:
                    messages_text.append(content)

            return "\n---\n".join(messages_text) if messages_text else "暂无对话历史"

        except Exception as e:
            logger.debug(f"获取消息历史失败: {e}")
            return "暂无对话历史"

    async def _get_last_bot_reply(self, event: AstrMessageEvent) -> str:
        """获取上次机器人的回复消息"""
        try:
            curr_cid = await self.context.conversation_manager.get_curr_conversation_id(event.unified_msg_origin)
            if not curr_cid:
                return None

            conversation = await self.context.conversation_manager.get_conversation(event.unified_msg_origin, curr_cid)
            if not conversation or not conversation.history:
                return None

            context = json.loads(conversation.history)

            # 从后往前查找最后一条assistant消息
            for msg in reversed(context):
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
                if role == "assistant" and content.strip():
                    return content

            return None

        except Exception as e:
            logger.debug(f"获取上次bot回复失败: {e}")
            return None

    def _update_active_state(self, event: AstrMessageEvent, judge_result: JudgeResult):
        """更新主动回复状态"""
        chat_id = event.unified_msg_origin
        chat_state = self._get_chat_state(chat_id)

        # 更新回复相关状态
        chat_state.last_reply_time = time.time()
        chat_state.total_replies += 1
        chat_state.total_messages += 1

        # 精力消耗（回复后精力下降）
        chat_state.energy = max(0.1, chat_state.energy - self.energy_decay_rate)

        logger.debug(f"更新主动状态: {chat_id[:20]}... | 精力: {chat_state.energy:.2f}")

    def _update_passive_state(self, event: AstrMessageEvent, judge_result: JudgeResult):
        """更新被动状态（未回复）"""
        chat_id = event.unified_msg_origin
        chat_state = self._get_chat_state(chat_id)

        # 更新消息计数
        chat_state.total_messages += 1

        # 精力恢复（不回复时精力缓慢恢复）
        chat_state.energy = min(1.0, chat_state.energy + self.energy_recovery_rate)

        logger.debug(f"更新被动状态: {chat_id[:20]}... | 精力: {chat_state.energy:.2f} | 原因: {judge_result.reasoning[:30]}...")

    async def _enter_silent_wait(self, event: AstrMessageEvent, judge_result: JudgeResult):
        """进入静默等待状态"""
        chat_id = event.unified_msg_origin
        user_id = event.get_sender_id()
        user_key = f"{chat_id}_{user_id}"
        current_time = time.time()
        
        # 记录当前消息
        message_info = {
            'timestamp': current_time,
            'content': event.message_str,
            'event': event
        }
        
        # 检查用户是否已经处于等待状态
        if user_key in self.user_wait_states:
            # 用户已有等待任务，取消原任务并重置计时器
            wait_state = self.user_wait_states[user_key]
            if wait_state.timer_task and not wait_state.timer_task.done():
                wait_state.timer_task.cancel()
            
            # 添加新消息到累积列表
            wait_state.accumulated_messages.append(message_info)
            wait_state.waiting_since = current_time
            
            # 创建新的计时器任务
            wait_state.timer_task = asyncio.create_task(
                self._wait_and_reply(user_key, event, judge_result)
            )
            
            logger.info(f"🔄 用户 {user_id[:5]}... 重置等待计时器，累积消息数: {len(wait_state.accumulated_messages)}")
        else:
            # 用户首次进入等待状态
            timer_task = asyncio.create_task(
                self._wait_and_reply(user_key, event, judge_result)
            )
            
            # 存储等待状态
            self.user_wait_states[user_key] = UserWaitState(
                user_id=user_id,
                start_time=current_time,
                accumulated_messages=[message_info],
                timer_task=timer_task,
                waiting_since=current_time
            )
            
            logger.info(f"⏳ 用户 {user_id[:5]}... 进入静默等待状态，预计等待 {self.silent_wait_duration} 秒")

    async def _wait_and_reply(self, user_key: str, event: AstrMessageEvent, judge_result: JudgeResult):
        """等待指定时长后，合并消息并触发回复"""
        try:
            # 等待指定时长
            await asyncio.sleep(self.silent_wait_duration)
            
            # 检查用户状态是否仍然存在
            if user_key not in self.user_wait_states:
                return
            
            wait_state = self.user_wait_states[user_key]
            
            # 检查是否需要继续等待（防止竞争条件）
            if time.time() - wait_state.waiting_since < self.silent_wait_duration - 0.1:
                return
            
            # 合并用户累积的消息
            combined_event = self._combine_user_messages(wait_state.accumulated_messages)
            
            if combined_event:
                logger.info(f"🚀 静默等待结束，准备回复用户 {wait_state.user_id[:5]}... 的 {len(wait_state.accumulated_messages)} 条消息")
                
                # 设置唤醒标志为真，调用LLM
                combined_event.is_at_or_wake_command = True
                
                # 更新主动回复状态
                self._update_active_state(combined_event, judge_result)
                
                logger.info(f"💖 心流设置唤醒标志 | {combined_event.unified_msg_origin[:20]}... | 评分:{judge_result.overall_score:.2f} | 合并消息数:{len(wait_state.accumulated_messages)}")
                
                # 触发大型LLM生成回复
                try:
                    # 通过context将带有唤醒标志的事件发送到事件队列，触发系统使用大型LLM生成回复
                    await self.context.get_event_queue().put(combined_event)
                    logger.info(f"✅ 成功触发大型LLM回复用户 {wait_state.user_id[:5]}...")
                    # 立即清空累积的消息，避免新对话混淆
                    wait_state.accumulated_messages = []
                    logger.info(f"🧹 已清空用户 {wait_state.user_id[:5]}... 的累积消息")
                except Exception as e:
                    logger.error(f"触发大型LLM回复失败: {e}")
            
        except asyncio.CancelledError:
            # 任务被取消（用户发送了新消息）
            logger.info(f"🚫 等待任务被取消: {user_key}")
        except Exception as e:
            logger.error(f"等待任务异常: {e}")
        finally:
            # 清理等待状态（只有当任务未被重置时才清理）
            if user_key in self.user_wait_states:
                wait_state = self.user_wait_states[user_key]
                # 检查当前任务是否是正在执行的任务
                if wait_state.timer_task and wait_state.timer_task.done():
                    del self.user_wait_states[user_key]

    def _combine_user_messages(self, accumulated_messages: list) -> Optional[AstrMessageEvent]:
        """合并用户累积的消息"""
        if not accumulated_messages:
            return None
        
        # 获取第一条消息的事件对象作为基础
        base_event = accumulated_messages[0]['event']
        
        # 合并所有消息内容
        combined_content = "\n".join([msg['content'] for msg in accumulated_messages])
        
        # 创建一个新的事件对象副本
        # 注意：这里我们直接修改原始事件对象的消息内容，因为无法直接创建新的AstrMessageEvent
        base_event.message_str = combined_content
        
        logger.debug(f"📝 合并消息完成，原消息数: {len(accumulated_messages)}，合并后长度: {len(combined_content)}")
        
        return base_event

    # 管理员命令：查看心流状态
    @filter.command("heartflow")
    async def heartflow_status(self, event: AstrMessageEvent):
        """查看心流状态"""

        chat_id = event.unified_msg_origin
        chat_state = self._get_chat_state(chat_id)

        status_info = f"""
🔮 心流状态报告

📊 **当前状态**
- 群聊ID: {event.unified_msg_origin}
- 精力水平: {chat_state.energy:.2f}/1.0 {'🟢' if chat_state.energy > 0.7 else '🟡' if chat_state.energy > 0.3 else '🔴'}
- 上次回复: {self._get_minutes_since_last_reply(chat_id)}分钟前

📈 **历史统计**
- 总消息数: {chat_state.total_messages}
- 总回复数: {chat_state.total_replies}
- 回复率: {(chat_state.total_replies / max(1, chat_state.total_messages) * 100):.1f}%

⚙️ **配置参数**
- 回复阈值: {self.reply_threshold}
- 判断提供商: {self.judge_provider_name}
- 最大重试次数: {self.judge_max_retries}
- 白名单模式: {'✅ 开启' if self.whitelist_enabled else '❌ 关闭'}
- 白名单群聊数: {len(self.chat_whitelist) if self.whitelist_enabled else 0}

🧠 **智能缓存**
- 系统提示词缓存: {len(self.system_prompt_cache)} 个

🎯 **评分权重**
- 内容相关度: {self.weights['relevance']:.0%}
- 回复意愿: {self.weights['willingness']:.0%}
- 社交适宜性: {self.weights['social']:.0%}
- 时机恰当性: {self.weights['timing']:.0%}
- 对话连贯性: {self.weights['continuity']:.0%}

🎯 **插件状态**: {'✅ 已启用' if self.config.get('enable_heartflow', False) else '❌ 已禁用'}
"""

        event.set_result(event.plain_result(status_info))

    # 管理员命令：重置心流状态
    @filter.command("heartflow_reset")
    async def heartflow_reset(self, event: AstrMessageEvent):
        """重置心流状态"""

        chat_id = event.unified_msg_origin
        if chat_id in self.chat_states:
            del self.chat_states[chat_id]

        event.set_result(event.plain_result("✅ 心流状态已重置"))
        logger.info(f"心流状态已重置: {chat_id}")

    # 管理员命令：查看系统提示词缓存
    @filter.command("heartflow_cache")
    async def heartflow_cache_status(self, event: AstrMessageEvent):
        """查看系统提示词缓存状态"""
        
        cache_info = "🧠 系统提示词缓存状态\n\n"
        
        if not self.system_prompt_cache:
            cache_info += "📭 当前无缓存记录"
        else:
            cache_info += f"📝 总缓存数量: {len(self.system_prompt_cache)}\n\n"
            
            for cache_key, cache_data in self.system_prompt_cache.items():
                original_len = len(cache_data.get("original", ""))
                summarized_len = len(cache_data.get("summarized", ""))
                persona_id = cache_data.get("persona_id", "unknown")
                
                cache_info += f"🔑 **缓存键**: {cache_key}\n"
                cache_info += f"👤 **人格ID**: {persona_id}\n"
                cache_info += f"📏 **压缩率**: {original_len} -> {summarized_len} ({(1-summarized_len/max(1,original_len))*100:.1f}% 压缩)\n"
                cache_info += f"📄 **精简内容**: {cache_data.get('summarized', '')[:100]}...\n\n"
        
        event.set_result(event.plain_result(cache_info))

    # 管理员命令：清除系统提示词缓存
    @filter.command("heartflow_cache_clear")
    async def heartflow_cache_clear(self, event: AstrMessageEvent):
        """清除系统提示词缓存"""
        
        cache_count = len(self.system_prompt_cache)
        self.system_prompt_cache.clear()
        
        event.set_result(event.plain_result(f"✅ 已清除 {cache_count} 个系统提示词缓存"))
        logger.info(f"系统提示词缓存已清除，共清除 {cache_count} 个缓存")

    async def _get_persona_system_prompt(self, event: AstrMessageEvent) -> str:
        """获取当前对话的人格系统提示词"""
        try:
            # 获取当前对话
            curr_cid = await self.context.conversation_manager.get_curr_conversation_id(event.unified_msg_origin)
            if not curr_cid:
                # 如果没有对话ID，使用默认人格
                default_persona_name = self.context.provider_manager.selected_default_persona["name"]
                return self._get_persona_prompt_by_name(default_persona_name)

            conversation = await self.context.conversation_manager.get_conversation(event.unified_msg_origin, curr_cid)
            if not conversation:
                # 如果没有对话对象，使用默认人格
                default_persona_name = self.context.provider_manager.selected_default_persona["name"]
                return self._get_persona_prompt_by_name(default_persona_name)

            # 获取人格ID
            persona_id = conversation.persona_id

            if not persona_id:
                # persona_id 为 None 时，使用默认人格
                persona_id = self.context.provider_manager.selected_default_persona["name"]
            elif persona_id == "[%None]":
                # 用户显式取消人格时，不使用任何人格
                return ""

            return self._get_persona_prompt_by_name(persona_id)

        except Exception as e:
            logger.debug(f"获取人格系统提示词失败: {e}")
            return ""

    def _get_persona_prompt_by_name(self, persona_name: str) -> str:
        """根据人格名称获取人格提示词"""
        try:
            # 从provider_manager中查找人格
            for persona in self.context.provider_manager.personas:
                if persona["name"] == persona_name:
                    return persona.get("prompt", "")

            logger.debug(f"未找到人格: {persona_name}")
            return ""

        except Exception as e:
            logger.debug(f"获取人格提示词失败: {e}")
            return ""
