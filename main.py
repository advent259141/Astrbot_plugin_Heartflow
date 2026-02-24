import json
import re
import time
import datetime
from collections import deque
from typing import Dict
from dataclasses import dataclass, field

import astrbot.api.star as star
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api import logger
from astrbot.api.message_components import Plain


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
class RawMessage:
    """原始群聊消息条目"""
    sender_name: str
    sender_id: str
    content: str
    timestamp: float
    is_bot: bool = False


@dataclass
class ChatState:
    """群聊状态数据类"""
    energy: float = 1.0
    last_reply_time: float = 0.0
    last_reset_date: str = ""
    total_messages: int = 0
    total_replies: int = 0


def _extract_json(text: str) -> dict:
    """从模型返回的文本中稳健地提取 JSON 对象。

    依次尝试：
    1. 直接解析
    2. 去除 markdown 代码块后解析
    3. 正则提取第一个 {...} 子串后解析
    """
    text = text.strip()

    # 1. 直接尝试
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 2. 去除 markdown 代码块
    cleaned = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # 3. 正则提取最外层 {...}
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        return json.loads(match.group())

    raise ValueError(f"无法从文本中提取有效 JSON: {text[:200]}")


def _clamp_score(v) -> float:
    """将模型返回的分数值钉位到 [0, 10]。"""
    try:
        return max(0.0, min(10.0, float(v)))
    except (TypeError, ValueError):
        return 0.0


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
        self.judge_context_count = self.config.get("judge_context_count", self.context_messages_count)
        self.min_reply_interval = self.config.get("min_reply_interval_seconds", 0)
        self.whitelist_enabled = self.config.get("whitelist_enabled", False)
        self.chat_whitelist = self.config.get("chat_whitelist", [])

        # 群聊状态管理
        self.chat_states: Dict[str, ChatState] = {}

        # 原始群聊消息缓冲区：{unified_msg_origin: deque[RawMessage]}
        # 记录所有群聊原始消息（无论是否触发 LLM），用于判断上下文
        self._raw_msg_buffer: Dict[str, deque] = {}
        self._raw_msg_buffer_size = max(self.context_messages_count, self.judge_context_count) * 4  # 缓冲区保留更多条以备用

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
            
            # 获取当前人格ID作为缓存键（仅用 persona_id，不包含 cid）
            # cid 随对话切换会变，但提示词是按人格存的，缓存键不应包含 cid
            conversation = await self.context.conversation_manager.get_conversation(event.unified_msg_origin, curr_cid)
            persona_id = (conversation.persona_id if conversation else None) or "default"

            # 构建缓存键
            cache_key = persona_id
            
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

            logger.info(f"创建新的精简系统提示词: [{cache_key}] | 原长度:{len(original_prompt)} -> 新长度:{len(summarized_prompt)}")
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
                result_data = _extract_json(content)
                summarized = result_data.get("summarized_persona", "")

                if summarized and len(summarized.strip()) > 10:
                    return summarized.strip()
                else:
                    logger.warning("小模型返回的总结内容为空或过短")
                    return original_prompt

            except (json.JSONDecodeError, ValueError):
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
        chat_context = self._build_chat_context(event)
        recent_messages = self._get_recent_messages(event)
        last_bot_reply = self._get_last_bot_reply(event)

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
            # 构建完整的判断提示词，将系统提示直接整合到prompt中
            complete_judge_prompt = "你是一个专业的群聊回复决策系统，能够准确判断消息价值和回复时机。"
            if persona_system_prompt:
                complete_judge_prompt += f"\n\n你正在为以下角色的机器人做决策：\n{persona_system_prompt}"
            complete_judge_prompt += "\n\n**重要提醒：你必须严格按照JSON格式返回结果，不要包含任何其他内容！请不要进行对话，只返回JSON！**\n\n"
            complete_judge_prompt += judge_prompt

            # 提前计算对话历史上下文（循环外只算一次）
            recent_contexts = self._get_recent_contexts(event)

            # 重试机制：使用配置的重试次数
            max_retries = self.judge_max_retries + 1
            if self.judge_max_retries == 0:
                max_retries = 1

            for attempt in range(max_retries):
                try:
                    llm_response = await judge_provider.text_chat(
                        prompt=complete_judge_prompt,
                        contexts=recent_contexts,
                        image_urls=[],
                    )

                    content = llm_response.completion_text.strip()
                    logger.debug(f"小参数模型原始返回内容: {content[:200]}...")

                    judge_data = _extract_json(content)

                    # 直接从 JSON 根对象获取分数，并钉位到 [0, 10]
                    relevance = _clamp_score(judge_data.get("relevance", 0))
                    willingness = _clamp_score(judge_data.get("willingness", 0))
                    social = _clamp_score(judge_data.get("social", 0))
                    timing = _clamp_score(judge_data.get("timing", 0))
                    continuity = _clamp_score(judge_data.get("continuity", 0))
                    
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
                    
                except (json.JSONDecodeError, ValueError) as e:
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

    def _record_raw_message(self, event: AstrMessageEvent, is_bot: bool = False) -> None:
        """将消息写入原始消息缓冲区"""
        umo = event.unified_msg_origin
        if umo not in self._raw_msg_buffer:
            self._raw_msg_buffer[umo] = deque(maxlen=self._raw_msg_buffer_size)
        self._raw_msg_buffer[umo].append(RawMessage(
            sender_name=event.get_sender_name(),
            sender_id=str(event.get_sender_id()),
            content=event.message_str,
            timestamp=time.time(),
            is_bot=is_bot,
        ))

    def _get_raw_buffer(self, umo: str) -> list[RawMessage]:
        """获取缓冲区中的消息列表（时间顺序）"""
        return list(self._raw_msg_buffer.get(umo, []))

    @filter.event_message_type(filter.EventMessageType.GROUP_MESSAGE, priority=1000)
    async def on_group_message(self, event: AstrMessageEvent):
        """群聊消息处理入口"""

        # 检查基本条件
        if not self._should_process_message(event):
            return

        # 第一时间记录原始消息，无论是否最终触发 LLM
        self._record_raw_message(event, is_bot=False)

        try:
            # 小参数模型判断是否需要回复
            judge_result = await self.judge_with_tiny_model(event)

            if judge_result.should_reply:
                logger.info(f"🔥 心流触发主动回复 | {event.unified_msg_origin[:20]}... | 评分:{judge_result.overall_score:.2f}")

                # 设置唤醒标志为真，调用LLM
                event.is_at_or_wake_command = True
                # 标记为心流触发，供 on_llm_request 钉入角色提示
                event.set_extra("heartflow_triggered", True)

                # 更新主动回复状态
                self._update_active_state(event, judge_result)
                logger.info(f"💖 心流设置唤醒标志 | {event.unified_msg_origin[:20]}... | 评分:{judge_result.overall_score:.2f} | {judge_result.reasoning[:50]}...")
                
                # 不需要yield任何内容，让核心系统处理
                return
            else:
                # 记录被动状态
                logger.debug(f"心流判断不通过 | {event.unified_msg_origin[:20]}... | 评分:{judge_result.overall_score:.2f} | 原因: {judge_result.reasoning[:30]}...")
                self._update_passive_state(event, judge_result)

        except Exception as e:
            logger.error(f"心流插件处理消息异常: {e}")
            import traceback
            logger.error(traceback.format_exc())

    @filter.after_message_sent()
    async def on_after_message_sent(self, event: AstrMessageEvent):
        """在消息发送后将机器人的回复写入原始消息缓冲区，以便后续判断参考"""
        if not self.config.get("enable_heartflow", False):
            return

        result = event.get_result()
        if result is None or not result.chain:
            return

        # 提取回复的纯文本内容
        reply_text = "".join(
            comp.text for comp in result.chain if isinstance(comp, Plain)
        ).strip()
        if not reply_text:
            return

        umo = event.unified_msg_origin
        if umo not in self._raw_msg_buffer:
            self._raw_msg_buffer[umo] = deque(maxlen=self._raw_msg_buffer_size)
        self._raw_msg_buffer[umo].append(RawMessage(
            sender_name="bot",
            sender_id="bot",
            content=reply_text,
            timestamp=time.time(),
            is_bot=True,
        ))
        logger.debug(f"机器人回复已写入缓冲区: {umo[:20]}... | {reply_text[:40]}...")

    @filter.on_llm_request()
    async def on_llm_request(self, event: AstrMessageEvent, req):
        """心流触发时，在 LLM 请求前注入一条提示，让大模型知道自己是主动参与群聊的"""
        if not event.get_extra("heartflow_triggered"):
            return
        if not req or not hasattr(req, "system_prompt"):
            return
        note = "（注意：本次是你主动参与群聊的，不是用户叫你。回复应自然随意，像普通群成员一样加入话题。）"
        req.system_prompt = (req.system_prompt or "") + "\n" + note

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

        # 冷却时间校验：防止短时间内连续触发
        if self.min_reply_interval > 0:
            minutes = self._get_minutes_since_last_reply(event.unified_msg_origin)
            elapsed_seconds = minutes * 60
            if elapsed_seconds < self.min_reply_interval:
                logger.debug(f"冷却中，距上次回复还有 {self.min_reply_interval - elapsed_seconds:.0f}s")
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
            # 每日重置时恒复一些精力
            state.energy = min(1.0, state.energy + 0.2)

        # 基于时间流逝自然恢复精力（距上次回复每过 5 分钟回复 1% 精力）
        if state.last_reply_time > 0:
            elapsed_minutes = (time.time() - state.last_reply_time) / 60.0
            time_recovery = elapsed_minutes * (self.energy_recovery_rate * 5)
            state.energy = min(1.0, state.energy + time_recovery)
            state.last_reply_time = time.time()  # 重置计时起点，避免重复累加

        return state

    def _get_minutes_since_last_reply(self, chat_id: str) -> int:
        """获取距离上次回复的分钟数"""
        chat_state = self._get_chat_state(chat_id)

        if chat_state.last_reply_time == 0:
            return 999  # 从未回复过

        return int((time.time() - chat_state.last_reply_time) / 60)

    def _get_recent_contexts(self, event: AstrMessageEvent) -> list:
        """从原始消息缓冲区获取最近对话上下文（用于传递给小参数模型）。

        使用本地缓冲区而非 conversation_manager，以便包含所有群聊消息，
        而不仅仅是触发过 LLM 的消息。
        """
        msgs = self._get_raw_buffer(event.unified_msg_origin)
        # 排除当前这条消息（已被 _record_raw_message 写入），取之前的若干条
        if msgs and msgs[-1].content == event.message_str:
            msgs = msgs[:-1]
        recent = msgs[-self.context_messages_count:] if len(msgs) > self.context_messages_count else msgs

        contexts = []
        for m in recent:
            role = "assistant" if m.is_bot else "user"
            contexts.append({"role": role, "content": m.content})
        return contexts

    def _get_recent_messages(self, event: AstrMessageEvent) -> str:
        """从原始消息缓冲区获取最近的消息历史（用于小参数模型判断）。

        包含所有群聊成员的消息，而非仅 LLM 处理过的消息。
        """
        msgs = self._get_raw_buffer(event.unified_msg_origin)
        # 排除当前这条消息（已被 _record_raw_message 写入），取之前的若干条
        if msgs and msgs[-1].content == event.message_str:
            msgs = msgs[:-1]
        recent = msgs[-self.context_messages_count:] if len(msgs) > self.context_messages_count else msgs

        if not recent:
            return "暂无对话历史"

        lines = []
        for m in recent:
            prefix = "[机器人]" if m.is_bot else f"[{m.sender_name}]"
            lines.append(f"{prefix}: {m.content}")
        return "\n".join(lines)

    def _get_last_bot_reply(self, event: AstrMessageEvent) -> str | None:
        """从原始消息缓冲区获取上次机器人的回复内容。"""
        msgs = self._get_raw_buffer(event.unified_msg_origin)
        for m in reversed(msgs):
            if m.is_bot and m.content.strip():
                return m.content
        return None

    def _build_chat_context(self, event: AstrMessageEvent) -> str:
        """构建群聊上下文摘要信息。"""
        chat_state = self._get_chat_state(event.unified_msg_origin)

        # 检查上次机器人回复后群里有没有人接话（评估回复质量）
        msgs = self._get_raw_buffer(event.unified_msg_origin)
        post_reply_engagement = ""
        found_bot = False
        user_msgs_after_bot = 0
        for m in reversed(msgs):
            if m.is_bot:
                found_bot = True
                break
            user_msgs_after_bot += 1
        if found_bot:
            if user_msgs_after_bot >= 3:
                post_reply_engagement = "（上次回复后群里进行了热烈讨论）"
            elif user_msgs_after_bot == 0:
                post_reply_engagement = "（上次回复后无人接话）"

        if chat_state.total_messages > 100:
            activity_level = "高"
        elif chat_state.total_messages > 20:
            activity_level = "中"
        else:
            activity_level = "低"

        context_info = f"最近活跃度: {activity_level}\n"
        context_info += f"历史回复率: {(chat_state.total_replies / max(1, chat_state.total_messages) * 100):.1f}%\n"
        context_info += f"当前时间: {datetime.datetime.now().strftime('%H:%M')}"

        if post_reply_engagement:
            context_info += f"\n回复效果: {post_reply_engagement}"
            
        return context_info

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
    @filter.permission_type(filter.PermissionType.ADMIN)
    @filter.command("heartflow_reset")
    async def heartflow_reset(self, event: AstrMessageEvent):
        """重置心流状态"""

        chat_id = event.unified_msg_origin
        if chat_id in self.chat_states:
            del self.chat_states[chat_id]

        event.set_result(event.plain_result("✅ 心流状态已重置"))
        logger.info(f"心流状态已重置: {chat_id}")

    # 管理员命令：查看系统提示词缓存
    @filter.permission_type(filter.PermissionType.ADMIN)
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
    @filter.permission_type(filter.PermissionType.ADMIN)
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
            persona_mgr = self.context.persona_manager

            # 获取当前对话，尝试拿到会话绑定的 persona_id
            curr_cid = await self.context.conversation_manager.get_curr_conversation_id(event.unified_msg_origin)
            persona_id: str | None = None
            if curr_cid:
                conversation = await self.context.conversation_manager.get_conversation(event.unified_msg_origin, curr_cid)
                if conversation:
                    persona_id = conversation.persona_id

            # 用户显式取消人格
            if persona_id == "[%None]":
                return ""

            if persona_id:
                # 直接通过 PersonaManager 查询数据库
                try:
                    persona = await persona_mgr.get_persona(persona_id)
                    return persona.system_prompt or ""
                except ValueError:
                    logger.debug(f"未找到人格 {persona_id}，回退到默认人格")

            # 无 persona_id 或查询失败，使用默认人格
            default_persona = await persona_mgr.get_default_persona_v3(event.unified_msg_origin)
            return default_persona.get("prompt", "")

        except Exception as e:
            logger.debug(f"获取人格系统提示词失败: {e}")
            return ""
