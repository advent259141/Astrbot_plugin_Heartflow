# 心流插件（Heartflow）

基于双模型架构的 AstrBot 群聊主动回复插件。小模型负责判断当前消息是否值得参与，大模型继续使用 AstrBot 当前会话的提供商和人格生成回复。

## 功能

- 使用独立的小模型从内容、意愿、社交、时机和连贯性五个维度评分
- 按群隔离上下文、精力、冷却和统计状态
- 保存普通群聊和直接 @ 机器人的对话，自动排除命令消息
- 使用每群异步锁，避免并发消息同时突破冷却限制
- 将判断规则放在系统提示词中，并严格校验模型返回的 JSON
- 支持群聊白名单、评分权重、上下文数量、判断超时和最短回复间隔
- 限制群聊状态与原文消息缓存，避免长期运行后无界占用内存

## 兼容性

- AstrBot `>=4.27,<5`
- 使用 AstrBot 已有的 `aiohttp` 调用 JEV，无需额外安装 JEV SDK

AstrBot 4.27 已经提供概率式群聊主动回复。如果使用本插件，请关闭 AstrBot 的 `provider_ltm_settings.active_reply.enable`，避免两套触发策略同时生效；插件检测到冲突时也会在日志中警告一次。

## 安装

推荐直接在 AstrBot WebUI 的插件管理页面通过仓库地址安装：

```text
https://github.com/advent259141/Astrbot_plugin_Heartflow
```

手动安装时，将仓库克隆到 AstrBot 的 `data/plugins/`：

```bash
cd AstrBot/data/plugins
git clone https://github.com/advent259141/Astrbot_plugin_Heartflow.git
```

重启 AstrBot 或在插件管理页面重载插件。

## 必要配置

1. 在 AstrBot 中配置一个成本较低、响应较快的聊天模型提供商。
2. 将 `judge_provider_name` 设置为该提供商的 ID。
3. 打开 `enable_heartflow`。
4. 建议先设置 `min_reply_interval_seconds`，再逐步调整回复阈值。

### 使用 TypeSafe JEV 判断

将 `judge_backend` 切换为 `jev`，填写 `jev_api_key`（或在 AstrBot 进程中设置
`TYPESAFE_API_KEY`），然后重载插件。JEV 模式无需配置 `judge_provider_name`。

```json
{
  "enable_heartflow": true,
  "judge_backend": "jev",
  "jev_api_key": "你的 TypeSafe API Key",
  "jev_base_url": "https://api.typesafe.ai/v1",
  "jev_model": "jev-latest",
  "jev_timeout_seconds": 10,
  "reply_threshold": 0.6
}
```

JEV 调用[原生 System One 接口](https://docs.typesafe.ai/api/system-one)，
一次请求包含五个独立的 `Score` 问题。每项使用五档评分（0–4），映射为
原有的 0–10 分后，继续应用已有权重、回复阈值、精力和冷却规则。
`confidence` 单独记录为五项置信度的加权平均，不参与回复阈值比较。
JEV 只做决策，实际回复仍由 AstrBot 当前会话的大模型生成。

两种后端都直接使用完整人格原文，不再压缩提示词或额外调用模型生成人格摘要。JEV 不请求文本理由。
`judge_include_reasoning` 仅控制固定的“JEV 五维评分”标记；`judge_max_retries`
只作用于原有 LLM 后端，JEV 每次判断只请求一次。
请求同时受 `jev_timeout_seconds` 和整轮 `judge_timeout_seconds` 限制。
缺少密钥、接口错误、超时或返回值非法时，跳过本次主动回复。
`jev_base_url` 也可填写完整的 `/v1/systemone` 地址，不能使用 Chat Completions 接口。
修改后端、模型或密钥后需重载插件。

### JEV 的完整对话上下文

JEV 每次请求的 `state.persona` 包含完整人格，`state.history` 包含逐条群聊历史，
每条保留 `role`（user/assistant）、`sender`、`sender_id`、`content` 原文和 `timestamp`。
历史按从旧到新排列，`state.current_message` 单独携带当前消息，避免重复。
`chat_summary` 仅为活跃度等辅助统计，不会代替原文历史；五个评分问题都会看到同一份历史。

`judge_context_count` 默认 10，设为 0 时传入插件当前群缓存的全部历史。
这里包括普通未唤醒消息和机器人已发出的回复，缓存来自插件本次运行期间的群聊记录，
不读取数据库中的全部旧会话。缓存仍受 `context_messages_count` 等设置计算出的容量限制。
不裁剪人格或单条消息原文；请求过长时可调小历史条数。历史图片保存消息标记，当前消息的图片按下述方式发送。

### 图片与图文消息

JEV 后端默认启用 `jev_enable_images`。纯图片和图文消息都会把当前消息的图片
通过原生 `assets` 传给 JEV，与人格、群聊上下文一起评分；本地历史仅保存图片消息标记。
`jev_max_images` 默认 4（范围 1–8），超出时选择当前消息最近的图片。
关闭图片判断后，纯图片不触发心流，图文消息只按文字评分。

图片功能遵循[官方实验性图片接口](https://docs.typesafe.ai/images)。请求会附带
`TypeSafe-Danger-Zone-Images: true`。插件读取 AstrBot 图片组件，支持本地、URL、Base64
来源，校正 EXIF 方向、将动画取首帧、转为 JPEG，最长边缩至 1568，短边不足 280 时留白补齐。
原始图片不会被修改。中转服务需要支持 `assets`，并透传图片请求头；仅文本调用成功
不代表该中转支持图片。图片加载或接口失败时，跳过本次主动回复，不伪造图片描述。

## 主要配置

| 配置 | 默认值 | 说明 |
| --- | ---: | --- |
| `enable_heartflow` | `false` | 是否启用心流主动回复 |
| `judge_provider_name` | 空 | 用于评分的小模型提供商 ID |
| `reply_threshold` | `0.6` | 综合评分达到该值才触发回复 |
| `energy_decay_rate` | `0.1` | 每次成功生成主动回复后消耗的精力 |
| `energy_recovery_rate` | `0.02` | 未回复以及每经过 5 分钟恢复的精力 |
| `context_messages_count` | `5` | 本地上下文缓冲基数 |
| `judge_context_count` | `10` | 两种后端传入的历史条数；`0` 表示当前群缓存的全部历史，不包含当前消息 |
| `judge_timeout_seconds` | `30` | 整轮判断（含格式重试）的超时秒数，范围 `5-120` |
| `min_reply_interval_seconds` | `0` | 两次主动触发的最短间隔；`0` 表示不限制 |
| `whitelist_enabled` | `false` | 是否仅处理白名单群聊 |
| `chat_whitelist` | `[]` | 通过 `/sid` 获取并填写完整会话 ID |
| `judge_max_retries` | `3` | 判断结果格式错误时的重试次数，范围 `0-5` |
| `max_tracked_chats` | `1000` | 内存中最多保留的群聊状态与消息缓冲数 |

五项评分权重默认分别为：内容相关度 25%、回复意愿 20%、社交适宜性 20%、时机 15%、连贯性 20%。权重不必手动保证总和为 1，插件会自动归一化；全部为 0 时会回退到默认权重。

## 管理命令

以下命令仅限 AstrBot 管理员：

- `/heartflow`：查看当前群聊的精力、回复统计和配置
- `/heartflow_reset`：重置当前群聊状态和本地消息缓冲

## 工作方式

1. 插件记录当前群聊消息；斜杠命令不会进入上下文。
2. 同一群聊的消息按顺序交给判断模型评分。
3. 综合评分超过阈值且不在冷却期时，插件唤醒 AstrBot 的正常大模型回复流程。
4. 只有大模型成功返回内容后，插件才扣除精力并增加回复次数；生成失败会释放本次冷却预留。
5. AstrBot 发出的普通或流式 LLM 最终回复会写回本地上下文，命令和其他非 LLM 输出不会污染记录。

普通 LLM 判断后端使用标准对话结构：System 包含完整人格原文、群聊状态、判断说明和评分/输出要求；`contexts` 按 user/assistant 角色传入最近历史；当前消息以 `[发送者]: 正文` 作为最后一条 User 消息，不再包装为 JSON。历史和当前消息中的指令不能改变判断任务。模型返回的五项分数必须是 JSON 数字且位于 `0-10`，否则会在总超时预算内重试。

所有内存状态会在插件重载或 AstrBot 重启后清空。

## 许可证

本插件遵循仓库中的许可证文件。
