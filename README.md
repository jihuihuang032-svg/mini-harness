# Mini Harness

Mini Harness 是一个小巧但完整的编码 Agent 框架。它将模型与执行系统分开:

- 模型决定下一步动作。
- 框架负责验证和执行工具。
- 工具结果反馈给模型。
- 循环持续进行,直到模型返回最终答案。

首个版本面向 OpenAI 兼容 API,因此可以配置用于 DeepSeek、Qwen、Kimi、GLM 等兼容服务。

## 当前能力

- CLI 入口,支持 `doctor`、`eval`、`init`、`list-evals`、`run`、`show-eval`、`show-run` 及历史命令
- OpenAI 兼容的 chat completions 适配器
- 对模型 API 瞬时失败的重试处理
- 通过 `harness.json` 支持工作区配置文件
- 内置 DeepSeek、Qwen、Kimi、GLM、Doubao 的 Provider 预设
- OpenAI 兼容 API 的流式模型输出
- 用于本地演示和测试的离线 mock 模型
- 面向无原生工具调用能力模型的 JSON 动作协议
- 将 OpenAI 兼容的原生 `tool_calls` 规范化为框架动作协议
- 可选:基于当前激活的工具 schema 生成 OpenAI 兼容的原生工具请求 payload
- 可选:通过 `response_format` 启用 OpenAI 兼容的 JSON 模式
- 规划动作:`plan` 和 `todo_update`
- 结构化动作校验
- 运行时注入工具 schema
- 运行时工具参数校验,支持默认值和类型检查
- 将仓库摘要注入初始上下文
- 可配置字符预算的消息窗口裁剪
- 反馈给模型前对工具结果进行压缩
- 工作区路径沙箱
- 命令风险评估,包含拒绝类和需审批类
- CLI 审批模式:`never`、`on-request`、`auto`
- 针对命令风险和审批决策的结构化命令审计事件
- 工具权限配置:`full`、`review`、`read-only`
- 命令策略配置:`default`、`strict`
- 工具路由器
- 工具:
  - `list_files`
  - `read_file`,支持可选行范围
  - `write_file`
  - `search_text`,支持字面量/正则匹配、匹配数量限制及可选上下文
  - `apply_patch`
  - `run_command`
  - `git_status`
  - `git_diff`
- 带序号的有序 JSONL 运行 trace
- 每个 trace 中包含非敏感运行配置快照
- 运行历史列表和 trace 检查
- 每次运行的工作区变更摘要,存储于 `.harness/changes`
- 运行 checkpoint,包含最新消息和计划状态,存储于 `.harness/checkpoints`
- 轻量级 HTTP API 服务器,支持运行、provider、健康检查及持久化异步任务提交
- 内嵌 Web 控制台,用于 mock 任务提交和 trace 时间线查看
- 输出截断
- 最大步数限制

## 快速开始

无需任何 API Key 即可运行离线 smoke 演示:

```powershell
python -m harness --mock "Inspect this project"
```

等价的显式模块形式为 `python -m harness.cli ...`。若已作为包安装,控制台命令为 `mini-harness`。

该包将提示词 markdown 作为包数据打包,因此安装后运行与源码树运行使用相同的系统提示词。

通过 `python -m harness --version` 或 `mini-harness --version` 查看已安装版本。

在工作区初始化配置文件:

```powershell
python -m harness.cli init --workspace E:\some-project --env
```

无需调用模型即可检查本地配置:

```powershell
python -m harness.cli doctor --workspace E:\some-project --mock
```

`doctor` 会检查工作区/配置状态、打包的提示词资源、默认工具路由、provider 设置、配置文件、重试设置及 trace 标志。

运行 JSONL 评估套件:

```powershell
python -m harness eval --mock examples/smoke.jsonl
```

也支持显式子命令形式:

```powershell
python -m harness.cli run --mock "Inspect this project"
```

也可以把较长的任务说明放到 UTF-8 文本文件中，避免命令行转义和换行问题:

```powershell
python -m harness.cli run --mock --task-file task.md
```
从脚本或 CI 调用框架时使用 JSON 输出:

```powershell
python -m harness.cli run --mock --json "Inspect this project"
```

流式模式在为框架组装完整 JSON 动作的同时,会打印模型输出块:

```powershell
python -m harness.cli run --mock --stream "Inspect this project"
```

使用真实模型时,将 `.env.example` 复制为 `.env` 并填入 provider 设置,或直接设置环境变量。

```powershell
python -m harness.cli run "Read this project and add a short README improvement"
```

本地 `.env`、`harness.json`、`.harness/`、缓存及构建输出会被项目 `.gitignore` 忽略。

也可将 `harness.json.example` 复制为 `harness.json`,将项目默认值保留在工作区:

```powershell
python -m harness.cli run --config harness.json "Inspect this project"
```

可将框架指向其他工作区:

```powershell
python -m harness.cli run --workspace E:\some-project "Find failing tests and fix them"
```

被归类为 `approval_required` 的命令默认会被拒绝。使用 `on-request` 进行交互式确认:

```powershell
python -m harness.cli run --approval on-request "Run the project checks and fix failures"
```

`--approval auto` 仅用于测试或可信本地演示,不适用于不可信任务。

使用工具配置文件限制单次运行的动作范围:

```powershell
python -m harness.cli run --mock --tool-profile read-only "Review this project without editing files"
```

配置文件:

- `full`:暴露所有内置工具。
- `review`:允许读取、搜索、`git_status`、`git_diff` 和 `run_command`,但禁止文件写入和 patch 应用。
- `read-only`:仅允许 `list_files`、`read_file`、`search_text`、`git_status` 和 `git_diff`。

使用命令配置文件限制 `run_command` 可执行的内容:

```powershell
python -m harness.cli run --mock --command-profile strict "Run safe checks only"
```

- `default`:拒绝已知破坏性命令,对依赖安装、提交、推送和管道下载要求审批。
- `strict`:保留相同的破坏性命令检查,然后拒绝不在狭窄只读/检查白名单内的 shell 命令。

无需运行任务即可查看当前激活的工具范围:

```powershell
python -m harness list-profiles
python -m harness list-tools --tool-profile read-only
python -m harness list-tools --tool-profile read-only --json
```

## 项目配置

`harness.json` 是可选的工作区级配置文件。默认情况下,CLI 查找 `<workspace>/harness.json`,未提供工作区时查找 `./harness.json`。使用 `--config <path>` 指定具体文件。

用以下命令创建初始配置:

```powershell
python -m harness.cli init
```

使用 `--env` 同时生成 `.env` 模板。除非提供 `--force`,否则已有文件会被跳过。

支持的字段:

```json
{
  "provider": "deepseek",
  "model": "deepseek-chat",
  "base_url": "https://api.deepseek.com",
  "workspace": ".",
  "max_steps": 20,
  "timeout_seconds": 60,
  "model_max_retries": 2,
  "model_retry_backoff_seconds": 1,
  "max_tool_output_chars": 12000,
  "max_context_chars": 60000,
  "max_summary_files": 120,
  "max_run_tokens": 0,
  "temperature": 0,
  "approval": "never",
  "tool_profile": "full",
  "command_profile": "default",
  "native_tools": false,
  "json_mode": false,
  "trace_messages": false,
  "trace_model_responses": false
}
```

优先级顺序:CLI 标志 > 环境变量 > `harness.json` > provider 默认值 > 内置默认值。API Key 通常应放在环境变量中,例如 `DEEPSEEK_API_KEY`;`api_key` 仅在受控的本地环境中支持写入配置。

`native_tools` 默认禁用。设为 `true` 时,真实模型请求会包含根据当前激活的工具配置生成的 OpenAI 兼容 `tools`。对于仅能可靠遵循 JSON 动作提示词的 provider,应保持关闭。

`json_mode` 默认禁用。设为 `true` 时,真实模型请求会包含 `response_format: {"type": "json_object"}`,以在兼容 provider 上鼓励严格的 JSON 动作输出。对于拒绝该 OpenAI 兼容选项的 provider,应保持关闭。

模型请求默认会对瞬时失败进行重试:HTTP 408、409、425、429、5xx 以及临时 URL 错误。认证和错误请求会立即失败。

`max_run_tokens` 默认为 `0`,即禁用 token 预算停止。设为大于零时,agent 会在 provider 上报的 `usage.total_tokens` 超过预算后停止。

`trace_messages` 默认为 `false`。模型请求 trace 默认只记录消息数量、角色、字符总数和 SHA-256 哈希;仅在你确实希望将完整提示词消息写入磁盘时才设为 `true`。

`trace_model_responses` 默认为 `false`。模型响应 trace 默认只记录响应长度和 SHA-256 哈希;仅在你确实希望将原始模型动作 JSON 写入磁盘时才设为 `true`。

## Provider 预设

列出内置 provider 预设:

```powershell
python -m harness.cli list-providers
```

使用预设运行:

```powershell
$env:DEEPSEEK_API_KEY="..."
python -m harness.cli run --provider deepseek "Inspect this project"
```

运行前检查 provider 配置:

```powershell
python -m harness.cli doctor --provider deepseek
```

支持的预设名称:

```text
deepseek
qwen
kimi
glm
doubao
```

预设值会填充 `base_url`、默认模型和 provider 专属的 API Key 环境变量。显式的 `HARNESS_BASE_URL`、`HARNESS_MODEL` 和 `HARNESS_API_KEY` 仍会覆盖预设默认值。

## 评估运行器

评估文件为 JSONL。每行一个用例:

```json
{"id":"smoke","task":"Inspect this project","expect_contains":"Offline mock run completed"}
```

`expect_contains` 可以是字符串或字符串列表。当运行正常结束且每个期望字符串都出现在最终内容中时,用例通过。使用 `--json` 获取机器可读报告,使用 `--fail-fast` 在首次失败后停止。

每次评估会在 `.harness/evals/<eval_id>.json` 下写入报告。

```powershell
python -m harness eval --mock examples/smoke.jsonl
python -m harness.cli list-evals
python -m harness.cli show-eval <eval_id>
```

## 运行历史

每次运行会在 `.harness/logs` 下写入有序 JSONL trace。trace 包含一个非敏感的 `run_config` 事件,记录 provider、模型、配置文件、重试设置、流式模式、原生工具模式、提示词/工具 schema 指纹以及适用的恢复来源。除非显式启用 trace 标志,否则它存储的是哈希和计数,而非 API Key、原始提示词文本或原始模型输出。兼容的模型响应元数据(如 `usage`、`finish_reason`、响应 id 和模型名)会以 `model_response_metadata` 形式记录。运行摘要会聚合模型用量和工具用量,包括总工具调用数、失败调用数和按工具统计的计数。运行还会在执行前后捕获工作区文件变更,排除 `.harness`、`.git`、缓存、虚拟环境和依赖目录。紧凑的变更摘要写入 `.harness/changes/<run_id>.json`,并作为 `workspace_changes` trace 事件记录。最新的循环 checkpoint 写入 `.harness/checkpoints/<run_id>.json`。

基于命令的工具还会发出 `command_audit` 事件,包含命令、风险分类、审批决策、返回码和成功标志。这些事件不包含 stdout 和 stderr,因此可以在不重复大段命令输出的情况下检查与安全相关的决策。

列出近期运行:

```powershell
python -m harness.cli list-runs
```

查看单次运行:

```powershell
python -m harness.cli show-run <run_id>
```

查看单次运行的工作区变更:

```powershell
python -m harness.cli show-changes <run_id>
```

查看单次运行最新持久化的消息和计划状态:

```powershell
python -m harness.cli show-checkpoint <run_id>
```

从非完成状态的 checkpoint 恢复:

```powershell
python -m harness.cli resume --mock <run_id>
```

检查类命令可使用 `--json` 获取机器可读输出。`list-runs --json` 在可用时会包含最新的工作区变更摘要以及聚合的模型用量和工具用量。

`run --json` 和 `resume --json` 会打印机器可读的任务结果,包含 `run_id`、`content`、`steps` 和 `changes`。JSON 输出时有意禁用流式,以保持 stdout 为有效 JSON。

## HTTP 服务器

启动轻量级 API 服务器:

```powershell
python -m harness.cli server --host 127.0.0.1 --port 8765
```

打开控制台:

```text
http://127.0.0.1:8765/
```

端点:

```text
GET  /
GET  /console
GET  /health
GET  /providers
GET  /runs?limit=20
GET  /runs/<run_id>
GET  /runs/<run_id>/changes
GET  /runs/<run_id>/checkpoint
GET  /tasks
GET  /tasks/<task_id>
GET  /tasks/<task_id>/events
POST /tasks
```

`POST /tasks` 立即返回一个异步任务记录。任务快照会追加到 `.harness/tasks.jsonl`,因此完成和失败的任务记录在服务器重启后仍然可见。默认使用 mock 运行:

```json
{
  "task": "Inspect this project",
  "mock": true,
  "stream": true
}
```

真实 provider 支持的任务需要显式选择:

```json
{
  "task": "Inspect this project",
  "mock": false,
  "provider": "deepseek",
  "stream": true
}
```

通过异步任务 API 从 checkpoint 恢复:

```json
{
  "resume_from": "20260713T030942Z-2bb0d1a0",
  "mock": true,
  "stream": false
}
```

匹配的 API Key 必须在服务器进程环境中可用,例如 `DEEPSEEK_API_KEY`。使用 `GET /tasks/<task_id>` 轮询状态,`GET /tasks/<task_id>/events` 获取 Server-Sent Events 快照,`GET /runs/<run_id>` 检查 trace,`GET /runs/<run_id>/changes` 检查工作区变更产物,`GET /runs/<run_id>/checkpoint` 检查最新持久化的循环状态。

若服务器在任务仍处于 `queued` 或 `running` 状态时重启,恢复后的任务会被标记为 `failed` 并带有中断错误。框架不会静默恢复部分执行的工作。

## 上下文控制

以下环境变量控制上下文大小:

```text
HARNESS_MAX_CONTEXT_CHARS=60000
HARNESS_MAX_TOOL_OUTPUT_CHARS=12000
HARNESS_MAX_SUMMARY_FILES=120
HARNESS_MAX_RUN_TOKENS=0
HARNESS_MODEL_MAX_RETRIES=2
HARNESS_MODEL_RETRY_BACKOFF_SECONDS=1
HARNESS_APPROVAL=never
HARNESS_TOOL_PROFILE=full
HARNESS_COMMAND_PROFILE=default
HARNESS_NATIVE_TOOLS=false
HARNESS_JSON_MODE=false
HARNESS_TRACE_MESSAGES=false
HARNESS_TRACE_MODEL_RESPONSES=false
```

框架目前使用字符预算而非模型专属 tokenizer,以保持首个实现与 provider 无关。

## 规划协议

对于非平凡任务,模型可在使用工具前先提交计划:

```json
{
  "type": "plan",
  "items": [
    {"id": "1", "content": "Inspect relevant files", "status": "in_progress"},
    {"id": "2", "content": "Make the change", "status": "pending"}
  ]
}
```

之后可更新 todo 状态:

```json
{
  "type": "todo_update",
  "items": [
    {"id": "1", "status": "completed"},
    {"id": "2", "status": "in_progress"}
  ]
}
```

计划状态会记录在 JSONL trace 中,因此可在运行后通过 `show-run` 检查进度决策。

## 模型动作协议

模型必须返回恰好一个 JSON 对象。

工具调用:

```json
{
  "type": "tool_call",
  "tool": "read_file",
  "args": {
    "path": "README.md"
  }
}
```

最终答案:

```json
{
  "type": "final",
  "content": "Task completed."
}
```

框架在路由工具调用前会校验动作结构。若模型返回的 JSON 无效或动作无效,错误会反馈到循环中,以便模型自行恢复。

对于 OpenAI 兼容响应,原生 `tool_calls` 会在 agent 循环处理前被规范化为相同的 `tool_call` 动作结构。流式模式也可将首个流式工具调用 delta 组装为 `tool_call` 动作。若启用 `native_tools`,请求还会将激活的工具 schema 作为 OpenAI 兼容函数工具包含在内,并设置 `tool_choice: "auto"`。

## 项目结构

```text
harness/
  actions.py            动作解析器和校验器
  agent.py              agent 循环
  cli.py                CLI 入口
  config.py             环境变量和 harness.json 加载
  doctor.py             工作区/provider 诊断
  eval_runner.py        JSONL 评估用例解析与断言
  init_project.py       工作区初始化模板
  context.py            仓库摘要和上下文预算管理
  schema.py             轻量级工具参数 schema 校验
  messages.py           消息辅助函数
  models/               模型适配器,包括 mock 模型
  tools/                工具定义、schema 和路由器
  runtime/              工作区、策略、审批、执行、日志、run/task/change/checkpoint 存储
  server.py             轻量级 HTTP API 服务器和内嵌控制台
  prompts/system.md     系统提示词
docs/                   架构说明
tests/                  smoke/单元测试
```

## 设计说明

这有意地不仅仅是一个聊天机器人包装器。重要的工程面在于模型周围的运行时:安全的工作区访问、动作解析、工具路由、shell 策略、审批控制、日志、运行历史、checkpoint、工作区变更追踪、上下文控制以及对失败动作的恢复。

