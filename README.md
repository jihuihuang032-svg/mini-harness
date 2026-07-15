# Mini Harness

Mini Harness 是一个轻量但形态完整的 coding-agent harness。它不是一个聊天机器人，也不是一份 prompt 文档，而是包在大模型外面的执行系统：接收任务、组织上下文、调用模型、解析动作、路由工具、记录 trace，并在安全策略约束下循环推进任务。

这个项目的目标是用尽量少的依赖实现一个可运行、可展示、可继续扩展的 coding agent harness。第一版面向 OpenAI-compatible API，因此可以接 DeepSeek、Qwen、Kimi、GLM、豆包等国内大模型。

## 核心能力

- CLI 入口：`run`、`resume`、`preview-run`、`doctor`、`init`、`eval`、`list-runs`、`show-run`、`show-changes`、`show-checkpoint`、`list-tools`、`list-providers`、`server`
- OpenAI-compatible chat completions 模型适配层
- 内置 provider preset：`deepseek`、`qwen`、`kimi`、`glm`、`doubao`
- 离线 mock 模型，方便无 API Key 演示和测试
- JSON action 协议，适合不支持原生 tool calling 的模型
- 可选 OpenAI-compatible native `tool_calls`
- 可选 `response_format: {"type": "json_object"}` JSON mode
- agent loop：模型输出 action，harness 执行工具，再把结果反馈给模型
- 规划动作：`plan`、`todo_update`
- 工具参数校验、默认值处理、错误反馈
- 工作区路径沙箱，所有文件访问限制在 workspace 内
- shell 命令风险评估和审批模式
- 工具权限 profile：`full`、`review`、`read-only`
- 命令策略 profile：`default`、`strict`
- 运行配置预览：不用调用模型即可查看一次 run 会启用哪些配置和工具
- JSONL trace：记录 action、tool result、token usage、workspace changes、checkpoint
- run checkpoint：支持从未完成 run 恢复
- eval runner：用 JSONL 用例回归测试 harness 行为
- 轻量 HTTP API 和内嵌 Web console

## 项目结构

```text
mini-harness/
  harness/
    cli.py                  # CLI 参数解析和命令分发
    agent.py                # agent loop
    actions.py              # 模型 action 解析和校验
    config.py               # 环境变量 / harness.json / provider preset 配置加载
    context.py              # workspace 摘要和上下文预算控制
    server.py               # HTTP API + Web console
    models/
      base.py
      mock.py
      openai_compatible.py
      providers.py
    tools/
      base.py
      file_tools.py
      shell_tools.py
      git_tools.py
    runtime/
      workspace.py          # 路径沙箱
      policy.py             # shell 风险策略
      approval.py           # 审批决策
      executor.py           # 命令执行器
      logger.py             # JSONL trace
      checkpoint.py         # run checkpoint
      run_store.py          # run 查询
      run_config.py         # 非敏感运行配置快照
    prompts/
      system.md             # 注入给模型的系统提示词
  tests/
  examples/
  harness.json.example
  .env.example
```

## 快速开始

无需 API Key，可以先跑离线 mock：

```powershell
python -m harness --mock "Inspect this project"
```

等价的显式命令：

```powershell
python -m harness.cli run --mock "Inspect this project"
```

如果安装为包，可以使用控制台命令：

```powershell
pip install -e .
mini-harness --version
mini-harness run --mock "Inspect this project"
```

## 初始化配置

在一个 workspace 里生成配置模板：

```powershell
python -m harness.cli init --workspace E:\some-project --env
```

会生成：

- `harness.json`：workspace 级配置
- `.env`：环境变量模板

不想写入文件时，可以只做配置检查：

```powershell
python -m harness.cli doctor --workspace E:\some-project --mock
```

## 接国内大模型

推荐先用 provider preset：

```powershell
$env:HARNESS_PROVIDER="deepseek"
$env:DEEPSEEK_API_KEY="你的 key"
python -m harness.cli run --provider deepseek "Read this project and summarize the architecture"
```

支持的 provider：

```powershell
python -m harness.cli list-providers
```

也可以用通用 OpenAI-compatible 配置：

```powershell
$env:HARNESS_BASE_URL="https://api.deepseek.com"
$env:HARNESS_MODEL="deepseek-chat"
$env:HARNESS_API_KEY="你的 key"
python -m harness.cli run "Inspect this project"
```

常见国内 provider 的 key 环境变量：

- DeepSeek：`DEEPSEEK_API_KEY`
- Qwen / DashScope：`DASHSCOPE_API_KEY`
- Kimi / Moonshot：`MOONSHOT_API_KEY`
- GLM / Zhipu：`ZHIPU_API_KEY`
- 豆包 / Ark：`ARK_API_KEY`

## 运行前预览配置

`preview-run` 不调用模型，只展示真实 run 会使用的非敏感配置、工具范围和 schema hash。这个命令适合在真实消耗 token 之前检查配置：

```powershell
python -m harness.cli preview-run --mock --tool-profile read-only
python -m harness.cli preview-run --provider deepseek --json
```

输出中不会包含 API Key 或完整 system prompt，只包含 hash、长度、工具名和非敏感配置。

## 常用 CLI 命令

从文件读取较长任务：

```powershell
python -m harness.cli run --mock --task-file task.md
```

输出机器可读 JSON：

```powershell
python -m harness.cli run --mock --json "Inspect this project"
```

流式打印模型输出块：

```powershell
python -m harness.cli run --mock --stream "Inspect this project"
```

控制单次运行成本：

```powershell
python -m harness.cli run --provider deepseek --max-steps 6 --max-run-tokens 20000 "Inspect this project"
```

查看工具配置：

```powershell
python -m harness.cli list-profiles
python -m harness.cli list-tools --tool-profile read-only
python -m harness.cli list-tools --tool-profile read-only --json
```

查看 run 历史：

```powershell
python -m harness.cli list-runs
python -m harness.cli show-run <run_id>
python -m harness.cli show-run --summary <run_id>
python -m harness.cli show-changes <run_id>
python -m harness.cli show-checkpoint <run_id>
```

从 checkpoint 恢复：

```powershell
python -m harness.cli resume --mock <run_id>
```

## 工具系统

内置工具：

- `list_files`：列出 workspace 内文件
- `read_file`：读取文件，支持行范围
- `write_file`：写入文件，受路径沙箱限制
- `search_text`：文本搜索，支持字面量 / 正则、匹配数量限制和上下文
- `apply_patch`：应用 unified diff patch
- `run_command`：执行 shell 命令，经过命令策略和审批控制
- `git_status`：查看 git 状态
- `git_diff`：查看 git diff

工具 profile：

- `full`：暴露所有内置工具
- `review`：允许读、搜、git 检查和命令执行，禁止写文件和 patch
- `read-only`：只允许 `list_files`、`read_file`、`search_text`、`git_status`、`git_diff`

命令 profile：

- `default`：拒绝已知破坏性命令，对依赖安装、提交、推送和管道下载要求审批
- `strict`：在 default 基础上，只允许窄范围的只读 / 检查命令

审批模式：

- `never`：默认模式，遇到需要审批的命令直接拒绝
- `on-request`：交互式询问用户是否批准
- `auto`：自动批准，仅建议用于本地测试或可信演示

示例：

```powershell
python -m harness.cli run --mock --tool-profile read-only "Review this project without editing files"
python -m harness.cli run --mock --command-profile strict "Run safe checks only"
python -m harness.cli run --approval on-request "Run checks and fix failures"
```

## 配置文件

`harness.json` 是可选的 workspace 级配置。CLI 默认查找 `<workspace>/harness.json`，也可以用 `--config` 指定：

```powershell
python -m harness.cli run --config harness.json --mock "Inspect this project"
```

示例字段见 `harness.json.example`：

```json
{
  "provider": "deepseek",
  "model": "deepseek-chat",
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

优先级：CLI 参数 > 环境变量 > `harness.json` > provider preset > 内置默认值。

`max_run_tokens` 默认为 `0`，表示不按 provider 上报 token 预算停止。设置为大于 0 后，agent 会在累计 `usage.total_tokens` 超过预算时停止。

`trace_messages` 和 `trace_model_responses` 默认关闭，避免把完整提示词或模型原始响应写入磁盘。需要调试时再开启。

## 模型动作协议

如果模型不支持原生 tool calling，Mini Harness 会要求模型输出 JSON action。

调用工具：

```json
{
  "type": "tool_call",
  "tool": "read_file",
  "args": {
    "path": "README.md"
  }
}
```

完成任务：

```json
{
  "type": "final",
  "content": "任务完成说明"
}
```

规划任务：

```json
{
  "type": "plan",
  "steps": ["阅读项目", "修改代码", "运行测试"]
}
```

## Eval 回归测试

eval 文件是 JSONL，每行一个用例。示例见 `examples/smoke.jsonl`。

```powershell
python -m harness.cli eval --mock examples/smoke.jsonl
python -m harness.cli list-evals
python -m harness.cli show-eval <eval_id>
```

## HTTP API 和 Web Console

启动本地服务：

```powershell
python -m harness.cli server --workspace . --host 127.0.0.1 --port 8765
```

浏览器打开：

```text
http://127.0.0.1:8765/
```

运行前预览 server 当前配置：

```text
http://127.0.0.1:8765/preview-run?mock=true&stream=false
```
主要接口：

- `GET /health`
- `GET /providers`
- `GET /preview-run`
- `GET /runs?limit=20`
- `GET /runs/<run_id>`
- `GET /runs/<run_id>/changes`
- `GET /runs/<run_id>/checkpoint`
- `GET /tasks`
- `GET /tasks/<task_id>`
- `GET /tasks/<task_id>/events`
- `POST /tasks`

提交 mock 任务：

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8765/tasks -ContentType 'application/json' -Body '{"task":"Inspect this project","mock":true}'
```

提交真实模型任务：

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8765/tasks -ContentType 'application/json' -Body '{"task":"Inspect this project","mock":false,"provider":"deepseek"}'
```

## 测试

运行全量测试：

```powershell
python -m unittest discover -s tests
```

编译检查：

```powershell
python -m compileall harness tests
```

当前测试覆盖 CLI、agent loop、配置加载、工具沙箱、patch、checkpoint、eval、server、streaming、trace 和任务队列。

## 安全边界

Mini Harness 是本地执行系统，不应该直接暴露到公网或给不可信用户使用。默认安全边界包括：

- 文件路径限制在 workspace 内
- `.env`、`harness.json`、`.harness/` 默认被 `.gitignore` 忽略
- API Key 不写入 run config trace
- shell 命令经过风险策略判断
- 默认 `approval=never`，需要审批的命令不会自动执行
- 工具输出会截断，避免上下文被超长结果撑爆

## 开发路线

这个项目当前已经可用，后续适合继续增强：

- 把 CLI 和 server 的 runtime 构建逻辑进一步抽到公共运行时模块
- 增加 server 侧的工具预览和配置预览接口
- 增强 Web console 的 run 详情和 checkpoint 恢复体验
- 增加更多 provider 兼容性测试
- 增加浏览器工具、IDE 集成或多 agent 编排