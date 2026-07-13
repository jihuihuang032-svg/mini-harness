# Architecture

Mini Harness is organized around a strict separation between the model and the runtime.

```text
user task
  -> Context manager
  -> Agent loop
  -> Model adapter
  -> JSON action
  -> Action validator
  -> Tool router
  -> Runtime policy / approval / workspace sandbox
  -> Compressed tool result
  -> Context manager
  -> Agent loop
  -> Ordered run trace
  -> Workspace change summary
  -> Run store
  -> Optional HTTP API
```

## Core Modules

### Agent Loop

`harness.agent.Agent` owns the task loop:

1. Build the initial messages from the system prompt, runtime tool schemas, repository summary, and user task.
2. Ask the context manager to trim messages for the current budget.
3. Call the model adapter.
4. Parse and validate the assistant response as a JSON action.
5. Execute valid tool calls through the tool router.
6. Compress tool results before appending them back into the conversation.
7. Feed invalid action errors back to the model for recovery.
8. Stop on `final` or after `max_steps`.

The agent does not directly read files, run commands, or know provider-specific API details.

### Context Manager

`harness.context.ContextManager` keeps the conversation within a practical budget.

Current responsibilities:

- Build a repository summary from sampled workspace files.
- Ignore noisy directories such as `.git`, `.harness`, `__pycache__`, `node_modules`, and virtual environments.
- Inject repository context into the initial messages.
- Keep system prompt, repository context, and original task at the head of the message list.
- Prefer recent messages when trimming older conversation history.
- Compress large tool results before they are logged and sent back to the model.

The current implementation uses character budgets. A future version can swap in provider-specific token counting without changing the agent loop.

### Planning Layer

`harness.planning.PlanState` stores the current todo list for a run.

The model can emit:

- `plan`: replace the current todo list with a short execution plan.
- `todo_update`: update status or content for existing todo items.

Allowed statuses are `pending`, `in_progress`, `completed`, and `blocked`.

Plan snapshots are included in `plan_updated`, `tool_result`, and `final` trace events. This gives later UI or service layers a structured progress model without parsing prose.

### Action Protocol

`harness.actions` defines the portable action protocol used by models without native function calling.

Supported actions:

- `tool_call`: execute one named tool with object args.
- `final`: finish the run with a human-readable summary.

`action_from_openai_message()` normalizes OpenAI-compatible native `tool_calls` into the same `tool_call` action shape. Keeping action parsing separate from the loop means native tool calling and JSON action prompts share the same tool router and runtime policy path.

### Model Adapter

`harness.models.openai_compatible.OpenAICompatibleClient` calls `/chat/completions` on an OpenAI-compatible endpoint. `harness.models.providers` stores provider presets for common compatible APIs, including base URL, default model, and provider-specific API key environment variable.

This keeps domestic model providers replaceable through configuration:

- `HARNESS_BASE_URL`
- `HARNESS_API_KEY`
- `HARNESS_MODEL`
- `HARNESS_PROVIDER`

The same values can be declared in a workspace-level `harness.json`. Runtime precedence is CLI flags, environment variables, `harness.json`, provider defaults, and then built-in defaults. API keys are intentionally environment-first so project config files can be committed without secrets.

`harness.models.mock.MockModelClient` is a deterministic offline model. It is used by `--mock` and tests to prove the harness loop works without an external API.

`OpenAICompatibleClient.complete()` accepts either plain text JSON actions or native `tool_calls` from compatible `/chat/completions` endpoints. `OpenAICompatibleClient.stream_complete()` supports Server-Sent Events for text chunks and can assemble the first streamed tool-call delta into the same JSON action. The agent records each emitted chunk as `model_stream_chunk`, then parses the assembled response as the normal JSON action. Future adapters can support provider-specific response formats without changing tool code.

The adapter stores non-sensitive response metadata on `last_response_metadata` after each request. The agent records it as `model_response_metadata` when present, including fields such as response id, model, finish reason, and provider-reported usage. This keeps token/cost analysis separate from the assistant content.

When `native_tools` is enabled, the client also converts the active `ToolRouter.specs()` list into an OpenAI-compatible `tools` request payload and sets `tool_choice` to `auto`. The switch is off by default because some compatible domestic endpoints are more reliable with the prompt-level JSON action protocol than with native function calling.

When `json_mode` is enabled, the client adds `response_format: {"type": "json_object"}` to compatible chat-completions requests. This keeps the default broad compatibility path unchanged while giving providers that support JSON mode a stricter output channel for the harness action protocol.

The adapter retries transient model API failures before surfacing an error to the agent loop. Retryable failures include HTTP 408, 409, 425, 429, 5xx, and temporary URL errors. Non-retryable provider errors such as invalid credentials fail immediately. Retry count and backoff are controlled by `model_max_retries` and `model_retry_backoff_seconds`.

### Tool Router

`harness.tools.base.ToolRouter` maps a model action to a concrete tool implementation.

Each tool exposes:

- `name`
- `description`
- `args_schema`
- `run(args)`

The agent injects `ToolRouter.specs()` into the system prompt, so the prompt and actual tool registry stay aligned. The same filtered specs are reused for native tools payloads when that mode is enabled.

Tool arguments are validated by `harness.schema.validate_args` before execution. The current validator supports the JSON Schema subset used by local tools: object roots, properties, required fields, defaults, and primitive type checks. Tools return structured dictionaries instead of free-form text. This keeps the next model turn easier to reason about and makes logs easier to inspect.

`read_file` supports `start_line`, `max_lines`, and optional line-number rendering so the agent can inspect large files in stable chunks instead of relying only on whole-file truncation.

`search_text` supports literal and regex matching, case-sensitivity control, maximum match limits, and optional surrounding line context. This keeps code search useful while bounding result size before it re-enters the model context.

Tool profiles constrain both tool exposure and execution:

- `full`: expose every built-in tool.
- `review`: expose read/search tools, `git_status`, `git_diff`, and `run_command`, but not file mutation tools.
- `read-only`: expose only `list_files`, `read_file`, `search_text`, `git_status`, and `git_diff`.

If a model calls a tool outside the active profile anyway, the router returns a structured denial instead of executing it. The active profile is recorded as a `tool_profile` trace event.

### Runtime

The runtime layer is responsible for safety and observability:

- `Workspace` resolves all paths and blocks path escapes.
- `CommandPolicy` classifies commands as `low`, `approval_required`, or `denied`.
- `ApprovalController` decides whether `approval_required` commands are allowed.
- `CommandExecutor` runs approved commands in the workspace with timeout and output truncation.
- `RunLogger` writes ordered JSONL traces under `.harness/logs`.
- `RunStore` lists, loads, and summarizes JSONL traces for CLI history commands, including aggregated model usage when providers report token counts and aggregated tool usage from `tool_result` events.
- `TaskStore` appends async task snapshots to `.harness/tasks.jsonl` and restores the latest snapshot for each task id.
- `WorkspaceChangeTracker` captures before/after file states and writes compact per-run summaries under `.harness/changes`.
- `RunCheckpointStore` writes the latest task, step, messages, status, and plan state under `.harness/checkpoints`.

Approval modes:

- `never`: default; refuse commands that require approval.
- `on-request`: prompt the CLI user before running approval-required commands.
- `auto`: approve automatically for tests or trusted demos.

Command policy profiles:

- `default`: deny known destructive commands and require approval for dependency changes, commits, pushes, and piped downloads.
- `strict`: apply default risky-pattern checks, then deny commands outside a narrow read-only/check allowlist such as `git diff`, `git status`, `python -m unittest`, `python -m compileall`, directory listing, and `rg`.

Each trace record includes `schema_version`, `seq`, `ts`, `run_id`, `kind`, and `payload` so later service/UI layers can replay runs in order.

Run entry points record a non-secret `run_config` event near the start of each trace. It captures provider, model, workspace, profiles, retry settings, native tool mode, JSON mode, stream mode, tool count, tool names, prompt/tool-schema SHA-256 fingerprints, max run token budget, and resume source. API keys and raw prompt text are intentionally excluded.

`model_request` events omit full prompt messages by default. They record message count, roles, character total, and a SHA-256 hash so requests can be compared without writing complete prompts to disk. Setting `trace_messages` to `true` stores full request messages for explicit debugging sessions.

Command-backed tool results include command risk and approval data. The agent also emits a separate `command_audit` event with command, risk classification, approval decision, return code, and success flag, excluding stdout and stderr to keep safety audit records compact.

The agent records a `run_finished` lifecycle event for terminal outcomes:

- `completed`: the model returned `final`.
- `stopped`: the loop reached `max_steps` or provider-reported token usage exceeded `max_run_tokens`.
- `failed`: an unexpected exception escaped the loop.

`RunStore` uses this event for run history status, while still treating older traces with a `final` event as completed.

Run entry points record a `workspace_snapshot` event before the agent loop and a `workspace_changes` event after it finishes or raises. Change tracking does not require a git repository; it hashes workspace files directly while ignoring `.harness`, `.git`, caches, virtual environments, and dependency directories.

The agent also writes a checkpoint after initial context creation and after each meaningful loop transition. Checkpoints are separate from JSONL traces: traces are append-only audit events, while checkpoints are compact latest-state artifacts for inspection and resume support. A resumed run gets a new run id and records a `resumed_from` trace event instead of overwriting the source run.

### CLI Surface

`harness.cli` exposes these commands:

- `doctor`: validate workspace, config, provider, API-key, approval, and tool-profile settings without calling a model.
- `eval`: run JSONL benchmark cases through the same harness execution path, report pass/fail results, and persist an eval report under `.harness/evals`.
- `init`: create starter `harness.json` and optional `.env` files.
- `run`: execute a coding-agent task.
- `resume`: continue from a non-completed run checkpoint using a new run id.
- `list-runs`: show recent traces in `.harness/logs`.
- `list-evals`: show recent eval reports in `.harness/evals`.
- `show-eval`: inspect one eval report by id.
- `show-run`: inspect one trace by run id.
- `show-changes`: inspect the per-run workspace change artifact.
- `show-checkpoint`: inspect the latest persisted agent-loop state for a run.

The legacy form `python -m harness.cli --mock "task"` is preserved by routing it to `run` internally.
Use `run --stream` to print model chunks as they arrive while preserving the same action loop. `run` prints the final changed-file count, and `list-runs` includes changed-file, tool-call, and token counts in the history table.
All workspace-aware commands accept `--config` to load a specific JSON config file. Without it, config is discovered as `<workspace>/harness.json` or `./harness.json`.

### HTTP API

`harness.server` provides a dependency-free HTTP API based on `ThreadingHTTPServer`.

Endpoints:

- `GET /` and `GET /console`: embedded web console.
- `GET /health`: server health and workspace path.
- `GET /providers`: built-in provider presets.
- `GET /runs?limit=20`: recent run summaries.
- `GET /runs/{run_id}`: raw ordered trace records.
- `GET /runs/{run_id}/changes`: compact workspace change artifact for the run.
- `GET /runs/{run_id}/checkpoint`: latest persisted agent-loop checkpoint.
- `GET /tasks`: async task records restored from `.harness/tasks.jsonl`.
- `GET /tasks/{task_id}`: one async task record.
- `GET /tasks/{task_id}/events`: Server-Sent Events containing task status snapshots, trace records, and a final `done` event.
- `POST /tasks`: submit a mock task by default, a provider-backed task when `mock:false` and `provider` are supplied, or a checkpoint resume task when `resume_from` is supplied, then immediately return task id plus run id.

`harness.runtime.task_queue.TaskQueue` runs submitted tasks in background daemon threads and persists each status transition through `TaskStore`. On server startup, completed and failed records are restored as-is. Records left in `queued` or `running` are converted to `failed` with an interruption error because the previous process owned their execution state. Provider-backed tasks are explicit opt-in and rely on environment-provided API keys. This keeps the default server behavior safe for local demos while allowing real model execution when configured.

The embedded console is plain HTML/CSS/JavaScript served from the same process. It submits mock tasks, follows task SSE events, and renders task/run/trace timelines without a build step.

## Safety Boundaries

The current safety boundary is practical, not absolute:

- File paths must stay inside the workspace.
- Shell commands run with the current user permissions.
- Destructive commands are denied.
- Dependency installs, git commits, git pushes, and piped downloads require approval.
- Tool arguments are schema-validated before execution.
- Tool output is truncated before it is fed back to the model.
- The loop has a maximum step count.
- Context size is capped before each model request.

Next hardening steps:

- Add per-tool permission declarations.
- Add sandbox backends for subprocess isolation.

## Roadmap

### Milestone 1: CLI Harness

- JSON action loop
- OpenAI-compatible API
- Core coding tools
- Workspace sandbox
- Logs and tests

### Milestone 2: Better Context Management

- Repository summary
- File relevance ranking
- Tool result compression
- Conversation compaction
- Token budget tracking

### Milestone 3: Service Mode

- FastAPI server
- Run/session database
- Async task execution
- WebSocket step streaming
- API for runs, steps, logs, and diff review

### Milestone 4: Web Console

- Task submission
- Live step timeline
- Tool result viewer
- Diff viewer
- Approval prompts
- Model/provider settings

### Milestone 5: Advanced Agent Features

- Native tool calling adapter
- Multi-model fallback
- Planner/executor split
- Eval harness for benchmark tasks
- MCP-style external tools





