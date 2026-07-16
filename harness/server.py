"""HTTP server: 把 harness 暴露为 REST API 和 SSE 流式接口。

接口概览:
    GET  /                       -> 内置 console HTML 单页可视化
    GET  /health                 -> 健康检查
    GET  /providers              -> 列出内置 provider 预设
    GET  /runs                   -> 列出最近 run
    GET  /runs/<id>              -> 加载某个 run 的完整事件流
    GET  /runs/<id>/changes      -> 加载某个 run 的文件变更
    GET  /runs/<id>/checkpoint   -> 加载某个 run 的最新 checkpoint
    GET  /tasks                  -> 列出所有任务
    GET  /tasks/<id>             -> 查询单个任务
    GET  /tasks/<id>/events      -> SSE 流式订阅任务进度
    POST /tasks                  -> 提交新任务、继续追问或恢复运行

实现基于 Python 标准库 http.server, 不依赖额外 Web 框架。
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from harness.agent import Agent, render_system_prompt
from harness.config import HarnessConfig
from harness.messages import Message
from harness.models.mock import MockModelClient
from harness.models.openai_compatible import OpenAICompatibleClient
from harness.models.providers import PROVIDER_PRESETS, provider_names
from harness.runtime.approval import ApprovalController
from harness.runtime.change_tracker import WorkspaceChangeTracker
from harness.runtime.checkpoint import RunCheckpointStore
from harness.runtime.executor import CommandExecutor
from harness.runtime.logger import RunLogger
from harness.runtime.policy import CommandPolicy
from harness.runtime.run_config import run_config_snapshot
from harness.runtime.run_store import RunStore
from harness.runtime.task_queue import TaskQueue
from harness.runtime.task_store import TaskStore
from harness.runtime.workspace import Workspace
from harness.tools import build_default_router

@dataclass(frozen=True)
class ServerRuntime:
    """Runtime dependencies shared by server preview and task execution."""

    config: HarnessConfig
    approval: ApprovalController
    executor: CommandExecutor
    router: object
    tool_specs: list[dict[str, object]]
    system_prompt: str


class HarnessServer:
    """HTTP server 的应用层: 封装业务方法, Handler 只负责 HTTP 协议。

    把业务逻辑从 HTTP 解析中拆出来, 便于测试直接调用 app.run_mock_task 等方法。
    """
    def __init__(self, workspace: str | None = None, config_path: str | None = None) -> None:
        self.config = HarnessConfig.offline(workspace, config_path)
        self.config_path = config_path
        self.workspace = Workspace(self.config.workspace)
        self.store = RunStore(self.workspace.logs_dir)
        self.checkpoints = RunCheckpointStore(self.workspace.checkpoints_dir)
        self.task_store = TaskStore(self.workspace.tasks_path)
        self.tasks = TaskQueue(self.run_task, store=self.task_store)

    def api_map(self) -> list[dict[str, object]]:
        """Return a human-readable map of HTTP endpoints exposed by this server."""
        return [
            {
                "method": "GET",
                "path": "/",
                "purpose": "打开内置 Web 控制台页面。",
                "params": [],
                "response": "text/html 控制台页面",
                "java_analogy": "类似 @GetMapping(\"/\") 返回前端页面。",
            },
            {
                "method": "GET",
                "path": "/health",
                "purpose": "检查本地后端和 workspace 是否可用。",
                "params": [],
                "response": "{ ok, workspace }",
                "java_analogy": "类似 health check Controller。",
            },
            {
                "method": "GET",
                "path": "/providers",
                "purpose": "列出 DeepSeek、Qwen 等可选模型供应商预设。",
                "params": [],
                "response": "ProviderPreset[]",
                "java_analogy": "类似查询下拉选项的 @GetMapping。",
            },
            {
                "method": "GET",
                "path": "/preview-run",
                "purpose": "预览一次运行会使用的工具、策略、模型和 prompt 摘要, 不真正执行任务。",
                "params": ["mock:boolean", "stream:boolean", "provider:string?"],
                "response": "RunPreview",
                "java_analogy": "类似运行前的配置预检接口。",
            },
            {
                "method": "GET",
                "path": "/runs?limit=20",
                "purpose": "读取最近保存过的 agent run 列表。",
                "params": ["limit:int"],
                "response": "RunSummary[]",
                "java_analogy": "类似分页查询历史任务。",
            },
            {
                "method": "GET",
                "path": "/runs/{run_id}",
                "purpose": "读取某次 run 的完整 trace 事件流, 中间时间线主要来自这里。",
                "params": ["run_id:path"],
                "response": "TraceEvent[]",
                "java_analogy": "类似根据 ID 查询任务执行明细。",
            },
            {
                "method": "GET",
                "path": "/runs/{run_id}/changes",
                "purpose": "读取某次 run 对 workspace 造成的文件变更。",
                "params": ["run_id:path"],
                "response": "WorkspaceChanges",
                "java_analogy": "类似查询任务产物或变更清单。",
            },
            {
                "method": "GET",
                "path": "/runs/{run_id}/checkpoint",
                "purpose": "读取某次 run 保存的上下文检查点, 用于恢复或继续追问。",
                "params": ["run_id:path"],
                "response": "RunCheckpoint",
                "java_analogy": "类似查询可恢复的任务快照。",
            },
            {
                "method": "GET",
                "path": "/tasks",
                "purpose": "读取异步任务列表。",
                "params": [],
                "response": "TaskRecord[]",
                "java_analogy": "类似查询任务队列。",
            },
            {
                "method": "GET",
                "path": "/tasks/{task_id}",
                "purpose": "读取单个异步任务状态和对应 run 摘要。",
                "params": ["task_id:path"],
                "response": "TaskRecord",
                "java_analogy": "类似根据 ID 查询任务状态。",
            },
            {
                "method": "GET",
                "path": "/tasks/{task_id}/events",
                "purpose": "用 SSE 流式推送任务状态和 trace 事件, 页面实时刷新依赖它。",
                "params": ["task_id:path", "timeout:float"],
                "response": "text/event-stream",
                "java_analogy": "类似 Spring SseEmitter 或 WebFlux stream。",
            },
            {
                "method": "POST",
                "path": "/tasks",
                "purpose": "提交新任务、恢复历史 run, 或基于历史 run 继续追问。",
                "params": ["task:string?", "resume_from:string?", "follow_up:string?", "mock:boolean", "provider:string?", "stream:boolean"],
                "response": "202 TaskRecord",
                "java_analogy": "类似 @PostMapping 接收 @RequestBody 创建异步任务。",
            },
            {
                "method": "GET",
                "path": "/api-map",
                "purpose": "返回这份接口地图, 供页面展示和学习。",
                "params": [],
                "response": "ApiEndpoint[]",
                "java_analogy": "类似简化版 Swagger/OpenAPI 描述。",
            },
        ]
    def health(self) -> dict[str, object]:
        return {"ok": True, "workspace": str(self.workspace.root)}

    def console_html(self) -> str:
        return CONSOLE_HTML.replace("__WORKSPACE__", str(self.workspace.root))

    def providers(self) -> list[dict[str, str]]:
        return [PROVIDER_PRESETS[name].to_dict() for name in provider_names()]

    def _build_runtime(self, mock: bool = True, provider: str | None = None, trace_model_responses: bool | None = None) -> ServerRuntime:
        config = self.config if mock else HarnessConfig.from_env(str(self.workspace.root), provider, self.config_path)
        if trace_model_responses is not None:
            config = replace(config, trace_model_responses=trace_model_responses)
        approval = ApprovalController("never")
        executor = CommandExecutor(
            self.workspace,
            CommandPolicy.default(config.command_profile),
            config.timeout_seconds,
            config.max_tool_output_chars,
            approval,
        )
        router = build_default_router(
            self.workspace,
            executor,
            config.max_tool_output_chars,
            tool_profile=config.tool_profile,
        )
        tool_specs = router.specs()
        return ServerRuntime(
            config=config,
            approval=approval,
            executor=executor,
            router=router,
            tool_specs=tool_specs,
            system_prompt=render_system_prompt(router),
        )

    def _run_config_payload(
        self,
        runtime: ServerRuntime,
        *,
        mock: bool,
        stream: bool,
        resume_from: str | None = None,
    ) -> dict[str, object]:
        return run_config_snapshot(
            runtime.config,
            mode="mock" if mock else "model",
            stream=stream,
            tool_profile=runtime.config.tool_profile,
            command_profile=runtime.config.command_profile,
            approval=runtime.approval.mode,
            tool_specs=runtime.tool_specs,
            system_prompt=runtime.system_prompt,
            resume_from=resume_from,
        )

    def preview_run(self, mock: bool = True, provider: str | None = None, stream: bool = False, trace_model_responses: bool | None = None) -> dict[str, object]:
        runtime = self._build_runtime(mock=mock, provider=provider, trace_model_responses=trace_model_responses)
        return {
            **self._run_config_payload(runtime, mock=mock, stream=stream),
            "tools": runtime.tool_specs,
        }

    def list_runs(self, limit: int = 20) -> list[dict[str, object]]:
        return [summary.to_dict() for summary in self.store.list_runs(limit=limit)]

    def load_run(self, run_id: str) -> list[dict[str, Any]]:
        return self.store.load_run(run_id)

    def load_run_changes(self, run_id: str) -> dict[str, object]:
        return self.store.load_changes(run_id)

    def load_run_checkpoint(self, run_id: str) -> dict[str, object]:
        return self.checkpoints.load(run_id)

    def list_tasks(self) -> list[dict[str, object]]:
        return [task.to_dict() for task in self.tasks.list()]

    def load_task(self, task_id: str) -> dict[str, object]:
        task = self.tasks.get(task_id).to_dict()
        task["run_summary"] = self._run_summary_if_exists(str(task["run_id"]))
        return task

    def submit_mock_task(self, task: str, stream: bool = False, trace_model_responses: bool = False) -> dict[str, object]:
        if not task:
            raise ValueError("Task is required.")
        return self.tasks.submit(task, stream=stream, mode="mock", metadata={"mock": True, "trace_model_responses": trace_model_responses}).to_dict()

    def submit_model_task(self, task: str, provider: str | None, stream: bool = False, trace_model_responses: bool = False) -> dict[str, object]:
        if not task:
            raise ValueError("Task is required.")
        if not provider:
            raise ValueError("Field 'provider' is required when mock=false.")
        return self.tasks.submit(
            task,
            stream=stream,
            mode="model",
            provider=provider,
            metadata={"mock": False, "provider": provider, "trace_model_responses": trace_model_responses},
        ).to_dict()

    def submit_resume_task(
        self,
        resume_from: str,
        mock: bool = True,
        provider: str | None = None,
        stream: bool = False,
        trace_model_responses: bool = False,
    ) -> dict[str, object]:
        checkpoint = self.checkpoints.load_state(resume_from)
        if checkpoint.status == "completed":
            raise ValueError(f"Checkpoint is already completed: {resume_from}")
        if mock is False and not provider:
            raise ValueError("Field 'provider' is required when mock=false.")
        metadata: dict[str, object] = {"mock": mock, "resume_from": resume_from, "trace_model_responses": trace_model_responses}
        if provider is not None:
            metadata["provider"] = provider
        return self.tasks.submit(
            checkpoint.task,
            stream=stream,
            mode="model" if mock is False else "mock",
            provider=provider,
            metadata=metadata,
        ).to_dict()

    def submit_followup_task(
        self,
        source_run_id: str,
        follow_up: str,
        mock: bool = True,
        provider: str | None = None,
        stream: bool = False,
        trace_model_responses: bool = False,
    ) -> dict[str, object]:
        if not follow_up.strip():
            raise ValueError("Field 'follow_up' is required.")
        self.checkpoints.load_state(source_run_id)
        if mock is False and not provider:
            raise ValueError("Field 'provider' is required when mock=false.")
        metadata: dict[str, object] = {"mock": mock, "source_run_id": source_run_id, "follow_up": follow_up, "trace_model_responses": trace_model_responses}
        if provider is not None:
            metadata["provider"] = provider
        return self.tasks.submit(
            follow_up,
            stream=stream,
            mode="model" if mock is False else "mock",
            provider=provider,
            metadata=metadata,
        ).to_dict()
    def task_events(
        self,
        task_id: str,
        poll_interval: float = 0.1,
        timeout_seconds: float = 30.0,
    ) -> list[dict[str, object]]:
        deadline = time.time() + timeout_seconds
        emitted_sequences: set[int] = set()
        events: list[dict[str, object]] = []
        while time.time() < deadline:
            task = self.load_task(task_id)
            events.append({"event": "task", "data": task})
            for record in self._load_run_if_exists(str(task["run_id"])):
                seq = record.get("seq")
                if isinstance(seq, int) and seq not in emitted_sequences:
                    emitted_sequences.add(seq)
                    events.append({"event": "trace", "data": record})
            if task["status"] in {"completed", "failed"}:
                events.append({"event": "done", "data": task})
                return events
            time.sleep(poll_interval)
        events.append({"event": "timeout", "data": self.load_task(task_id)})
        return events

    def run_task(self, task: str, stream: bool, run_id: str, metadata: dict[str, object]) -> dict[str, object]:
        """TaskQueue 的 runner 回调: 根据 metadata 分发到 mock、model 或追问流程。

        这个方法会在后台线程执行, 负责把异步任务连接到实际 agent run。
        """
        follow_up = metadata.get("follow_up")
        source_run_id = metadata.get("source_run_id")
        trace_model_responses = bool(metadata.get("trace_model_responses", False))
        if isinstance(follow_up, str) and isinstance(source_run_id, str):
            if metadata.get("mock", True) is False:
                provider = metadata.get("provider")
                provider_name = provider if isinstance(provider, str) else None
                return self.run_model_followup(
                    follow_up,
                    source_run_id,
                    provider_name,
                    stream=stream,
                    run_id=run_id,
                    trace_model_responses=trace_model_responses,
                )
            return self.run_mock_followup(follow_up, source_run_id, stream=stream, run_id=run_id, trace_model_responses=trace_model_responses)
        resume_from = metadata.get("resume_from")
        resume_run_id = resume_from if isinstance(resume_from, str) else None
        if metadata.get("mock", True) is False:
            provider = metadata.get("provider")
            provider_name = provider if isinstance(provider, str) else None
            return self.run_model_task(task, provider_name, stream=stream, run_id=run_id, resume_from=resume_run_id, trace_model_responses=trace_model_responses)
        return self.run_mock_task(task, stream=stream, run_id=run_id, resume_from=resume_run_id, trace_model_responses=trace_model_responses)

    def run_mock_followup(
        self,
        follow_up: str,
        source_run_id: str,
        stream: bool = False,
        run_id: str | None = None,
        trace_model_responses: bool = False,
    ) -> dict[str, object]:
        return self._run_followup_task(
            follow_up,
            source_run_id,
            mock=True,
            provider=None,
            stream=stream,
            run_id=run_id,
            trace_model_responses=trace_model_responses,
        )

    def run_model_followup(
        self,
        follow_up: str,
        source_run_id: str,
        provider: str | None,
        stream: bool = False,
        run_id: str | None = None,
        trace_model_responses: bool = False,
    ) -> dict[str, object]:
        return self._run_followup_task(
            follow_up,
            source_run_id,
            mock=False,
            provider=provider,
            stream=stream,
            run_id=run_id,
            trace_model_responses=trace_model_responses,
        )

    def _run_followup_task(
        self,
        follow_up: str,
        source_run_id: str,
        mock: bool,
        provider: str | None,
        stream: bool,
        run_id: str | None,
        trace_model_responses: bool,
    ) -> dict[str, object]:
        if not follow_up.strip():
            raise ValueError("Field 'follow_up' is required.")
        loaded_checkpoint = self.checkpoints.load_state(source_run_id)
        runtime = self._build_runtime(mock=mock, provider=provider, trace_model_responses=trace_model_responses)
        logger = RunLogger(self.workspace.logs_dir, run_id=run_id)
        tracker = WorkspaceChangeTracker(self.workspace)
        before = tracker.capture()
        logger.event("workspace_snapshot", {"phase": "before", "file_count": len(before.files)})
        chunks: list[str] = []
        logger.event(
            "conversation_follow_up",
            {
                "source_run_id": source_run_id,
                "source_status": loaded_checkpoint.status,
                "follow_up": follow_up,
            },
        )
        logger.event(
            "run_config",
            self._run_config_payload(runtime, mock=mock, stream=stream, resume_from=source_run_id),
        )
        logger.event("tool_profile", {"profile": runtime.config.tool_profile})
        logger.event("command_profile", {"profile": runtime.config.command_profile})
        try:
            messages = list(loaded_checkpoint.messages)
            messages.append(Message("user", "Follow-up request:\n" + follow_up))
            model = MockModelClient() if mock else OpenAICompatibleClient(runtime.config, runtime.tool_specs)
            agent = Agent(
                config=runtime.config,
                model=model,
                tools=runtime.router,
                logger=logger,
                workspace=self.workspace,
                plan=loaded_checkpoint.plan,
                checkpoint_store=self.checkpoints,
                stream=stream,
                stream_callback=chunks.append if stream else None,
            )
            result = agent.resume(
                task=follow_up,
                messages=messages,
                completed_steps=0,
                source_run_id=source_run_id,
            )
        finally:
            after = tracker.capture()
            changes = tracker.compare(before, after)
            changes_path = tracker.save(logger.run_id, changes)
            logger.event("workspace_changes", {**changes.to_dict(), "path": self.workspace.relative(changes_path)})
        return {
            "run_id": logger.run_id,
            "mode": "mock" if mock else "model",
            "provider": "mock" if mock else runtime.config.provider,
            "model": None if mock else runtime.config.model,
            "content": result.content,
            "steps": result.steps,
            "stream_chunks": chunks,
            "source_run_id": source_run_id,
            "follow_up": follow_up,
            "changes": changes.to_dict(),
        }
    def run_mock_task(
        self,
        task: str,
        stream: bool = False,
        run_id: str | None = None,
        resume_from: str | None = None,
        trace_model_responses: bool = False,
    ) -> dict[str, object]:
        if not task:
            raise ValueError("Task is required.")
        loaded_checkpoint = self.checkpoints.load_state(resume_from) if resume_from is not None else None
        if loaded_checkpoint is not None and loaded_checkpoint.status == "completed":
            raise ValueError(f"Checkpoint is already completed: {resume_from}")
        logger = RunLogger(self.workspace.logs_dir, run_id=run_id)
        tracker = WorkspaceChangeTracker(self.workspace)
        before = tracker.capture()
        logger.event("workspace_snapshot", {"phase": "before", "file_count": len(before.files)})
        runtime = self._build_runtime(mock=True, trace_model_responses=trace_model_responses)
        chunks: list[str] = []
        logger.event("run_config", self._run_config_payload(runtime, mock=True, stream=stream, resume_from=resume_from))
        logger.event("tool_profile", {"profile": runtime.config.tool_profile})
        logger.event("command_profile", {"profile": runtime.config.command_profile})
        try:
            agent = Agent(
                config=runtime.config,
                model=MockModelClient(calls=loaded_checkpoint.step if loaded_checkpoint is not None else 0),
                tools=runtime.router,
                logger=logger,
                workspace=self.workspace,
                plan=loaded_checkpoint.plan if loaded_checkpoint is not None else None,
                checkpoint_store=self.checkpoints,
                stream=stream,
                stream_callback=chunks.append if stream else None,
            )
            result = (
                agent.resume(
                    task=loaded_checkpoint.task,
                    messages=loaded_checkpoint.messages,
                    completed_steps=loaded_checkpoint.step,
                    source_run_id=loaded_checkpoint.run_id,
                )
                if loaded_checkpoint is not None
                else agent.run(task)
            )
        finally:
            after = tracker.capture()
            changes = tracker.compare(before, after)
            changes_path = tracker.save(logger.run_id, changes)
            logger.event("workspace_changes", {**changes.to_dict(), "path": self.workspace.relative(changes_path)})
        return {
            "run_id": logger.run_id,
            "mode": "mock",
            "provider": "mock",
            "content": result.content,
            "steps": result.steps,
            "stream_chunks": chunks,
            "changes": changes.to_dict(),
        }

    def run_model_task(
        self,
        task: str,
        provider: str | None,
        stream: bool = False,
        run_id: str | None = None,
        resume_from: str | None = None,
        trace_model_responses: bool = False,
    ) -> dict[str, object]:
        if not task:
            raise ValueError("Task is required.")
        runtime = self._build_runtime(mock=False, provider=provider, trace_model_responses=trace_model_responses)
        config = runtime.config
        loaded_checkpoint = self.checkpoints.load_state(resume_from) if resume_from is not None else None
        if loaded_checkpoint is not None and loaded_checkpoint.status == "completed":
            raise ValueError(f"Checkpoint is already completed: {resume_from}")
        logger = RunLogger(self.workspace.logs_dir, run_id=run_id)
        tracker = WorkspaceChangeTracker(self.workspace)
        before = tracker.capture()
        logger.event("workspace_snapshot", {"phase": "before", "file_count": len(before.files)})
        chunks: list[str] = []
        logger.event("run_config", self._run_config_payload(runtime, mock=False, stream=stream, resume_from=resume_from))
        logger.event("tool_profile", {"profile": runtime.config.tool_profile})
        logger.event("command_profile", {"profile": runtime.config.command_profile})
        try:
            agent = Agent(
                config=config,
                model=OpenAICompatibleClient(runtime.config, runtime.tool_specs),
                tools=runtime.router,
                logger=logger,
                workspace=self.workspace,
                plan=loaded_checkpoint.plan if loaded_checkpoint is not None else None,
                checkpoint_store=self.checkpoints,
                stream=stream,
                stream_callback=chunks.append if stream else None,
            )
            result = (
                agent.resume(
                    task=loaded_checkpoint.task,
                    messages=loaded_checkpoint.messages,
                    completed_steps=loaded_checkpoint.step,
                    source_run_id=loaded_checkpoint.run_id,
                )
                if loaded_checkpoint is not None
                else agent.run(task)
            )
        finally:
            after = tracker.capture()
            changes = tracker.compare(before, after)
            changes_path = tracker.save(logger.run_id, changes)
            logger.event("workspace_changes", {**changes.to_dict(), "path": self.workspace.relative(changes_path)})
        return {
            "run_id": logger.run_id,
            "mode": "model",
            "provider": config.provider,
            "model": config.model,
            "content": result.content,
            "steps": result.steps,
            "stream_chunks": chunks,
            "changes": changes.to_dict(),
        }

    def _load_run_if_exists(self, run_id: str) -> list[dict[str, Any]]:
        try:
            return self.load_run(run_id)
        except ValueError:
            return []

    def _run_summary_if_exists(self, run_id: str) -> dict[str, object] | None:
        try:
            return self.store.summarize_run(run_id).to_dict()
        except ValueError:
            return None


def create_handler(app: HarnessServer) -> type[BaseHTTPRequestHandler]:
    """构造 HTTP 请求处理器类, 通过闭包捕获 app 实例。

    ThreadingHTTPServer 需要 handler 类而不是实例, 所以这里返回一个内部类。
    """
    class Handler(BaseHTTPRequestHandler):
        server_version = "MiniHarnessHTTP/0.1"

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            try:
                if parsed.path in {"/", "/console"}:
                    self._html(200, app.console_html())
                    return
                if parsed.path == "/health":
                    self._json(200, app.health())
                    return
                if parsed.path == "/providers":
                    self._json(200, app.providers())
                    return
                if parsed.path == "/api-map":
                    self._json(200, {"endpoints": app.api_map()})
                    return
                if parsed.path == "/preview-run":
                    query = parse_qs(parsed.query)
                    provider = query.get("provider", [None])[0]
                    self._json(
                        200,
                        app.preview_run(
                            mock=self._query_bool(query, "mock", True),
                            provider=provider,
                            stream=self._query_bool(query, "stream", False),
                            trace_model_responses=self._query_bool(query, "trace_model_responses", False),
                        ),
                    )
                    return
                if parsed.path == "/runs":
                    query = parse_qs(parsed.query)
                    limit = int(query.get("limit", ["20"])[0])
                    self._json(200, app.list_runs(limit=limit))
                    return
                if parsed.path == "/tasks":
                    self._json(200, app.list_tasks())
                    return
                if parsed.path.startswith("/tasks/") and parsed.path.endswith("/events"):
                    task_id = parsed.path.removeprefix("/tasks/").removesuffix("/events").strip("/")
                    query = parse_qs(parsed.query)
                    timeout = float(query.get("timeout", ["30"])[0])
                    self._sse(app.task_events(task_id, timeout_seconds=timeout))
                    return
                if parsed.path.startswith("/tasks/"):
                    task_id = parsed.path.removeprefix("/tasks/")
                    self._json(200, app.load_task(task_id))
                    return
                if parsed.path.startswith("/runs/") and parsed.path.endswith("/changes"):
                    run_id = parsed.path.removeprefix("/runs/").removesuffix("/changes").strip("/")
                    self._json(200, app.load_run_changes(run_id))
                    return
                if parsed.path.startswith("/runs/") and parsed.path.endswith("/checkpoint"):
                    run_id = parsed.path.removeprefix("/runs/").removesuffix("/checkpoint").strip("/")
                    self._json(200, app.load_run_checkpoint(run_id))
                    return
                if parsed.path.startswith("/runs/"):
                    run_id = parsed.path.removeprefix("/runs/")
                    self._json(200, app.load_run(run_id))
                    return
                self._json(404, {"error": "Not found"})
            except Exception as exc:
                self._json(400, {"error": str(exc)})

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            try:
                if parsed.path != "/tasks":
                    self._json(404, {"error": "Not found"})
                    return
                payload = self._read_json()
                task = payload.get("task")
                resume_from = payload.get("resume_from")
                mock = self._json_bool(payload, "mock", True)
                stream = self._json_bool(payload, "stream", False)
                trace_model_responses = self._json_bool(payload, "trace_model_responses", False)
                if resume_from is not None:
                    if not isinstance(resume_from, str):
                        self._json(400, {"error": "Field 'resume_from' must be a string."})
                        return
                    provider = payload.get("provider")
                    if provider is not None and not isinstance(provider, str):
                        self._json(400, {"error": "Field 'provider' must be a string."})
                        return
                    follow_up = payload.get("follow_up")
                    if follow_up is not None:
                        if not isinstance(follow_up, str):
                            self._json(400, {"error": "Field 'follow_up' must be a string."})
                            return
                        self._json(
                            202,
                            app.submit_followup_task(
                                resume_from,
                                follow_up,
                                mock=mock,
                                provider=provider,
                                stream=stream,
                                trace_model_responses=trace_model_responses,
                            ),
                        )
                        return
                    self._json(202, app.submit_resume_task(resume_from, mock=mock, provider=provider, stream=stream, trace_model_responses=trace_model_responses))
                    return
                if not isinstance(task, str):
                    self._json(400, {"error": "Field 'task' must be a string."})
                    return
                if mock is False:
                    provider = payload.get("provider")
                    if provider is not None and not isinstance(provider, str):
                        self._json(400, {"error": "Field 'provider' must be a string."})
                        return
                    self._json(202, app.submit_model_task(task, provider, stream=stream, trace_model_responses=trace_model_responses))
                    return
                self._json(202, app.submit_mock_task(task, stream=stream, trace_model_responses=trace_model_responses))
            except Exception as exc:
                self._json(400, {"error": str(exc)})

        def log_message(self, format: str, *args: object) -> None:
            return

        def _query_bool(self, query: dict[str, list[str]], name: str, default: bool) -> bool:
            raw = query.get(name, [str(default).lower()])[0].strip().lower()
            if raw in {"1", "true", "yes", "on"}:
                return True
            if raw in {"0", "false", "no", "off"}:
                return False
            raise ValueError(f"Query parameter {name!r} must be a boolean.")

        def _json_bool(self, payload: dict[str, object], name: str, default: bool) -> bool:
            value = payload.get(name, default)
            if isinstance(value, bool):
                return value
            raise ValueError(f"Field {name!r} must be a boolean.")

        def _read_json(self) -> dict[str, object]:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length).decode("utf-8") if length else "{}"
            data = json.loads(raw)
            if not isinstance(data, dict):
                raise ValueError("JSON request body must be an object.")
            return data

        def _json(self, status: int, payload: object) -> None:
            body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _html(self, status: int, html: str) -> None:
            body = html.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _sse(self, events: list[dict[str, object]]) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            for event in events:
                name = event.get("event", "message")
                data = json.dumps(event.get("data", {}), ensure_ascii=False)
                payload = f"event: {name}\ndata: {data}\n\n".encode("utf-8")
                self.wfile.write(payload)
                self.wfile.flush()

    return Handler


def serve(workspace: str | None, host: str, port: int, config_path: str | None = None) -> None:
    """Start the blocking local HTTP server."""
    app = HarnessServer(workspace, config_path)
    server = ThreadingHTTPServer((host, port), create_handler(app))
    print(f"Mini Harness server listening on http://{host}:{port}")
    print(f"Workspace: {Path(app.workspace.root)}")
    server.serve_forever()


CONSOLE_HTML = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Mini Harness 控制台</title>
  <style>
    :root { --bg:#f4f6f8; --panel:#fff; --line:#d8dde6; --text:#172033; --muted:#667085; --accent:#1f6feb; --ok:#0f7b4f; --warn:#9a5b00; --bad:#b42318; }
    * { box-sizing: border-box; }
    body { margin:0; color:var(--text); background:var(--bg); font:14px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif; }
    header { height:58px; display:flex; align-items:center; justify-content:space-between; gap:16px; padding:0 18px; border-bottom:1px solid var(--line); background:var(--panel); }
    h1,h2,h3,p { margin:0; } h1{font-size:16px;} h2{font-size:14px;} h3{font-size:12px;color:var(--muted);text-transform:uppercase;}
    button,textarea,select{font:inherit;} button{height:34px;border:1px solid #1d5fd0;border-radius:6px;padding:0 12px;background:var(--accent);color:#fff;font-weight:650;cursor:pointer;} button.secondary{background:#fff;color:var(--text);border-color:var(--line);} button:disabled{opacity:.55;cursor:not-allowed;}
    textarea,select{width:100%;border:1px solid var(--line);border-radius:6px;background:#fff;color:var(--text);} textarea{min-height:110px;resize:vertical;padding:10px;} select{height:34px;padding:0 8px;} label{display:block;margin-bottom:6px;font-size:12px;font-weight:700;color:var(--muted);}
    .workspace{min-width:0;color:var(--muted);font-size:12px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;}.layout{display:grid;grid-template-columns:340px minmax(420px,1fr) 390px;min-height:calc(100vh - 58px);} .sidebar,.center,.detail{min-width:0;padding:14px}.sidebar,.center{border-right:1px solid var(--line)}.sidebar{background:var(--panel)}
    .section{margin-bottom:14px;border:1px solid var(--line);border-radius:8px;background:var(--panel);overflow:hidden}.section-head{display:flex;align-items:center;justify-content:space-between;gap:10px;padding:11px 12px;border-bottom:1px solid var(--line);background:#f9fafb}.section-body,.detail-block{padding:12px}.field{margin-bottom:10px}.row{display:flex;gap:8px;align-items:center}.split{display:grid;grid-template-columns:1fr 1fr;gap:8px}.muted{color:var(--muted);font-size:12px}.mono,pre{font-family:ui-monospace,SFMono-Regular,Consolas,monospace;font-size:12px}pre{margin:0;white-space:pre-wrap;overflow-wrap:anywhere}
    .template-grid{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin:8px 0 10px}.template-btn{height:auto;min-height:34px;padding:7px 8px;background:#fff;color:var(--text);border-color:var(--line);font-size:12px;text-align:left}.template-btn:hover{background:#eef5ff;border-color:#b7cff8}.stats{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px;margin-bottom:14px}.stat{border:1px solid var(--line);border-radius:8px;background:var(--panel);padding:10px 12px}.stat strong{display:block;margin-top:4px;font-size:20px}.list{max-height:300px;overflow:auto}.timeline{max-height:calc(100vh - 224px);overflow:auto}.history-item,.event-item{width:100%;padding:10px 12px;border:0;border-bottom:1px solid var(--line);border-radius:0;background:transparent;color:var(--text);text-align:left;cursor:pointer}.history-item:hover,.event-item:hover,.history-item.active,.event-item.active{background:#eef5ff}.history-title,.event-title{display:flex;align-items:center;justify-content:space-between;gap:8px;margin-bottom:5px}.truncate{min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
    .badge{display:inline-flex;align-items:center;justify-content:center;min-width:68px;height:22px;padding:0 8px;border-radius:999px;background:#eef2f7;color:#344054;font-size:12px;font-weight:700;white-space:nowrap}.badge.completed{background:#e9f8f0;color:var(--ok)}.badge.failed{background:#fff0ee;color:var(--bad)}.badge.running,.badge.stopped{background:#fff4d6;color:var(--warn)}.badge.queued{background:#eaf2ff;color:#1959bd}.empty{padding:24px 12px;color:var(--muted);text-align:center}.kv{display:grid;grid-template-columns:118px 1fr;gap:6px 10px;margin-top:8px;font-size:12px}.kv div:nth-child(odd){color:var(--muted)}.conversation{display:grid;gap:8px;max-height:220px;overflow:auto}.message{border:1px solid var(--line);border-radius:7px;padding:8px 9px;background:#fff}.message.user{border-color:#b7cff8;background:#f3f7ff}.message.assistant{border-color:#c7e4d3;background:#f3fbf6}.message-role{margin-bottom:4px;color:var(--muted);font-size:11px;font-weight:750;text-transform:uppercase}.changes{display:grid;gap:8px}.change-row{display:flex;justify-content:space-between;gap:10px;border:1px solid var(--line);border-radius:6px;padding:8px 9px}.change-path{overflow-wrap:anywhere}.event-explain{font-size:13px;line-height:1.6}.event-explain .explain-title{font-weight:750;margin-bottom:8px}.event-explain .explain-row{display:grid;grid-template-columns:88px 1fr;gap:6px 10px;margin-top:4px}.event-explain .explain-row span:first-child{color:var(--muted)}.event-explain details{margin-top:10px}.event-explain summary{cursor:pointer;color:var(--accent);font-weight:650}.call-chain{display:grid;gap:8px}.call-row{border:1px solid var(--line);border-radius:7px;background:#fff;padding:8px 9px}.call-title{display:flex;align-items:center;gap:8px;margin-bottom:4px;font-weight:750}.call-note{color:var(--muted);font-size:12px;line-height:1.5}.method.internal{background:#fff4d6;color:var(--warn)}.loop-io{display:grid;gap:8px}.io-card{border:1px solid var(--line);border-radius:7px;background:#fff;padding:8px 9px}.io-card strong{display:block;font-size:12px}.io-card span{display:block;margin-top:4px;color:var(--muted);font-size:12px;line-height:1.5}.io-card pre{margin-top:8px;max-height:220px;overflow:auto;background:#f9fafb;border:1px solid var(--line);border-radius:6px;padding:8px}.flow{display:grid;grid-template-columns:repeat(7,minmax(0,1fr));gap:8px}.flow-node{border:1px solid var(--line);border-radius:8px;background:#fff;padding:9px 8px;min-height:72px}.flow-node.active{border-color:var(--accent);background:#eef5ff}.flow-node.done{border-color:#b7e0c9;background:#f3fbf6}.flow-title{font-weight:750;font-size:12px}.flow-count{margin-top:6px;color:var(--muted);font-size:12px}.flow-note{margin-top:4px;color:var(--muted);font-size:11px;line-height:1.35}.api-list{display:grid;gap:8px;max-height:360px;overflow:auto}.api-card{border:1px solid var(--line);border-radius:7px;background:#fff;padding:9px}.api-head{display:flex;align-items:center;gap:8px;margin-bottom:6px}.method{display:inline-flex;align-items:center;justify-content:center;min-width:42px;height:20px;border-radius:5px;background:#eaf2ff;color:#1959bd;font-size:11px;font-weight:800}.method.post{background:#e9f8f0;color:var(--ok)}.api-path{font-weight:750;overflow-wrap:anywhere}.api-meta{display:grid;gap:4px;font-size:12px;color:var(--muted)}.preview-grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:8px}.preview-cell{border:1px solid var(--line);border-radius:7px;background:#fff;padding:8px}.preview-cell span{display:block;color:var(--muted);font-size:11px}.preview-cell strong{display:block;margin-top:4px;font-size:13px;overflow-wrap:anywhere}.tool-chips{display:flex;flex-wrap:wrap;gap:6px;margin-top:10px}.tool-chip{border:1px solid var(--line);border-radius:999px;padding:4px 8px;background:#f9fafb;font-size:12px}.tool-schema-list{display:grid;gap:8px;margin-top:10px}.tool-schema{border:1px solid var(--line);border-radius:7px;background:#fff;padding:8px}.tool-schema summary{cursor:pointer;font-weight:750}.tool-schema-desc{margin-top:6px;color:var(--muted);font-size:12px}.tool-schema pre{margin-top:8px;max-height:160px;overflow:auto;background:#f9fafb;border:1px solid var(--line);border-radius:6px;padding:8px}.hash-row{display:grid;grid-template-columns:120px 1fr;gap:6px 10px;margin-top:10px;font-size:12px}.hash-row span:nth-child(odd){color:var(--muted)}.report-grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:8px}.report-card{border:1px solid var(--line);border-radius:7px;background:#fff;padding:9px}.report-card span{display:block;color:var(--muted);font-size:11px}.report-card strong{display:block;margin-top:4px;font-size:18px}.report-text{margin-top:10px;border:1px solid var(--line);border-radius:7px;background:#f9fafb;padding:10px;font-size:13px;line-height:1.6}.report-actions{display:flex;align-items:center;justify-content:space-between;gap:8px;margin-top:10px}.copy-status{color:var(--muted);font-size:12px}.report-script{margin-top:8px;max-height:180px;overflow:auto;border:1px solid var(--line);border-radius:7px;background:#fff;padding:10px}
    @media(max-width:1160px){.layout{grid-template-columns:320px 1fr}.detail{grid-column:1/-1;border-top:1px solid var(--line)}.center{border-right:0}.flow{grid-template-columns:repeat(4,minmax(0,1fr))}}@media(max-width:760px){header{height:auto;padding:12px 14px;align-items:flex-start;flex-direction:column}.layout{grid-template-columns:1fr}.sidebar,.center{border-right:0;border-bottom:1px solid var(--line)}.stats,.preview-grid,.report-grid{grid-template-columns:repeat(2,minmax(0,1fr))}.flow{grid-template-columns:repeat(2,minmax(0,1fr))}}
  </style>
</head>
<body>
  <header><h1>Mini Harness 控制台</h1><div class="workspace" title="__WORKSPACE__">__WORKSPACE__</div></header>
  <main class="layout">
    <aside class="sidebar">
      <div class="section"><div class="section-head"><h2>任务输入</h2><span id="health" class="badge queued">本地</span></div><div class="section-body">
        <div class="field"><label for="task">任务</label><textarea id="task">检查这个项目并总结结构</textarea></div>
        <div class="split"><div class="field"><label for="mode">模式</label><select id="mode"><option value="mock">Mock 演示</option><option value="model">真实模型</option></select></div><div class="field"><label for="provider">模型供应商</label><select id="provider"></select></div></div><div class="field"><label><input id="traceModelResponses" type="checkbox"> 记录模型原文到原始事件 JSON</label><p class="muted">开启后本次 run 的 model_response.payload.content 会显示完整模型返回。</p></div>
        <div class="field"><label>演示任务模板</label><div class="template-grid"><button class="template-btn" type="button" data-template="inspect">项目结构巡检</button><button class="template-btn" type="button" data-template="trace">解释 Harness 流程</button><button class="template-btn" type="button" data-template="safety">检查安全策略</button><button class="template-btn" type="button" data-template="resume">观察模型返回</button></div><p class="muted">点击模板只会填充任务，不会自动运行。</p></div>
        <div class="row"><button id="submit">运行</button><button id="refresh" class="secondary">刷新</button></div><p id="activeTask" class="muted" style="margin-top:10px;">当前没有运行中的任务。</p>
      </div></div>
      <div class="section"><div class="section-head"><h2>对话</h2><span id="conversationRun" class="muted">-</span></div><div class="section-body">
        <div id="conversation" class="conversation"><div class="empty">请选择或运行一个任务。</div></div>
        <div class="field" style="margin-top:10px;"><label for="followup">继续追问</label><textarea id="followup" style="min-height:74px;" placeholder="基于当前选中的 run 继续追问"></textarea></div>
        <div class="row"><button id="sendFollowup" class="secondary">发送</button></div>
      </div></div>
      <div class="section"><div class="section-head"><h2>任务列表</h2><span id="taskCount" class="muted">0</span></div><div id="taskList" class="list"><div class="empty">还没有任务。</div></div></div>
      <div class="section"><div class="section-head"><h2>后端接口地图</h2><span id="apiCount" class="muted">0</span></div><div id="apiMap" class="section-body"><div class="empty">正在加载接口。</div></div></div>
    </aside>
    <section class="center"><div class="section"><div class="section-head"><h2>Harness 流程图</h2><span id="flowHint" class="muted">选择 run 后自动高亮</span></div><div class="section-body"><div id="flow" class="flow"></div></div></div><div class="section"><div class="section-head"><h2>运行前预检</h2><span id="previewBadge" class="badge queued">preview</span></div><div id="runPreview" class="section-body"><div class="empty">正在读取运行配置。</div></div></div><div class="stats"><div class="stat" title="运行 + 发送追问 创建的总 run 数"><span class="muted">运行次数</span><strong id="runCount">0</strong></div><div class="stat" title="当前 run 里模型返回动作的轮数"><span class="muted">模型步数</span><strong id="eventCount">0</strong></div><div class="stat" title="当前 run 调用了多少次 read_file、run_command 等工具"><span class="muted">工具调用</span><strong id="toolCount">0</strong></div><div class="stat" title="当前 run 修改了多少个文件"><span class="muted">文件变更</span><strong id="changeCount">0</strong></div></div>
      <div class="section"><div class="section-head"><h2>本次运行报告</h2><span id="runReportStatus" class="muted">未选择 run</span></div><div id="runReport" class="section-body"><div class="empty">选择或运行一个任务后生成报告。</div></div></div>
      <div class="section"><div class="section-head"><h2>执行过程</h2><span id="selectedRun" class="muted">未选择 run</span></div><div id="timeline" class="timeline"><div class="empty">运行一个任务，或选择历史 run。</div></div></div>
    </section>
    <aside class="detail"><div class="section"><div class="section-head"><h2>当前步骤</h2><span id="runStatus" class="badge">空闲</span></div><div id="summary" class="detail-block"><p class="muted">选择中间的某一步，这里会按小白能看懂的方式解释它。</p></div></div>
      <div class="section"><div class="section-head"><h2>这一步做了什么</h2><span id="eventKind" class="muted">-</span></div><div class="detail-block"><div id="eventDetail" class="event-explain">请从执行过程中选择一个步骤。</div></div></div>
      <div class="section"><div class="section-head"><h2>接口链路</h2><span id="apiTraceKind" class="muted">-</span></div><div class="detail-block"><div id="apiTrace" class="event-explain">选择步骤后会显示前端、后端和模型/工具的调用关系。</div></div></div>
      <div class="section"><div class="section-head"><h2>关键数据</h2><span id="eventDataKind" class="muted">-</span></div><div class="detail-block"><div id="eventData" class="event-explain">这里会展示和这一步最相关的字段。</div></div></div>
      <div class="section"><div class="section-head"><h2>循环输入输出</h2><span id="loopIoKind" class="muted">-</span></div><div class="detail-block"><div id="loopIo" class="event-explain">选择步骤后会显示这一步在 agent loop 中的输入和输出。</div></div></div>
      <div class="section"><div class="section-head"><h2>原始事件 JSON</h2><span id="rawEventKind" class="muted">-</span></div><div class="detail-block"><pre id="rawEventJson">请选择执行过程中的一个步骤。</pre></div></div>
      <div class="section"><div class="section-head"><h2>工作区变更</h2><span id="changesBadge" class="badge">0</span></div><div id="changes" class="detail-block"><p class="muted">还没有加载文件变更。</p></div></div>
    </aside>
  </main>
  <script>
    const state={tasks:[],runs:[],events:[],apiMap:[],preview:null,selectedRunId:null,selectedEventSeq:null,source:null,selectedSummary:null};
    const FLOW_STAGES=[['start','接收目标','创建 run'],['context','准备上下文','配置/工具/快照'],['model','调用模型','请求 /chat/completions'],['parse','解析动作','计划/动作/错误'],['tool','执行工具','文件/命令/git'],['memory','保存状态','checkpoint/变更'],['done','结束输出','final/finished']];
    const DEMO_TASKS={inspect:'请像代码审查员一样巡检这个项目：先列出目录结构，再说明 harness 的核心模块、Web 页面入口、模型适配层、工具系统和运行时安全策略。',trace:'请运行一次面向面试展示的讲解：解释从用户提交任务到模型返回动作、工具执行、trace 记录、checkpoint 保存和最终回复的完整 harness 流程。',safety:'请重点检查这个 harness 的安全设计：workspace 路径限制、命令策略、工具 schema、变更追踪和日志记录分别在哪里体现。',resume:'请专门观察一次模型返回和动作解析：说明 model_response 为什么默认只记录哈希和长度，什么时候会在原始事件 JSON 中出现完整 content。'};
    const el=(id)=>document.getElementById(id);
    async function requestJson(url,options){const r=await fetch(url,options);const t=await r.text();if(!r.ok)throw new Error(t||r.statusText);return t?JSON.parse(t):null;}
    async function refreshAll(){const [health,providers,tasks,runs,apiMap]=await Promise.all([requestJson('/health'),requestJson('/providers'),requestJson('/tasks'),requestJson('/runs?limit=20'),requestJson('/api-map')]);el('health').textContent=health.ok?'就绪':'异常';el('health').className='badge '+(health.ok?'completed':'failed');renderProviders(providers);await refreshRunPreview();state.tasks=tasks.slice().reverse();state.runs=runs.slice().reverse();state.apiMap=apiMap.endpoints||apiMap;renderTasks();renderApiMap();renderRunStats();renderFlow();if(!state.selectedRunId&&state.runs.length)await loadRun(state.runs[0].run_id);}
    function renderProviders(providers){if(el('provider').children.length)return;providers.forEach((p)=>{const o=document.createElement('option');o.value=p.name;o.textContent=`${p.name} (${p.default_model})`;el('provider').appendChild(o);});}
    async function refreshRunPreview(){
      const mock=el('mode').value==='mock';
      const provider=encodeURIComponent(el('provider').value||'');
      const providerQuery=provider?`&provider=${provider}`:'';
      const traceQuery=`&trace_model_responses=${el('traceModelResponses').checked}`;
      state.preview=await requestJson(`/preview-run?mock=${mock}&stream=true${providerQuery}${traceQuery}`);
      renderRunPreview();
    }
    function renderRunPreview(){
      const preview=state.preview;
      if(!preview){el('runPreview').innerHTML='<div class="empty">正在读取运行配置。</div>';return;}
      el('previewBadge').textContent=preview.mode||'preview';
      el('previewBadge').className='badge '+(preview.mode==='model'?'running':'completed');
      const tools=Array.isArray(preview.tool_names)?preview.tool_names:[];
      const toolSpecs=Array.isArray(preview.tools)?preview.tools:[];
      const schemas=toolSpecs.map((tool)=>{
        const args=tool.args_schema||{};
        return `<details class="tool-schema"><summary>${escapeHtml(tool.name||'unknown')}</summary><div class="tool-schema-desc">${escapeHtml(tool.description||'')}</div><pre>${escapeHtml(JSON.stringify(args,null,2))}</pre></details>`;
      }).join('');
      el('runPreview').innerHTML=`<div class="preview-grid"><div class="preview-cell"><span>模型</span><strong>${escapeHtml(preview.provider)} / ${escapeHtml(preview.model||'mock')}</strong></div><div class="preview-cell"><span>工具策略</span><strong>${escapeHtml(preview.tool_profile)}</strong></div><div class="preview-cell"><span>命令策略</span><strong>${escapeHtml(preview.command_profile)}</strong></div><div class="preview-cell"><span>人工确认</span><strong>${escapeHtml(preview.approval)}</strong></div></div><div class="tool-chips">${tools.map((tool)=>`<span class="tool-chip">${escapeHtml(tool)}</span>`).join('')}</div><div class="hash-row"><span>工具协议哈希</span><span class="mono">${escapeHtml(preview.tool_schema_sha256||'-')}</span><span>系统提示词哈希</span><span class="mono">${escapeHtml(preview.system_prompt_sha256||'-')}</span><span>最大步数</span><span>${escapeHtml(preview.max_steps??'-')}</span><span>工作区</span><span class="mono">${escapeHtml(preview.workspace||'-')}</span></div><div class="section" style="margin-top:12px;margin-bottom:0;"><div class="section-head"><h2>工具协议详情</h2><span class="muted">${toolSpecs.length} tools</span></div><div class="section-body"><div class="tool-schema-list">${schemas||'<div class="empty">没有工具 schema。</div>'}</div></div></div>`;
    }    function renderApiMap(){
      el('apiCount').textContent=String(state.apiMap.length);
      if(!state.apiMap.length){el('apiMap').innerHTML='<div class="empty">没有接口信息。</div>';return;}
      el('apiMap').innerHTML=`<div class="api-list">${state.apiMap.map((api)=>{
        const method=String(api.method||'GET').toLowerCase();
        const params=Array.isArray(api.params)&&api.params.length?api.params.join(', '):'无';
        return `<div class="api-card"><div class="api-head"><span class="method ${method}">${escapeHtml(api.method)}</span><span class="api-path mono">${escapeHtml(api.path)}</span></div><div class="api-meta"><div>${escapeHtml(api.purpose||'')}</div><div>入参: <span class="mono">${escapeHtml(params)}</span></div><div>响应: <span class="mono">${escapeHtml(api.response||'-')}</span></div><div>Java 对照: ${escapeHtml(api.java_analogy||'-')}</div></div></div>`;
      }).join('')}</div>`;
    }
    function renderTasks(){el('taskCount').textContent=String(state.tasks.length);if(!state.tasks.length){el('taskList').innerHTML='<div class="empty">还没有任务。</div>';return;}el('taskList').innerHTML='';state.tasks.forEach((task)=>{const b=document.createElement('button');b.className='history-item'+(task.run_id===state.selectedRunId?' active':'');b.innerHTML=`<div class="history-title"><strong class="truncate">${escapeHtml(task.task)}</strong><span class="${statusClass(task.status)}">${escapeHtml(task.status)}</span></div><div class="muted mono">${escapeHtml(task.task_id)}<br>${escapeHtml(task.run_id)}</div>`;b.addEventListener('click',async()=>{const d=await requestJson(`/tasks/${encodeURIComponent(task.task_id)}`);showTaskSummary(d);await loadRun(d.run_id);});el('taskList').appendChild(b);});}
    function runReportMetrics(){
      const events=state.events||[];
      const toolResults=events.filter((e)=>e.kind==='tool_result');
      const failedTools=toolResults.filter((e)=>e.payload?.result?.ok===false).length;
      const modelRequests=events.filter((e)=>e.kind==='model_request').length;
      const modelResponses=events.filter((e)=>e.kind==='model_response').length;
      const checkpoints=events.filter((e)=>e.kind==='checkpoint_saved').length;
      const audits=events.filter((e)=>e.kind==='command_audit').length;
      const changes=events.find((e)=>e.kind==='workspace_changes')?.payload?.changed_count??state.selectedSummary?.changes?.changed_count??0;
      const finished=events.find((e)=>e.kind==='run_finished')?.payload?.status||state.selectedSummary?.status||'-';
      return {events,toolResults,failedTools,modelRequests,modelResponses,checkpoints,audits,changes,finished};
    }
    function renderRunReport(){
      const m=runReportMetrics();
      el('runReportStatus').textContent=state.selectedRunId||'未选择 run';
      if(!state.selectedRunId||!m.events.length){
        el('runReport').innerHTML='<div class="empty">选择或运行一个任务后生成报告。</div>';
        return;
      }
      const cards=[
        ['模型请求',m.modelRequests],['模型返回',m.modelResponses],['工具成功',m.toolResults.length-m.failedTools],['工具失败',m.failedTools],
        ['Checkpoint',m.checkpoints],['命令审计',m.audits],['文件变更',m.changes],['最终状态',m.finished]
      ];
      const talk=runReportTalk(m);
      const script=runReportMarkdown(m);
      el('runReport').innerHTML=`<div class="report-grid">${cards.map(([k,v])=>`<div class="report-card"><span>${escapeHtml(k)}</span><strong>${escapeHtml(v)}</strong></div>`).join('')}</div><div class="report-text">${escapeHtml(talk)}</div><div class="report-actions"><button class="secondary" type="button" onclick="copyRunReport()">复制讲解稿</button><span id="copyReportStatus" class="copy-status">Markdown 讲解稿已生成</span></div><pre id="reportScript" class="report-script">${escapeHtml(script)}</pre>`;
    }
    function runReportTalk(m){
      return `这次 run 展示了一个完整的 coding-agent harness: 它记录了 ${m.modelRequests} 次模型请求、${m.toolResults.length} 次工具执行、${m.checkpoints} 次状态保存, 并在结束时汇总 ${m.changes} 个文件变更。面试时可以用这段报告说明: 模型只负责决策, 真正的文件、命令、安全策略和日志追踪都由 harness 运行时接管。`;
    }
    function runReportMarkdown(m){
      const tools=m.toolResults.map((event)=>event.payload?.action?.tool).filter(Boolean);
      const uniqueTools=[...new Set(tools)];
      return `# Mini Harness 运行讲解稿\n\n- run_id: ${state.selectedRunId||'-'}\n- 最终状态: ${m.finished}\n- 模型请求/返回: ${m.modelRequests}/${m.modelResponses}\n- 工具执行: ${m.toolResults.length} 次, 失败 ${m.failedTools} 次\n- 使用工具: ${uniqueTools.length?uniqueTools.join(', '):'暂无'}\n- Checkpoint: ${m.checkpoints} 次\n- 命令审计: ${m.audits} 次\n- 文件变更: ${m.changes} 个\n\n讲解重点:\n1. 这个项目不是普通聊天页面, 而是一个 coding-agent harness。\n2. 模型只输出下一步决策, 文件读取、命令执行、安全策略、日志追踪由 harness 运行时接管。\n3. 页面能把一次 run 拆成流程图、trace 时间线、步骤详情、接口地图和运行报告, 方便解释 agent loop。`;
    }
    async function copyRunReport(){
      const text=el('reportScript')?.textContent||'';
      if(!text)return;
      try{
        await navigator.clipboard.writeText(text);
        el('copyReportStatus').textContent='已复制到剪贴板';
      }catch{
        el('copyReportStatus').textContent='复制失败, 可以手动选中下方文本';
      }
    }

    function renderRunStats(){const s=state.runs.find((r)=>r.run_id===state.selectedRunId);el('runCount').textContent=String(state.runs.length);el('eventCount').textContent=String(s?.step_count??maxStepFromEvents(state.events));el('toolCount').textContent=String(s?.tools?.total_calls??toolCallsFromEvents(state.events));el('changeCount').textContent=String(s?.changes?.changed_count??0);renderRunReport();}
    function maxStepFromEvents(events){return Math.max(0,...events.map((e)=>Number(e.payload?.step||0)));}
    function toolCallsFromEvents(events){return events.filter((e)=>e.kind==='tool_result').length;}
    async function sendFollowup(){const followUp=el('followup').value.trim();if(!followUp)return;if(!state.selectedRunId){alert('请先选择一个 run，再继续追问。');return;}el('sendFollowup').disabled=true;try{const submitted=await requestJson('/tasks',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({resume_from:state.selectedRunId,follow_up:followUp,mock:el('mode').value==='mock',provider:el('provider').value,stream:true,trace_model_responses:el('traceModelResponses').checked})});el('followup').value='';el('activeTask').textContent=`追问任务 ${submitted.task_id} -> run ${submitted.run_id}`;state.selectedRunId=submitted.run_id;state.selectedSummary={run_id:submitted.run_id,task:followUp,status:submitted.status||'queued'};state.events=[];state.selectedEventSeq=null;renderConversation(state.selectedSummary);renderTimeline();followTask(submitted.task_id);await refreshAll();}finally{el('sendFollowup').disabled=false;}}
    async function submitTask(){const task=el('task').value.trim();if(!task)return;el('submit').disabled=true;try{const submitted=await requestJson('/tasks',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({task,mock:el('mode').value==='mock',provider:el('provider').value,stream:true,trace_model_responses:el('traceModelResponses').checked})});el('activeTask').textContent=`任务 ${submitted.task_id} -> run ${submitted.run_id}`;state.selectedRunId=submitted.run_id;state.selectedSummary={run_id:submitted.run_id,task,status:submitted.status||'queued'};state.events=[];state.selectedEventSeq=null;renderConversation(state.selectedSummary);renderTimeline();followTask(submitted.task_id);await refreshAll();}finally{el('submit').disabled=false;}}
    function followTask(taskId){if(state.source)state.source.close();state.source=new EventSource(`/tasks/${encodeURIComponent(taskId)}/events`);state.source.addEventListener('task',(e)=>showTaskSummary(JSON.parse(e.data)));state.source.addEventListener('trace',(e)=>appendTrace(JSON.parse(e.data)));state.source.addEventListener('done',async(e)=>{showTaskSummary(JSON.parse(e.data));state.source.close();await refreshAll();if(state.selectedRunId)await loadRun(state.selectedRunId);});state.source.addEventListener('timeout',(e)=>showTaskSummary(JSON.parse(e.data)));state.source.onerror=()=>{if(state.source)state.source.close();};}
    function appendTrace(record){if(!state.events.some((e)=>e.seq===record.seq)){state.events.push(record);renderTimeline();}}
    async function loadRun(runId){if(state.selectedRunId!==runId)state.selectedEventSeq=null;state.selectedRunId=runId;const [events,changes]=await Promise.all([requestJson(`/runs/${encodeURIComponent(runId)}`),requestJson(`/runs/${encodeURIComponent(runId)}/changes`)]);state.events=events;const summary=state.runs.find((r)=>r.run_id===runId)||summarizeFromEvents(runId,events,changes);state.selectedSummary=summary;showRunSummary(summary);renderConversation(summary);renderTimeline();renderChanges(changes);renderTasks();renderRunStats();}
    function summarizeFromEvents(runId,events,changes){const final=events.find((e)=>e.kind==='final');const finished=events.find((e)=>e.kind==='run_finished');const started=events.find((e)=>e.kind==='run_started');return{run_id:runId,task:started?.payload?.task||'',status:finished?.payload?.status||'running',event_count:events.length,step_count:Math.max(0,...events.map((e)=>Number(e.payload?.step||0))),final:final?.payload?.content||null,changes};}
    function compactTimelineEvents(events){
      const compact=[];
      const streamCounts=new Map();
      for(const event of events){
        if(event.kind==='model_stream_chunk'){
          const step=event.payload?.step??'?';
          const key=String(step);
          streamCounts.set(key,(streamCounts.get(key)||0)+1);
          continue;
        }
        if(event.kind==='model_response'){
          const step=event.payload?.step??'?';
          const count=streamCounts.get(String(step));
          if(count){
            compact.push({seq:`stream-${step}`,kind:'stream_summary',ts:event.ts,payload:{step,count}});
          }
        }
        compact.push(event);
      }
      return compact;
    }

    function renderTimeline(){
      el('selectedRun').textContent=state.selectedRunId||'未选择 run';
      el('eventCount').textContent=String(state.events.length);
      renderFlow();
      if(!state.events.length){
        el('timeline').innerHTML='<div class="empty">正在等待事件。</div>';
        return;
      }
      const visibleEvents=compactTimelineEvents(state.events);
      el('timeline').innerHTML='';
      visibleEvents.forEach((event)=>{
        const b=document.createElement('button');
        b.className='event-item'+(event.seq===state.selectedEventSeq?' active':'');
        b.innerHTML=`<div class="event-title"><strong class="truncate">${escapeHtml(eventTitle(event))}</strong><span class="badge">#${escapeHtml(event.seq??'')}</span></div><div class="muted mono">${escapeHtml(eventSubtitle(event))}</div>`;
        b.addEventListener('click',()=>selectEvent(event));
        el('timeline').appendChild(b);
      });
      if(!state.selectedEventSeq)selectEvent(visibleEvents[visibleEvents.length-1]);
    }
    function eventTitle(event){
      const p=event.payload||{};
      if(event.kind==='run_started')return p.task||'运行开始';
      if(event.kind==='workspace_snapshot')return p.phase==='before'?'扫描工作区':'工作区快照';
      if(event.kind==='run_config')return'运行配置';
      if(event.kind==='tool_profile')return`工具范围：${p.profile||'-'}`;
      if(event.kind==='command_profile')return`命令策略：${p.profile||'-'}`;
      if(event.kind==='context_summary')return'上下文摘要';
      if(event.kind==='checkpoint_saved')return`保存检查点 step ${p.step??''}`;
      if(event.kind==='model_request')return`请求模型 step ${p.step??''}`;
      if(event.kind==='stream_summary')return`模型流式输出 step ${p.step}（${p.count} 段）`;
      if(event.kind==='model_response')return`模型返回 step ${p.step??''}`;
      if(event.kind==='model_response_metadata')return'模型用量统计';
      if(event.kind==='action_error')return'动作解析错误';
      if(event.kind==='plan_updated')return'计划已更新';
      if(event.kind==='tool_result')return`工具调用：${p.action?.tool||'未知工具'}`;
      if(event.kind==='command_audit')return'命令审计';
      if(event.kind==='final')return'最终回复';
      if(event.kind==='run_finished')return`运行结束：${p.status||'-'}`;
      if(event.kind==='workspace_changes')return`${p.changed_count??0} 个文件变更`;
      if(event.kind==='conversation_follow_up')return'继续追问';
      return event.kind||'事件';
    }

    function eventSubtitle(event){
      if(event.kind==='stream_summary')return'已折叠逐字输出，原始 chunk 仍在 trace 中';
      return `${event.kind} · ${event.ts||''}`;
    }
    function stageForEvent(event){
      const kind=event.kind||'';
      if(['run_started'].includes(kind))return'start';
      if(['workspace_snapshot','run_config','tool_profile','command_profile','context_summary'].includes(kind))return'context';
      if(['model_request','stream_summary','model_response','model_response_metadata'].includes(kind))return'model';
      if(['plan_updated','action_error'].includes(kind))return'parse';
      if(['tool_result','command_audit'].includes(kind))return'tool';
      if(['checkpoint_saved','workspace_changes','conversation_follow_up'].includes(kind))return'memory';
      if(['final','run_finished'].includes(kind))return'done';
      return'parse';
    }
    function renderFlow(){
      const counts=Object.fromEntries(FLOW_STAGES.map(([id])=>[id,0]));
      state.events.forEach((event)=>{counts[stageForEvent(event)]++;});
      const selected=state.events.find((event)=>event.seq===state.selectedEventSeq);
      const activeStage=selected?stageForEvent(selected):(state.events.length?stageForEvent(state.events[state.events.length-1]):null);
      el('flowHint').textContent=state.selectedRunId?'点击时间线步骤可联动右侧详情':'选择 run 后自动高亮';
      el('flow').innerHTML=FLOW_STAGES.map(([id,title,note])=>{
        const count=counts[id]||0;
        const cls='flow-node '+(id===activeStage?'active':count>0?'done':'');
        return `<div class="${cls}"><div class="flow-title">${escapeHtml(title)}</div><div class="flow-count">${count} 个事件</div><div class="flow-note">${escapeHtml(note)}</div></div>`;
      }).join('');
    }
    function selectEvent(event){
      state.selectedEventSeq=event.seq;
      el('eventKind').textContent=eventTitle(event);
      el('summary').innerHTML=eventOverview(event);
      el('eventDetail').innerHTML=eventExplanation(event);
      el('apiTraceKind').textContent=event.kind||'-';
      el('apiTrace').innerHTML=eventApiTrace(event);
      el('eventDataKind').textContent=event.kind||'-';
      el('eventData').innerHTML=eventDataView(event);
      el('loopIoKind').textContent=event.kind||'-';
      el('loopIo').innerHTML=eventLoopIo(event);
      el('rawEventKind').textContent=event.kind||'-';
      el('rawEventJson').textContent=JSON.stringify(event,null,2);
      document.querySelectorAll('.event-item').forEach((n)=>n.classList.remove('active'));
      const visibleEvents=compactTimelineEvents(state.events);
      const i=visibleEvents.findIndex((x)=>x.seq===event.seq);
      if(i>=0)el('timeline').children[i]?.classList.add('active');
      renderFlow();
    }
    function eventOverview(event){
      const p=event.payload||{};
      const step=p.step!==undefined?`第 ${p.step} 轮模型循环`:'运行级步骤';
      const source=eventSourceLabel(event);
      return `<div class="explain-title">${escapeHtml(eventTitle(event))}</div><div class="kv"><div>它在流程里</div><div>${escapeHtml(step)}</div><div>由谁产生</div><div>${escapeHtml(source)}</div><div>事件类型</div><div class="mono">${escapeHtml(event.kind||'-')}</div><div>事件序号</div><div class="mono">#${escapeHtml(event.seq??'-')}</div></div>`;
    }

    function eventSourceLabel(event){
      if(event.kind==='model_request'||event.kind==='model_response'||event.kind==='model_response_metadata'||event.kind==='stream_summary')return'模型适配层';
      if(event.kind==='tool_result'||event.kind==='command_audit')return'工具执行层';
      if(event.kind==='run_config'||event.kind==='tool_profile'||event.kind==='command_profile')return'运行配置层';
      if(event.kind==='workspace_snapshot'||event.kind==='workspace_changes')return'工作区沙箱层';
      if(event.kind==='checkpoint_saved'||event.kind==='conversation_follow_up')return'上下文管理层';
      if(event.kind==='plan_updated')return'计划管理层';
      if(event.kind==='action_error')return'动作解析层';
      return'Agent 主循环';
    }

    function eventExplanation(event){
      const p=event.payload||{};
      let lines=[];
      if(event.kind==='run_started'){
        lines=['Harness 接收到你的任务, 创建一次新的 run。','从这里开始, 后端会不断执行“准备上下文 -> 问模型 -> 解析动作 -> 调工具 -> 继续问模型”的循环。'];
      }else if(event.kind==='workspace_snapshot'){
        lines=['Harness 先扫描工作区当前文件状态。','这样运行结束后, 它可以判断哪些文件被新增、修改或删除。'];
      }else if(event.kind==='run_config'){
        lines=['Harness 把这次运行的配置固定下来。','这里能看到用哪个模型、最多循环几步、开放了哪些工具、是否开启流式输出。'];
      }else if(event.kind==='tool_profile'){
        lines=['这一步决定模型能使用哪些工具。','例如只读模式下模型只能看文件和查 git diff, 不能改文件。'];
      }else if(event.kind==='command_profile'){
        lines=['这一步决定 shell 命令的安全策略。','危险命令会被限制或需要确认, 避免模型随意破坏工作区。'];
      }else if(event.kind==='context_summary'){
        lines=['Harness 整理要发给模型的上下文。','上下文可能包含任务、历史消息、计划、工具说明和必要的运行状态。'];
      }else if(event.kind==='model_request'){
        lines=['Harness 准备向大模型提问。','真正的模型 API 调用发生在后端模型适配层, OpenAI-compatible provider 会请求 /chat/completions。','前端没有直接调用 DeepSeek 或其他模型, 前端只是在看后端记录下来的过程。'];
      }else if(event.kind==='stream_summary'){
        lines=['模型正在一小段一小段地流式返回内容。','中间时间线把逐字片段折叠成一条, 这样你不会看到几十上百条重复的小事件。'];
      }else if(event.kind==='model_response'){
        lines=['模型这一轮已经返回完整内容。','模型返回不是加密的；当前 trace 默认只保存长度和 SHA-256 哈希，避免日志过大或泄露提示词/业务内容。','如果开启 trace_model_responses，原文会出现在原始事件 JSON 的 payload.content 中。','接下来 Harness 会把模型回复解析成结构化动作, 比如更新计划、调用工具、或输出最终答案。'];
      }else if(event.kind==='model_response_metadata'){
        lines=['这是模型返回附带的用量信息。','如果供应商返回 token 用量, Harness 会记录在这里, 方便后续统计成本和上下文长度。'];
      }else if(event.kind==='plan_updated'){
        lines=['模型更新了任务计划。','计划不是最终答案, 它更像 agent 给自己列的待办清单, 用来决定后面先做什么。'];
      }else if(event.kind==='tool_result'){
        lines=['模型要求 Harness 调用一个工具, Harness 执行后把结果记录在这里。','工具可能是读文件、搜索文本、执行命令、应用补丁或查看 git diff。','模型本身不能直接碰你的文件系统, 必须通过 Harness 提供的工具。'];
      }else if(event.kind==='command_audit'){
        lines=['这是 shell 命令的安全审计。','Harness 会检查命令是否符合当前策略, 再决定允许、拒绝或要求确认。'];
      }else if(event.kind==='action_error'){
        lines=['模型返回的动作格式不符合 Harness 约定。','Harness 会把错误反馈给模型, 让模型下一轮按正确 JSON/tool 格式重试。'];
      }else if(event.kind==='checkpoint_saved'){
        lines=['Harness 保存当前对话、计划和步骤状态。','这就是你能继续追问或从某次 run 恢复的基础。'];
      }else if(event.kind==='conversation_follow_up'){
        lines=['这是一次基于历史 run 的追问。','Harness 会加载上一次保存的 checkpoint, 再把你的新问题追加进去继续跑。'];
      }else if(event.kind==='final'){
        lines=['模型认为任务已经完成, 输出最终回复。','左侧“对话”展示的助手回复主要来自这里。'];
      }else if(event.kind==='workspace_changes'){
        lines=['Harness 对比运行前后的工作区快照。','这里能看到本次 run 有没有新增、修改或删除文件。'];
      }else if(event.kind==='run_finished'){
        lines=['这次 run 结束了。','状态可能是 completed、failed 或 stopped。stopped 通常表示达到最大循环次数但还没拿到 final。'];
      }else{
        lines=['这是 Harness 记录的一条内部事件。','可以结合下面的关键数据和原始 JSON 看它携带了什么信息。'];
      }
      const body=lines.map((line)=>`<p>${escapeHtml(line)}</p>`).join('');
      return `<div class="explain-title">通俗解释</div>${body}`;
    }

    function eventApiTrace(event){
      const rows=[
        ['GET','/tasks/{task_id}/events','页面实时等待任务进度时, 用 SSE 收到这条 trace。'],
        ['GET','/runs/{run_id}','刷新页面或点击历史 run 时, 从这个接口回放完整事件流。']
      ];
      if(event.kind==='run_started'){
        rows.unshift(['POST','/tasks','你点击“运行”时提交任务, 后端创建 task 和 run。']);
      }else if(event.kind==='conversation_follow_up'){
        rows.unshift(['POST','/tasks','你点击“发送”继续追问时提交 resume_from 和 follow_up。']);
      }else if(event.kind==='model_request'||event.kind==='model_response'||event.kind==='model_response_metadata'||event.kind==='stream_summary'){
        rows.push(['INTERNAL','ModelAdapter','Harness 后端模型适配层把消息发给 OpenAI-compatible 接口。']);
        rows.push(['POST','/chat/completions','这是 DeepSeek/Qwen 等真实大模型 API 的标准路径, 不是浏览器直接调用。']);
      }else if(event.kind==='tool_result'){
        const tool=event.payload?.action?.tool||'tool';
        rows.push(['INTERNAL',`ToolRouter -> ${tool}`,'模型只给出工具名和参数, 后端按工具 schema 执行并记录结果。']);
      }else if(event.kind==='command_audit'){
        rows.push(['INTERNAL','CommandPolicy / CommandExecutor','执行 shell 前先走安全策略审计, 再决定是否允许。']);
      }else if(event.kind==='workspace_changes'){
        rows.push(['GET','/runs/{run_id}/changes','右侧“工作区变更”从这个接口单独读取变更清单。']);
      }else if(event.kind==='checkpoint_saved'){
        rows.push(['GET','/runs/{run_id}/checkpoint','继续追问或恢复运行时会使用这份 checkpoint。']);
      }else if(event.kind==='run_config'){
        rows.push(['GET','/preview-run','页面上方“运行前预检”用这个接口提前展示模型、工具和安全策略。']);
      }
      return `<div class="call-chain">${rows.map(([method,path,note])=>{
        const cls=String(method).toLowerCase();
        return `<div class="call-row"><div class="call-title"><span class="method ${cls}">${escapeHtml(method)}</span><span class="api-path mono">${escapeHtml(path)}</span></div><div class="call-note">${escapeHtml(note)}</div></div>`;
      }).join('')}</div>`;
    }

    function eventDataView(event){
      const p=event.payload||{};
      const rows=[];
      rows.push(['发生时间',event.ts||'-']);
      if(p.step!==undefined)rows.push(['模型轮次',`第 ${p.step} 轮`]);
      if(event.kind==='run_started'){
        rows.push(['用户任务',p.task||'-']);
        rows.push(['run_id',event.run_id||'-']);
      }else if(event.kind==='run_config'){
        rows.push(['运行模式',p.mode||'-']);
        rows.push(['模型供应商',p.provider||'-']);
        rows.push(['模型名称',p.model||'-']);
        rows.push(['最大循环步数',String(p.max_steps??'-')]);
        rows.push(['可用工具数量',String(p.tool_count??'-')]);
        rows.push(['工具列表',Array.isArray(p.tool_names)?p.tool_names.join(', '):'-']);
      }else if(event.kind==='model_request'){
        rows.push(['发送给模型',`${p.message_count??'-'} 条消息`]);
        rows.push(['上下文大小',`${p.messages_chars??'-'} 字符`]);
        rows.push(['实际模型接口','后端模型适配层 -> /chat/completions']);
      }else if(event.kind==='stream_summary'){
        rows.push(['折叠片段数',String(p.count??0)]);
        rows.push(['说明','原始 model_stream_chunk 仍保存在 trace 中']);
      }else if(event.kind==='model_response'){
        rows.push(['返回长度',`${p.content_chars??'-'} 字符`]);
        rows.push(['内容哈希',p.content_sha256||'-']);
        rows.push(['原文是否记录',p.trace_model_responses?'已记录在 payload.content':'默认关闭，只记录长度和哈希']);
        rows.push(['模型原文',p.content||'未写入 trace；这不是加密，开启 trace_model_responses 后会记录']);
      }else if(event.kind==='model_response_metadata'){
        rows.push(['供应商',p.provider||'-']);
        rows.push(['模型',p.model||'-']);
        rows.push(['用量',compactJson(p.usage)]);
      }else if(event.kind==='plan_updated'){
        rows.push(['计划项',compactJson(p.items||p.plan||p)]);
      }else if(event.kind==='tool_result'){
        rows.push(['工具名称',p.action?.tool||'-']);
        rows.push(['工具参数',compactJson(p.action?.args)]);
        rows.push(['执行结果',p.result?.ok===false?'失败':'成功']);
        rows.push(['结果摘要',compactJson(p.result)]);
      }else if(event.kind==='command_audit'){
        rows.push(['命令',p.command||p.action?.args?.command||'-']);
        rows.push(['审计结果',p.decision||p.status||'-']);
        rows.push(['原因',p.reason||'-']);
      }else if(event.kind==='action_error'){
        rows.push(['错误',p.result?.error||p.error||'-']);
        rows.push(['原始动作',compactJson(p.action)]);
      }else if(event.kind==='checkpoint_saved'){
        rows.push(['保存路径',p.path||'-']);
        rows.push(['已完成步数',String(p.step??'-')]);
      }else if(event.kind==='conversation_follow_up'){
        rows.push(['来源 run',p.source_run_id||'-']);
        rows.push(['追问内容',p.follow_up||'-']);
      }else if(event.kind==='final'){
        rows.push(['最终回复',p.content||'-']);
      }else if(event.kind==='workspace_changes'){
        rows.push(['变更文件数',String(p.changed_count??0)]);
        rows.push(['新增',compactJson(p.added||[])]);
        rows.push(['修改',compactJson(p.modified||[])]);
        rows.push(['删除',compactJson(p.deleted||[])]);
      }else if(event.kind==='run_finished'){
        rows.push(['结束状态',p.status||'-']);
        rows.push(['错误',p.error||'-']);
      }else{
        rows.push(['payload',compactJson(p)]);
      }
      return rows.map(([k,v])=>`<div class="explain-row"><span>${escapeHtml(k)}</span><span>${escapeHtml(truncateText(v,420))}</span></div>`).join('');
    }

    function eventLoopIo(event){
      const p=event.payload||{};
      const cards=[];
      if(event.kind==='model_request'){
        cards.push(['输入给模型',`${p.message_count??'-'} 条消息, ${p.messages_chars??'-'} 字符。默认不写完整 prompt, 只写 roles 和 messages_sha256。`,compactJson({roles:p.message_roles, messages_sha256:p.messages_sha256, trace_messages:p.trace_messages})]);
        cards.push(['下一步','等待模型返回 JSON action / tool call / final。','']);
      }else if(event.kind==='model_response'){
        const raw=p.content||'未记录原文；默认只记录 content_chars 和 content_sha256。';
        cards.push(['模型输出',p.trace_model_responses?'已记录完整模型返回。':'未记录完整模型返回；不是加密，是日志策略。',raw]);
        cards.push(['下一步','Harness 会把这段文本交给 parse_action 解析成动作。',compactJson({content_chars:p.content_chars, content_sha256:p.content_sha256, trace_model_responses:p.trace_model_responses})]);
      }else if(event.kind==='plan_updated'){
        cards.push(['解析出的动作','模型输出被解析为计划更新。',compactJson(p.action)]);
        cards.push(['回灌给模型','Harness 把计划记录结果追加进 messages, 下一轮模型会看到。',compactJson(p.plan)]);
      }else if(event.kind==='tool_result'){
        cards.push(['解析出的动作',`模型请求调用工具 ${p.action?.tool||'-'}。`,compactJson(p.action)]);
        cards.push(['工具返回','Harness 执行工具并压缩结果, 再作为 Tool result 回灌给模型。',compactJson(p.result)]);
      }else if(event.kind==='action_error'){
        cards.push(['解析失败','模型返回不符合动作协议, Harness 不会直接执行。',compactJson(p.result||p.error)]);
        cards.push(['回灌给模型','错误会作为 Tool result 反馈给模型, 让它下一轮修正格式。','']);
      }else if(event.kind==='final'){
        cards.push(['最终输出','模型输出 final, agent loop 结束。',p.content||'']);
      }else{
        cards.push(['当前事件','这一步不是模型动作边界, 主要用于记录运行状态或安全审计。',compactJson(p)]);
      }
      return `<div class="loop-io">${cards.map(([title,note,raw])=>`<div class="io-card"><strong>${escapeHtml(title)}</strong><span>${escapeHtml(note)}</span>${raw?`<pre>${escapeHtml(truncateText(raw,1200))}</pre>`:''}</div>`).join('')}</div>`;
    }

    function compactJson(value){
      if(value===undefined||value===null)return'-';
      if(typeof value==='string')return value;
      try{return JSON.stringify(value,null,2);}catch{return String(value);}
    }

    function truncateText(value,max){
      const text=String(value);
      return text.length>max?text.slice(0,max)+'...':text;
    }
    function fillDemoTask(name){const task=DEMO_TASKS[name];if(!task)return;el('task').value=task;el('task').focus();el('activeTask').textContent='已填充演示任务，确认后点击运行。';}
    function assistantText(summary){if(summary?.final)return summary.final;if(summary?.content)return summary.content;if(summary?.error)return summary.error;if(summary?.status==='stopped')return'运行已停止：通常是达到最大步数，但模型还没有输出 final。可以提高 max_steps，或把任务描述得更具体。';if(summary?.status==='failed')return'运行失败，请查看事件详情中的错误信息。';return'等待模型回复。';}
    function renderConversation(summary){const runId=summary?.run_id||state.selectedRunId;el('conversationRun').textContent=runId||'-';if(!summary){el('conversation').innerHTML='<div class="empty">请选择或运行一个任务。</div>';return;}el('conversation').innerHTML=`<div class="message user"><div class="message-role">用户</div><div>${escapeHtml(summary.task||'已选择的 run')}</div></div><div class="message assistant"><div class="message-role">助手</div><div>${escapeHtml(assistantText(summary))}</div></div>`;}
    function showTaskSummary(task){const s=task.run_summary||{};showRunSource({...s,run_id:task.run_id,status:task.status,task_id:task.task_id,mode:task.mode});renderConversation({...s,task:task.task,content:task.result?.content,error:task.error||s.error,status:s.status||task.status,run_id:task.run_id});}
    function showRunSummary(run){showRunSource(run);}
    function showRunSource(run){
      el('runStatus').textContent=run.status||'running';
      el('runStatus').className=statusClass(run.status||'running');
      if(!state.selectedEventSeq){
        el('summary').innerHTML=`<p class="muted">当前 run: <span class="mono">${escapeHtml(run.run_id||'-')}</span></p><p class="muted" style="margin-top:8px;">点击中间“执行过程”的任一步, 右边会解释 Harness 这一刻到底在做什么。</p>`;
      }
    }
    function renderChanges(changes){const changed=Number(changes?.changed_count||0);el('changesBadge').textContent=String(changed);el('changesBadge').className='badge '+(changed>0?'running':'completed');const rows=[];for(const group of ['added','modified','deleted']){const files=Array.isArray(changes?.[group])?changes[group]:[];files.forEach((file)=>rows.push({group,file}));}if(!rows.length){el('changes').innerHTML='<p class="muted">没有工作区文件变更。</p>';return;}el('changes').innerHTML=`<div class="changes">${rows.map((r)=>`<div class="change-row"><span class="change-path mono">${escapeHtml(r.file)}</span><span class="badge">${escapeHtml(r.group)}</span></div>`).join('')}</div>`;}
    function statusClass(status){return`badge ${String(status||'').toLowerCase()}`;} function escapeHtml(value){return String(value).replace(/[&<>"']/g,(ch)=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch]));}
    document.querySelectorAll('[data-template]').forEach((button)=>button.addEventListener('click',()=>fillDemoTask(button.dataset.template)));el('submit').addEventListener('click',()=>submitTask().catch((e)=>alert(e.message)));el('sendFollowup').addEventListener('click',()=>sendFollowup().catch((e)=>alert(e.message)));el('refresh').addEventListener('click',()=>refreshAll().catch((e)=>alert(e.message)));el('mode').addEventListener('change',()=>refreshRunPreview().catch((e)=>alert(e.message)));el('provider').addEventListener('change',()=>refreshRunPreview().catch((e)=>alert(e.message)));el('traceModelResponses').addEventListener('change',()=>refreshRunPreview().catch((e)=>alert(e.message)));refreshAll().catch((e)=>{el('health').textContent='异常';el('health').className='badge failed';el('summary').innerHTML=`<pre>${escapeHtml(e.message)}</pre>`;});
  </script>
</body>
</html>
"""
