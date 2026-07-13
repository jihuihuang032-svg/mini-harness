"""HTTP server:把 harness 暴露为 REST API + SSE 流式接口。

接口概览:
    GET  /                  -> 内置 console HTML(单页可视化)
    GET  /health            -> 健康检查
    GET  /providers         -> 列出内置 provider 预设
    GET  /runs              -> 列出最近 run
    GET  /runs/<id>         -> 加载某 run 的完整事件流
    GET  /runs/<id>/changes -> 加载某 run 的文件变更
    GET  /runs/<id>/checkpoint -> 加载某 run 的最新 checkpoint
    GET  /tasks             -> 列出所有任务
    GET  /tasks/<id>        -> 查询单个任务
    GET  /tasks/<id>/events -> SSE 流式订阅任务进度
    POST /tasks             -> 提交新任务(mock / model / resume)

实现基于 Python 标准库 http.server,无第三方依赖。
"""

from __future__ import annotations

import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from harness.agent import Agent, render_system_prompt
from harness.config import HarnessConfig
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


class HarnessServer:
    """HTTP server 的应用层:封装所有业务方法,Handler 只负责 HTTP 协议。

    把业务逻辑从 HTTP 解析中拆出来,便于测试(可以直接调 app.run_mock_task)。
    """
    def __init__(self, workspace: str | None = None, config_path: str | None = None) -> None:
        self.config = HarnessConfig.offline(workspace, config_path)
        self.config_path = config_path
        self.workspace = Workspace(self.config.workspace)
        self.store = RunStore(self.workspace.logs_dir)
        self.checkpoints = RunCheckpointStore(self.workspace.checkpoints_dir)
        self.task_store = TaskStore(self.workspace.tasks_path)
        self.tasks = TaskQueue(self.run_task, store=self.task_store)

    def health(self) -> dict[str, object]:
        return {"ok": True, "workspace": str(self.workspace.root)}

    def console_html(self) -> str:
        return CONSOLE_HTML.replace("__WORKSPACE__", str(self.workspace.root))

    def providers(self) -> list[dict[str, str]]:
        return [PROVIDER_PRESETS[name].to_dict() for name in provider_names()]

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
        return self.tasks.get(task_id).to_dict()

    def submit_mock_task(self, task: str, stream: bool = False) -> dict[str, object]:
        if not task:
            raise ValueError("Task is required.")
        return self.tasks.submit(task, stream=stream, mode="mock", metadata={"mock": True}).to_dict()

    def submit_model_task(self, task: str, provider: str | None, stream: bool = False) -> dict[str, object]:
        if not task:
            raise ValueError("Task is required.")
        if not provider:
            raise ValueError("Field 'provider' is required when mock=false.")
        return self.tasks.submit(
            task,
            stream=stream,
            mode="model",
            provider=provider,
            metadata={"mock": False, "provider": provider},
        ).to_dict()

    def submit_resume_task(
        self,
        resume_from: str,
        mock: bool = True,
        provider: str | None = None,
        stream: bool = False,
    ) -> dict[str, object]:
        checkpoint = self.checkpoints.load_state(resume_from)
        if checkpoint.status == "completed":
            raise ValueError(f"Checkpoint is already completed: {resume_from}")
        if mock is False and not provider:
            raise ValueError("Field 'provider' is required when mock=false.")
        metadata: dict[str, object] = {"mock": mock, "resume_from": resume_from}
        if provider is not None:
            metadata["provider"] = provider
        return self.tasks.submit(
            checkpoint.task,
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
        """TaskQueue 的 runner 回调:根据 metadata 分发到 mock 或 model 实现。

        这个方法是 TaskQueue.submit 的实际执行体,在 daemon thread 内被调用。
        """
        resume_from = metadata.get("resume_from")
        resume_run_id = resume_from if isinstance(resume_from, str) else None
        if metadata.get("mock", True) is False:
            provider = metadata.get("provider")
            provider_name = provider if isinstance(provider, str) else None
            return self.run_model_task(task, provider_name, stream=stream, run_id=run_id, resume_from=resume_run_id)
        return self.run_mock_task(task, stream=stream, run_id=run_id, resume_from=resume_run_id)

    def run_mock_task(
        self,
        task: str,
        stream: bool = False,
        run_id: str | None = None,
        resume_from: str | None = None,
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
        executor = CommandExecutor(
            self.workspace,
            CommandPolicy.default(self.config.command_profile),
            self.config.timeout_seconds,
            self.config.max_tool_output_chars,
            ApprovalController("never"),
        )
        router = build_default_router(
            self.workspace,
            executor,
            self.config.max_tool_output_chars,
            tool_profile=self.config.tool_profile,
        )
        tool_specs = router.specs()
        system_prompt = render_system_prompt(router)
        chunks: list[str] = []
        logger.event(
            "run_config",
            run_config_snapshot(
                self.config,
                mode="mock",
                stream=stream,
                tool_profile=self.config.tool_profile,
                command_profile=self.config.command_profile,
                approval="never",
                tool_specs=tool_specs,
                system_prompt=system_prompt,
                resume_from=resume_from,
            ),
        )
        logger.event("tool_profile", {"profile": self.config.tool_profile})
        logger.event("command_profile", {"profile": self.config.command_profile})
        try:
            agent = Agent(
                config=self.config,
                model=MockModelClient(calls=loaded_checkpoint.step if loaded_checkpoint is not None else 0),
                tools=router,
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
    ) -> dict[str, object]:
        if not task:
            raise ValueError("Task is required.")
        config = HarnessConfig.from_env(str(self.workspace.root), provider, self.config_path)
        loaded_checkpoint = self.checkpoints.load_state(resume_from) if resume_from is not None else None
        if loaded_checkpoint is not None and loaded_checkpoint.status == "completed":
            raise ValueError(f"Checkpoint is already completed: {resume_from}")
        logger = RunLogger(self.workspace.logs_dir, run_id=run_id)
        tracker = WorkspaceChangeTracker(self.workspace)
        before = tracker.capture()
        logger.event("workspace_snapshot", {"phase": "before", "file_count": len(before.files)})
        executor = CommandExecutor(
            self.workspace,
            CommandPolicy.default(config.command_profile),
            config.timeout_seconds,
            config.max_tool_output_chars,
            ApprovalController("never"),
        )
        router = build_default_router(
            self.workspace,
            executor,
            config.max_tool_output_chars,
            tool_profile=config.tool_profile,
        )
        tool_specs = router.specs()
        system_prompt = render_system_prompt(router)
        chunks: list[str] = []
        logger.event(
            "run_config",
            run_config_snapshot(
                config,
                mode="model",
                stream=stream,
                tool_profile=config.tool_profile,
                command_profile=config.command_profile,
                approval="never",
                tool_specs=tool_specs,
                system_prompt=system_prompt,
                resume_from=resume_from,
            ),
        )
        logger.event("tool_profile", {"profile": config.tool_profile})
        logger.event("command_profile", {"profile": config.command_profile})
        try:
            agent = Agent(
                config=config,
                model=OpenAICompatibleClient(config, tool_specs),
                tools=router,
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


def create_handler(app: HarnessServer) -> type[BaseHTTPRequestHandler]:
    """构造 HTTP 请求处理器类(闭包捕获 app 实例)。

    返回 type 而不是实例,因为 ThreadingHTTPServer 需要的是类(它会自己实例化)。
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
                mock = payload.get("mock", True)
                stream = bool(payload.get("stream", False))
                if resume_from is not None:
                    if not isinstance(resume_from, str):
                        self._json(400, {"error": "Field 'resume_from' must be a string."})
                        return
                    provider = payload.get("provider")
                    if provider is not None and not isinstance(provider, str):
                        self._json(400, {"error": "Field 'provider' must be a string."})
                        return
                    self._json(202, app.submit_resume_task(resume_from, mock=mock is not False, provider=provider, stream=stream))
                    return
                if not isinstance(task, str):
                    self._json(400, {"error": "Field 'task' must be a string."})
                    return
                if mock is False:
                    provider = payload.get("provider")
                    if provider is not None and not isinstance(provider, str):
                        self._json(400, {"error": "Field 'provider' must be a string."})
                        return
                    self._json(202, app.submit_model_task(task, provider, stream=stream))
                    return
                self._json(202, app.submit_mock_task(task, stream=stream))
            except Exception as exc:
                self._json(400, {"error": str(exc)})

        def log_message(self, format: str, *args: object) -> None:
            return

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
    """启动 HTTP server(阻塞,直到 Ctrl+C 或 kill)。"""
    app = HarnessServer(workspace, config_path)
    server = ThreadingHTTPServer((host, port), create_handler(app))
    print(f"Mini Harness server listening on http://{host}:{port}")
    print(f"Workspace: {Path(app.workspace.root)}")
    server.serve_forever()


CONSOLE_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Mini Harness Console</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f6f7f9;
      --panel: #ffffff;
      --line: #d8dde6;
      --text: #1f2937;
      --muted: #667085;
      --accent: #2563eb;
      --ok: #047857;
      --bad: #b42318;
      --code: #111827;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font: 14px/1.45 ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--text);
      background: var(--bg);
    }
    header {
      height: 56px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 0 20px;
      border-bottom: 1px solid var(--line);
      background: var(--panel);
    }
    h1 {
      margin: 0;
      font-size: 16px;
      font-weight: 650;
    }
    main {
      display: grid;
      grid-template-columns: 360px 1fr;
      min-height: calc(100vh - 56px);
    }
    aside {
      border-right: 1px solid var(--line);
      background: var(--panel);
      padding: 16px;
    }
    section {
      padding: 16px;
    }
    label {
      display: block;
      margin: 0 0 6px;
      font-weight: 600;
    }
    textarea {
      width: 100%;
      min-height: 118px;
      resize: vertical;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 10px;
      font: inherit;
      background: #fff;
    }
    button {
      border: 1px solid #1d4ed8;
      border-radius: 6px;
      background: var(--accent);
      color: white;
      padding: 8px 12px;
      font-weight: 600;
      cursor: pointer;
    }
    button.secondary {
      color: var(--text);
      background: #fff;
      border-color: var(--line);
    }
    .row {
      display: flex;
      gap: 8px;
      align-items: center;
      margin-top: 10px;
    }
    .meta {
      color: var(--muted);
      font-size: 12px;
      overflow-wrap: anywhere;
    }
    .grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 16px;
    }
    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      min-height: 180px;
      overflow: hidden;
    }
    .panel h2 {
      margin: 0;
      padding: 12px 14px;
      border-bottom: 1px solid var(--line);
      font-size: 14px;
    }
    .list {
      max-height: 420px;
      overflow: auto;
    }
    .item {
      padding: 10px 14px;
      border-bottom: 1px solid var(--line);
    }
    .item:last-child { border-bottom: 0; }
    .status {
      display: inline-block;
      min-width: 76px;
      padding: 2px 6px;
      border-radius: 999px;
      background: #eef2ff;
      color: #3730a3;
      font-size: 12px;
      text-align: center;
    }
    .status.completed { background: #ecfdf3; color: var(--ok); }
    .status.failed { background: #fef3f2; color: var(--bad); }
    pre {
      margin: 0;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      color: var(--code);
      font: 12px/1.45 ui-monospace, SFMono-Regular, Consolas, "Liberation Mono", monospace;
    }
    @media (max-width: 860px) {
      main { grid-template-columns: 1fr; }
      aside { border-right: 0; border-bottom: 1px solid var(--line); }
      .grid { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <header>
    <h1>Mini Harness Console</h1>
    <div class="meta">__WORKSPACE__</div>
  </header>
  <main>
    <aside>
      <label for="task">Task</label>
      <textarea id="task">Inspect this project</textarea>
      <label style="margin-top:10px;" for="mode">Mode</label>
      <select id="mode" style="width:100%;height:34px;border:1px solid var(--line);border-radius:6px;">
        <option value="mock">Mock</option>
        <option value="model">Real provider</option>
      </select>
      <label style="margin-top:10px;" for="provider">Provider</label>
      <select id="provider" style="width:100%;height:34px;border:1px solid var(--line);border-radius:6px;"></select>
      <div class="row">
        <button id="submit">Run Task</button>
        <button class="secondary" id="refresh">Refresh</button>
      </div>
      <p class="meta" id="active">No active task.</p>
    </aside>
    <section>
      <div class="grid">
        <div class="panel">
          <h2>Tasks</h2>
          <div class="list" id="tasks"></div>
        </div>
        <div class="panel">
          <h2>Runs</h2>
          <div class="list" id="runs"></div>
        </div>
      </div>
      <div class="panel" style="margin-top:16px;">
        <h2>Event Timeline</h2>
        <div class="list" id="events"></div>
      </div>
    </section>
  </main>
  <script>
    const $ = (id) => document.getElementById(id);
    let source = null;

    async function json(url, options) {
      const response = await fetch(url, options);
      if (!response.ok) throw new Error(await response.text());
      return response.json();
    }

    function renderItem(container, html) {
      const div = document.createElement("div");
      div.className = "item";
      div.innerHTML = html;
      container.prepend(div);
    }

    function statusClass(status) {
      return "status " + (status || "").toLowerCase();
    }

    async function refresh() {
      const providers = await json("/providers");
      if (!$("provider").children.length) {
        providers.forEach(provider => {
          const option = document.createElement("option");
          option.value = provider.name;
          option.textContent = `${provider.name} (${provider.default_model})`;
          $("provider").appendChild(option);
        });
      }
      const tasks = await json("/tasks");
      $("tasks").innerHTML = "";
      tasks.reverse().forEach(task => {
        renderItem($("tasks"), `<span class="${statusClass(task.status)}">${task.status}</span>
          <div>${escapeHtml(task.task)}</div>
          <div class="meta">${task.task_id}<br>${task.run_id}</div>`);
      });
      const runs = await json("/runs?limit=20");
      $("runs").innerHTML = "";
      runs.reverse().forEach(run => {
        renderItem($("runs"), `<span class="${statusClass(run.status)}">${run.status}</span>
          <div>${escapeHtml(run.final || "")}</div>
          <div class="meta">${run.run_id}<br>${run.event_count} events</div>`);
      });
    }

    async function submitTask() {
      const task = $("task").value.trim();
      if (!task) return;
      const result = await json("/tasks", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({
          task,
          mock: $("mode").value === "mock",
          provider: $("provider").value,
          stream: true
        })
      });
      $("active").textContent = `Task ${result.task_id} -> run ${result.run_id}`;
      $("events").innerHTML = "";
      follow(result.task_id);
      await refresh();
    }

    function follow(taskId) {
      if (source) source.close();
      source = new EventSource(`/tasks/${taskId}/events`);
      source.addEventListener("task", (event) => addEvent("task", JSON.parse(event.data)));
      source.addEventListener("trace", (event) => addEvent("trace", JSON.parse(event.data)));
      source.addEventListener("done", async (event) => {
        addEvent("done", JSON.parse(event.data));
        source.close();
        await refresh();
      });
      source.addEventListener("timeout", (event) => addEvent("timeout", JSON.parse(event.data)));
    }

    function addEvent(type, data) {
      renderItem($("events"), `<div><strong>${type}</strong></div><pre>${escapeHtml(JSON.stringify(data, null, 2))}</pre>`);
    }

    function escapeHtml(value) {
      return String(value).replace(/[&<>"']/g, ch => ({
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        '"': "&quot;",
        "'": "&#39;"
      }[ch]));
    }

    $("submit").addEventListener("click", () => submitTask().catch(err => alert(err.message)));
    $("refresh").addEventListener("click", () => refresh().catch(err => alert(err.message)));
    refresh().catch(err => console.error(err));
  </script>
</body>
</html>
"""
