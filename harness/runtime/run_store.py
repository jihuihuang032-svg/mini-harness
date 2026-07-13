"""Run trace storage and query helpers.

RunLogger writes append-only JSONL traces under .harness/logs. RunStore reads those
traces and derives summaries for CLI/server views. Some newer code may also pass a
separate .harness/runs directory for persisted summaries, so the constructor keeps
that argument optional for compatibility.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from harness.runtime.logger import TRACE_SCHEMA_VERSION


def _empty_changes() -> dict[str, object]:
    return {"changed_count": 0, "added": [], "modified": [], "deleted": []}


def _empty_usage() -> dict[str, int]:
    return {"response_count": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


def _empty_tools() -> dict[str, object]:
    return {"total_calls": 0, "failed_calls": 0, "tool_names": [], "by_tool": {}}


@dataclass(frozen=True)
class RunSummary:
    run_id: str
    task: str = ""
    status: str = "running"
    step_count: int = 0
    event_count: int = 0
    started_at: str = ""
    finished_at: str = ""
    final: str | None = None
    error: str | None = None
    changes: dict[str, object] = field(default_factory=_empty_changes)
    usage: dict[str, int] | None = None
    tools: dict[str, object] = field(default_factory=_empty_tools)
    trace_schema_version: int = TRACE_SCHEMA_VERSION

    @property
    def last_event_at(self) -> str:
        return self.finished_at

    @property
    def final_message(self) -> str | None:
        return self.final

    def to_dict(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "task": self.task,
            "status": self.status,
            "step_count": self.step_count,
            "event_count": self.event_count,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "final": self.final,
            "final_message": self.final,
            "error": self.error,
            "changes": self.changes,
            "usage": self.usage,
            "tools": self.tools,
            "trace_schema_version": self.trace_schema_version,
        }


class RunStore:
    def __init__(self, logs_dir: Path, runs_dir: Path | None = None) -> None:
        self.logs_dir = logs_dir
        self.runs_dir = runs_dir or logs_dir.parent / "runs"
        self.changes_dir = logs_dir.parent / "changes"

    def write_summary(self, summary: RunSummary) -> Path:
        run_dir = self.runs_dir / summary.run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        path = run_dir / "summary.json"
        path.write_text(json.dumps(summary.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def list_runs(self, limit: int | None = None) -> list[RunSummary]:
        summaries: list[RunSummary] = []
        seen: set[str] = set()
        for trace_path in sorted(self.logs_dir.glob("*.jsonl"), reverse=True):
            run_id = trace_path.stem
            try:
                summaries.append(self.summarize_run(run_id))
                seen.add(run_id)
            except ValueError:
                continue
            if limit is not None and len(summaries) >= limit:
                return summaries
        if self.runs_dir.exists():
            for entry in sorted(self.runs_dir.iterdir(), reverse=True):
                if limit is not None and len(summaries) >= limit:
                    break
                if not entry.is_dir() or entry.name in seen:
                    continue
                summary_path = entry / "summary.json"
                if not summary_path.exists():
                    continue
                try:
                    summaries.append(_load_summary(summary_path))
                except ValueError:
                    continue
        return summaries

    def load_summary(self, run_id: str) -> RunSummary:
        return self.summarize_run(run_id)

    def summarize_run(self, run_id: str) -> RunSummary:
        events = self.load_run(run_id)
        return _summarize_events(run_id, events, _changes_from_events(events))

    def load_run(self, run_id: str) -> list[dict[str, Any]]:
        return self.load_events(run_id)

    def load_events(self, run_id: str) -> list[dict[str, Any]]:
        path = self._trace_path_for(run_id)
        if not path.exists():
            raise ValueError(f"Run trace not found: {run_id}")
        events: list[dict[str, Any]] = []
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid trace JSON line {line_number}: {exc}") from exc
            if not isinstance(raw, dict):
                raise ValueError(f"Trace line {line_number} must be an object.")
            events.append(raw)
        return events

    def load_changes(self, run_id: str) -> dict[str, object]:
        safe = self._safe_run_id(run_id)
        artifact = self.changes_dir / f"{safe}.json"
        if artifact.exists():
            try:
                raw = json.loads(artifact.read_text(encoding="utf-8"))
                if isinstance(raw, dict):
                    return raw
            except json.JSONDecodeError:
                pass
        return _changes_from_events(self.load_run(run_id))

    def _trace_path_for(self, run_id: str) -> Path:
        return self.logs_dir / f"{self._safe_run_id(run_id)}.jsonl"

    def _safe_run_id(self, run_id: str) -> str:
        safe = Path(run_id).name
        if safe != run_id:
            raise ValueError(f"Invalid run id: {run_id}")
        return safe


def _changes_from_events(events: list[dict[str, Any]]) -> dict[str, object]:
    changes = _empty_changes()
    for event in events:
        if event.get("kind") == "workspace_changes" and isinstance(event.get("payload"), dict):
            changes = event["payload"]
    return changes


def _summarize_events(run_id: str, events: list[dict[str, Any]], changes: dict[str, object]) -> RunSummary:
    task = ""
    status = "running"
    final: str | None = None
    error: str | None = None
    started_at = events[0].get("ts", "") if events else ""
    finished_at = events[-1].get("ts", "") if events else ""
    step_count = 0
    trace_schema_version = TRACE_SCHEMA_VERSION
    usage = _empty_usage()
    by_tool: dict[str, dict[str, int]] = {}
    total_calls = 0
    failed_calls = 0

    for event in events:
        payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
        kind = event.get("kind")
        if isinstance(event.get("schema_version"), int):
            trace_schema_version = event["schema_version"]
        if kind == "run_started":
            trace_schema_version = int(payload.get("trace_schema_version", trace_schema_version))
            if payload.get("task") is not None:
                task = str(payload.get("task"))
        elif kind == "model_request":
            raw_step = payload.get("step")
            if isinstance(raw_step, int) and raw_step > step_count:
                step_count = raw_step
        elif kind == "model_response_metadata":
            raw_usage = payload.get("usage")
            if isinstance(raw_usage, dict):
                usage["response_count"] += 1
                for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                    value = raw_usage.get(key)
                    if isinstance(value, int) and not isinstance(value, bool):
                        usage[key] += value
        elif kind == "tool_result":
            action = payload.get("action")
            result = payload.get("result")
            tool_name = "unknown"
            if isinstance(action, dict) and isinstance(action.get("tool"), str):
                tool_name = action["tool"]
            ok = isinstance(result, dict) and result.get("ok") is True
            total_calls += 1
            if not ok:
                failed_calls += 1
            stats = by_tool.setdefault(tool_name, {"calls": 0, "failures": 0})
            stats["calls"] += 1
            if not ok:
                stats["failures"] += 1
        elif kind == "final":
            raw_final = payload.get("content")
            if isinstance(raw_final, str):
                final = raw_final
        elif kind == "run_finished":
            raw_status = payload.get("status")
            if isinstance(raw_status, str):
                status = raw_status
            raw_error = payload.get("error")
            if isinstance(raw_error, str):
                error = raw_error
            finished_at = str(event.get("ts", finished_at))

    return RunSummary(
        run_id=run_id,
        task=task,
        status=status,
        step_count=step_count,
        event_count=len(events),
        started_at=str(started_at),
        finished_at=str(finished_at),
        final=final,
        error=error,
        changes=changes,
        usage=usage if usage["response_count"] > 0 else None,
        tools={
            "total_calls": total_calls,
            "failed_calls": failed_calls,
            "tool_names": sorted(by_tool),
            "by_tool": by_tool,
        },
        trace_schema_version=trace_schema_version,
    )


def _load_summary(path: Path) -> RunSummary:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid summary JSON in {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"Summary must be an object: {path}")
    return RunSummary(
        run_id=_required_str(raw, "run_id"),
        task=_optional_str(raw, "task") or "",
        status=_optional_str(raw, "status") or "running",
        step_count=_optional_int(raw, "step_count") or 0,
        event_count=_optional_int(raw, "event_count") or 0,
        started_at=_optional_str(raw, "started_at") or "",
        finished_at=_optional_str(raw, "finished_at") or "",
        final=_optional_str(raw, "final") or _optional_str(raw, "final_message"),
        error=_optional_str(raw, "error"),
        changes=_optional_dict(raw, "changes") or _empty_changes(),
        usage=_optional_dict(raw, "usage"),
        tools=_optional_dict(raw, "tools") or _empty_tools(),
        trace_schema_version=_optional_int(raw, "trace_schema_version") or TRACE_SCHEMA_VERSION,
    )


def _required_str(raw: dict[str, Any], key: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Summary field {key!r} must be a non-empty string.")
    return value


def _optional_str(raw: dict[str, Any], key: str) -> str | None:
    value = raw.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"Summary field {key!r} must be a string or null.")
    return value


def _optional_int(raw: dict[str, Any], key: str) -> int | None:
    value = raw.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"Summary field {key!r} must be an integer.")
    return value


def _optional_dict(raw: dict[str, Any], key: str) -> dict[str, Any] | None:
    value = raw.get(key)
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError(f"Summary field {key!r} must be an object.")
    return value


