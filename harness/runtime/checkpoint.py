"""运行检查点:每步保存运行状态,便于崩溃后 resume。

每次 Agent 主循环执行一个 step,都会调用 RunCheckpointStore.save
把当前 messages + plan + step 落盘到 <workspace>/.harness/checkpoints/<run_id>.json。
崩溃或主动中断后,Agent.resume 会读 checkpoint 接着跑。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from harness.messages import Message
from harness.planning import PlanState


# checkpoint 文件格式版本号
CHECKPOINT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class RunCheckpoint:
    """单次检查点的数据:run_id/task/step/status/messages/plan。"""
    run_id: str
    task: str
    step: int
    status: str
    messages: list[Message]
    plan: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        """序列化为可写入 JSON 的 dict。"""
        return {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "run_id": self.run_id,
            "task": self.task,
            "step": self.step,
            "status": self.status,
            "messages": [message.to_api() for message in self.messages],
            "plan": self.plan,
        }


class RunCheckpointStore:
    """检查点存储:每个 run 一个 JSON 文件。"""

    def __init__(self, checkpoints_dir: Path) -> None:
        self.checkpoints_dir = checkpoints_dir
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)

    def save(self, checkpoint: RunCheckpoint) -> Path:
        """保存检查点(覆盖式写,每个 run 只保留最新)。"""
        path = self._path_for_run(checkpoint.run_id)
        path.write_text(json.dumps(checkpoint.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def load(self, run_id: str) -> dict[str, object]:
        """读取检查点原始 dict(不做类型校验)。"""
        path = self._path_for_run(run_id)
        if not path.exists():
            raise ValueError(f"Checkpoint not found: {run_id}")
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid checkpoint JSON in {path}: {exc}") from exc
        if not isinstance(raw, dict):
            raise ValueError(f"Checkpoint must be a JSON object: {path}")
        return raw

    def load_state(self, run_id: str) -> "LoadedCheckpoint":
        """读取检查点并解析为强类型 LoadedCheckpoint(供 Agent.resume 使用)。"""
        raw = self.load(run_id)
        task = _required_str(raw, "task")
        step = _required_int(raw, "step")
        status = _required_str(raw, "status")
        messages = _messages_from_raw(raw.get("messages"))
        plan_raw = raw.get("plan", {})
        if not isinstance(plan_raw, dict):
            raise ValueError(f"Checkpoint field 'plan' must be an object: {run_id}")
        return LoadedCheckpoint(
            run_id=_required_str(raw, "run_id"),
            task=task,
            step=step,
            status=status,
            messages=messages,
            plan=PlanState.from_snapshot(plan_raw),
        )

    def _path_for_run(self, run_id: str) -> Path:
        """根据 run_id 构造文件路径,并防止路径穿越。

        Path(run_id).name 只取最后一段,避免 run_id 形如 "../../etc/passwd" 越界。
        """
        safe = Path(run_id).name
        if safe != run_id:
            raise ValueError(f"Invalid run id: {run_id}")
        return self.checkpoints_dir / f"{safe}.json"


@dataclass(frozen=True)
class LoadedCheckpoint:
    """已解析的检查点:用于 Agent.resume 接续运行。"""
    run_id: str
    task: str
    step: int
    status: str
    messages: list[Message]
    plan: PlanState


def _messages_from_raw(raw: object) -> list[Message]:
    """从 raw 列表解析出 Message 列表(带类型校验)。"""
    if not isinstance(raw, list):
        raise ValueError("Checkpoint field 'messages' must be a list.")
    messages: list[Message] = []
    for index, item in enumerate(raw, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Checkpoint message {index} must be an object.")
        role = item.get("role")
        content = item.get("content")
        if not isinstance(role, str) or not isinstance(content, str):
            raise ValueError(f"Checkpoint message {index} requires string role and content.")
        messages.append(Message(role=role, content=content))
    return messages


def _required_str(raw: dict[str, object], key: str) -> str:
    """从 raw 取必填字符串字段。"""
    value = raw.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Checkpoint field {key!r} must be a non-empty string.")
    return value


def _required_int(raw: dict[str, object], key: str) -> int:
    """从 raw 取必填整数字段(拒绝 bool,因为 Python 中 True 是 int 子类)。"""
    value = raw.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"Checkpoint field {key!r} must be an integer.")
    return value
