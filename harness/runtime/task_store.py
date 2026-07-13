"""任务存储:TaskQueue 的 JSONL 持久化实现。

文件布局:.harness/tasks.jsonl,每行一条 TaskRecord 快照。
由于同一 task 会被多次 append(状态变化时各写一次),
load_latest 用 dict 去重,只保留每个 task_id 的最后一条。
"""

from __future__ import annotations

import json
from pathlib import Path

from harness.runtime.task_queue import TaskRecord


class TaskStore:
    """TaskPersistence 的 JSONL 实现。"""

    def __init__(self, path: Path) -> None:
        self.path = path
        # 父目录不存在则创建,parents=True 递归创建
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: TaskRecord) -> None:
        """追加一条记录到 JSONL 文件。

        用 "a" 模式打开文件:不存在则创建,存在则在末尾追加。
        ensure_ascii=False:让中文字符直接写入,而不是 \\uXXXX 转义。
        """
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")

    def load_latest(self) -> list[TaskRecord]:
        """读取所有记录,返回每个 task_id 的最新一条。

        实现思路:逐行解析,用 dict 按 task_id 覆盖,最终取 values()。
        """
        if not self.path.exists():
            return []
        latest: dict[str, TaskRecord] = {}
        # enumerate(splitlines(), start=1):逐行带行号迭代
        for line_number, line in enumerate(self.path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue  # 跳过空行
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                # 行损坏:抛带行号的错误,便于定位
                raise ValueError(f"Invalid task store JSON line {line_number}: {exc}") from exc
            if not isinstance(raw, dict):
                raise ValueError(f"Task store record line {line_number} must be an object.")
            record = task_record_from_dict(raw)
            # 直接覆盖:后写的总是更新的
            latest[record.task_id] = record
        return list(latest.values())


def task_record_from_dict(raw: dict[str, object]) -> TaskRecord:
    """从 dict 解析 TaskRecord,带字段类型校验和默认值。"""
    task_id = _required_str(raw, "task_id")
    run_id = _required_str(raw, "run_id")
    task = _required_str(raw, "task")
    stream = bool(raw.get("stream", False))
    # 状态字段:不在白名单内则降级为 failed(防止脏数据)
    status = str(raw.get("status", "queued"))
    if status not in {"queued", "running", "completed", "failed"}:
        status = "failed"
    metadata = raw.get("metadata", {})
    result = raw.get("result")
    return TaskRecord(
        task_id=task_id,
        run_id=run_id,
        task=task,
        stream=stream,
        mode=str(raw.get("mode", "mock")),
        provider=raw.get("provider") if isinstance(raw.get("provider"), str) else None,
        metadata=metadata if isinstance(metadata, dict) else {},
        status=status,  # type: ignore[arg-type]  # mypy 忽略:已校验过白名单
        created_at=str(raw.get("created_at", "")),
        updated_at=str(raw.get("updated_at", "")),
        result=result if isinstance(result, dict) else None,
        error=raw.get("error") if isinstance(raw.get("error"), str) else None,
    )


def _required_str(raw: dict[str, object], key: str) -> str:
    """从 raw 取必填字符串字段(缺失或非字符串则抛错)。"""
    value = raw.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Task store record missing string field {key!r}.")
    return value
