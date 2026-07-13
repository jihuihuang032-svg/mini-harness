"""运行 trace 日志器。

每次 run 对应一个 .jsonl 文件,记录所有事件:
    - run_started / run_finished
    - model_request / model_response_metadata
    - action / tool_result
    - workspace_changes
    - final

JSONL = JSON Lines,每行一个 JSON 对象,便于流式追加和后续解析。
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


# trace 文件格式版本号,后续格式变更时递增
TRACE_SCHEMA_VERSION = 1


class RunLogger:
    """运行 trace 日志器,每个 run 一个文件,事件按顺序追加。"""

    def __init__(self, logs_dir: Path, run_id: str | None = None) -> None:
        logs_dir.mkdir(parents=True, exist_ok=True)
        # run_id 不传则自动生成:时间戳 + uuid 前 8 位
        self.run_id = run_id or new_run_id()
        self.path = logs_dir / f"{self.run_id}.jsonl"
        # sequence 是事件序号,用于后续按顺序解析
        self.sequence = 0
        # 构造时即记录 run_started 事件
        self.event("run_started", {"trace_schema_version": TRACE_SCHEMA_VERSION})

    def event(self, kind: str, payload: dict[str, object]) -> dict[str, object]:
        """记录一条事件,以 JSONL 形式追加到文件。

        @param kind: 事件类型(如 model_response_metadata / tool_result)
        @param payload: 事件数据
        @return: 完整记录(含元数据)
        """
        self.sequence += 1
        record = {
            "schema_version": TRACE_SCHEMA_VERSION,
            "seq": self.sequence,
            "ts": datetime.now(timezone.utc).isoformat(),
            "run_id": self.run_id,
            "kind": kind,
            "payload": payload,
        }
        # 追加模式写入,每次 event 都立即落盘(便于崩溃后恢复)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        return record


def new_run_id() -> str:
    """生成 run_id:UTC 时间戳 + uuid4 前 8 位,保证全局唯一且可读。"""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
