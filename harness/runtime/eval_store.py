"""评测报告存储:save / list / load / summarize。

每次 eval 跑完一组测试用例,会生成一个完整 report(ok/total/passed/failed + 每条用例详情),
落到 .harness/evals/<eval_id>.json。EvalStore 提供查询接口:
    - save:写入新报告
    - list_evals:列出最近的报告(只返回摘要)
    - load:读取完整报告
    - summarize:返回单个报告的摘要
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4


@dataclass(frozen=True)
class EvalSummary:
    """评测摘要:不带详情,只统计 ok/total/passed/failed。"""
    eval_id: str
    path: Path
    created_at: str
    ok: bool
    total: int
    passed: int
    failed: int

    def to_dict(self) -> dict[str, object]:
        return {
            "eval_id": self.eval_id,
            "path": str(self.path),
            "created_at": self.created_at,
            "ok": self.ok,
            "total": self.total,
            "passed": self.passed,
            "failed": self.failed,
        }


class EvalStore:
    """评测报告存储:管理 .harness/evals/ 下的 JSON 文件。"""

    def __init__(self, evals_dir: Path) -> None:
        self.evals_dir = evals_dir

    def save(self, report: dict[str, object], eval_id: str | None = None) -> dict[str, object]:
        """保存一份评测报告,返回写入磁盘的完整 dict(含 eval_id/created_at)。"""
        self.evals_dir.mkdir(parents=True, exist_ok=True)
        # eval_id 不传则自动生成:时间戳 + uuid 前 8 位
        actual_id = eval_id or new_eval_id()
        path = self._path_for_eval(actual_id)
        # 拷贝一份再补充元数据,避免修改调用方传入的 dict
        stored = dict(report)
        stored["eval_id"] = actual_id
        stored["created_at"] = datetime.now(timezone.utc).isoformat()
        # indent=2:格式化输出,便于人工查看
        path.write_text(json.dumps(stored, ensure_ascii=False, indent=2), encoding="utf-8")
        return stored

    def list_evals(self, limit: int | None = None) -> list[EvalSummary]:
        """列出最近的评测摘要,按文件名倒序(新的在前)。"""
        if not self.evals_dir.exists():
            return []
        summaries: list[EvalSummary] = []
        # glob("*.json") 匹配所有 .json 文件;sorted(reverse=True) 让新的在前
        for path in sorted(self.evals_dir.glob("*.json"), reverse=True):
            try:
                # path.stem 是文件名去掉扩展名,即 eval_id
                summaries.append(self.summarize(path.stem))
            except ValueError:
                # 单个文件损坏不阻塞整个列表
                continue
            if limit is not None and len(summaries) >= limit:
                break
        return summaries

    def load(self, eval_id: str) -> dict[str, Any]:
        """读取完整评测报告。"""
        path = self._path_for_eval(eval_id)
        if not path.exists():
            raise ValueError(f"Eval report not found: {eval_id}")
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid eval report JSON in {path}: {exc}") from exc
        if not isinstance(raw, dict):
            raise ValueError(f"Eval report must be an object: {path}")
        return raw

    def summarize(self, eval_id: str) -> EvalSummary:
        """读取报告并提取摘要。"""
        report = self.load(eval_id)
        path = self._path_for_eval(eval_id)
        return EvalSummary(
            eval_id=str(report.get("eval_id", eval_id)),
            path=path,
            created_at=str(report.get("created_at", "")),
            ok=bool(report.get("ok", False)),
            total=_int_field(report, "total"),
            passed=_int_field(report, "passed"),
            failed=_int_field(report, "failed"),
        )

    def _path_for_eval(self, eval_id: str) -> Path:
        """根据 eval_id 构造文件路径,并防止路径穿越。

        Path(eval_id).name 只取最后一段,避免 eval_id 形如 "../../etc/passwd" 越界。
        """
        safe = Path(eval_id).name
        if safe != eval_id:
            raise ValueError(f"Invalid eval id: {eval_id}")
        return self.evals_dir / f"{safe}.json"


def new_eval_id() -> str:
    """生成 eval_id:UTC 时间戳 + uuid4 前 8 位,保证全局唯一且可读。"""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]


def _int_field(report: dict[str, Any], name: str) -> int:
    """从报告取整数字段(缺失或非 int 返回 0,容忍脏数据)。"""
    value = report.get(name, 0)
    return value if isinstance(value, int) else 0
