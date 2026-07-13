"""评测运行器:加载 JSONL 测试用例并评估 agent 输出。

eval 的本质:给定 (task, expect_contains) 一组用例,让 agent 跑一遍,
然后检查 agent 输出是否包含期望的关键词(expect_contains)。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class EvalCase:
    """单条评测用例:id + 任务 + 期望包含的关键词列表。"""
    id: str
    task: str
    expect_contains: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {"id": self.id, "task": self.task, "expect_contains": self.expect_contains}


@dataclass(frozen=True)
class EvalResult:
    """单条评测结果:是否通过 + agent 输出 + 缺失的关键词。"""
    id: str
    ok: bool
    run_id: str | None
    steps: int | None
    content: str
    error: str | None = None
    missing: list[str] = field(default_factory=list)  # 没在 content 中找到的 expect_contains 项

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "ok": self.ok,
            "run_id": self.run_id,
            "steps": self.steps,
            "content": self.content,
            "error": self.error,
            "missing": self.missing,
        }


def load_eval_cases(path: Path) -> list[EvalCase]:
    """从 JSONL 文件加载评测用例,每行一个 {"id", "task", "expect_contains"}。"""
    cases: list[EvalCase] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid eval JSONL line {line_number}: {exc}") from exc
        if not isinstance(raw, dict):
            raise ValueError(f"Eval case line {line_number} must be an object.")
        case_id = raw.get("id", f"case-{line_number}")
        task = raw.get("task")
        expected = raw.get("expect_contains", [])
        if not isinstance(case_id, str) or not case_id:
            raise ValueError(f"Eval case line {line_number} field 'id' must be a non-empty string.")
        if not isinstance(task, str) or not task:
            raise ValueError(f"Eval case line {line_number} field 'task' must be a non-empty string.")
        # expect_contains 既可以是字符串也可以是字符串列表,这里统一转列表
        if isinstance(expected, str):
            expected = [expected]
        if not isinstance(expected, list) or not all(isinstance(item, str) for item in expected):
            raise ValueError(f"Eval case line {line_number} field 'expect_contains' must be a string or string list.")
        cases.append(EvalCase(id=case_id, task=task, expect_contains=expected))
    return cases


def evaluate_case(case: EvalCase, content: str, run_id: str | None, steps: int | None, error: str | None = None) -> EvalResult:
    """评估单条用例:检查 agent 输出是否包含所有期望关键词。

    判定 ok 的三个条件:
        1. 无 error
        2. content 不是因 max_steps 提前停止
        3. expect_contains 全部出现在 content 中
    """
    missing = [expected for expected in case.expect_contains if expected not in content]
    stopped = content.startswith("Stopped after reaching max_steps=")
    ok = error is None and not stopped and not missing
    return EvalResult(id=case.id, ok=ok, run_id=run_id, steps=steps, content=content, error=error, missing=missing)
