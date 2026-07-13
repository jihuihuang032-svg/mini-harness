from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from harness.eval_runner import evaluate_case, load_eval_cases


class EvalRunnerTests(unittest.TestCase):
    def test_example_smoke_eval_file_loads(self) -> None:
        cases = load_eval_cases(Path("examples/smoke.jsonl"))

        self.assertEqual(len(cases), 1)
        self.assertEqual(cases[0].id, "mock-smoke")
        self.assertIn("Offline mock run completed", cases[0].expect_contains)

    def test_load_eval_cases_from_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cases.jsonl"
            path.write_text(
                '{"id":"smoke","task":"Inspect this project","expect_contains":["Offline mock run completed"]}\n',
                encoding="utf-8",
            )

            cases = load_eval_cases(path)

            self.assertEqual(len(cases), 1)
            self.assertEqual(cases[0].id, "smoke")
            self.assertEqual(cases[0].expect_contains, ["Offline mock run completed"])

    def test_evaluate_case_reports_missing_expectations(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cases.jsonl"
            path.write_text('{"id":"missing","task":"demo","expect_contains":"needle"}\n', encoding="utf-8")
            case = load_eval_cases(path)[0]

            result = evaluate_case(case, content="haystack", run_id="run-1", steps=1)

            self.assertFalse(result.ok)
            self.assertEqual(result.missing, ["needle"])

    def test_evaluate_case_treats_stopped_as_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cases.jsonl"
            path.write_text('{"id":"stopped","task":"demo"}\n', encoding="utf-8")
            case = load_eval_cases(path)[0]

            result = evaluate_case(case, content="Stopped after reaching max_steps=1.", run_id="run-1", steps=1)

            self.assertFalse(result.ok)


if __name__ == "__main__":
    unittest.main()
