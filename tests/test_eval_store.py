from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from harness.runtime.eval_store import EvalStore


class EvalStoreTests(unittest.TestCase):
    def test_save_list_and_load_eval_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = EvalStore(Path(tmp))

            report = store.save({"ok": True, "total": 1, "passed": 1, "failed": 0, "results": []}, eval_id="eval-1")
            summaries = store.list_evals()
            loaded = store.load("eval-1")

            self.assertEqual(report["eval_id"], "eval-1")
            self.assertEqual(len(summaries), 1)
            self.assertEqual(summaries[0].eval_id, "eval-1")
            self.assertTrue(summaries[0].ok)
            self.assertEqual(loaded["passed"], 1)

    def test_rejects_path_like_eval_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = EvalStore(Path(tmp))

            with self.assertRaises(ValueError):
                store.load("../bad")


if __name__ == "__main__":
    unittest.main()
