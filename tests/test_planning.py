from __future__ import annotations

import unittest

from harness.planning import PlanState


class PlanStateTests(unittest.TestCase):
    def test_replace_accepts_string_and_object_items(self) -> None:
        plan = PlanState()
        snapshot = plan.replace([
            "Inspect files",
            {"id": "custom", "content": "Run tests", "status": "pending"},
        ])

        self.assertEqual(snapshot["items"][0], {"id": "1", "content": "Inspect files", "status": "pending"})
        self.assertEqual(snapshot["items"][1], {"id": "custom", "content": "Run tests", "status": "pending"})

    def test_replace_reuses_existing_content_when_model_sends_status_only_items(self) -> None:
        plan = PlanState()
        plan.replace([
            {"id": "1", "content": "Inspect files", "status": "in_progress"},
            {"id": "2", "content": "Summarize result", "status": "pending"},
        ])

        snapshot = plan.replace([
            {"id": "1", "status": "completed"},
            {"id": "2", "status": "in_progress"},
        ])

        self.assertEqual(snapshot["items"][0], {"id": "1", "content": "Inspect files", "status": "completed"})
        self.assertEqual(snapshot["items"][1], {"id": "2", "content": "Summarize result", "status": "in_progress"})
    def test_update_changes_status(self) -> None:
        plan = PlanState()
        plan.replace([{"id": "1", "content": "Inspect", "status": "in_progress"}])
        snapshot = plan.update([{"id": "1", "status": "completed"}])

        self.assertEqual(snapshot["items"][0]["status"], "completed")
        self.assertEqual(snapshot["counts"]["completed"], 1)

    def test_update_rejects_unknown_id(self) -> None:
        plan = PlanState()
        plan.replace(["Inspect"])
        with self.assertRaises(ValueError):
            plan.update([{"id": "missing", "status": "completed"}])

    def test_rejects_invalid_status(self) -> None:
        plan = PlanState()
        with self.assertRaises(ValueError):
            plan.replace([{"id": "1", "content": "Inspect", "status": "done"}])


if __name__ == "__main__":
    unittest.main()
