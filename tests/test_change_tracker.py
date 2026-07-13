from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from harness.runtime.change_tracker import WorkspaceChangeTracker
from harness.runtime.workspace import Workspace


class ChangeTrackerTests(unittest.TestCase):
    def test_detects_added_modified_and_deleted_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "keep.txt").write_text("before", encoding="utf-8")
            (root / "delete.txt").write_text("delete me", encoding="utf-8")
            (root / ".harness" / "logs").mkdir(parents=True)
            (root / ".harness" / "logs" / "ignored.jsonl").write_text("ignored", encoding="utf-8")
            tracker = WorkspaceChangeTracker(Workspace(root))
            before = tracker.capture()

            (root / "keep.txt").write_text("after", encoding="utf-8")
            (root / "added.txt").write_text("new", encoding="utf-8")
            (root / "delete.txt").unlink()
            after = tracker.capture()

            changes = tracker.compare(before, after)

            self.assertEqual(changes.added, ["added.txt"])
            self.assertEqual(changes.modified, ["keep.txt"])
            self.assertEqual(changes.deleted, ["delete.txt"])
            self.assertEqual(changes.changed_count, 3)

    def test_saves_change_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Workspace(Path(tmp))
            tracker = WorkspaceChangeTracker(workspace)
            before = tracker.capture()
            (workspace.root / "added.txt").write_text("new", encoding="utf-8")
            changes = tracker.compare(before, tracker.capture())

            path = tracker.save("run-1", changes)

            self.assertEqual(path, workspace.changes_dir / "run-1.json")
            self.assertIn('"changed_count": 1', path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
