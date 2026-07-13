from __future__ import annotations

import unittest
from pathlib import Path


class ProjectFilesTests(unittest.TestCase):
    def test_gitignore_protects_runtime_and_secret_files(self) -> None:
        patterns = set(Path(".gitignore").read_text(encoding="utf-8").splitlines())

        self.assertIn(".env", patterns)
        self.assertIn("!.env.example", patterns)
        self.assertIn("harness.json", patterns)
        self.assertIn("!harness.json.example", patterns)
        self.assertIn(".harness/", patterns)
        self.assertIn("__pycache__/", patterns)
        self.assertIn("dist/", patterns)


if __name__ == "__main__":
    unittest.main()
