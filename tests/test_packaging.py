from __future__ import annotations

import tomllib
import unittest
from pathlib import Path

from harness import __version__


class PackagingTests(unittest.TestCase):
    def test_prompt_markdown_is_included_as_package_data(self) -> None:
        pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))

        package_data = pyproject["tool"]["setuptools"]["package-data"]

        self.assertIn("prompts/*.md", package_data["harness"])

    def test_package_version_matches_project_metadata(self) -> None:
        pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))

        self.assertEqual(__version__, pyproject["project"]["version"])


if __name__ == "__main__":
    unittest.main()
