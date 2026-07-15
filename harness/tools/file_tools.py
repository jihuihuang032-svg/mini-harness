"""Built-in file tools.

All file operations go through Workspace.resolve so model actions cannot escape the
configured workspace. Tool output is truncated before returning to the agent loop.
"""

from __future__ import annotations

import re
import subprocess

from harness.runtime.executor import _truncate
from harness.runtime.workspace import Workspace


class ListFilesTool:
    name = "list_files"
    description = "List files under a workspace directory."
    args_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "default": "."},
            "limit": {"type": "integer", "default": 200},
        },
    }

    def __init__(self, workspace: Workspace) -> None:
        self.workspace = workspace

    def run(self, args: dict[str, object]) -> dict[str, object]:
        path = self.workspace.resolve(str(args.get("path", ".")))
        if not path.exists():
            return {"ok": False, "error": f"Path does not exist: {path}"}
        if path.is_file():
            return {"ok": True, "files": [self.workspace.relative(path)]}
        limit = int(args.get("limit", 200))
        files: list[str] = []
        for child in sorted(path.rglob("*")):
            if ".harness" in child.parts:
                continue
            files.append(self.workspace.relative(child))
            if len(files) >= limit:
                break
        return {"ok": True, "files": files, "truncated": len(files) >= limit}


class ReadFileTool:
    name = "read_file"
    description = "Read a UTF-8 text file from the workspace, optionally selecting a line range."
    args_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "start_line": {"type": "integer", "default": 1},
            "max_lines": {"type": "integer", "default": 0},
            "line_numbers": {"type": "boolean", "default": False},
        },
        "required": ["path"],
    }

    def __init__(self, workspace: Workspace, max_output_chars: int) -> None:
        self.workspace = workspace
        self.max_output_chars = max_output_chars

    def run(self, args: dict[str, object]) -> dict[str, object]:
        path = self.workspace.resolve(_required_str(args, "path"))
        text = path.read_text(encoding="utf-8")
        start_line = int(args.get("start_line", 1))
        max_lines = int(args.get("max_lines", 0))
        if start_line < 1:
            return {"ok": False, "error": "start_line must be >= 1"}
        if max_lines < 0:
            return {"ok": False, "error": "max_lines must be >= 0"}

        lines = text.splitlines()
        start_index = min(start_line - 1, len(lines))
        end_index = len(lines) if max_lines == 0 else min(start_index + max_lines, len(lines))
        selected = lines[start_index:end_index]
        if bool(args.get("line_numbers", False)):
            selected = [f"{line_number}: {line}" for line_number, line in enumerate(selected, start=start_line)]
        rendered = "\n".join(selected)
        content = _truncate(rendered, self.max_output_chars)
        return {
            "ok": True,
            "path": self.workspace.relative(path),
            "content": content,
            "start_line": start_line,
            "end_line": start_line + len(selected) - 1 if selected else start_line - 1,
            "total_lines": len(lines),
            "truncated": end_index < len(lines) or len(content) < len(rendered),
        }


class WriteFileTool:
    name = "write_file"
    description = "Write a UTF-8 text file inside the workspace."
    args_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "content": {"type": "string"},
        },
        "required": ["path", "content"],
    }

    def __init__(self, workspace: Workspace) -> None:
        self.workspace = workspace

    def run(self, args: dict[str, object]) -> dict[str, object]:
        path = self.workspace.resolve(_required_str(args, "path"))
        content = _required_str(args, "content")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return {"ok": True, "path": self.workspace.relative(path), "bytes": len(content.encode("utf-8"))}


class SearchTextTool:
    name = "search_text"
    description = "Search workspace text files with literal or regex matching and optional line context."
    args_schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "path": {"type": "string", "default": "."},
            "regex": {"type": "boolean", "default": False},
            "case_sensitive": {"type": "boolean", "default": True},
            "max_matches": {"type": "integer", "default": 100},
            "context_lines": {"type": "integer", "default": 0},
        },
        "required": ["query"],
    }

    def __init__(self, workspace: Workspace, max_output_chars: int) -> None:
        self.workspace = workspace
        self.max_output_chars = max_output_chars

    def run(self, args: dict[str, object]) -> dict[str, object]:
        query = _required_str(args, "query")
        path = self.workspace.resolve(str(args.get("path", ".")))
        if not path.exists():
            return {"ok": False, "error": f"Path does not exist: {path}"}
        regex = bool(args.get("regex", False))
        case_sensitive = bool(args.get("case_sensitive", True))
        max_matches = int(args.get("max_matches", 100))
        context_lines = int(args.get("context_lines", 0))
        if max_matches < 1:
            return {"ok": False, "error": "max_matches must be >= 1"}
        if context_lines < 0:
            return {"ok": False, "error": "context_lines must be >= 0"}

        matcher = _build_matcher(query, regex=regex, case_sensitive=case_sensitive)
        if isinstance(matcher, dict):
            return matcher

        matches: list[dict[str, object]] = []
        files = [path] if path.is_file() else path.rglob("*")
        for file_path in files:
            if not file_path.is_file() or ".harness" in file_path.parts:
                continue
            try:
                lines = file_path.read_text(encoding="utf-8").splitlines()
            except UnicodeDecodeError:
                continue
            for line_number, line in enumerate(lines, start=1):
                if matcher(line):
                    match: dict[str, object] = {
                        "path": self.workspace.relative(file_path),
                        "line": line_number,
                        "text": line,
                    }
                    if context_lines:
                        match["context"] = _line_context(lines, line_number, context_lines)
                    matches.append(match)
                    if len(matches) >= max_matches:
                        return {
                            "ok": True,
                            "matches": matches,
                            "match_count": len(matches),
                            "truncated": True,
                            "preview": _truncate(str(matches), self.max_output_chars),
                        }
        return {
            "ok": True,
            "matches": matches,
            "match_count": len(matches),
            "truncated": False,
            "preview": _truncate(str(matches), self.max_output_chars),
        }


class ApplyPatchTool:
    name = "apply_patch"
    description = "Apply a unified diff patch inside the workspace using git apply."
    args_schema = {
        "type": "object",
        "properties": {"patch": {"type": "string"}},
        "required": ["patch"],
    }

    def __init__(self, workspace: Workspace, max_output_chars: int) -> None:
        self.workspace = workspace
        self.max_output_chars = max_output_chars

    def run(self, args: dict[str, object]) -> dict[str, object]:
        patch = _required_str(args, "patch")
        invalid_path = _invalid_patch_path(patch, self.workspace)
        if invalid_path is not None:
            return {"ok": False, "error": f"Patch path escapes workspace: {invalid_path}"}
        completed = subprocess.run(
            ["git", "apply", "--whitespace=nowarn", "-"],
            cwd=self.workspace.root,
            input=patch,
            text=True,
            capture_output=True,
            timeout=30,
        )
        return {
            "ok": completed.returncode == 0,
            "returncode": completed.returncode,
            "stdout": _truncate(completed.stdout, self.max_output_chars),
            "stderr": _truncate(completed.stderr, self.max_output_chars),
        }


def _required_str(args: dict[str, object], name: str) -> str:
    value = args.get(name)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Missing required string arg: {name}")
    return value


def _build_matcher(query: str, regex: bool, case_sensitive: bool):
    if regex:
        flags = 0 if case_sensitive else re.IGNORECASE
        try:
            pattern = re.compile(query, flags)
        except re.error as exc:
            return {"ok": False, "error": f"Invalid regex: {exc}"}
        return pattern.search
    needle = query if case_sensitive else query.lower()

    def literal_match(line: str) -> bool:
        haystack = line if case_sensitive else line.lower()
        return needle in haystack

    return literal_match


def _invalid_patch_path(patch: str, workspace: Workspace) -> str | None:
    for path in _patch_paths(patch):
        try:
            workspace.resolve(path)
        except PermissionError:
            return path
    return None


def _patch_paths(patch: str) -> list[str]:
    paths: list[str] = []
    for line in patch.splitlines():
        if line.startswith("diff --git "):
            parts = line.split()
            raw_paths = parts[2:4]
        elif line.startswith("--- ") or line.startswith("+++ "):
            raw_paths = [line[4:].split("\t", 1)[0].strip()]
        else:
            continue
        for raw_path in raw_paths:
            if raw_path == "/dev/null":
                continue
            if raw_path.startswith(("a/", "b/")):
                raw_path = raw_path[2:]
            paths.append(raw_path)
    return paths


def _line_context(lines: list[str], line_number: int, radius: int) -> list[dict[str, object]]:
    start = max(1, line_number - radius)
    end = min(len(lines), line_number + radius)
    return [{"line": number, "text": lines[number - 1]} for number in range(start, end + 1)]
