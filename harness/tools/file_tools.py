"""文件类工具实现。

提供 5 个工具:
    - ListFilesTool:列目录
    - ReadFileTool:读文件(支持行范围)
    - WriteFileTool:写文件
    - SearchTextTool:文本搜索(字面量或正则)
    - ApplyPatchTool:应用 unified diff 补丁(通过 git apply)

所有文件操作都被限制在 workspace 内(workspace.resolve 会做沙箱检查)。
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

from harness.runtime.executor import _truncate
from harness.runtime.workspace import Workspace


class ListFilesTool:
    """list_files:列出工作区某目录下的所有文件(递归)。"""
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
            # 单文件:直接返回这一项
            return {"ok": True, "files": [self.workspace.relative(path)]}
        limit = int(args.get("limit", 200))
        files: list[str] = []
        for child in sorted(path.rglob("*")):  # rglob("*") 递归遍历
            # 跳过 .harness 内部目录,避免污染输出
            if ".harness" in child.parts:
                continue
            files.append(self.workspace.relative(child))
            if len(files) >= limit:
                break
        return {"ok": True, "files": files, "truncated": len(files) >= limit}


class ReadFileTool:
    """read_file:读 UTF-8 文本文件,可选行范围、是否带行号。"""
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
        max_lines = int(args.get("max_lines", 0))  # 0 表示读到末尾
        if start_line < 1:
            return {"ok": False, "error": "start_line must be >= 1"}
        if max_lines < 0:
            return {"ok": False, "error": "max_lines must be >= 0"}

        lines = text.splitlines()
        # 把 1-based 行号转成 0-based 索引
        start_index = min(start_line - 1, len(lines))
        end_index = len(lines) if max_lines == 0 else min(start_index + max_lines, len(lines))
        selected = lines[start_index:end_index]
        # 可选:加行号前缀(类似 cat -n)
        if bool(args.get("line_numbers", False)):
            selected = [f"{line_number}: {line}" for line_number, line in enumerate(selected, start=start_line)]
        rendered = "\n".join(selected)
        # 截断到字符上限,防止单次读耗尽上下文
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
    """write_file:写 UTF-8 文本文件(覆盖)。"""
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
        # 自动创建父目录,类似 Java Files.createDirectories
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return {"ok": True, "path": self.workspace.relative(path), "bytes": len(content.encode("utf-8"))}


class SearchTextTool:
    """search_text:文本搜索,支持字面量/正则、大小写敏感、上下文行。"""
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
        # 构造匹配器:正则用 pattern.search,字面量用 in 判断
        matcher = _build_matcher(query, regex=regex, case_sensitive=case_sensitive)
        if isinstance(matcher, dict):
            # 构造匹配器失败(如非法正则),直接返回错误
            return matcher

        matches: list[dict[str, object]] = []
        files = [path] if path.is_file() else path.rglob("*")
        for file_path in files:
            if not file_path.is_file() or ".harness" in file_path.parts:
                continue
            try:
                lines = file_path.read_text(encoding="utf-8").splitlines()
            except UnicodeDecodeError:
                # 二进制文件跳过
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
    """apply_patch:应用 unified diff 补丁(通过 git apply)。"""
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
        # 通过 stdin 把 patch 内容传给 git apply,避免写临时文件
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
    """从 args 取必填字符串,缺失或非字符串则抛 ValueError(被 router 捕获)。"""
    value = args.get(name)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Missing required string arg: {name}")
    return value


def _build_matcher(query: str, regex: bool, case_sensitive: bool):
    """构造一个匹配函数(line -> bool)。

    正则模式:用 re.compile + pattern.search
    字面量模式:用 in 判断
    构造失败时返回一个错误 dict(而不是抛异常),便于上层直接 return。
    """
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


def _line_context(lines: list[str], line_number: int, radius: int) -> list[dict[str, object]]:
    """取匹配行附近若干行的上下文(类似 grep -C)。"""
    start = max(1, line_number - radius)
    end = min(len(lines), line_number + radius)
    return [{"line": number, "text": lines[number - 1]} for number in range(start, end + 1)]
