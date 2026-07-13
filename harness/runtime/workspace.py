"""工作区:Agent 操作文件的沙箱根。

所有文件类工具(list_files/read_file/write_file 等)都通过 Workspace.resolve
解析路径,resolve 会强制把路径限制在 workspace.root 内,防止 "../" 越界。
"""

from __future__ import annotations

from pathlib import Path


class Workspace:
    """工作区,封装 root 路径与各子目录位置。"""

    def __init__(self, root: Path) -> None:
        # resolve() 把相对路径转绝对路径,并消除 ".." / 符号链接
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        # .harness 目录用于存放 logs/tasks/changes/checkpoints/evals
        self.logs_dir = self.root / ".harness" / "logs"
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.tasks_path = self.root / ".harness" / "tasks.jsonl"
        self.changes_dir = self.root / ".harness" / "changes"
        self.checkpoints_dir = self.root / ".harness" / "checkpoints"
        self.evals_dir = self.root / ".harness" / "evals"

    def resolve(self, path: str | Path) -> Path:
        """把任意路径解析为 workspace 内的绝对路径,做沙箱检查。

        - 相对路径:相对 workspace.root
        - 绝对路径:直接用
        - 路径跳出 workspace -> 抛 PermissionError(沙箱核心边界)
        """
        candidate = (self.root / path).resolve() if not Path(path).is_absolute() else Path(path).resolve()
        # 所有文件路径都必须落在 workspace 内,这是文件工具的基础沙箱边界
        if candidate != self.root and self.root not in candidate.parents:
            raise PermissionError(f"Path escapes workspace: {path}")
        return candidate

    def relative(self, path: Path) -> str:
        """把绝对路径转回相对 workspace.root 的字符串(给模型看)。"""
        return str(path.resolve().relative_to(self.root))
