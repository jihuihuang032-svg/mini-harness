"""工作区文件变更追踪。

Agent 运行前后各拍一次快照(文件大小 + sha256),
对比得到 added/modified/deleted 列表,落盘到 .harness/changes/<run_id>.json。
用于事后审计:Agent 这次到底改了哪些文件。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from harness.runtime.workspace import Workspace


# 快照时忽略的目录(这些目录的内容不应算作"工作区变更")
IGNORED_DIRS = {".git", ".harness", "__pycache__", "node_modules", ".venv", "venv"}


@dataclass(frozen=True)
class FileState:
    """单文件快照:相对路径 + 大小 + sha256。"""
    path: str
    size: int
    sha256: str

    def to_dict(self) -> dict[str, object]:
        return {"path": self.path, "size": self.size, "sha256": self.sha256}


@dataclass(frozen=True)
class WorkspaceSnapshot:
    """整个工作区的快照:文件路径 -> FileState。"""
    files: dict[str, FileState]

    def to_dict(self) -> dict[str, object]:
        return {"file_count": len(self.files), "files": [state.to_dict() for state in self.files.values()]}


@dataclass(frozen=True)
class WorkspaceChanges:
    """两次快照的对比结果:added/modified/deleted 三个列表。"""
    added: list[str]
    modified: list[str]
    deleted: list[str]

    @property
    def changed_count(self) -> int:
        """总变更文件数。"""
        return len(self.added) + len(self.modified) + len(self.deleted)

    def to_dict(self) -> dict[str, object]:
        return {
            "changed_count": self.changed_count,
            "added": self.added,
            "modified": self.modified,
            "deleted": self.deleted,
        }


class WorkspaceChangeTracker:
    """工作区变更追踪器:capture / compare / save。"""

    def __init__(self, workspace: Workspace) -> None:
        self.workspace = workspace

    def capture(self) -> WorkspaceSnapshot:
        """扫描整个工作区,计算每个文件的 size+sha256。"""
        files: dict[str, FileState] = {}
        for path in sorted(self.workspace.root.rglob("*")):
            if not path.is_file() or self._is_ignored(path):
                continue
            relative = self.workspace.relative(path)
            files[relative] = FileState(path=relative, size=path.stat().st_size, sha256=_sha256(path))
        return WorkspaceSnapshot(files=files)

    def compare(self, before: WorkspaceSnapshot, after: WorkspaceSnapshot) -> WorkspaceChanges:
        """对比两次快照,得到 added/modified/deleted。"""
        before_paths = set(before.files)
        after_paths = set(after.files)
        added = sorted(after_paths - before_paths)        # after 有 before 没有
        deleted = sorted(before_paths - after_paths)      # before 有 after 没有
        modified = sorted(
            path
            for path in before_paths & after_paths  # 两边都有的,比对 size+sha256
            if before.files[path].sha256 != after.files[path].sha256
            or before.files[path].size != after.files[path].size
        )
        return WorkspaceChanges(added=added, modified=modified, deleted=deleted)

    def save(self, run_id: str, changes: WorkspaceChanges) -> Path:
        """把变更结果落盘到 .harness/changes/<run_id>.json。"""
        self.workspace.changes_dir.mkdir(parents=True, exist_ok=True)
        path = self.workspace.changes_dir / f"{run_id}.json"
        path.write_text(json.dumps(changes.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def _is_ignored(self, path: Path) -> bool:
        """判断路径是否落在忽略目录内。"""
        relative_parts = path.relative_to(self.workspace.root).parts
        return any(part in IGNORED_DIRS for part in relative_parts)


def _sha256(path: Path) -> str:
    """计算文件 sha256,按 1MB 块流式读取避免大文件 OOM。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        # iter(callable, sentinel):反复调用 callable 直到返回 sentinel
        # 这里反复读 1MB,直到读到空 bytes(b"")
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
