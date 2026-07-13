"""工具子包入口。

提供 build_default_router 工厂:把所有内置工具注册到 ToolRouter,
并按 tool_profile 决定哪些工具对模型可见/可执行。
类似 Spring 的 @Configuration 类:集中装配所有 Tool Bean。
"""
from __future__ import annotations

from harness.runtime.executor import CommandExecutor
from harness.runtime.workspace import Workspace
from harness.tools.base import ToolProfile, ToolRouter
from harness.tools.file_tools import ApplyPatchTool, ListFilesTool, ReadFileTool, SearchTextTool, WriteFileTool
from harness.tools.git_tools import GitDiffTool, GitStatusTool
from harness.tools.shell_tools import RunCommandTool


def build_default_router(
    workspace: Workspace,
    executor: CommandExecutor,
    max_output_chars: int,
    tool_profile: ToolProfile = "full",
) -> ToolRouter:
    """构造默认工具路由器,注册全部内置工具。

    @param workspace: 工作区,文件类工具操作文件的根
    @param executor: 命令执行器,shell/git 类工具用它跑命令
    @param max_output_chars: 单次工具输出字符上限
    @param tool_profile: 工具权限配置 full/review/read-only
    """
    router = ToolRouter(profile=tool_profile)
    # 文件类工具
    router.register(ListFilesTool(workspace))
    router.register(ReadFileTool(workspace, max_output_chars))
    router.register(WriteFileTool(workspace))
    router.register(SearchTextTool(workspace, max_output_chars))
    router.register(ApplyPatchTool(workspace, max_output_chars))
    # Shell / Git 类工具
    router.register(RunCommandTool(executor))
    router.register(GitStatusTool(executor))
    router.register(GitDiffTool(executor))
    return router
