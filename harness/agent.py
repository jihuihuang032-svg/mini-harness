"""Agent 主循环模块。

本模块是整个框架的核心,类似 Spring 中的 Service 层。
Agent 持有模型客户端(ModelClient)、工具路由器(ToolRouter)、日志器等依赖,
反复执行"模型生成动作 -> 框架执行动作 -> 把结果回填给模型"的循环,
直到模型返回 final 动作或达到步数/token 上限。
"""

from __future__ import annotations  # 允许在类型注解中使用尚未定义的类名,类似 Java 前向引用

import json
import hashlib
from collections.abc import Callable
from dataclasses import dataclass  # dataclass 类似 Java 的 record,自动生成 __init__ 等方法
from pathlib import Path

from harness.actions import parse_action
from harness.config import HarnessConfig
from harness.context import ContextBudget, ContextManager
from harness.messages import Message
from harness.models.base import ModelClient
from harness.planning import PlanState
from harness.runtime.checkpoint import RunCheckpoint, RunCheckpointStore
from harness.runtime.logger import RunLogger
from harness.runtime.workspace import Workspace
from harness.tools.base import ToolRouter


# 流式回调签名:每收到一个 chunk 就回调一次,类型相当于 Java 的 Consumer<String>
StreamCallback = Callable[[str], None]


@dataclass(frozen=True)  # frozen=True 表示不可变,类似 Java 的 final record
class AgentResult:
    """Agent 一次运行的最终结果。"""
    content: str  # 模型给出的最终文本
    steps: int   # 实际执行的步数


class Agent:
    """编码 Agent 主体。

    依赖通过构造器注入(类似 Spring 构造器注入):
        - config:全局配置
        - model:模型客户端(决定调真实 LLM 还是 mock)
        - tools:工具路由器(决定哪些工具对模型可见、可执行)
        - logger:运行 trace 日志器
        - workspace:工作区(沙箱根目录)
        - context:上下文管理器(负责消息裁剪、工具结果压缩)
        - plan:计划状态(可选,从 checkpoint 恢复时复用)
        - checkpoint_store:检查点存储,每步落盘便于恢复
    """

    def __init__(
        self,
        config: HarnessConfig,
        model: ModelClient,
        tools: ToolRouter,
        logger: RunLogger,
        workspace: Workspace | None = None,  # X | None 相当于 Java 的 Optional<X>
        context: ContextManager | None = None,
        plan: PlanState | None = None,
        checkpoint_store: RunCheckpointStore | None = None,
        stream: bool = False,
        stream_callback: StreamCallback | None = None,
    ) -> None:
        self.config = config
        self.model = model
        self.tools = tools
        self.logger = logger
        # None 时使用默认实现,等价于 Java 中的 `workspace != null ? workspace : new Workspace(...)`
        self.workspace = workspace or Workspace(config.workspace)
        self.context = context or ContextManager(
            self.workspace,
            ContextBudget(
                max_message_chars=config.max_context_chars,
                max_tool_result_chars=config.max_tool_output_chars,
                max_summary_files=config.max_summary_files,
            ),
        )
        self.plan = plan or PlanState()
        self.checkpoint_store = checkpoint_store or RunCheckpointStore(self.workspace.checkpoints_dir)
        self.stream = stream
        self.stream_callback = stream_callback
        self.total_model_tokens = 0  # 累计 token 用量,用于 max_run_tokens 预算控制

    def run(self, task: str) -> AgentResult:
        """从零开始执行一个新任务。失败时记录 run_finished 事件并重新抛出。"""
        try:
            return self._run(task)
        except Exception as exc:
            self.logger.event("run_finished", {"status": "failed", "error": str(exc), "plan": self.plan.snapshot()})
            raise

    def resume(self, task: str, messages: list[Message], completed_steps: int, source_run_id: str) -> AgentResult:
        """从一个已存在的 checkpoint 恢复运行。

        与 run 的区别:messages/completed_steps 来自旧 run,新 run 的步数上限
        在旧基础上继续累加 max_steps。
        """
        try:
            self.logger.event(
                "resumed_from",
                {
                    "run_id": source_run_id,
                    "completed_steps": completed_steps,
                    "message_count": len(messages),
                    "plan": self.plan.snapshot(),
                },
            )
            return self._run_loop(task, list(messages), completed_steps=completed_steps, max_step=completed_steps + self.config.max_steps)
        except Exception as exc:
            self.logger.event("run_finished", {"status": "failed", "error": str(exc), "plan": self.plan.snapshot()})
            raise

    def _run(self, task: str) -> AgentResult:
        """新建运行的内部入口:组装初始消息 -> 写入 checkpoint -> 进入主循环。"""
        # 初始上下文 = system 提示词 + 仓库摘要 + 用户任务,类似 Java 中拼装 Prompt 模板
        messages = self.context.initial_messages(render_system_prompt(self.tools), task)
        self.logger.event("context_summary", {"summary": self.context.repo_summary})
        self._save_checkpoint(task, 0, "running", messages)
        return self._run_loop(task, messages, completed_steps=0, max_step=self.config.max_steps)

    def _run_loop(self, task: str, messages: list[Message], completed_steps: int, max_step: int) -> AgentResult:
        """Agent 主循环。

        每一步:
            1. 裁剪上下文到字符预算内
            2. 调用模型得到响应
            3. 解析为动作(动作协议由 actions.parse_action 定义)
            4. 根据 type 分发:final -> 结束;plan/todo_update -> 更新计划;
               tool_call -> 执行工具;解析失败 -> 把错误反馈回循环
            5. 落盘 checkpoint
        """
        # 已完成步数 >= 上限,直接停止(恢复场景下 completed_steps 可能已经接近上限)
        if completed_steps >= max_step:
            content = f"Stopped after reaching max_steps={max_step}."
            self._save_checkpoint(task, max_step, "stopped", messages)
            self.logger.event("run_finished", {"status": "stopped", "reason": "max_steps", "max_steps": max_step})
            return AgentResult(content=content, steps=max_step)
        # 主循环只负责编排:准备上下文、调用模型、解析动作、执行工具并保存检查点。
        for step in range(completed_steps + 1, max_step + 1):
            # 进入模型前先裁剪上下文,防止历史消息超过字符预算
            model_messages = self.context.prepare_for_model(messages)
            self.logger.event("model_request", self._model_request_payload(step, model_messages))
            # 调用模型得到响应文本(非流式返回完整字符串,流式则拼回字符串)
            response_text = self._complete_model(model_messages, step)
            self.logger.event("model_response", self._model_response_payload(step, response_text))
            # 记录模型响应元数据(usage 等),并累加 token 用量
            self._log_model_response_metadata(step)
            # token 预算检查:超过 max_run_tokens 立即停止
            if self._token_budget_exceeded():
                content = (
                    f"Stopped after exceeding max_run_tokens={self.config.max_run_tokens} "
                    f"(used {self.total_model_tokens})."
                )
                self._save_checkpoint(task, step, "stopped", messages)
                self.logger.event(
                    "token_budget_exceeded",
                    {"step": step, "max_run_tokens": self.config.max_run_tokens, "total_tokens": self.total_model_tokens},
                )
                self.logger.event(
                    "run_finished",
                    {
                        "status": "stopped",
                        "reason": "token_budget",
                        "max_run_tokens": self.config.max_run_tokens,
                        "total_tokens": self.total_model_tokens,
                    },
                )
                return AgentResult(content=content, steps=step)

            try:
                # 模型输出不可信,必须先按动作协议解析;失败时把错误反馈给模型继续修正。
                action = parse_action(response_text)
                action_log = action.to_log_dict()
            except ValueError as exc:
                # 解析失败:把错误塞回上下文,让模型下一轮自我修复
                result = {"ok": False, "error": f"Invalid model action: {exc}"}
                compressed = self.context.compress_tool_result(result)
                self.logger.event("action_error", {"step": step, "result": compressed})
                messages.append(Message("assistant", response_text))
                messages.append(Message("user", "Tool result:\n" + json.dumps(compressed, ensure_ascii=False)))
                self._save_checkpoint(task, step, "running", messages)
                continue

            # ---- 动作分发:final / plan / todo_update / tool_call ----
            if action.type == "final":
                # 模型认为任务完成,直接结束
                content = action.content or ""
                messages.append(Message("assistant", response_text))
                self._save_checkpoint(task, step, "completed", messages)
                self.logger.event("final", {"step": step, "content": content, "plan": self.plan.snapshot()})
                self.logger.event("run_finished", {"status": "completed", "step": step, "plan": self.plan.snapshot()})
                return AgentResult(content=content, steps=step)

            if action.type == "plan":
                # 替换整个计划
                result = self.plan.replace(action.items or [])
                self.logger.event("plan_updated", {"step": step, "action": action_log, "plan": result})
                messages.append(Message("assistant", json.dumps(action_log, ensure_ascii=False)))
                messages.append(Message("user", "Plan recorded:\n" + json.dumps(result, ensure_ascii=False)))
                self._save_checkpoint(task, step, "running", messages)
                continue

            if action.type == "todo_update":
                # 增量更新 todo 状态
                try:
                    result = self.plan.update(action.items or [])
                except ValueError as exc:
                    result = {"ok": False, "error": str(exc), "plan": self.plan.snapshot()}
                self.logger.event("plan_updated", {"step": step, "action": action_log, "plan": result})
                messages.append(Message("assistant", json.dumps(action_log, ensure_ascii=False)))
                messages.append(Message("user", "Plan updated:\n" + json.dumps(result, ensure_ascii=False)))
                self._save_checkpoint(task, step, "running", messages)
                continue

            # 工具调用是模型影响真实工作区的边界,结果进入上下文前要先压缩。
            result = self.tools.call(action.tool or "", action.args or {})
            compressed = self.context.compress_tool_result(result)
            self.logger.event("tool_result", {"step": step, "action": action_log, "result": compressed, "plan": self.plan.snapshot()})
            # 如果是命令类工具,额外记一条审计事件(不含 stdout/stderr)
            self._log_command_audit(step, action_log, result)
            messages.append(Message("assistant", json.dumps(action_log, ensure_ascii=False)))
            messages.append(Message("user", "Tool result:\n" + json.dumps(compressed, ensure_ascii=False)))
            self._save_checkpoint(task, step, "running", messages)

        # 循环结束仍未返回 final,说明触达步数上限
        content = f"Stopped after reaching max_steps={max_step}."
        self._save_checkpoint(task, max_step, "stopped", messages)
        self.logger.event("run_finished", {"status": "stopped", "reason": "max_steps", "max_steps": max_step})
        return AgentResult(content=content, steps=max_step)

    def _complete_model(self, messages: list[Message], step: int) -> str:
        """统一封装流式与非流式调用,对外都返回完整字符串。"""
        if not self.stream:
            return self.model.complete(messages)
        chunks: list[str] = []
        for chunk in self.model.stream_complete(messages):
            chunks.append(chunk)
            self.logger.event("model_stream_chunk", {"step": step, "content": chunk})
            if self.stream_callback is not None:
                self.stream_callback(chunk)
        return "".join(chunks)

    def _model_request_payload(self, step: int, messages: list[Message]) -> dict[str, object]:
        """构造 model_request 事件 payload。

        默认只记哈希和计数,避免把提示词原文写入磁盘;
        只有 trace_messages=True 时才写入完整 messages。
        """
        api_messages = [message.to_api() for message in messages]
        payload: dict[str, object] = {
            "step": step,
            "message_count": len(messages),
            "messages_chars": sum(len(message.content) for message in messages),
            "message_roles": [message.role for message in messages],
            "messages_sha256": _messages_sha256(api_messages),
            "stream": self.stream,
            "trace_messages": self.config.trace_messages,
        }
        if self.config.trace_messages:
            payload["messages"] = api_messages
        return payload

    def _model_response_payload(self, step: int, content: str) -> dict[str, object]:
        """构造 model_response 事件 payload,默认只记长度和哈希。"""
        payload: dict[str, object] = {
            "step": step,
            "content_chars": len(content),
            "content_sha256": _text_sha256(content),
            "trace_model_responses": self.config.trace_model_responses,
        }
        if self.config.trace_model_responses:
            payload["content"] = content
        return payload

    def _log_model_response_metadata(self, step: int) -> None:
        """记录模型响应元数据(usage、finish_reason 等),并累加 token 用量。

        getattr(obj, name, default) 类似 Java 反射读取字段,缺省时返回 default。
        """
        metadata = getattr(self.model, "last_response_metadata", None)
        if isinstance(metadata, dict) and metadata:
            self.logger.event("model_response_metadata", {"step": step, **metadata})
            usage = metadata.get("usage")
            if isinstance(usage, dict):
                total_tokens = usage.get("total_tokens")
                # isinstance(True, int) 在 Python 里为 True,所以单独排除 bool
                if isinstance(total_tokens, int) and not isinstance(total_tokens, bool):
                    self.total_model_tokens += total_tokens

    def _token_budget_exceeded(self) -> bool:
        """max_run_tokens=0 表示禁用 token 预算停止。"""
        return self.config.max_run_tokens > 0 and self.total_model_tokens > self.config.max_run_tokens

    def _log_command_audit(self, step: int, action: dict[str, object], result: dict[str, object]) -> None:
        """从工具结果中提取命令审计字段,记录 command_audit 事件。

        只有 run_command / git_status / git_diff 这类命令工具的 result 才有
        command/risk/approval 字段,其它工具直接跳过。
        """
        command = result.get("command")
        risk = result.get("risk")
        approval = result.get("approval")
        if not isinstance(command, str) or not isinstance(risk, dict) or not isinstance(approval, dict):
            return
        # 命令审计只记录安全决策和退出码,不复制可能很大的 stdout/stderr。
        self.logger.event(
            "command_audit",
            {
                "step": step,
                "tool": action.get("tool", ""),
                "command": command,
                "ok": result.get("ok") is True,
                "returncode": result.get("returncode"),
                "risk": risk,
                "approval": approval,
            },
        )

    def _save_checkpoint(self, task: str, step: int, status: str, messages: list[Message]) -> None:
        """每步落盘 checkpoint,便于事后 resume 恢复。"""
        path = self.checkpoint_store.save(
            RunCheckpoint(
                run_id=self.logger.run_id,
                task=task,
                step=step,
                status=status,
                messages=list(messages),
                plan=self.plan.snapshot(),
            )
        )
        self.logger.event("checkpoint_saved", {"step": step, "status": status, "path": self.workspace.relative(path)})


def render_system_prompt(tools: ToolRouter) -> str:
    """读取打包的系统提示词模板,并把当前工具 schema 注入到 {{TOOL_SPECS}} 占位符处。"""
    path = Path(__file__).parent / "prompts" / "system.md"
    prompt = path.read_text(encoding="utf-8")
    return prompt.replace("{{TOOL_SPECS}}", json.dumps(tools.specs(), ensure_ascii=False, indent=2))


def _messages_sha256(messages: list[dict[str, str]]) -> str:
    """对消息列表做 SHA-256,用于 trace 中识别重复/变化的请求,而不暴露原文。"""
    raw = json.dumps(messages, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return _text_sha256(raw)


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
