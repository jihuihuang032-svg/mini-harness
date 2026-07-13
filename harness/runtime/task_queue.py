"""任务队列:HTTP server 用它异步执行任务。

submit() 创建 TaskRecord 后,启动一个 daemon thread 跑 _run,
_run 调用 runner 回调(实际执行 agent),完成后更新状态并持久化。
崩溃后再次启动时,把未完成的 queued/running 任务标记为 failed。
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Literal, Protocol
from uuid import uuid4

from harness.runtime.logger import new_run_id


# 任务状态:排队 / 执行中 / 已完成 / 失败
TaskStatus = Literal["queued", "running", "completed", "failed"]
# runner 回调签名:(task, stream, run_id, metadata) -> result_dict
TaskRunner = Callable[[str, bool, str, dict[str, object]], dict[str, object]]


class TaskPersistence(Protocol):
    """任务持久化接口(Protocol,类似 Java interface,供 TaskStore 实现)。

    TaskQueue 只依赖这两个方法,不关心具体实现(JSONL 文件 / sqlite / 内存都行)。
    """

    def append(self, record: "TaskRecord") -> None:
        """追加一条任务记录快照。"""

    def load_latest(self) -> list["TaskRecord"]:
        """加载每个 task_id 的最新一条记录。"""


def _now() -> str:
    """当前 UTC 时间 ISO 字符串。"""
    return datetime.now(timezone.utc).isoformat()


@dataclass
class TaskRecord:
    """任务记录:一次 submit 对应一条,生命周期内多次更新。

    mutable(不加 frozen=True),因为 _run 会不断更新 status/result/error。
    """
    task_id: str
    run_id: str
    task: str
    stream: bool
    mode: str = "mock"
    provider: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)
    status: TaskStatus = "queued"
    created_at: str = field(default_factory=_now)    # 默认值工厂:每次构造调 _now
    updated_at: str = field(default_factory=_now)
    result: dict[str, object] | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, object]:
        """序列化为可写入 JSON 的 dict。"""
        return {
            "task_id": self.task_id,
            "run_id": self.run_id,
            "task": self.task,
            "stream": self.stream,
            "mode": self.mode,
            "provider": self.provider,
            "metadata": self.metadata,
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "result": self.result,
            "error": self.error,
        }


class TaskQueue:
    """线程安全的任务队列:submit / get / list。

    @param runner: 实际执行任务的回调(在 server.py 里被注入成 agent run)
    @param store:  可选持久化(None 表示纯内存,主要用于测试)
    """

    def __init__(self, runner: TaskRunner, store: TaskPersistence | None = None) -> None:
        self.runner = runner
        self.store = store
        # threading.Lock 是 Java synchronized 的等价物,保护 _tasks 字典
        self._lock = threading.Lock()
        self._tasks: dict[str, TaskRecord] = {}
        if self.store is not None:
            # 启动时从持久化恢复:把上次崩溃时未完成的任务标记为 failed
            for record in self.store.load_latest():
                if record.status in {"queued", "running"}:
                    record.status = "failed"
                    record.error = "Task was interrupted before server restart."
                    record.updated_at = _now()
                    self.store.append(record)
                self._tasks[record.task_id] = record

    def submit(
        self,
        task: str,
        stream: bool = False,
        mode: str = "mock",
        provider: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> TaskRecord:
        """提交任务:创建 record -> 入队 -> 启动 daemon thread 执行。"""
        record = TaskRecord(
            task_id=uuid4().hex,        # uuid4 生成随机 UUID,hex 取十六进制字符串
            run_id=new_run_id(),
            task=task,
            stream=stream,
            mode=mode,
            provider=provider,
            metadata=metadata or {},
        )
        # with 语句:进入时获取锁,退出时自动释放(类似 Java synchronized 块)
        with self._lock:
            self._tasks[record.task_id] = record
            self._persist(record)
        # daemon=True:主线程退出时这些后台线程也会被强制结束(不阻塞进程关闭)
        thread = threading.Thread(target=self._run, args=(record.task_id,), daemon=True)
        thread.start()
        return record

    def get(self, task_id: str) -> TaskRecord:
        """查询单个任务:返回副本(防止外部修改内部状态)。"""
        with self._lock:
            record = self._tasks.get(task_id)
            if record is None:
                raise ValueError(f"Task not found: {task_id}")
            return _copy_record(record)

    def list(self) -> list[TaskRecord]:
        """列出所有任务:返回副本列表。"""
        with self._lock:
            return [_copy_record(record) for record in self._tasks.values()]

    def _run(self, task_id: str) -> None:
        """任务执行逻辑(在 daemon thread 内运行)。

        1. 加锁:标记 running,提取任务参数
        2. 释放锁后调 runner(可能很慢,不能持锁)
        3. 加锁:根据结果更新 status/result/error
        """
        # 1. 标记 running,提取参数(持锁)
        with self._lock:
            record = self._tasks[task_id]
            record.status = "running"
            record.updated_at = _now()
            self._persist(record)
            task = record.task
            stream = record.stream
            run_id = record.run_id
            metadata = dict(record.metadata)
        # 2. 实际执行(不持锁,否则其它任务会被阻塞)
        try:
            result = self.runner(task, stream, run_id, metadata)
        except Exception as exc:
            # 异常:标记 failed 并记录错误信息
            with self._lock:
                record = self._tasks[task_id]
                record.status = "failed"
                record.error = str(exc)
                record.updated_at = _now()
                self._persist(record)
            return
        # 3. 成功:标记 completed 并保存结果
        with self._lock:
            record = self._tasks[task_id]
            record.status = "completed"
            record.result = result
            record.updated_at = _now()
            self._persist(record)

    def _persist(self, record: TaskRecord) -> None:
        """持久化一条记录快照(如果 store 不为 None)。"""
        if self.store is not None:
            self.store.append(_copy_record(record))


def _copy_record(record: TaskRecord) -> TaskRecord:
    """深拷贝 TaskRecord(防止外部修改内部状态)。

    对于嵌套的 dict(result/metadata),用 dict() 浅拷贝已足够,
    因为外层不期望被修改,内层不期望被深度修改。
    """
    return TaskRecord(
        task_id=record.task_id,
        run_id=record.run_id,
        task=record.task,
        stream=record.stream,
        mode=record.mode,
        provider=record.provider,
        metadata=dict(record.metadata),
        status=record.status,
        created_at=record.created_at,
        updated_at=record.updated_at,
        result=dict(record.result) if record.result is not None else None,
        error=record.error,
    )
