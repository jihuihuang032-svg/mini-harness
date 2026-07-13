"""Runtime 子包入口。

提供 Agent 运行时所需的基础设施:
    - workspace:工作区(沙箱根目录)
    - policy:命令风险评估
    - approval:审批控制器
    - executor:命令执行器(组合 policy + approval)
    - logger:trace 日志(JSONL)
    - checkpoint:运行检查点(便于 resume)
    - change_tracker:工作区文件变更追踪
    - run_config / run_store / task_queue / task_store:运行/任务存储
    - eval_store:评测报告存储
"""
