"""命令行入口:python -m harness 时被调用。

类似 Java 的 public static void main(String[] args):
    - __name__ == "__main__" 表示该文件被直接运行(而非 import)
    - main() 返回退出码,raise SystemExit 让 Python 以该码退出
"""

from __future__ import annotations

from harness.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
