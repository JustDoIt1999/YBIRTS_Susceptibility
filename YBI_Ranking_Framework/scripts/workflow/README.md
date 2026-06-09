# Workflow Files

当前提供两个正式 workflow 入口，它们服务的是**同一条** YBI_Ranking_Framework 主流程，不是两套不同 pipeline。

1. **Python 入口（跨平台，推荐 Windows 使用）**:
   - 脚本：`run_full_pipeline.py`
   - 使用方式：`python scripts/workflow/run_full_pipeline.py`
   - 适用平台：Windows / macOS / Linux

2. **Shell 入口（推荐 macOS / Linux 使用）**:
   - 脚本：`run_full_pipeline.sh`
   - 使用方式：`bash scripts/workflow/run_full_pipeline.sh`
   - 适用平台：macOS / Linux

配套 workflow YAML：

- `full_pipeline.yaml`

职责：

- 一键执行 YBI_Ranking_Framework 核心的三步流水线：`make-folds` -> `sweep-lambda` -> `select-lambda --train`
- 支持 `--skip-existing` 自动检测并跳过已完成步骤（用于断点续传）
- 支持通过命令行参数如 `--batch-size`、`--rank-loss-mode` 覆盖默认配置

说明：

- 两个入口的目标产物、默认目录规则和步骤顺序应保持一致。
- 选择原则只与平台和用户习惯有关，不代表不同实验路线。
- 运行前都需要先进入可用的 Python 环境，并确保 `cvrank` 依赖已安装。
