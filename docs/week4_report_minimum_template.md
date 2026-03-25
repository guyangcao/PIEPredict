# Week 4 报告最小结构模板（可复现版）

> 目标：保证所有结论都能从同一套实验产物（`evaluation_summary*.csv/json` + run artifacts）追溯。

## 1) Baseline 复现结果

- **实验配置**
  - 数据划分（`protocol` + `eval_split`）：
  - Baseline 模型：
  - `K`：
  - seeds（例如 `42,43,44`）：
  - 评价脚本版本（git commit）：
- **结果（建议引用 `evaluation_summary_mean_std.csv`）**
  - Intention：`F1` / `AUC`
  - Trajectory：`minADE` / `minFDE`
- **可追溯证据**
  - 汇总文件路径：
  - 对应 run artifacts 目录：

## 2) Multi-future 方法描述（含 Best-of-K）

- **方法说明（简述）**
  - 同一历史轨迹生成 `K` 条未来假设。
  - 使用 Best-of-K：按最小轨迹误差（`minADE/minFDE`）评估多模态覆盖能力。
- **实验配置（必须显式写出）**
  - 数据划分：
  - `K` 值集合（例如 `1,5,10`）：
  - seed 数量：
  - 评价脚本版本（git commit）：

## 3) 定量表（F1/AUC + minADE/minFDE）

> 要求：每一行都能映射到 `evaluation_summary.csv` 中的 `(model_name, K, seed, protocol)`。

| Model | Protocol | Split | K | #Seeds | F1 (mean±std) | AUC (mean±std) | minADE (mean±std) | minFDE (mean±std) | Eval Script Commit |
|---|---|---|---:|---:|---|---|---|---|---|
| Baseline |  |  |  |  |  |  |  |  |  |
| Multi-future |  |  |  |  |  |  |  |  |  |

## 4) 定性图（同一历史下 K 条未来）

- 每组可视化至少包含：
  - 同一历史轨迹
  - `K` 条预测未来轨迹
  - GT 未来轨迹
- 图注必须包含：
  - 数据划分
  - `K`
  - seed（或 seed 集合）
  - 评价脚本版本

## 5) 失败案例（2–3 个）

> 必须覆盖以下类别（可各 1 个）：遮挡、路口、犹豫行人。

- **Case A（遮挡）**：
  - 现象：
  - 误差表现（minADE/minFDE/F1/AUC 关联）：
  - 可能原因：
- **Case B（路口）**：
  - 现象：
  - 误差表现：
  - 可能原因：
- **Case C（犹豫行人）**：
  - 现象：
  - 误差表现：
  - 可能原因：

> 用这些失败案例作为 Week 5-6 consistency regularizer 的动机铺垫。

## 6) 结论与下一步（必须逐字覆盖）

- 下一步 1：加入 mixture weights `π_k`。
- 下一步 2：实现 intent-trajectory consistency loss。

## 7) 数字可追溯性检查（提交前 Checklist）

- [ ] 所有表格数字均来自同一份 `evaluation_summary.csv` 或 `evaluation_summary_mean_std.csv`。
- [ ] 所有图表都标注了：数据划分、`K`、seed 数、评价脚本版本。
- [ ] 报告中的每个结论可回溯到 `K{K}_seed{seed}/metrics.json`。
- [ ] 禁止手工抄写后再改值；仅允许脚本导出或直接引用。
