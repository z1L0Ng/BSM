# Strict Incumbent 修复与完整评估计划

## 目标与边界
- 仅做模型优化与诊断，不改论文核心约束含义与目标函数语义。
- 严格模式默认：禁 fallback、禁缩 horizon。
- 不在本线程执行完整实验；本文件供后续线程直接执行。

## 已落地代码能力（本轮完成）
- Reposition 求解参数新增并全链路暴露：
  - `reposition_presolve`
  - `reposition_use_lp_primal_start`
  - `reposition_lp_warm_start_mode`
- strict 语义下新增 LP 可行暖启动（零派单可行起点）：
  - 通过 `PStart/Start` 为 `x/y/r/b/mu/served/full_stock` 设置初值
  - 不改变原目标函数与核心约束语义
- 诊断审计扩展：
  - `iter_count`、`bar_iter_count`、`obj_val`、`obj_bound`
  - `lp_warm_start_*` 相关字段（是否启用/是否生效/设置变量数）
  - build 子阶段计时（候选生成/变量/表达式/约束/目标）
- 参数扫脚本扩展（小规模 8-12 组可控）：
  - `scripts/run_reposition_solver_sweep.py` 支持 `presolve`、`lp_warm_start_mode` 维度

## 下一步执行计划（由实验线程执行）

### 阶段 A：incumbent 可得性专项（先过门槛）
固定 case：
- `06:00-22:00, inv=5, chargers=5, swap=6, reposition=gurobi, charging=fcfs, strict`

建议命令（8-12 组）：
```bash
export PROJ_DATA=/opt/anaconda3/envs/bsm/share/proj
export PROJ_LIB=/opt/anaconda3/envs/bsm/share/proj

/opt/anaconda3/envs/bsm/bin/python scripts/run_reposition_solver_sweep.py \
  --config configs/ev_yellow_2025_11_eval_6_22_fcfs.yaml \
  --max-steps 1 \
  --max-run-seconds-per-run 240 \
  --inventory 5 --chargers 5 --swap-capacity 6 \
  --methods 2,1,0 \
  --crossovers 0,1 \
  --time-limits 10,15,20 \
  --numeric-focuses 0,1 \
  --presolves -1,1 \
  --lp-warm-start-modes 2,1 \
  --reposition-use-lp-primal-start \
  --max-combos 12 \
  --output-csv results/diagnostics/reposition_solver_param_sweep_strict_v2.csv \
  --output-summary-json results/diagnostics/reposition_solver_param_sweep_strict_v2.summary.json
```

阶段 A 验收：
- 至少 1 组 `sol_count > 0`（step=0）。
- 若 12 组全失败，必须输出失败归因统计（`status/error` 聚合）和迭代统计对照（`iter/bar_iter/obj_bound`）。

### 阶段 B：质量不劣化回归（在 A 通过后）
对“首个拿到 incumbent 且 wall-time 最小”的组合，按同一 case 连跑 3 次，和当前 strict 基线对比：
- `total_served` 不降低
- `battery_swap_success_ratio` 不降低
- `deadline_miss_ratio` 不升高
- `total_waiting_vehicles` 不升高

### 阶段 C：完整 evaluation（最终）
1. 锁定参数：
- 固定阶段 A/B 通过的 strict 参数组，冻结到评估配置文件。

2. 运行矩阵：
- 算法组合矩阵保持既定实验协议（不在此文件更改业务 KPI 定义）。
- 每组合建议至少 3 次重复（中位数作为主结果）。

3. 报告结构：
- 结论摘要（3-5 行）
- 前后证据表（总耗时、build/solve 分解、模型规模、KPI）
- 风险与后续建议（只列未闭环项）

## 判定与回退原则
- 禁止用 fallback/缩 horizon 通过验收。
- 若 A 阶段失败，不进入完整矩阵；先在 strict 下继续求解可行性专项。
- 允许继续做等价重构（缓存/预聚合/表达式复用），但必须保证目标与核心约束语义不变。
