# Gurobi reposition 瓶颈与优化报告

## 结论摘要
- 在固定 case（06:00-22:00，inv=5，chargers=5，swap=6，reposition=gurobi，charging=fcfs）下，主要瓶颈是 **模型构建阶段**，不是 optimize 阶段。
- 关闭预聚合时，单步 build 中位数约 70.29s；开启预聚合后降至 11.04s（-84.29%）。
- 单步总耗时（run wall-time，中位数）从 81.22s 降至 21.98s（-72.93%）。
- optimize 时间基本不变（约 10.04s），说明改动主要命中“重复表达式构建”热点。
- 在严格语义（禁 fallback/禁缩 horizon）下，两组都在 step=0 出现 `status=9` 且无 incumbent，质量指标暂无法做完整对齐验证。

## 证据表格（前后对比）

### 口径与复现命令
- Python: `/opt/anaconda3/envs/bsm/bin/python`
- 环境变量: `PROJ_DATA=/opt/anaconda3/envs/bsm/share/proj`, `PROJ_LIB=/opt/anaconda3/envs/bsm/share/proj`
- 时间约束: `--max-run-seconds 240`（防止基线爆时）
- 运行口径: 每组 3 次，取中位数；每次 `--max-steps 1`

```bash
# BEFORE: 关闭预聚合（保持 dense 变量构造）
python scripts/run_reposition_diagnostic.py \
  --config configs/ev_yellow_2025_11_eval_6_22_fcfs.yaml \
  --reposition-solver gurobi --charging-solver fcfs \
  --inventory 5 --chargers 5 --swap-capacity 6 \
  --solver-time-limit-sec 10 \
  --max-steps 1 --max-run-seconds 240 \
  --legacy-dense --disable-preaggregation \
  --output-csv results/diagnostics/reposition_perf_before_run1.csv \
  --output-summary-json results/diagnostics/reposition_perf_before_run1.json

# AFTER: 开启预聚合（同语义）
python scripts/run_reposition_diagnostic.py \
  --config configs/ev_yellow_2025_11_eval_6_22_fcfs.yaml \
  --reposition-solver gurobi --charging-solver fcfs \
  --inventory 5 --chargers 5 --swap-capacity 6 \
  --solver-time-limit-sec 10 \
  --max-steps 1 --max-run-seconds 240 \
  --legacy-dense \
  --output-csv results/diagnostics/reposition_perf_after_run1.csv \
  --output-summary-json results/diagnostics/reposition_perf_after_run1.json
```

### 性能对比（中位数）

| 指标 | Before（禁预聚合） | After（开预聚合） | 变化 |
|---|---:|---:|---:|
| run_wall_time_sec | 81.22 | 21.98 | -72.93% |
| reposition_build_time_sec | 70.29 | 11.04 | -84.29% |
| reposition_optimize_time_sec | 10.04 | 10.04 | +0.06% |
| NumVars | 894,385 | 894,385 | 0 |
| NumConstrs | 325,420 | 325,420 | 0 |
| NumNZs | 6,220,445 | 6,220,445 | 0 |

数据来源：
- `results/diagnostics/reposition_perf_before_run{1..3}.csv/.json`
- `results/diagnostics/reposition_perf_after_run{1..3}.csv/.json`
- `results/diagnostics/reposition_perf_compare_summary.json`

### 结果质量（served / swap_success / miss / waiting）
- Before 与 After 在严格语义下均在 step=0 报 `status=9, sol_count=0` 并抛出异常，因此未产生有效 episode 级 KPI。
- 本轮结论聚焦于“真实瓶颈定位与构建性能修复”，质量回归需在后续拿到 incumbent 后再补齐。

## 风险与后续建议
- 风险 1：严格模式下无 incumbent（`status=9`）仍会阻断完整仿真，说明当前 case 在 `solver_time_limit_sec=10` 下求解阶段仍是瓶颈之一。
- 风险 2：本轮已显著降低 build 时间，但 optimize 阶段无改善，后续需继续做求解参数与模型数值稳定性优化。
- 建议 1：在不改语义前提下，继续做求解参数扫（如 method/crossover/数值容忍度）并记录同口径证据。
- 建议 2：在严格模式下增加“单步可行解可得性”专项诊断（日志化 barrier/simplex 迭代统计），先解决 no-incumbent，再做完整 KPI 回归。
- 建议 3：当前 `--max-run-seconds` 已纳入诊断流程，建议后续所有 baseline 命令默认带该约束，避免运行时间失控。
