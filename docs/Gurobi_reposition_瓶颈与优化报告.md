# Gurobi reposition 瓶颈与优化报告

## 结论摘要
- 固定 case（06:00-22:00，inv=5，chargers=5，swap=6，gurobi+fcfs）下，build 瓶颈已被命中：预聚合后 build 中位数 11.04s，相比禁预聚合 70.29s，下降 84.29%。
- 本轮 strict 参数扫（12 组：method/crossover/time_limit/numeric_focus）全部 `status=9, sol_count=0`，说明当前阻塞点不在这组参数组合，而在“首个 incumbent 可得性”。
- 在 strict 下，随着 `time_limit` 从 10→15→20 秒，optimize 中位数近似线性增长（10.03→15.07→20.06s），而 build 基本稳定（约 10.9s）。
- 新增 build 子阶段审计后，热点位于表达式聚合（约 43%）与变量构建（约 29%）；约束下发约 26%，候选集准备约 0.39s。
- 本轮仅做等价重构/诊断增强/求解参数接口暴露，未改目标函数与核心约束语义，未启用 fallback 或缩 horizon。

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

### 表 1：建模优化前后（中位数）

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

### 表 2：Strict 参数扫（12 组）汇总

| 分组口径 | 结果 |
|---|---|
| 参数维度 | `method={2,1}, crossover={0,1}, time_limit={10,15,20}, numeric_focus={0,1}`（按优先级选取 12 组） |
| 固定 case | `06-22, inv=5, chargers=5, swap=6, reposition=gurobi, charging=fcfs, strict` |
| incumbent 可得性 | `0/12`（全部 `sol_count=0`） |
| 状态码分布 | `status=9`：12/12 |
| 失败归因 | `RuntimeError: ... no solution, status=9`：12/12 |
| 最快组合（仅 wall-time） | `method=2, crossover=0, time_limit=10, numeric_focus=0`，`run_wall_time=21.71s`（仍无 incumbent） |

### 表 3：按 time_limit 的中位数（Strict 参数扫）

| time_limit_sec | run_wall_time_sec | step_build_time_sec | step_optimize_time_sec |
|---:|---:|---:|---:|
| 10 | 21.81 | 10.89 | 10.03 |
| 15 | 26.88 | 10.93 | 15.07 |
| 20 | 31.88 | 10.90 | 20.06 |

来源：
- `results/diagnostics/reposition_solver_param_sweep.csv`
- `results/diagnostics/reposition_solver_param_sweep.summary.json`

### 表 4：build 子阶段分解（step=0，strict，time_limit=10）

| 子阶段 | 耗时（秒） | build 内占比 |
|---|---:|---:|
| candidate_prep | 0.39 | - |
| vars_build | 3.13 | 29.05% |
| expressions_build | 4.63 | 42.96% |
| constraints_build | 2.78 | 25.78% |
| objective_build | 0.24 | 2.21% |
| total_build | 10.90 | 100% |

来源：
- `results/diagnostics/_smoke_diag.csv`

## 风险与后续建议
- 风险 1：strict 下 `sol_count=0` 会阻断 episode KPI（`served/swap_success/miss/waiting`）回归；当前仅能完成性能与失败归因层面验证。
- 风险 2：表达式构建虽已显著降耗，但仍占 build 最大头；若不继续做复用/缓存，后续扩规模会再次放大。
- 风险 3：求解阶段随 `time_limit` 线性增长但无 incumbent，说明“仅延长时限”不是有效策略。
- 建议 1：下一轮做“incumbent 可得性专项诊断”（保语义）：输出 Gurobi 迭代日志摘要（presolve/迭代数/终止原因）并补 8-12 组参数扫，仅针对可行解可得性。
- 建议 2：在不改目标与约束语义前提下，继续做表达式复用（重复 quicksum 的缓存化）并以 build 子阶段中位数做验收。
- 建议 3：所有基线与参数扫维持 `--max-run-seconds`（当前 240s）硬限制，避免诊断运行时间失控。
