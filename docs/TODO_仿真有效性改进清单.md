# TODO：仿真有效性改进清单（执行版）

更新时间：2026-03-13  
状态：阶段一至阶段三已完成，当前进入 P0 异常诊断与全量实验前 gate

## 已完成阶段（归档）

### 第一阶段：指标与日志

- [x] `total_served`
- [x] `battery_swap_success_ratio`
- [x] `deadline_miss_ratio`（换电请求未满足比例）
- [x] `charging_deadline_miss_ratio`（充电任务逾期比例）
- [x] `idle_driving_distance`
- [x] `waiting_time_for_battery_slots`
- [x] `timeseries.csv` 与 `summary.json` 稳定输出

### 第二阶段：参数矩阵

- [x] `inventory` sweep
- [x] `chargers` sweep
- [x] `swap_capacity` sweep
- [x] 强瓶颈参数网格（stress grid）

### 第三阶段：基线与组合

- [x] `reposition_solver=heuristic`
- [x] `reposition_solver=ideal`
- [x] `charging_solver=fcfs`
- [x] 基线组合结果文件与图表产出

## 当前最高优先（P0）：先解决稳定性与诊断可信度

### A. 3756 车辆场景卡死/无解边界条件

- [ ] 复现并记录 `gurobi_reposition_policy` 在 3756 车辆下每时段求解状态（含 `model.Status`、`SolCount`）
- [ ] 修复“边界条件处理”而非简单退化：避免 `status=4/9` 直接导致整轮中断
- [ ] 给出可审计日志字段：每时段是否触发降级、降级原因、降级次数
- [ ] 验收标准：诊断 episode 可完整跑完且无 silent fallback

### B. `charging_deadline_miss_ratio` 异常（algorithm+FCFS ≈ 1.0）

- [ ] 单独输出并核对分子分母：`charging_deadline_misses` 与 `successful_swaps`
- [ ] 对比 `algorithm+FCFS` 与 `heuristic+FCFS` 的 `successful_swaps` 规模差
- [ ] 判断是否为“分母过小放大比值”还是“充电任务真实大量逾期”
- [ ] 形成 1 页 root-cause 结论后再推进全量矩阵

### C. 全量实验 gate（未通过不得启动 162-run）

- [ ] 通过 A 与 B 的诊断 gate
- [ ] 明确主方法与消融口径：`reposition × charging = {gurobi, heuristic, ideal} × {gurobi, fcfs}`
- [ ] 先跑 stress grid 的 6 组合补齐充电侧证据
- [ ] 再跑全量参数矩阵

## 次优先（P1）：论文证据补齐

### D. 充电策略贡献证据

- [ ] stress grid 增补 `* + gurobi charging` 组合
- [ ] 生成对比表：`served / swap_success / deadline_miss / charging_deadline_miss / peak_power`

### E. 24h 时序图

- [ ] 选中等压力参数，跑完整 24h episode
- [ ] 输出 `charging_power_kw`、`station_total_power_kw` 时序
- [ ] 对比 `ours+gurobi charging` vs `heuristic+fcfs`

## 文档一致性（并行进行）

- [x] `论文公式一致性核对表.md` 增加“已知风险与 gate 条件”
- [x] `代码审查报告_Paper对齐核查.md` 同步当前代码真实状态（区分“已修复”与“待修复”）
- [x] `E-Taxi_换电站技术规格说明书_完整版.md` 明确工程扩展：等待队列、Eq.(14) 软约束、FCFS 非抢占

## 本周执行顺序（固定）

1. 先完成 A（卡死边界条件）  
2. 再完成 B（指标分母诊断）  
3. 通过 gate 后执行 C（stress 6 组合 -> 全量矩阵）  
4. 最后收敛 D/E 与文档统一
