# Evaluation Phase-B 回填与出图说明（2026-03-26）

适用时机：试验 agent 回传 corrected CCB（`reposition_solver=gurobi`）后执行。  
本阶段目标：回填 CCB 数字并导出新版 4 张主图（PDF）。

## 1) 输入文件

- Primary（27 runs）  
  `results/evaluation_runs/20260320_214439_main_only_3day/agg/main_matrix_raw.csv`
- Ablation（54 runs）  
  `results/evaluation_runs/20260321_223203_incremental_fix_maintext/agg/ablation_6combo_3day_all.csv`
- Corrected CCB（9 runs）  
  `results/evaluation_runs/<ts>_capacity_cap90_3day_fix/agg/cap90_ccb_raw.csv`

## 2) 执行命令（出图 + 门禁）

```bash
python scripts/plot_evaluation_phaseb_pdf.py \
  --primary-csv results/evaluation_runs/20260320_214439_main_only_3day/agg/main_matrix_raw.csv \
  --ablation-csv results/evaluation_runs/20260321_223203_incremental_fix_maintext/agg/ablation_6combo_3day_all.csv \
  --ccb-csv results/evaluation_runs/<ts>_capacity_cap90_3day_fix/agg/cap90_ccb_raw.csv \
  --output-dir results/evaluation_runs/<ts>_phaseb_figures_pdf
```

## 3) 预期输出

- `fig1_performance_served.pdf`
- `fig2_performance_peak_power.pdf`
- `fig3_overhead.pdf`
- `fig4_ablation_variants.pdf`
- `phaseb_summary.json`

## 4) 门禁条件（脚本自动检查）

- primary 行数 = 27
- ablation 行数 = 54
- corrected CCB 行数 = 9
- corrected CCB 的 `reposition_solver` 唯一值必须是 `gurobi`
- corrected CCB 的 `(sim_date, group_id)` 必须唯一

## 5) 回填到 LaTeX

更新文件：`docs/Evaluation_section_draft_2026-03-25.tex`

- Table 1：把 CCB 行的 `TBD` 替换为 `phaseb_summary.json` 中的均值
- Table 2：把 CCB 行的 `TBD` 替换为 `phaseb_summary.json` 中 overhead 均值
- Results 段：补 CCB 相对 COP/FDB 的变化幅度
- 图路径：将主文核心图替换为上述 4 个 PDF 文件

