# CDC-Style Evaluation Snippets (Overleaf, 2026-03-22)

This document provides a strict, paper-ready evaluation write-up and figure templates for Overleaf.
Scope is evaluation-only. No method implementation changes are assumed.

## 1) Naming Standard (Paper-facing)

Use the following names consistently in text, tables, and figures.

| Paper Name | Abbrev. | Internal Combo ID |
|---|---|---|
| Coordinated Optimization Policy | COP | `algorithm_plus_gurobi` |
| Fleet Dispatch Baseline | FDB | `algorithm_plus_fcfs` |
| Heuristic Dispatch Baseline | HDB | `heuristic_plus_fcfs` |
| Oracle Dispatch Baseline | ODB | `ideal_plus_fcfs` |
| Capacity-Constrained Charging Baseline | CCB | `algorithm_plus_fcfs` with capped chargers |

Notes:
- Internal combo IDs remain in scripts/logs for reproducibility.
- The paper body should avoid implementation-style names such as `gurobi+fcfs`.

## 2) Evaluation Body (LaTeX)

```latex
\section{Evaluation}

\subsection{Baseline Evaluation Setup}
We use NYC yellow taxi trip records (November 2025) to construct time-binned passenger demand.
The simulation window is fixed to 06:00--22:00 with 15-minute slots and receding horizon $H=64$.
Because real battery-swapping-station operation logs are not available in this study, station-side operations are simulated using parameterized station resources.
Each method uses the same demand input, date set, random-seed policy, and resource settings.
The primary evaluation covers three dates (2025-11-01, 2025-11-05, 2025-11-09), producing 9 matched (date, resource-setting) units per pairwise method comparison.

We compare:
\begin{itemize}
  \item \textbf{COP}: Coordinated Optimization Policy (proposed),
  \item \textbf{FDB}: Fleet Dispatch Baseline (fleet-level dispatch + FCFS charging),
  \item \textbf{HDB}: Heuristic Dispatch Baseline,
  \item \textbf{ODB}: Oracle Dispatch Baseline.
\end{itemize}
In a separate capacity-constrained scenario, we additionally evaluate \textbf{CCB}, which follows fleet-level dispatch with FCFS charging under capped charger concurrency (e.g., 90\% active chargers).

The primary endpoint is total served.
To characterize operational trade-offs, we additionally report swap success, deadline miss, charging miss, waiting, and station power.
For paired significance on total served, we use two-sided sign tests over matched units.

\subsection{Primary Results}
\textbf{Summary of outcome.}
COP achieves statistically significant gains in total served against HDB and ODB across matched units, while exhibiting non-negligible quality trade-offs in swap/deadline-related metrics.

\textbf{Aggregate service performance (27 runs).}
\begin{itemize}
  \item COP: $77613.67 \pm 6455.29$
  \item HDB: $36284.33 \pm 3285.06$
  \item ODB: $67840.44 \pm 6051.73$
\end{itemize}
Relative to baselines, COP improves total served by $+113.9\%$ over HDB and $+14.4\%$ over ODB.

\textbf{Paired consistency and significance (9 matched units per baseline).}
COP is higher in all 9 matched units against HDB and also higher in all 9 matched units against ODB.
Sign tests on total served give:
\begin{itemize}
  \item COP vs HDB: mean paired delta $= +41329.33$, $p=0.0039$;
  \item COP vs ODB: mean paired delta $= +9773.22$, $p=0.0039$.
\end{itemize}

\textbf{Trade-off profile.}
The service gain is accompanied by a clear shift in operational quality:
\begin{itemize}
  \item swap success ratio: COP $=0.1531$, HDB $=0.8969$, ODB $=1.0000$;
  \item deadline miss ratio: COP $=0.8469$, HDB $=0.1031$, ODB $=0.0000$;
  \item charging miss ratio: COP $=0.0715$, HDB $=0.1089$, ODB $=0.4404$.
\end{itemize}
Therefore, the evidence supports a service-priority operating regime rather than uniform dominance across all metrics.

\textbf{Solver reliability.}
No no-incumbent reposition event is observed in this evaluation (reposition_event_no_incumbent_count=0), indicating stable solver behavior under the tested protocol.

\subsection{Ablation Evidence}
Ablation covers 54 runs (3 dates $\times$ 3 groups $\times$ 6 combinations) using a two-factor decomposition: dispatch policy and charging policy.
Results indicate that dispatch policy choice is the dominant source of total served variation, while charging solver changes are secondary and mixed in sign.
In particular, moving from coordinated dispatch to heuristic dispatch causes a substantial service drop regardless of charging mode.

\subsection{Protocol-Aligned Time-Series Check}
We perform a protocol-aligned case replay on (2025-11-05, G2) comparing COP and HDB.
The cumulative served delta is $+44947$ (COP minus HDB), consistent with the aggregate service advantage.
Time-resolved traces also show higher waiting and power demand under COP at multiple intervals, matching the table-level trade-off pattern.
```

## 3) Figure Templates (LaTeX, PDF only)

Use single-column placement first; expand to two-column only if page packing requires.

```latex
\begin{figure}[t]
  \centering
  \includegraphics[width=0.98\linewidth]{figures_pdf/fig1_primary_served.pdf}
  \caption{Primary comparison on \texttt{total\_served}. Bars show mean performance over 9 matched units per method.}
  \label{fig:primary_served}
\end{figure}

\begin{figure}[t]
  \centering
  \includegraphics[width=0.98\linewidth]{figures_pdf/fig2_ablation_served.pdf}
  \caption{Ablation on \texttt{total\_served} using a dispatch-policy x charging-policy decomposition (54 runs total).}
  \label{fig:ablation_served}
\end{figure}

\begin{figure}[t]
  \centering
  \includegraphics[width=0.98\linewidth]{figures_pdf/fig3a_timeseries_served.pdf}
  \caption{Protocol-aligned time-series check (2025-11-05, G2): served demand per slot (COP vs HDB).}
  \label{fig:ts_served}
\end{figure}

\begin{figure}[t]
  \centering
  \includegraphics[width=0.98\linewidth]{figures_pdf/fig3b_timeseries_waiting.pdf}
  \caption{Protocol-aligned time-series check (2025-11-05, G2): waiting vehicles (COP vs HDB).}
  \label{fig:ts_waiting}
\end{figure}

\begin{figure}[t]
  \centering
  \includegraphics[width=0.98\linewidth]{figures_pdf/fig3c_timeseries_station_power.pdf}
  \caption{Protocol-aligned time-series check (2025-11-05, G2): station total power (COP vs HDB).}
  \label{fig:ts_power}
\end{figure}

\begin{figure}[t]
  \centering
  \includegraphics[width=0.98\linewidth]{figures_pdf/fig3d_timeseries_swap_success_ratio.pdf}
  \caption{Protocol-aligned time-series check (2025-11-05, G2): swap success ratio (COP vs HDB).}
  \label{fig:ts_swap_success}
\end{figure}

\begin{figure}[t]
  \centering
  \includegraphics[width=0.98\linewidth]{figures_pdf/fig3e_timeseries_charging_miss_ratio.pdf}
  \caption{Protocol-aligned time-series check (2025-11-05, G2): charging deadline miss ratio (COP vs HDB).}
  \label{fig:ts_charging_miss}
\end{figure}
```

## 4) PDF Figure Generation

From repository root:

```bash
python scripts/plot_paper_core3_pdf.py \
  --primary-csv results/evaluation_runs/20260320_214439_main_only_3day/agg/main_matrix_raw.csv \
  --ablation-csv /Users/zilongzeng/Research/BSM/results/evaluation_runs/20260321_223203_incremental_fix_maintext/agg/ablation_6combo_3day_all.csv \
  --timeseries-csv /Users/zilongzeng/Research/BSM/results/evaluation_runs/20260321_223203_incremental_fix_maintext/agg/timeseries_compare_alg_vs_heu_G2_2025-11-05.csv \
  --output-dir results/evaluation_runs/20260321_223203_incremental_fix_maintext/figures_pdf
```

Expected outputs:
- `fig1_primary_served.pdf`
- `fig2_ablation_served.pdf`
- `fig3a_timeseries_served.pdf`
- `fig3b_timeseries_waiting.pdf`
- `fig3c_timeseries_station_power.pdf`
- `fig3d_timeseries_swap_success_ratio.pdf`
- `fig3e_timeseries_charging_miss_ratio.pdf`
- `core3_summary.json`

## 5) Consistency Checklist Before Overleaf Update

- All figure legends and text use `COP/FDB/HDB/ODB/CCB` naming consistently.
- No implementation-style method names appear in the paper body.
- Protocol text is fixed and identical everywhere: `06:00--22:00`, `15-min`, `horizon=64`, dates `2025-11-01/05/09`, groups `G1/G2/G3`.
- Reported primary significance values match `paired_stats.csv` (`p=0.0039` for COP vs HDB and COP vs ODB).
- Time-series cumulative served delta matches `+44947` (COP minus HDB).

## 6) Prompt for Experiment Agent (Teacher-Feedback Increment)

Use this directly:

```text
Please execute `docs/Evaluation_老师反馈对齐执行计划_2026-03-25.md` end-to-end (Tasks A/B/C), and only do Evaluation increment work.

Hard constraints:
1) Do not rerun the 27-run primary matrix.
2) Reuse existing 54-run ablation output and extract FDB (`algorithm_plus_fcfs`) summaries.
3) Run only CCB 9 runs (3 dates x 3 settings) using algorithm dispatch + FCFS charging with cap90 integerized chargers (3->2, 5->4).
4) Do not modify algorithm implementation.

Required deliverables:
- method_summary_primary_plus_fdb.csv
- date_summary_primary_plus_fdb.csv
- cap90_ccb_raw.csv
- cap90_ccb_summary.csv
- evaluation_tables_for_paper.csv
- evaluation_notes_for_paper.md

Validation:
- CCB rows = 9
- primary rows = 27, ablation rows = 54 (check only)
- key columns non-null: total_served, charging_deadline_miss_ratio, total_waiting_vehicles, peak_station_power_kw
- explicitly state in notes: CCB is a capacity-constrained supplementary scenario and is not part of the primary paired significance conclusion.
```
