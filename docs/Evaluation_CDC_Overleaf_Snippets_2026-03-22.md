# CDC-Style Evaluation Snippets (Overleaf, 2026-03-22)

This document provides a strict, paper-ready evaluation write-up and figure templates for Overleaf.
Scope is evaluation-only. No method implementation changes are assumed.

## 1) Naming Standard (Paper-facing)

Use the following names consistently in text, tables, and figures.

| Paper Name | Abbrev. | Internal Combo ID |
|---|---|---|
| Coordinated Optimization Policy | COP | `algorithm_plus_gurobi` |
| Heuristic Dispatch Baseline | HDB | `heuristic_plus_fcfs` |
| Oracle Dispatch Baseline | ORB | `ideal_plus_fcfs` |

Notes:
- Internal combo IDs remain in scripts/logs for reproducibility.
- The paper body should avoid implementation-style names such as `gurobi+fcfs`.

## 2) Evaluation Body (LaTeX)

```latex
\section{Evaluation}

\subsection{Baseline Evaluation Setup}
We evaluate all methods under a unified protocol to ensure paired comparability across dispatch and charging decisions.
The simulation window is fixed to 06:00--22:00 with 15-minute slots and receding horizon $H=64$.
All methods use the same request stream, date set, random-seed policy, and resource grouping (Critical3: G1/G2/G3).
The primary evaluation covers three dates (2025-11-01, 2025-11-05, 2025-11-09), producing 9 matched \texttt{(date, group)} units per pairwise method comparison.

We compare:
\begin{itemize}
  \item \textbf{COP}: Coordinated Optimization Policy (proposed),
  \item \textbf{HDB}: Heuristic Dispatch Baseline,
  \item \textbf{ORB}: Oracle Dispatch Baseline.
\end{itemize}

The primary endpoint is \texttt{total\_served}.
To characterize operational trade-offs, we additionally report swap success, deadline miss, charging miss, waiting, and station power.
For paired significance on \texttt{total\_served}, we use two-sided sign tests over matched units.

\subsection{Primary Results}
\textbf{Summary of outcome.}
COP achieves statistically significant gains in \texttt{total\_served} against both baselines across all matched units, while exhibiting non-negligible quality trade-offs in swap/deadline related metrics.

\textbf{Aggregate service performance (27 runs).}
\begin{itemize}
  \item COP: $77613.67 \pm 6455.29$
  \item HDB: $36284.33 \pm 3285.06$
  \item ORB: $67840.44 \pm 6051.73$
\end{itemize}
Relative to baselines, COP improves \texttt{total\_served} by $+113.9\%$ over HDB and $+14.4\%$ over ORB.

\textbf{Paired consistency and significance (9 matched units per baseline).}
COP records Win/Tie/Loss = $9/0/0$ versus both HDB and ORB.
Sign tests on \texttt{total\_served} give:
\begin{itemize}
  \item COP vs HDB: mean paired delta $= +41329.33$, $p=0.0039$;
  \item COP vs ORB: mean paired delta $= +9773.22$, $p=0.0039$.
\end{itemize}

\textbf{Trade-off profile.}
The service gain is accompanied by a clear shift in operational quality:
\begin{itemize}
  \item swap success ratio: COP $=0.1531$, HDB $=0.8969$, ORB $=1.0000$;
  \item deadline miss ratio: COP $=0.8469$, HDB $=0.1031$, ORB $=0.0000$;
  \item charging miss ratio: COP $=0.0715$, HDB $=0.1089$, ORB $=0.4404$.
\end{itemize}
Therefore, the evidence supports a service-priority operating regime rather than uniform dominance across all metrics.

\textbf{Solver reliability.}
No no-incumbent reposition event is observed in this evaluation (\texttt{reposition\_event\_no\_incumbent\_count}=0), indicating stable solver behavior under the tested protocol.

\subsection{Ablation Evidence}
Ablation covers 54 runs (3 dates $\times$ 3 groups $\times$ 6 combinations) using a two-factor decomposition: dispatch policy and charging policy.
Results indicate that dispatch policy choice is the dominant source of \texttt{total\_served} variation, while charging solver changes are secondary and mixed in sign.
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

- All figure legends and text use `COP/HDB/ORB` only.
- No implementation-style method names appear in the paper body.
- Protocol text is fixed and identical everywhere: `06:00--22:00`, `15-min`, `horizon=64`, dates `2025-11-01/05/09`, groups `G1/G2/G3`.
- Reported primary significance values match `paired_stats.csv` (`p=0.0039` for both pairwise comparisons).
- Time-series cumulative served delta matches `+44947` (COP minus HDB).

## 6) Prompt for Experiment Agent (Incremental Re-check)

Use this directly:

```text
Please run an incremental verification pass for CDC evaluation outputs only.
Do not rerun full experiments.

Tasks:
1) Validate that main result files are unchanged in shape and key statistics:
   - primary rows = 27
   - ablation rows = 54
   - timeseries rows = 64
2) Re-run plotting script `scripts/plot_paper_core3_pdf.py` and confirm exactly seven PDF figures plus `core3_summary.json` are generated.
3) Verify `core3_summary.json` includes:
   - `timeseries_delta_served_sum_COP_minus_HDB = 44947`
   - all expected file names
4) Produce a short markdown report with:
   - pass/fail per check
   - any path mismatch or missing-column issue
   - final artifact directory

Output only the verification report and key command lines used.
```
