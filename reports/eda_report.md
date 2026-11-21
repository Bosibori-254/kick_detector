# Exploratory data analysis report

> Generated from the raw DataDRILL CSV. Correlations are observations, not causal claims.

- Dataset shape: 2337 rows x 28 columns.
- Missing values: 0 total.
- Exact duplicate rows: 5.
- Target: `ActiveGL` (proxy_from_threshold); 1501 normal / 836 proxy kick (35.8% positive).
- Proxy onset row: 1501; depth: 12650.15 ft.

## Strongest observed associations with the continuous target signal

| Variable | Pearson r |
| --- | ---: |
| ATVolume | 1.0000 |
| DPPress | -0.9494 |
| HLoad | 0.9471 |
| BTBR | 0.9266 |
| WellDepth | 0.9266 |
| WoBit | -0.9204 |
| ATMYP | 0.9043 |
| ATMPV | 0.9014 |

## Petroleum-engineering interpretation

- `ATVolume` is a near-exact copy of `ActiveGL` plus a nearly constant offset (~199.083), so its apparent association is redundancy, not independent evidence.
- Increasing `FOut` relative to `FIn` (positive `Flow_Imbalance`) is consistent with formation fluid entering the wellbore and increasing return flow.
- Pressure and mechanical variables (`DPPress`, `HLoad`, `WoBit`) change as the simulated formation pressure exceeds hydrostatic pressure. These are associations within one simulated scenario, not universal physical laws.
- Class imbalance is present but moderate; recall and precision are therefore more informative than accuracy.
- No timestamp or event identifier is available, so this EDA cannot validate early-warning lead time.

## Figures

- `results/figures/target_distribution.png`
- `results/figures/missing_values.png`
- `results/figures/correlation_heatmap.png`
- `results/figures/feature_distributions.png`
- `results/figures/boxplots_by_class.png`
- `results/figures/flow_pressure_relationships.png`
