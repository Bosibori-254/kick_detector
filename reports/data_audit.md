# DataDRILL kick-detection dataset audit

> Generated automatically by `python -m src.cli audit`. This is an experimental research prototype, not an operational well-control tool.

## 1. Provenance

- Source: https://zenodo.org/records/12759014
- Configured path: `data/raw/Kick_Detection.csv`
- MD5 observed: `8bbd611ca9d66c397d43c61a152bed93`
- MD5 published by Zenodo: `8bbd611ca9d66c397d43c61a152bed93`
- The raw CSV is read only and is never overwritten by this project.

## 2. Shape, types and completeness

- Rows: 2337; columns: 28.
- Duplicate rows: 6 (5 beyond the first copy), retained for inspection.
- Missing values: none detected.
- Infinite/non-finite values: none detected.
- IQR-rule outliers are recorded but retained: in drilling data, large excursions can be the kick event itself. Counts: CircFlow=40, FDensity=324, SMSpeed=172, BDepth=1, BSize=1, MVis=324, DPPress=974, FIn=48, ActiveGL=275, ATVolume=275, HLoad=82, WBoPress=815.
- Constant columns: ['CPress', 'MPS1', 'MPS2', 'MPS3', 'AMTD', 'STP'].
- Near-constant columns (<5 distinct values): ['BSize'].
- Non-numeric columns: none.

| Column | Dtype | Min | Median | Max | Distinct |
| --- | --- | ---: | ---: | ---: | ---: |
| CircFlow | float64 | 0 | 0.953155 | 0.965908 | 1843 |
| FRate | float64 | -3.01253 | -0.95497 | 0 | 1892 |
| FDensity | float64 | 0 | 78.5471 | 78.5471 | 838 |
| SMSpeed | float64 | -0.177619 | -0.00548131 | 0 | 1697 |
| BDepth | float64 | 0 | 12645.6 | 12665.4 | 1393 |
| BSize | float64 | 0 | 12.25 | 12.25 | 2 |
| MVis | float64 | 0 | 10.0005 | 10.0005 | 838 |
| FPress | float64 | 6157.12 | 7263.53 | 8224.05 | 1497 |
| DPPress | float64 | 1302.94 | 1683.49 | 1730.27 | 1592 |
| CPress | int64 | 0 | 0 | 0 | 1 |
| FIn | float64 | 0.952737 | 0.952945 | 0.953303 | 1105 |
| FOut | float64 | 0.952837 | 0.95497 | 3.01253 | 1892 |
| MRFlow | float64 | -2.05949 | -0.00199063 | 0.000103364 | 1944 |
| ActiveGL | float64 | -0.0825956 | -0.0279161 | 55.7161 | 2325 |
| ATVolume | float64 | 199 | 199.055 | 254.799 | 1142 |
| RoPen | float64 | 1.69897 | 162.385 | 242.797 | 2311 |
| WoBit | float64 | 1224.8 | 39666 | 39821 | 2246 |
| HLoad | float64 | 310776 | 310922 | 355229 | 1782 |
| WBoPress | float64 | 6682.45 | 6955.51 | 7060.65 | 1873 |
| BTBR | float64 | 12641.5 | 12645.6 | 12665.4 | 1392 |
| MPS1 | float64 | 99.0463 | 99.0463 | 99.0463 | 1 |
| MPS2 | float64 | 99.0463 | 99.0463 | 99.0463 | 1 |
| MPS3 | float64 | 59.6604 | 59.6604 | 59.6604 | 1 |
| AMTD | float64 | 78.5455 | 78.5455 | 78.5455 | 1 |
| ATMPV | float64 | 10.0001 | 10.0001 | 10.0003 | 20 |
| ATMYP | float64 | 20.0001 | 20.0001 | 20.0003 | 19 |
| STP | float64 | 13798.1 | 13798.1 | 13798.1 | 1 |
| WellDepth | float64 | 12641.5 | 12645.6 | 12665.4 | 1392 |

## 3. Target verification

- Configured target column: `ActiveGL`; mode: `proxy_from_threshold`.
- Rule: ActiveGL > 0 bbl (pit gain rather than pit loss).
- Class counts: 1501 negative / 836 positive (35.8% positive).
- First positive row index: 1501; depth there: 12650.15 ft (documented kick depth: 12650.0 ft).
- Label transitions: 1.
- The source documentation calls ActiveGL 'Active Gain Loss / pit gain-loss', an **indicator of a kick**. It is continuous, not a supplied binary class. The binary label used here is therefore an experimental proxy based on the physically meaningful change from pit loss (negative) to pit gain (positive), which coincides with the documented kick depth in the source paper. Supervisor confirmation is required.

Warnings:

- No authoritative kick label is supplied by the dataset. Labels are an EXPERIMENTAL PROXY from 'ActiveGL' and must be confirmed by a drilling supervisor before any research claim is made.

## 4. Leakage and redundancy findings

| Column(s) | Severity | Reason |
| --- | --- | --- |
| ActiveGL | critical | Source of the experimental target; using it as an input would be target leakage. |
| ATVolume | critical | Near-exact copy of ActiveGL plus a constant offset (~199.083); encodes the target. |
| MRFlow | moderate | Exactly FIn - FOut; redundant with the engineered Flow_Imbalance feature. |
| WellDepth/BDepth/BTBR | high | Kick is configured at 12650 ft; depth can memorise this single simulated scenario. |
| FPress | high | Configured cause of the simulated kick, not a conventional real-time measured kick indicator. |

Verified algebraic identities:

- `MRFlow_equals_FIn_minus_FOut`: {'max_abs_residual': 1.09999999997612e-06, 'holds': True}
- `FRate_is_negative_FOut`: {'correlation': -0.9997768438092364, 'max_abs_residual_all_rows': 0.9528908, 'max_abs_residual_after_first_row': 0.0, 'holds_after_first_row': True}
- `ATVolume_is_ActiveGL_plus_constant`: {'estimated_offset': 199.0827772, 'max_abs_residual': 6.279999999492247e-05, 'holds': True}

## 5. Temporal and event information

- Timestamp-like columns: none.
- Group/event columns: none.
- Sequence order plausible from monotonic depth: True.
- Validated early-warning lead time possible: **False**.

- No timestamp-like column exists; elapsed warning time cannot be calculated.
- No well/run/event identifier exists; only one continuous run is observable.
- 'WellDepth' is monotonic, so row order is a plausible sequence index.

## 6. Target correlations (observed, not causal)

| Feature | Pearson r with target signal |
| --- | ---: |
| ATVolume | 1.0000 |
| DPPress | -0.9494 |
| HLoad | 0.9471 |
| BTBR | 0.9266 |
| WellDepth | 0.9266 |
| WoBit | -0.9204 |
| ATMYP | 0.9043 |
| ATMPV | 0.9014 |
| MVis | -0.8683 |
| FDensity | -0.8508 |
| WBoPress | -0.8473 |
| FOut | 0.8428 |

Correlation does not establish causation. High correlations with `ATVolume` and depth-related columns are consequences of how the simulation records the event and are exactly why those variables are excluded from model inputs.
