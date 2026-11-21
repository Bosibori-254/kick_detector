# Temporal / early-warning feasibility assessment

This report answers whether the downloaded dataset supports a **validated early-warning**
analysis. It is based solely on the audited file header, the source documentation and
the source paper (Arifeen et al., arXiv:2409.19724).

## Evidence examined

| Requirement | Finding |
| --- | --- |
| Timestamp column | None found among the 28 columns. Row order is a plausible sequence index because depth increases monotonically, but elapsed seconds per row are unknown. |
| Well / run / scenario identifier | None found. The file is one continuous simulated drilling sequence. |
| Event identifier or kick onset time | None supplied. The paper documents a kick at approximately 12,650 ft. |
| Observations before the event | Yes: proxy pit gain changes from loss to gain at row 1501 (12,650.15 ft), leaving 1,501 preceding rows. |
| Observations after the event | Yes: 836 subsequent rows. |

## Decision

**Validated early-warning lead time cannot be calculated.** There are no timestamps and
no recorded kick onset time, so a warning measured in seconds or minutes would be
fabricated. The project therefore evaluates **kick-condition classification at the
observation time**, not confirmed pre-event detection.

## What was still done

- Row order was preserved and treated as a sequence for splitting purposes.
- Contiguous 25-row blocks were kept together in the train/test split to reduce leakage
  from adjacent near-duplicate measurements.
- A block purge gap was applied around the held-out blocks.
- Causal rolling features are implemented in `src/feature_engineering.py` but are disabled
  by default (`features.rolling.enabled: false`) because no time calibration exists.
- An Isolation Forest variant is trained only on normal training observations to provide a
  sequence-aware unsupervised comparison.

## What requires external information

To perform genuine early-warning research, the following are needed from the simulator
operator or a drilling supervisor:

1. A timestamp or sampling interval for every row.
2. The recorded kick onset time and a definition of the warning horizon.
3. Multiple runs/wells so that a model can be validated across scenarios.
4. Confirmation that a first warning must persist for N consecutive samples.

No detection lead-time number is reported anywhere in this project.
