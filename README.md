# qc-rolling7-lvn-atlas

## Why this PR

- QuantConnect history for futures can be keyed by `Symbol` objects or other index values depending on mapping/normalization output, so the canonical display string (for example `"/MES"`) is not guaranteed to exist as a direct pandas key.
- This update replaces brittle direct indexing assumptions with symbol-safe history selection (`xs` attempts across likely symbol representations) and graceful fallback logging when no match is found.
- MES is now added through `Futures.Indices.MICRO_SP_500_E_MINI`, and orders are sent to the mapped contract (`future.Mapped`) instead of the continuous canonical symbol.
