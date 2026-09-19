# Oman GT3A sample-level table

After local preparation, `data/oman/oman_gt3a_sample_level.csv` contains the
sample-level input to the external control. All rows derive from version 1.2 of
the public workbook in Hiroshima University repository record 2000060. The
table is deliberately excluded from this repository because the source record
does not state a redistribution licence.

| Column | Meaning |
|---|---|
| `Source row` | Row number in the archived source workbook |
| `Core`, `Section`, `Depth (m)` | Physical sample identifiers and measured depth |
| `Lithology`, `Sequence` | Descriptive grouping variables; not model predictors |
| `Bulk density` | Bulk density in g cm\(^{-3}\) |
| `Porosity (%)` | Target porosity in percentage points |
| `Wet Vp mean`, `Wet Vs mean` | Geometric directional means in km s\(^{-1}\) |
| `R35 mean` | Geometric directional mean of resistivity at 35 g L\(^{-1}\) NaCl |
| `Formation factor` | Geometric directional mean used only for sensitivity analysis |
| `log10 R35`, `log10 F` | Frozen electrical predictors |
| `Depth block` | One of eight contiguous blocks assigned before complete-case filtering |
| `Primary complete`, `F complete` | Complete-case masks for the primary and sensitivity analyses |

Directions are summarized within each physical sample and are never counted as
independent replicates.
