# Oman GT3A input preparation

The external control uses a derived sample-level table named
`oman_gt3a_sample_level.csv`. It is not redistributed in this repository
because the source record provides open access but does not state a licence
that authorizes redistribution of the measurements. Download version 1.2 of
the supporting workbook for Akamatsu and Katayama (2022) from Hiroshima
University repository record 2000060:

<https://hiroshima.repo.nii.ac.jp/records/2000060>

The expected source workbook has SHA-256
`dceaae2ee7e85d233349222447f3c17e50d726b580ee2c629bcc1aed9291ff8b`.
Then generate the local analysis table:

```bash
python scripts/prepare_oman_data.py \
  --source /path/to/Oman_GT3A_Supporting_information_original.xlsx
```

The preparation script verifies the source hash before reading it. Column
definitions and extraction rules are documented in
[`../../docs/oman-data-dictionary.md`](../../docs/oman-data-dictionary.md).

The repository's MIT License applies only to the software and original
repository content. It does not apply to the source measurements or to a local
derived table generated from them.
