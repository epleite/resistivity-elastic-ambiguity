# When does resistivity resolve elastic ambiguity?

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22844387.svg)](https://doi.org/10.5281/zenodo.22844387)

Reproducibility repository for:

**Emilson Pereira Leite**, “When does resistivity resolve elastic ambiguity? Testing partially coupled rock-physics inversion under model error”.

The study tests when electrical resistivity adds reliable porosity information beyond elastic velocity and density, and how that gain changes when the fitted rock-physics models omit relevant transport processes.

## Versioned release

The complete tested package is available from the [v1.0.0 release](https://github.com/epleite/resistivity-elastic-ambiguity/releases/tag/v1.0.0):

- `resistivity-elastic-ambiguity-v1.0.0.zip`: source code, environment files, automated tests, frozen synthetic results, aggregate Oman results, and scripts for all seven manuscript figures;
- `Leite_GMD_source_v0.2.zip`: manuscript and supplement source prepared for *Geoscientific Model Development*;
- `SHA256SUMS-release.txt`: integrity checks for both archives.

The default branch exposes the source code, tests, documentation, and citation metadata for inspection. Download and extract the complete release asset before running the full reproducibility checks.

## Validation

- Python 3.10–3.12;
- 50 local tests passed with the locally prepared Oman table;
- the clean release archive passes all available tests, with the Oman sample-level test skipped as designed;
- 50 frozen numerical-artifact hashes verified;
- seven manuscript figures regenerated from the archived results.

## Oman GT3A data

The source workbook is openly available from the Hiroshima University Institutional Repository, record 2000060. Its record does not state a redistribution licence. The sample-level measurements are therefore not redistributed. The release supplies a hash-checking preparation script that reconstructs the derived table after a local download of the official workbook.

## Citation and licence

Citation metadata are provided in `CITATION.cff`. The software is released under the MIT License. The archived v1.0.0 release is available at [https://doi.org/10.5281/zenodo.22844387](https://doi.org/10.5281/zenodo.22844387).

## Author

Emilson Pereira Leite  
Department of Geology and Natural Resources, Institute of Geosciences, University of Campinas (UNICAMP), Rua Carlos Gomes 250, 13083–855 Campinas, SP, Brazil  
ORCID: https://orcid.org/0000-0003-1691-6243  
Email: emilson@unicamp.br

## Funding

The author acknowledges support from a CNPq Research Productivity Fellowship.
