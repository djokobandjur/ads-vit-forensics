# Colab Quickstart Notebook

Cleaned repository notebook for reproducing ADS JSON artifacts and figures.

File: `colab_quickstart.ipynb`

Main updates:

- Replaced ad-hoc `/content/*_n6.py` calls with repository-standard script names.
- Centralized all Google Drive/model/dataset/output paths in one configuration cell.
- Added logged execution helper so each long experiment writes a Drive log file.
- Added dataset preparation, reference-index validation, figure generation, CPU-only verification, and final artifact inventory/zip cells.
- Cleared outputs and execution counts for clean GitHub display.
