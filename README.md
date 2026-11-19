# Rural Road Connectivity Progress Under PMGSY: Typology of States

**Author:** Madhumita Maan
**Roll No.:** 24020570008  
**Course:** Machine Learning (CSA), B.Sc. (Hons.) Computer Science, Semester VI  
**Institution:** Ramanujan College, University of Delhi  
**Allotted Problem:** Problem No. 40

## Overview
This repository analyses State/UT-wise road-length-completion figures across four financial years (`2021-22` to `2024-25`) under the Pradhan Mantri Gram Sadak Yojana (PMGSY) from `datafile.csv` ($N = 32$ States/UTs after removing the `Total` row and `Sl. No.`).

## Repository Structure
- `final_analysis.py`: End-to-end analysis script implementing data cleaning and quality auditing, dual-track preparation (median imputation + standardisation for clustering; observed-only values for linear trend slopes), K-Means and Ward hierarchical clustering (`k = 2..7`), Gaussian null-baseline comparison, 30-run bootstrap stability (ARI), complete-case and `log1p` scale sensitivity checks, and publication figure generation (`fig1_year_audit.png` to `fig6_state_trends.png`).
- `datafile.csv`: Source PMGSY dataset (33 rows including `Total`, 6 columns).
- `requirements_2.txt`: Python dependencies.
- `results/`: Generated CSV tables, JSON sensitivity summary, and figures.

## Reproduction Instructions
```bash
pip install -r requirements_2.txt
python final_analysis.py
```
