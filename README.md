# The Geometry of Omission: Type I, II, and III Identification in Correlated Data
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17392989.svg)](https://doi.org/10.5281/zenodo.17392989)

Rebecca Whitworth, PhD  
License: MIT

This repository contains simulation and analysis code accompanying the working paper  
_The Geometry of Omission: Type I, II, and III Identification in Correlated Data._  

**Disclaimer:** This repository is provided for research transparency and reproducibility.  
It is not intended for commercial or production use.  
The views expressed are those of the author and do not represent any affiliated institution.

**Citation:**
@software{whitworth2025geometry,
  author    = {Rebecca Whitworth},
  title     = {The Geometry of Omission: Type I, II, and III Identification in Correlated Data},
  month     = oct,
  year      = 2025,
  publisher = {Zenodo},
  version   = {v1.0.0a},
  doi       = {10.5281/zenodo.17392989},
  url       = {https://doi.org/10.5281/zenodo.17392989}
}

---

## Overview

This repository contains all replication code, figures, and supporting diagnostics for the paper  
**"The Geometry of Omission: Type I, II, and III Identification in Correlated Data" (Whitworth, 2025).**

The paper explores how common preprocessing steps that "remove group effects" (e.g., global or within-group demeaning) can unintentionally destroy structural information that machine learning models depend on.  
Through three simulated regimes, we demonstrate how omitted regional structure re-emerges as prediction bias and calibration drift.

---

## Repository Structure
geometry_of_omission/

├── geometry_of_omission.py        # Main replication script (reproduces all paper figures)

├── diagnostics_and_checks.py      # Optional diagnostics and exploratory analyses

├── figures/                       # Auto-generated output figures

├── requirements.txt               # Python package dependencies

├── environment.yml                # Conda environment for reproducibility

├── README.md

└── LICENSE

---

## Environment

All experiments were run on macOS (Apple Silicon) with Python 3.11.

**Core dependencies:**
- numpy ≥ 1.26  
- pandas ≥ 2.2  
- scikit-learn ≥ 1.5  
- matplotlib ≥ 3.8  
- torch ≥ 2.3  
- xgboost ≥ 2.1  
- seaborn ≥ 0.13  
- scipy ≥ 1.13  

To recreate the environment:
**Option 1:PIP**
pip install -r requirements.txt

**Option 2:CONDA**
conda env create -f environment.yml
conda activate geometry_of_omission

### Usage
To reproduce all figures in the paper:
python geometry_of_omission.py

By default, all figures are saved automatically to:
~/Documents/Geometry_of_Omission/figures/

(If you wish to save elsewhere, edit the export_dir variable near the top of
geometry_of_omission.py to point to your preferred directory.)

Each file name matches the figure numbering and title in the paper.

### Figures Produced by geometry_of_omission.py (Deterministic)
| Figure | Title | Filename |
|:--:|:--|:--|
| **1** | DGP Geometry across omission regimes | `fig_1_dgp_geometry.png` |
| **2a** | Groupwise Calibration (Type I) | `fig_2a_groupwise_calibration.png` |
| **2b** | Groupwise Prediction Bias (Type I) | `fig_2b_groupwise_bias.png` |
| **3** | Reliability (Calibration) Curves — Type I | `fig_3_reliability_quantile_shared.png` |
| **4a** | XGBoost — no region (Type II) | `fig_4a_typeII_XGB.png` |
| **4b** | XGBoost — with region (diagnostic, Type II) | `fig_4b_typeII_diagnostic.png` |
| **4c** | Logistic — no region (Type II) | `fig_4c_typeII_Logistic_no_region.png` |
| **4d** | Logistic — with region (Type II) | `fig_4d_typeII_Logistic_with_region.png` |
| **4e** | NeuralNet — no region (Type II) | `fig_4e_typeII_NeuralNet_no_region.png` |
| **4f** | NeuralNet — with region (Type II) | `fig_4f_typeII_NeuralNet_with_region.png` |
| **5a** | XGBoost — no region (Type III) | `fig_5a_typeIII_XGB.png` |
| **5b** | XGBoost — with region (diagnostic, Type III) | `fig_5b_typeIII_diagnostic.png` |
| **5c** | Logistic — no region (Type III) | `fig_5c_typeIII_Logistic_no_region.png` |
| **5d** | Logistic — with region (Type III) | `fig_5d_typeIII_Logistic_with_region.png` |
| **5e** | NeuralNet — no region (Type III) | `fig_5e_typeIII_NeuralNet_no_region.png` |
| **5f** | NeuralNet — with region (Type III) | `fig_5f_typeIII_NeuralNet_with_region.png` |
| **6** | Reliability across omission regimes (Types I–III) | `fig_6_reliability_across_regimes.png` |
| **A1** | Correlation geometry heatmap — Type I | `appendix_typeI_corr_heatmap.png` |
| **A2** | Correlation geometry heatmap — Type II | `appendix_typeII_corr_heatmap.png` |
| **A3** | Correlation geometry heatmap — Type III | `appendix_typeIII_corr_heatmap.png` |


### Supplemental Diagnostics
Additional validation and exploratory checks can be reproduced using:
python diagnostics_and_checks.py

This script includes:
	•	ROC curves and by-region AUC comparisons
	•	Boxplot diagnostics for predicted probabilities
	•	Regional correlation structure heatmaps
	•	Copula-based latent structure checks

These diagnostics support robustness claims and transparency but are not shown in the main paper.

### Citation
If you use this code or reproduce figures, please cite:

Whitworth, R. (2025). The Geometry of Omission: Type I, II, and III Identification in Correlated Data [v1.0.0a]. Zenodo. https://doi.org/10.5281/zenodo.17392989

### Notes for Reproducibility
- All simulations use fixed random seeds for full determinism.
- The Diagnostic (+region) model in figures refers to the oracle case where region indicators are explicitly included.
- Colors and figure titles are aligned with the paper to ensure 1-to-1 correspondence.
- Figures will render in color (as intended) for both interactive and PDF LaTeX inclusion.

### Contact
For correspondence:
Rebecca Whitworth
rebeccawhitworth [at] gmail.com
github.com/r-whitworth/
linkedin.com/in/rwhitworth

Version: October 2025.1
