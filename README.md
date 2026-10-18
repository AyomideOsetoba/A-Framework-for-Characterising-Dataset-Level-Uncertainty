# Dataset-level uncertainty in experimental condition-monitoring datasets: code

Code for the paper *A Framework for Characterising Dataset-Level Uncertainty in Experimental
Condition-Monitoring Datasets: A Rolling-Element Bearing Case Study* (Journal of Dynamics,
Monitoring and Diagnostics).

The code reads the raw vibration recordings, extracts squared-envelope-spectrum features, computes
the dataset-level uncertainty indicators with their statistical assessment and robustness checks,
and produces every data figure of the paper and the supplementary material.

## Contents

| File | Purpose |
|---|---|
| `uncertainty_core.py` | All classes (configuration, data loading, signal processing, indicators, statistics, plotting) |
| `01_extract_features.ipynb` | Raw Excel files to feature tables, plus peak-verification figures |
| `02_analysis.ipynb` | Indicators, bootstrap / jackknife / permutation test, screening sweep, variance components, task check, sample-size check, synthetic check |
| `03_paper_figures.ipynb` | All data figures, from the saved tables only |
| `outputs/features/` | Feature tables produced by `01_extract_features.ipynb` |
| `requirements.txt` | Python packages |

## Installation

```bash
pip install -r requirements.txt
```

## Data layout

```
<DATA_ROOT>/
    Healthy/  Test 1 ... Test 4/  Baseline | Visual | No Outerrace | Outerrace /  Test 1.xlsx ... Test 5.xlsx
    Damage/   Test 1 ... Test 4/  Baseline | Visual | No Outerrace | Outerrace /  Test 1.xlsx ... Test 5.xlsx
```

| Folder | Case | Intervention |
|---|---|---|
| `Baseline` | C1 | Baseline assembly |
| `Visual` | C2 | Preload removal and reinstallation |
| `No Outerrace` | C3 | Preload and inner race removal and reinstallation |
| `Outerrace` | C4 | Preload and full bearing removal and reinstallation |

Each Excel file holds one 10 s recording (R1 to R5) sampled at 48 kHz. The analysed accelerometer is
the `X - Axis` column (DAQ hardware channel 2); numeric data start on row 50 of the sheet.

## Running

1. `01_extract_features.ipynb`: set `DATA_ROOT` (and `N_JOBS`) and run all cells. This is the only
   step that needs the raw data (about 10 s per recording per process). Its feature tables are
   included in `outputs/features/`, so this step can be skipped.
2. Run `02_analysis.ipynb`.
3. Run `03_paper_figures.ipynb`.

All results are written to `outputs/`:

```
outputs/features/           features_full.csv (160 rows), features_segmented.csv (1600 rows)
outputs/peak_verification/  SES with the selected 1x and 2x BPFO peaks for every recording (step 1)
outputs/tables/             all result tables (CSV)
outputs/figures/            all figures (PNG, and PDF where useful)
```

## Indicators

| Symbol | Meaning |
|---|---|
| $D_{\mathrm{hull}}$ | Square root of the convex-hull area of a set of feature points (dispersion) |
| $D_\Sigma$ | $(\det\hat\Sigma)^{1/4}$ of the sample covariance (dispersion) |
| $B$ | Bias: distance between the centroid of case Cj and the centroid of C1 of the same test session |
| $\rho_\Sigma$ | Effect size $B/D_{\Sigma,\mathrm{pool}}$, with the pooled covariance of C1 and Cj |
| $\Delta/\delta$ | Task check: smallest healthy-to-damaged centroid distance over the largest distance between two centroids of the same health state |

An assembly state is one intervention case (C1 to C4) within one test session (T1 to T4). Whole
recordings are the unit of repetition in the bootstrap and in the exact permutation test.

## Processing settings

| Setting | Value |
|---|---|
| Envelope | Hilbert transform, squared envelope, mean removed |
| Spectrum | FFT, $N_{FFT}=2^{22}$, one-sided, scaled by $2/N$ ($N$ = number of samples) |
| Fault frequency | BPFO = 276.27 Hz (nominal 2000 rpm); features at 1x and 2x BPFO |
| Peak selection | maximum of the SES within ±5 Hz of each harmonic |
| Signal lengths | full 10 s recording, and ten non-overlapping 1 s segments |
| Hull screening | 85 % of the largest distance from the centroid, segmented sets only |

All settings are in the `Config` class of `uncertainty_core.py`. Random seeds are fixed per analysis,
so every run reproduces the same numbers.
