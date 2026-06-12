# Reproducibility files for “Whitened Moment Estimation and Spectrum Recovery under Heteroskedastic Noise”

## Overview

This repository contains the Python code used to reproduce the synthetic experiments in the manuscript:

> Whitened Moment Estimation and Spectrum Recovery under Heteroskedastic Noise

The observations are generated from

```text
Z = X S + U D^(1/2),
```

where the entries of `X` and `U` are independent standard normal variables in the simulations. The diagonal matrix

```text
D = diag(tau_1^2, ..., tau_d^2)
```

contains known unequal noise variances. The recovery target is the eigenvalue vector of the whitened signal covariance

```text
tilde_Sigma = D^(-1/2) Sigma D^(-1/2),
Sigma = S^T S.
```

The script performs four steps:

1. whiten the noisy observations;
2. estimate the first `K` moments with an increasing cycle estimator;
3. solve a support restricted linear program on `[1 / b_plus, 1]`;
4. compare the recovered spectrum with a shifted empirical baseline.

## Files

```text
.
├── README.md
├── Whitened_spectrum_experiment.py
├── requirements.txt
├── requirements-gurobi.txt
├── CITATION.cff
├── .gitignore
└── results/
       ├── README.md
       ├── d256_n256/
       └── d256_n128/
```

The simulation script writes CSV files, metadata, and empirical cumulative distribution function figures to a user-specified output directory.

## Software requirements

The code requires Python 3.9.12 or later and the packages listed in `requirements.txt`:

```text
numpy
pandas
matplotlib
scipy
```

Install them with:

```bash
python -m pip install -r requirements.txt
```

The manuscript values were generated with Gurobi. To use Gurobi, install the optional Python package and configure a valid Gurobi license:

```bash
python -m pip install -r requirements-gurobi.txt
```

SciPy is retained as a fallback backend. It can be used with `--solver scipy`, but the manuscript values should be reproduced with `--solver gurobi`.

Each run writes a `metadata.json` file containing the actual Python version, operating system, package versions, command line, and solver information.

## Simulation design

The manuscript reports the following settings:

- Dimension: `d = 256`
- Sample sizes: `n = 256` and `n = 128`
- Number of independent trials: `10`
- Number of estimated moments: `K = 7`
- Base random seed: `2026`
- Noise standard deviations: `tau_min = 0.3` and `tau_max = 1.0`
- Grid spacing: `eps = 1 / max(n, d)` unless `--eps` is specified
- Oracle upper bound in the synthetic experiments:

```text
b_plus = 1.1 * lambda_max(tilde_Sigma + I_d)
```

The three signal covariance settings are:

1. `identity`: `Sigma = I_d`.
2. `two_spike`: half of the eigenvalues of `Sigma` are `1` and half are `2`.
3. `toeplitz`: `Sigma_ij = 0.3^|i-j|`.

For the figures, the default horizontal axis is the square-root eigenvalue scale. The table errors are always computed on the eigenvalue scale.

## Random seeds

The script uses a base random seed of `2026`. For reproducibility, a fixed offset is added for each covariance setting:

```text
identity:  2026
two_spike: 2027
toeplitz:  2028
```

Within each covariance setting, the random number generator is initialized once. The ten trials are then generated sequentially from the same random number stream. The generated CSV files record both the base seed and the setting-specific seed.

## Reproduce the manuscript results

Run the two regimes in separate output directories to avoid overwriting files.

### Balanced regime: d = 256 and n = 256

```bash
python Whitened_spectrum_experiment.py \
  --solver gurobi \
  --d 256 \
  --n 256 \
  --k 7 \
  --num-runs 10 \
  --seed 2026 \
  --tau-min 0.3 \
  --tau-max 1.0 \
  --b-plus-multiplier 1.10 \
  --plot-scale sqrt \
  --output-dir results/d256_n256
```

### Under-sampled regime: d = 256 and n = 128

```bash
python Whitened_spectrum_experiment.py \
  --solver gurobi \
  --d 256 \
  --n 128 \
  --k 7 \
  --num-runs 10 \
  --seed 2026 \
  --tau-min 0.3 \
  --tau-max 1.0 \
  --b-plus-multiplier 1.10 \
  --plot-scale sqrt \
  --output-dir results/d256_n128
```

On Windows PowerShell, write each command on one line or replace each trailing backslash with a backtick.

## Generated outputs

Each output directory contains:

```text
all_runs.csv
summary.csv
metadata.json
identity_whitened_recovery_runs.csv
two_spike_whitened_recovery_runs.csv
toeplitz_whitened_recovery_runs.csv
identity_whitened_recovery_cdf.png
two_spike_whitened_recovery_cdf.png
toeplitz_whitened_recovery_cdf.png
fig_identity_cdf.pdf
fig_spiked_cdf.pdf
fig_toeplitz_cdf.pdf
```

The PDF figures from `results/d256_n128/` correspond to the under-sampled regime shown in the manuscript. The figures contain faint curves for individual trials and darker curves for the averages across ten trials.

The script also reports `reduction_percent` in `summary.csv`, computed relative to the empirical baseline.

## Manuscript results

The current manuscript reports the following mean normalized spectral errors. Sample standard deviations are shown in parentheses. The repository version fixes the endpoint handling of the support grid so that the maximum mesh size is at most `eps`. 
| Setting | Regime | Empirical baseline | Proposed estimate | Reduction |
|---|---|---:|---:|---:|
| Identity | `d = 256, n = 256` | 1.7261 (0.0263) | 0.9367 (0.1279) | 45.7% |
| Identity | `d = 256, n = 128` | 2.6167 (0.0315) | 1.2385 (0.2413) | 52.7% |
| Two-spike | `d = 256, n = 256` | 2.7018 (0.0246) | 0.9086 (0.0740) | 66.4% |
| Two-spike | `d = 256, n = 128` | 3.8561 (0.0439) | 1.1950 (0.2498) | 69.0% |
| Toeplitz | `d = 256, n = 256` | 1.5893 (0.0278) | 1.0895 (0.1123) | 31.4% |
| Toeplitz | `d = 256, n = 128` | 2.4265 (0.0379) | 1.2958 (0.2108) | 46.6% |

## Notes on the implementation

The manuscript defines the moment estimator using

```text
overline_Z = tilde_Z / sqrt(b_plus).
```

For computational convenience, the script first computes the increasing-cycle moments from `tilde_Z` and divides the moment of order `r` by `b_plus^r` before solving the linear program. These two implementations are algebraically equivalent.

The upper bound `b_plus` is an oracle quantity in the synthetic experiments because the population covariance is known. In applications, this upper bound must be specified or estimated separately.

## Data availability

The numerical results are generated from synthetic data. No external dataset is required.
The Python code used to generate the numerical results will be made publicly available upon acceptance.

```text
[The DOI will be added after the first archived release.]
```

## Citation

A `CITATION.cff` file is included. The DOI will be added after the first archived release.

## Contact

For questions about the code and numerical results, contact:

Jiaqi Chen  
School of Mathematics, Harbin Institute of Technology  
Email: chenjq1016@hit.edu.cn

## License

This code is released under the MIT License. See `LICENSE` for details.
