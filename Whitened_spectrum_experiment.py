#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Reproduce the synthetic experiments for noise-normalized spectrum recovery.

The script implements the numerical procedure described in the manuscript:
whitening, cycle based moment estimation, support restricted moment inversion,
and comparison with a naive shifted empirical spectrum.

The linear program can be solved with Gurobi or SciPy. The manuscript numbers
use Gurobi; SciPy is retained as a fallback backend. The simulations use an
oracle value of b_plus by default because the population covariance is known in
synthetic experiments.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".matplotlib"))

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ExperimentConfig:
    settings: list[str]
    k: int
    d: int
    n: int
    num_runs: int
    seed: int
    tau_min: float
    tau_max: float
    b_plus_multiplier: float
    eps: float | None
    timelimit: int
    solver: str
    plot_scale: str
    output_dir: str


def parse_args() -> ExperimentConfig:
    parser = argparse.ArgumentParser(
        description="Run whitened spectrum recovery simulations."
    )
    parser.add_argument(
        "--settings",
        nargs="+",
        default=["identity", "two_spike", "toeplitz"],
        choices=["identity", "two_spike", "toeplitz"],
    )
    parser.add_argument("--k", type=int, default=7)
    parser.add_argument("--d", type=int, default=256)
    parser.add_argument("--n", type=int, default=256)
    parser.add_argument("--num-runs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--tau-min", type=float, default=0.3)
    parser.add_argument("--tau-max", type=float, default=1.0)
    parser.add_argument("--b-plus-multiplier", type=float, default=1.10)
    parser.add_argument(
        "--eps",
        type=float,
        default=None,
        help="Grid spacing. Defaults to 1 / max(n, d).",
    )
    parser.add_argument("--timelimit", type=int, default=60)
    parser.add_argument(
        "--solver",
        choices=["auto", "scipy", "gurobi"],
        default="auto",
        help="LP solver. auto uses Gurobi when available and otherwise tries SciPy.",
    )
    parser.add_argument(
        "--plot-scale",
        choices=["sqrt", "eigenvalue"],
        default="sqrt",
        help="Scale used for CDF figures. Table errors are always eigenvalue-scale.",
    )
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Directory where CSV, metadata, and figures are written.",
    )
    ns = parser.parse_args()
    return ExperimentConfig(
        settings=ns.settings,
        k=ns.k,
        d=ns.d,
        n=ns.n,
        num_runs=ns.num_runs,
        seed=ns.seed,
        tau_min=ns.tau_min,
        tau_max=ns.tau_max,
        b_plus_multiplier=ns.b_plus_multiplier,
        eps=ns.eps,
        timelimit=ns.timelimit,
        solver=ns.solver,
        plot_scale=ns.plot_scale,
        output_dir=ns.output_dir,
    )


def covariance_matrix(setting: str, d: int) -> np.ndarray:
    if setting == "identity":
        return np.eye(d, dtype=float)
    if setting == "two_spike":
        eigvals = np.concatenate([np.ones(d // 2), 2.0 * np.ones(d - d // 2)])
        return np.diag(eigvals)
    if setting == "toeplitz":
        rho = 0.3
        idx = np.arange(d)
        return rho ** np.abs(idx[:, None] - idx[None, :])
    raise ValueError(f"Unknown setting: {setting}")


def display_name(setting: str) -> str:
    return {
        "identity": "Identity",
        "two_spike": "Two-spike",
        "toeplitz": "Toeplitz",
    }[setting]


def moment_est_from_data(z_whitened: np.ndarray, n: int, k: int, d: int) -> np.ndarray:
    """Estimate moments of the unscaled shifted whitened covariance.

    The manuscript defines the estimator on z_whitened / sqrt(b_plus). Here we
    estimate moments for z_whitened and divide by b_plus**r before the LP.
    These two implementations are algebraically equivalent.
    """
    a_mat = z_whitened @ z_whitened.T
    g_mat = a_mat.copy()
    for i in range(n):
        g_mat[i:n, i] = 0.0

    m_hat = np.zeros(k, dtype=float)
    for r in range(1, k + 1):
        if r == 1:
            numerator = float(np.trace(a_mat))
        else:
            numerator = float(np.trace(np.linalg.matrix_power(g_mat, r - 1) @ a_mat))
        denominator = float(d * math.comb(n, r))
        m_hat[r - 1] = numerator / denominator
    return m_hat


def spectrum_plus_with_lower_bound(
    moment_plus_vec: np.ndarray,
    k: int,
    d: int,
    n: int,
    b_plus: float,
    eps: float | None,
    timelimit: int,
    solver: str,
) -> np.ndarray:
    b = float(b_plus)
    moment_scaled = np.array(
        [float(moment_plus_vec[i]) / (b ** (i + 1)) for i in range(k)],
        dtype=float,
    )
    grid_eps = eps if eps is not None else 1.0 / max(d, n)
    if grid_eps <= 0:
        raise ValueError("Grid spacing must be positive.")

    lower = 1.0 / b
    grid_size = int((1.0 - lower) / grid_eps) + 1
    grid = lower + np.arange(grid_size, dtype=float) * grid_eps
    if grid[-1] < 1.0:
        grid = np.append(grid, 1.0)
    else:
        grid[-1] = 1.0
    grid_size = len(grid)

    powers = np.zeros((k, grid_size), dtype=float)
    for i in range(k):
        powers[i, :] = grid ** (i + 1)

    if solver == "auto":
        try:
            import gurobipy  # noqa: F401

            solver_to_use = "gurobi"
        except Exception:
            solver_to_use = "scipy"
    else:
        solver_to_use = solver

    if solver_to_use == "scipy":
        p_opt = solve_lp_scipy(powers, moment_scaled)
    elif solver_to_use == "gurobi":
        p_opt = solve_lp_gurobi(powers, moment_scaled, timelimit)
    else:
        raise ValueError(f"Unknown solver: {solver_to_use}")

    cumulative = np.cumsum(p_opt)
    targets = (np.arange(1, d + 1, dtype=float)) / (d + 1)
    indices = np.searchsorted(cumulative, targets, side="left")
    indices = np.minimum(indices, grid_size - 1)
    lam_scaled = grid[indices]
    lam_plus = b * lam_scaled
    if np.min(lam_plus) < 1.0 - 1e-8:
        raise AssertionError(
            "Support-restricted reconstruction returned lam_plus below 1. "
            "This contradicts the stated support [1/b_plus, 1]."
        )
    return lam_plus


def solve_lp_scipy(powers: np.ndarray, moment_scaled: np.ndarray) -> np.ndarray:
    """Solve min ||powers @ p - moment_scaled||_1 with scipy.optimize.linprog."""
    from scipy.optimize import linprog

    k, grid_size = powers.shape
    # Variables are [p_0, ..., p_{M-1}, u_0, ..., u_{K-1}],
    # where u_i bounds the absolute residual of moment i.
    c = np.concatenate([np.zeros(grid_size), np.ones(k)])
    a_ub = []
    b_ub = []
    for i in range(k):
        row = np.zeros(grid_size + k)
        row[:grid_size] = powers[i, :]
        row[grid_size + i] = -1.0
        a_ub.append(row)
        b_ub.append(moment_scaled[i])

        row = np.zeros(grid_size + k)
        row[:grid_size] = -powers[i, :]
        row[grid_size + i] = -1.0
        a_ub.append(row)
        b_ub.append(-moment_scaled[i])

    a_eq = np.zeros((1, grid_size + k))
    a_eq[0, :grid_size] = 1.0
    bounds = [(0.0, 1.0)] * grid_size + [(0.0, None)] * k

    result = linprog(
        c,
        A_ub=np.vstack(a_ub),
        b_ub=np.array(b_ub),
        A_eq=a_eq,
        b_eq=np.array([1.0]),
        bounds=bounds,
        method="highs",
    )
    if not result.success:
        raise RuntimeError(f"SciPy linprog failed: {result.message}")
    p_opt = np.maximum(result.x[:grid_size], 0.0)
    total = p_opt.sum()
    if total <= 0:
        raise RuntimeError("LP returned a zero probability vector.")
    return p_opt / total


def solve_lp_gurobi(
    powers: np.ndarray,
    moment_scaled: np.ndarray,
    timelimit: int,
) -> np.ndarray:
    import gurobipy as gp

    k, grid_size = powers.shape
    model = gp.Model()
    p = model.addMVar((grid_size,), lb=0, ub=1, vtype=gp.GRB.CONTINUOUS, name="p")
    model.addConstr(p.sum() == 1.0)

    z = model.addMVar((k,), lb=-gp.GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name="z")
    for i in range(k):
        model.addConstr(z[i] == (powers[i, :] @ p - moment_scaled[i]))

    obj = model.addVar(name="obj")
    model.addGenConstrNorm(obj, z.tolist(), 1.0, "normconstr")
    model.setObjective(obj, sense=gp.GRB.MINIMIZE)

    model.setParam("OutputFlag", 0)
    model.setParam("NumericFocus", 3)
    model.setParam("Method", 2)
    model.setParam("BarConvTol", 1e-10)
    model.setParam("FeasibilityTol", 1e-9)
    model.setParam("OptimalityTol", 1e-9)
    model.setParam("TimeLimit", timelimit)
    model.optimize()

    if model.Status not in (gp.GRB.OPTIMAL, gp.GRB.SUBOPTIMAL):
        raise RuntimeError(f"Gurobi failed with status={model.Status}.")

    return np.asarray(p.getAttr("X"), dtype=float)


def l1_eig_error(lam_hat: np.ndarray, lam_true: np.ndarray) -> float:
    return float(np.sum(np.abs(np.sort(lam_hat) - np.sort(lam_true))))


def plot_curves(
    setting: str,
    true_eigs: np.ndarray,
    empirical_curves: list[np.ndarray],
    recovered_curves: list[np.ndarray],
    output_path: Path,
    scale: str,
) -> None:
    cdf = np.arange(1, len(true_eigs) + 1) / len(true_eigs)
    true_curve = np.sort(np.sqrt(true_eigs) if scale == "sqrt" else true_eigs)

    plt.figure(figsize=(10, 6))
    for curve in empirical_curves:
        plt.step(curve, cdf, where="post", color="gray", alpha=0.20)
    plt.step(
        np.sort(np.mean(np.vstack(empirical_curves), axis=0)),
        cdf,
        where="post",
        color="gray",
        linewidth=2.0,
        label="Empirical baseline",
    )

    for curve in recovered_curves:
        plt.step(curve, cdf, where="post", color="blue", alpha=0.20)
    plt.step(
        np.sort(np.mean(np.vstack(recovered_curves), axis=0)),
        cdf,
        where="post",
        color="blue",
        linewidth=2.0,
        label="Proposed estimate",
    )

    plt.step(true_curve, cdf, where="post", color="black", linewidth=2.0, label="True spectrum")
    plt.xlabel("sqrt(eigenvalue)" if scale == "sqrt" else "eigenvalue")
    plt.ylabel("Empirical CDF")
    plt.title(f"{display_name(setting)} covariance under heteroskedastic diagonal noise")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def manuscript_figure_name(setting: str) -> str:
    return {
        "identity": "fig_identity_cdf.pdf",
        "two_spike": "fig_spiked_cdf.pdf",
        "toeplitz": "fig_toeplitz_cdf.pdf",
    }[setting]


def run_setting(setting: str, cfg: ExperimentConfig, output_dir: Path) -> pd.DataFrame:
    seed_offset = {
        "identity": 0,
        "two_spike": 1,
        "toeplitz": 2,
    }[setting]
    setting_seed = cfg.seed + seed_offset
    rng = np.random.default_rng(setting_seed)
    sigma = covariance_matrix(setting, cfg.d)
    chol = np.linalg.cholesky(sigma + 1e-12 * np.eye(cfg.d))

    tau = np.linspace(cfg.tau_min, cfg.tau_max, cfg.d)
    d_inv_sqrt = 1.0 / tau
    sigma_tilde = (d_inv_sqrt[:, None] * sigma) * d_inv_sqrt[None, :]
    eig_tilde_true = np.maximum(np.linalg.eigvalsh(sigma_tilde), 0.0)
    eig_plus_true = eig_tilde_true + 1.0
    b_plus = float(eig_plus_true.max()) * cfg.b_plus_multiplier

    empirical_curves: list[np.ndarray] = []
    recovered_curves: list[np.ndarray] = []
    rows = []

    for run in range(1, cfg.num_runs + 1):
        signal_noise_free = rng.standard_normal((cfg.n, cfg.d)) @ chol.T
        noise = rng.standard_normal((cfg.n, cfg.d)) * tau
        z_obs = signal_noise_free + noise
        z_tilde = z_obs * d_inv_sqrt

        moment_hat = moment_est_from_data(z_tilde, cfg.n, cfg.k, cfg.d)
        lam_plus_hat = spectrum_plus_with_lower_bound(
            moment_hat,
            cfg.k,
            cfg.d,
            cfg.n,
            b_plus=b_plus,
            eps=cfg.eps,
            timelimit=cfg.timelimit,
            solver=cfg.solver,
        )
        lam_recovered = np.maximum(lam_plus_hat - 1.0, 0.0)

        sample_plus = (z_tilde.T @ z_tilde) / cfg.n
        lam_empirical = np.maximum(np.linalg.eigvalsh(sample_plus) - 1.0, 0.0)

        rows.append(
            {
                "run": run,
                "sigma_setting": display_name(setting),
                "k": cfg.k,
                "d": cfg.d,
                "n": cfg.n,
                "num_runs": cfg.num_runs,
                "seed": cfg.seed,
                "setting_seed": setting_seed,
                "tau_min": cfg.tau_min,
                "tau_max": cfg.tau_max,
                "b_plus_multiplier": cfg.b_plus_multiplier,
                "b_plus_oracle": b_plus,
                "L1_empirical": l1_eig_error(lam_empirical, eig_tilde_true),
                "L1_recovered": l1_eig_error(lam_recovered, eig_tilde_true),
                "normalized_L1_empirical": l1_eig_error(lam_empirical, eig_tilde_true) / cfg.d,
                "normalized_L1_recovered": l1_eig_error(lam_recovered, eig_tilde_true) / cfg.d,
                "true_max_eig": float(eig_tilde_true.max()),
                "empirical_max_eig": float(lam_empirical.max()),
                "recovered_max_eig": float(lam_recovered.max()),
            }
        )

        if cfg.plot_scale == "sqrt":
            empirical_curves.append(np.sort(np.sqrt(lam_empirical)))
            recovered_curves.append(np.sort(np.sqrt(lam_recovered)))
        else:
            empirical_curves.append(np.sort(lam_empirical))
            recovered_curves.append(np.sort(lam_recovered))

    runs_df = pd.DataFrame(rows)
    prefix = f"{setting}_whitened_recovery"
    runs_df.to_csv(output_dir / f"{prefix}_runs.csv", index=False, encoding="utf-8-sig")
    plot_curves(
        setting,
        eig_tilde_true,
        empirical_curves,
        recovered_curves,
        output_dir / f"{prefix}_cdf.png",
        cfg.plot_scale,
    )
    plot_curves(
        setting,
        eig_tilde_true,
        empirical_curves,
        recovered_curves,
        output_dir / manuscript_figure_name(setting),
        cfg.plot_scale,
    )
    return runs_df


def write_metadata(cfg: ExperimentConfig, output_dir: Path) -> None:
    scipy_version = None
    gurobi_version = None
    try:
        import scipy

        scipy_version = scipy.__version__
    except Exception:
        pass
    try:
        import gurobipy as gp

        gurobi_version = ".".join(str(part) for part in gp.gurobi.version())
    except Exception:
        pass

    metadata = {
        "config": asdict(cfg),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "matplotlib": matplotlib.__version__,
        "scipy": scipy_version,
        "gurobipy": gurobi_version,
        "command": " ".join(sys.argv),
        "notes": [
            "b_plus_oracle is computed from the known synthetic population covariance.",
            "The manuscript values are generated with Gurobi; SciPy is a fallback backend.",
            "Table errors are eigenvalue-scale even when CDF plots use sqrt scale.",
            "CDF figures show faint individual-trial curves and darker mean curves.",
            "Setting-specific RNG seeds are base seed + 0, + 1, and + 2 for identity, two_spike, and toeplitz.",
        ],
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)


def main() -> None:
    cfg = parse_args()
    if cfg.d < 2:
        raise ValueError("--d must be at least 2.")
    if cfg.n < 1:
        raise ValueError("--n must be positive.")
    if not 1 <= cfg.k <= cfg.n:
        raise ValueError("--k must satisfy 1 <= k <= n.")
    if cfg.num_runs < 1:
        raise ValueError("--num-runs must be positive.")
    if cfg.tau_min <= 0:
        raise ValueError("--tau-min must be positive.")
    if cfg.tau_max < cfg.tau_min:
        raise ValueError("--tau-max must be at least --tau-min.")
    if cfg.eps is not None and cfg.eps <= 0:
        raise ValueError("--eps must be positive when provided.")
    if cfg.timelimit <= 0:
        raise ValueError("--timelimit must be positive.")
    if cfg.b_plus_multiplier < 1.0:
        raise ValueError("--b-plus-multiplier must be at least 1.0.")

    output_dir = Path(cfg.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    all_runs = []
    for setting in cfg.settings:
        all_runs.append(run_setting(setting, cfg, output_dir))

    combined = pd.concat(all_runs, ignore_index=True)
    summary = (
        combined.groupby("sigma_setting", sort=False)
        .agg(
            k=("k", "first"),
            d=("d", "first"),
            n=("n", "first"),
            num_runs=("num_runs", "first"),
            tau_min=("tau_min", "first"),
            tau_max=("tau_max", "first"),
            b_plus_multiplier=("b_plus_multiplier", "first"),
            normalized_L1_empirical_mean=("normalized_L1_empirical", "mean"),
            normalized_L1_empirical_std=("normalized_L1_empirical", lambda x: x.std(ddof=1)),
            normalized_L1_recovered_mean=("normalized_L1_recovered", "mean"),
            normalized_L1_recovered_std=("normalized_L1_recovered", lambda x: x.std(ddof=1)),
        )
        .reset_index()
    )
    summary["reduction_percent"] = 100.0 * (
        1.0
        - summary["normalized_L1_recovered_mean"]
        / summary["normalized_L1_empirical_mean"]
    )
    combined.to_csv(output_dir / "all_runs.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(output_dir / "summary.csv", index=False, encoding="utf-8-sig")
    write_metadata(cfg, output_dir)
    print(summary.to_string(index=False))
    print(f"Results written to: {output_dir}")


if __name__ == "__main__":
    main()
