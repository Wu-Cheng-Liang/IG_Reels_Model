#!/usr/bin/env python3
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, minimize


# =========================
# USER CONFIG
# =========================
DYNAMIC_CSV = Path("Reels_Data/reels_dynamic_info.csv")
STATIC_CSV = Path("Reels_Data/reels_static_info.csv")
OUTPUT_ROOT = Path("Output")

# "posted after 4/10"
POST_TIME_CUTOFF = pd.Timestamp("2026-04-09 23:59:59")

MAX_REELS = 20
MIN_OBS = 3

# Regularization / prior settings for p and q
# These are just weak anchors, not hard constraints.
P_PRIOR = 0.03
Q_PRIOR = 0.20
LAMBDA_PQ = 2e-3
LAMBDA_M = 1e-4
LEVEL_LOSS_WEIGHT = 0.15

EPS = 1e-12


@dataclass
class ModelFitSummary:
    p: float
    q: float
    M_train: float
    objective_value: float
    level_mape_pct: float
    level_rmse: float
    level_r2: float
    increment_mape_pct: float
    increment_rmse: float
    m_eff_start: float
    m_eff_end: float
    m_eff_median: float


@dataclass
class FitSummary:
    reels_shortcode: str
    kol_account: str | None
    post_time: str | None
    fit_time_zero: str
    used_true_post_time: bool
    first_observation_time: str
    last_observation_time: str
    first_observation_lag_hours: float
    n_obs: int
    duration_hours_since_fit_zero: float
    duration_days_since_fit_zero: float
    start_views_raw: float
    end_views_raw: float
    n_negative_growth_intervals_raw: int
    total_downward_correction_raw: float
    model: ModelFitSummary


def sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    x = np.clip(x, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-x))


def logit(p: float) -> float:
    p = float(np.clip(p, 1e-6, 1.0 - 1e-6))
    return np.log(p / (1.0 - p))


def softplus(x: np.ndarray | float) -> np.ndarray | float:
    x = np.asarray(x)
    out = np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)
    return out.item() if out.ndim == 0 else out


def bass_cumulative(t: np.ndarray, p: float, q: float, M: float) -> np.ndarray:
    """
    Standard closed-form Bass cumulative curve.
    t is in days and p, q are interpreted in the same unit.
    """
    t = np.asarray(t, dtype=float)

    if M <= 0 or p < 0 or q < 0:
        return np.full_like(t, np.nan, dtype=float)

    if p == 0 and q == 0:
        return np.zeros_like(t, dtype=float)

    if q == 0:
        return M * (1.0 - np.exp(-p * t))

    p_safe = max(float(p), EPS)
    e = np.exp(-(p + q) * t)
    return M * (1.0 - e) / (1.0 + (q / p_safe) * e)


def unpack_theta(z: np.ndarray, ymax: float) -> tuple[float, float, float]:
    """
    Map unconstrained latent parameters to constrained model parameters.

    p, q in (0, 1)
    M > ymax
    """
    p = float(np.clip(sigmoid(z[0]), 1e-6, 1.0 - 1e-6))
    q = float(np.clip(sigmoid(z[1]), 1e-6, 1.0 - 1e-6))

    # M = ymax * (1.001 + softplus(z2)) guarantees M > ymax
    extra_mult = float(softplus(z[2]))
    M = float(ymax * (1.001 + extra_mult))
    return p, q, M


def stable_bass_objective(
    z: np.ndarray,
    t: np.ndarray,
    y: np.ndarray,
    p_prior: float = P_PRIOR,
    q_prior: float = Q_PRIOR,
    lambda_pq: float = LAMBDA_PQ,
    lambda_m: float = LAMBDA_M,
    level_weight: float = LEVEL_LOSS_WEIGHT,
) -> float:
    """
    Stable fitting objective.

    Primary fit target:
      - interval growth: diff(y) vs diff(yhat), measured in log1p-space

    Secondary fit target:
      - cumulative level: y vs yhat, low weight

    Regularization:
      - weak penalty keeping p and q near chosen priors
      - very weak penalty discouraging extreme M
    """
    if len(t) != len(y):
        return 1e18

    if len(y) < 3:
        return 1e18

    ymax = float(np.max(y))
    if ymax <= 0:
        return 1e18

    p, q, M = unpack_theta(z, ymax)
    yhat = bass_cumulative(t, p, q, M)

    if np.any(~np.isfinite(yhat)):
        return 1e18

    dt = np.diff(t)
    if np.any(dt <= 0):
        return 1e18

    dy_obs = np.diff(y)
    dy_hat = np.diff(yhat)

    # Defensive clamp
    dy_obs = np.maximum(dy_obs, 0.0)
    dy_hat = np.maximum(dy_hat, 0.0)

    # Main loss on interval growth in log-space
    # This reduces domination by large later counts and stabilizes p/q.
    w = dt / max(float(np.mean(dt)), EPS)
    inc_resid = np.log1p(dy_obs) - np.log1p(dy_hat)
    inc_loss = float(np.mean(w * inc_resid ** 2))

    # Small cumulative-level term
    lvl_resid = np.log1p(np.maximum(y, 0.0)) - np.log1p(np.maximum(yhat, 0.0))
    level_loss = float(np.mean(lvl_resid ** 2))

    # Weak regularization toward reasonable interior values
    z0_prior = logit(p_prior)
    z1_prior = logit(q_prior)
    reg_pq = lambda_pq * ((z[0] - z0_prior) ** 2 + (z[1] - z1_prior) ** 2)
    reg_m = lambda_m * (z[2] ** 2)

    return float(inc_loss + level_weight * level_loss + reg_pq + reg_m)


def fit_stable_bass(
    t: np.ndarray,
    y: np.ndarray,
    p_prior: float = P_PRIOR,
    q_prior: float = Q_PRIOR,
    lambda_pq: float = LAMBDA_PQ,
    lambda_m: float = LAMBDA_M,
    level_weight: float = LEVEL_LOSS_WEIGHT,
) -> tuple[float, float, float, float]:
    """
    Fit stable constrained Bass model.

    Returns:
      p, q, M, objective_value
    """
    if len(t) < 3 or len(y) < 3:
        raise ValueError("Need at least 3 observations to fit.")

    # z0, z1 are logits for p and q
    # z2 controls M multiplier above ymax
    bounds = [
        (-12.0, 12.0),  # p latent
        (-12.0, 12.0),  # q latent
        (-6.0, 12.0),   # M latent
    ]

    de = differential_evolution(
        stable_bass_objective,
        bounds=bounds,
        args=(t, y, p_prior, q_prior, lambda_pq, lambda_m, level_weight),
        seed=42,
        polish=False,
        maxiter=120,
        popsize=20,
        tol=1e-7,
    )

    local = minimize(
        stable_bass_objective,
        x0=de.x,
        args=(t, y, p_prior, q_prior, lambda_pq, lambda_m, level_weight),
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 3000},
    )

    ymax = float(np.max(y))
    p, q, M = unpack_theta(local.x, ymax)
    obj = float(local.fun)
    return p, q, M, obj


def implied_M_from_rate(
    N: np.ndarray, rate: np.ndarray, p: float, q: float
) -> np.ndarray:
    """
    Solve interval-level effective M from:
        rate = (p + q*N/M) * (M - N)

    Rearranged quadratic:
        p M^2 + ((q-p)N - rate) M - q N^2 = 0
    """
    p = max(float(p), EPS)
    q = max(float(q), EPS)

    a = p
    b = (q - p) * N - rate
    c = -q * N * N

    disc = b * b - 4.0 * a * c
    disc = np.maximum(disc, 0.0)

    M = (-b + np.sqrt(disc)) / (2.0 * a)
    M = np.maximum(M, N * (1.0 + 1e-9))
    return M


def implied_M_series(
    t: np.ndarray, y: np.ndarray, p: float, q: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Infer interval-level effective M from observed cumulative views y(t).
    """
    dt = np.diff(t)
    dy = np.diff(y)

    if not np.all(dt > 0):
        raise ValueError("Non-positive time gap found.")

    rate = dy / dt
    N = y[:-1].copy()

    neg_mask = rate < 0
    if np.any(neg_mask):
        rate = rate.copy()
        rate[neg_mask] = 0.0

    M = implied_M_from_rate(N, rate, p, q)
    t_mid = 0.5 * (t[:-1] + t[1:])
    return t_mid, N, M


def smooth_log_series(x: np.ndarray, window: int = 9) -> np.ndarray:
    """
    Median smoothing in log space.
    """
    if len(x) == 0:
        return x
    w = min(int(window), len(x))
    s = pd.Series(np.log(np.maximum(x, 1e-12)))
    sm = s.rolling(window=w, center=True, min_periods=1).median()
    return np.exp(sm.to_numpy())


def level_metrics(y: np.ndarray, yhat: np.ndarray) -> tuple[float, float, float]:
    """
    Level metrics on cumulative views: MAPE (%), RMSE, R^2.
    """
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)

    denom = np.maximum(np.abs(y), 1e-9)
    mape = 100.0 * np.mean(np.abs(y - yhat) / denom)
    rmse = float(np.sqrt(np.mean((y - yhat) ** 2)))
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return float(mape), rmse, float(r2)


def increment_metrics(y: np.ndarray, yhat: np.ndarray) -> tuple[float, float]:
    """
    Increment metrics on interval growth.
    """
    dy = np.diff(np.asarray(y, dtype=float))
    dyhat = np.diff(np.asarray(yhat, dtype=float))

    dy = np.maximum(dy, 0.0)
    dyhat = np.maximum(dyhat, 0.0)

    denom = np.maximum(np.abs(dy), 1e-9)
    mape = 100.0 * np.mean(np.abs(dy - dyhat) / denom)
    rmse = float(np.sqrt(np.mean((dy - dyhat) ** 2)))
    return float(mape), rmse


def build_model_fit_summary(
    p: float,
    q: float,
    M_train: float,
    objective_value: float,
    y: np.ndarray,
    yhat: np.ndarray,
    M_smooth: np.ndarray,
) -> ModelFitSummary:
    lvl_mape, lvl_rmse, lvl_r2 = level_metrics(y, yhat)
    inc_mape, inc_rmse = increment_metrics(y, yhat)

    return ModelFitSummary(
        p=float(p),
        q=float(q),
        M_train=float(M_train),
        objective_value=float(objective_value),
        level_mape_pct=float(lvl_mape),
        level_rmse=float(lvl_rmse),
        level_r2=float(lvl_r2),
        increment_mape_pct=float(inc_mape),
        increment_rmse=float(inc_rmse),
        m_eff_start=float(M_smooth[0]),
        m_eff_end=float(M_smooth[-1]),
        m_eff_median=float(np.median(M_smooth)),
    )


def validate_input_columns(dynamic_df: pd.DataFrame, static_df: pd.DataFrame) -> None:
    dynamic_required = {"reels_shortcode", "views", "timestamp"}
    static_required = {"reels_shortcode", "post_time"}

    dyn_missing = dynamic_required - set(dynamic_df.columns)
    sta_missing = static_required - set(static_df.columns)

    if dyn_missing:
        raise ValueError(f"Missing required dynamic columns: {sorted(dyn_missing)}")
    if sta_missing:
        raise ValueError(f"Missing required static columns: {sorted(sta_missing)}")


def prepare_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load and minimally clean dynamic + static datasets.
    """
    if not DYNAMIC_CSV.exists():
        raise FileNotFoundError(f"Dynamic CSV not found: {DYNAMIC_CSV}")
    if not STATIC_CSV.exists():
        raise FileNotFoundError(f"Static CSV not found: {STATIC_CSV}")

    dynamic_df = pd.read_csv(DYNAMIC_CSV)
    static_df = pd.read_csv(STATIC_CSV)

    validate_input_columns(dynamic_df, static_df)

    if "kol_account" not in static_df.columns:
        static_df["kol_account"] = np.nan

    dynamic_df["views"] = pd.to_numeric(dynamic_df["views"], errors="coerce")
    dynamic_df["timestamp"] = pd.to_datetime(dynamic_df["timestamp"], errors="coerce")
    static_df["post_time"] = pd.to_datetime(static_df["post_time"], errors="coerce")

    dynamic_df = dynamic_df.dropna(subset=["reels_shortcode", "views", "timestamp"]).copy()
    static_df = static_df.dropna(subset=["reels_shortcode", "post_time"]).copy()

    dynamic_df["reels_shortcode"] = dynamic_df["reels_shortcode"].astype(str)
    static_df["reels_shortcode"] = static_df["reels_shortcode"].astype(str)

    return dynamic_df, static_df


def select_reels(dynamic_df: pd.DataFrame, static_df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep reels posted strictly after cutoff and rank by usable count.
    """
    eligible_static = static_df.loc[static_df["post_time"] > POST_TIME_CUTOFF].copy()

    merged = dynamic_df.merge(
        eligible_static[["reels_shortcode", "kol_account", "post_time"]],
        on="reels_shortcode",
        how="inner",
    )

    merged = (
        merged.sort_values(["reels_shortcode", "timestamp"])
        .drop_duplicates(subset=["reels_shortcode", "timestamp"], keep="last")
        .reset_index(drop=True)
    )

    counts = (
        merged.groupby(["reels_shortcode", "kol_account", "post_time"], dropna=False)
        .size()
        .reset_index(name="n_obs")
        .sort_values(["n_obs", "post_time", "reels_shortcode"], ascending=[False, False, True])
        .reset_index(drop=True)
    )

    counts = counts.loc[counts["n_obs"] >= MIN_OBS].copy()
    counts["rank"] = np.arange(1, len(counts) + 1)
    return counts.head(MAX_REELS).copy()


def fit_one_reel(
    g: pd.DataFrame,
    reels_shortcode: str,
    kol_account: str | None,
    post_time: pd.Timestamp | None,
    reel_outdir: Path,
) -> dict:
    """
    Fit one reel and write all outputs.
    """
    g = (
        g.sort_values("timestamp")
        .drop_duplicates(subset=["timestamp"], keep="last")
        .dropna(subset=["timestamp", "views"])
        .reset_index(drop=True)
    )

    if post_time is not None and pd.notna(post_time):
        # Keep only rows at or after official post time
        g = g.loc[g["timestamp"] >= post_time].copy().reset_index(drop=True)

    if len(g) < MIN_OBS:
        raise ValueError(f"Not enough usable observations: {len(g)}")

    first_obs_ts = g["timestamp"].iloc[0]
    last_obs_ts = g["timestamp"].iloc[-1]

    # Prefer true post_time if available and valid
    used_true_post_time = False
    if post_time is not None and pd.notna(post_time) and post_time <= first_obs_ts:
        fit_t0 = post_time
        used_true_post_time = True
        baseline_views = 0.0
    else:
        fit_t0 = first_obs_ts
        baseline_views = float(g["views"].iloc[0])

    t_hours = ((g["timestamp"] - fit_t0).dt.total_seconds() / 3600.0).to_numpy()
    t_days = t_hours / 24.0

    if np.any(t_days < 0):
        raise ValueError("Negative times after fit origin.")

    y_raw = g["views"].astype(float).to_numpy()
    y_mono = np.maximum.accumulate(y_raw)

    n_down = int(np.sum(np.diff(y_raw) < 0))
    down_mag = float(-np.sum(np.minimum(np.diff(y_raw), 0.0)))

    # If true post_time exists, fit on cumulative level directly.
    # Otherwise fit on growth since first observation.
    if used_true_post_time:
        y_fit = y_mono.copy()
    else:
        y_fit = y_mono - baseline_views

    if np.max(y_fit) <= 0:
        raise ValueError("Non-positive fitted target after preprocessing.")

    p, q, M_train, obj = fit_stable_bass(t_days, y_fit)
    yhat_fit = bass_cumulative(t_days, p, q, M_train)
    yhat = yhat_fit + baseline_views

    t_mid_days, N_left_fit, M_raw = implied_M_series(t_days, y_fit, p, q)
    M_smooth = smooth_log_series(M_raw, window=9)

    N_left_display = N_left_fit + baseline_views

    duration_hours = float(t_hours[-1])
    duration_days = float(t_days[-1])

    horizon_days = max(30.0, duration_days)
    t_dense_days = np.linspace(0.0, horizon_days, 1500)
    y_dense_fit = bass_cumulative(t_dense_days, p, q, M_train)
    y_dense = y_dense_fit + baseline_views

    ts_dense = fit_t0 + pd.to_timedelta(t_dense_days, unit="D")
    ts_mid = fit_t0 + pd.to_timedelta(t_mid_days, unit="D")

    summary = FitSummary(
        reels_shortcode=str(reels_shortcode),
        kol_account=None if pd.isna(kol_account) else str(kol_account),
        post_time=None if post_time is None or pd.isna(post_time) else str(post_time),
        fit_time_zero=str(fit_t0),
        used_true_post_time=bool(used_true_post_time),
        first_observation_time=str(first_obs_ts),
        last_observation_time=str(last_obs_ts),
        first_observation_lag_hours=float((first_obs_ts - fit_t0).total_seconds() / 3600.0),
        n_obs=int(len(g)),
        duration_hours_since_fit_zero=duration_hours,
        duration_days_since_fit_zero=duration_days,
        start_views_raw=float(y_mono[0]),
        end_views_raw=float(y_mono[-1]),
        n_negative_growth_intervals_raw=n_down,
        total_downward_correction_raw=down_mag,
        model=build_model_fit_summary(
            p=p,
            q=q,
            M_train=M_train,
            objective_value=obj,
            y=y_mono,
            yhat=yhat,
            M_smooth=M_smooth,
        ),
    )

    reel_outdir.mkdir(parents=True, exist_ok=True)

    g_out = g.copy()
    g_out["fit_time_zero"] = fit_t0
    g_out["t_hours_since_fit_zero"] = t_hours
    g_out["t_days_since_fit_zero"] = t_days
    g_out["views_raw"] = y_raw
    g_out["views_monotone"] = y_mono
    g_out["views_fit_target"] = y_fit
    g_out["fixed_M_bass_fit_views"] = yhat
    g_out.to_csv(reel_outdir / "observed_vs_fit.csv", index=False)

    dense_out = pd.DataFrame(
        {
            "timestamp": ts_dense,
            "t_days_since_fit_zero": t_dense_days,
            "fixed_M_bass_fit_views": y_dense,
        }
    )
    dense_out.to_csv(reel_outdir / "30d_projection.csv", index=False)

    m_eff_out = pd.DataFrame(
        {
            "timestamp_mid": ts_mid,
            "t_mid_days_since_fit_zero": t_mid_days,
            "views_left_display": N_left_display,
            "views_left_fit_scale": N_left_fit,
            "M_eff_raw": M_raw,
            "M_eff_smooth": M_smooth,
        }
    )
    m_eff_out.to_csv(reel_outdir / "effective_M.csv", index=False)

    with open(reel_outdir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(asdict(summary), f, indent=2)

    plt.figure(figsize=(10, 6))
    plt.plot(ts_dense, y_dense, label="Fixed-M Bass projection")
    plt.scatter(g["timestamp"], y_mono, s=18, label="Observed monotone views")
    if post_time is not None and pd.notna(post_time):
        plt.axvline(post_time, linestyle="--", alpha=0.7, label="post_time")
    plt.xlabel("timestamp")
    plt.ylabel("cumulative views")
    plt.title(f"Stable constrained Bass fit: {reels_shortcode}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(reel_outdir / "fixed_M_bass_fit_30d.png", dpi=180)
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(ts_mid, M_raw, alpha=0.25, label="Effective M raw")
    plt.plot(ts_mid, M_smooth, linewidth=2.0, label="Effective M smoothed")
    plt.scatter(ts_mid, N_left_display, s=10, alpha=0.5, label="Observed views (left interval)")
    if post_time is not None and pd.notna(post_time):
        plt.axvline(post_time, linestyle="--", alpha=0.7, label="post_time")
    plt.xlabel("timestamp")
    plt.ylabel("effective M")
    plt.title(f"Effective M from observed views: {reels_shortcode}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(reel_outdir / "effective_M.png", dpi=180)
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(ts_mid, M_raw, alpha=0.2, label="Effective M raw")
    plt.plot(ts_mid, M_smooth, linewidth=2.0, label="Effective M smoothed")
    plt.scatter(ts_mid, N_left_display, s=10, alpha=0.5, label="Observed views (left interval)")
    if post_time is not None and pd.notna(post_time):
        plt.axvline(post_time, linestyle="--", alpha=0.7, label="post_time")
    plt.yscale("log")
    plt.xlabel("timestamp")
    plt.ylabel("effective M (log scale)")
    plt.title(f"Effective M from observed views (log): {reels_shortcode}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(reel_outdir / "effective_M_log.png", dpi=180)
    plt.close()

    summary_row = {
        "reels_shortcode": summary.reels_shortcode,
        "kol_account": summary.kol_account,
        "post_time": summary.post_time,
        "fit_time_zero": summary.fit_time_zero,
        "used_true_post_time": summary.used_true_post_time,
        "first_observation_time": summary.first_observation_time,
        "last_observation_time": summary.last_observation_time,
        "first_observation_lag_hours": summary.first_observation_lag_hours,
        "n_obs": summary.n_obs,
        "duration_hours_since_fit_zero": summary.duration_hours_since_fit_zero,
        "duration_days_since_fit_zero": summary.duration_days_since_fit_zero,
        "start_views_raw": summary.start_views_raw,
        "end_views_raw": summary.end_views_raw,
        "n_negative_growth_intervals_raw": summary.n_negative_growth_intervals_raw,
        "total_downward_correction_raw": summary.total_downward_correction_raw,
        "p": summary.model.p,
        "q": summary.model.q,
        "M_train": summary.model.M_train,
        "objective_value": summary.model.objective_value,
        "level_mape_pct": summary.model.level_mape_pct,
        "level_rmse": summary.model.level_rmse,
        "level_r2": summary.model.level_r2,
        "increment_mape_pct": summary.model.increment_mape_pct,
        "increment_rmse": summary.model.increment_rmse,
        "m_eff_start": summary.model.m_eff_start,
        "m_eff_end": summary.model.m_eff_end,
        "m_eff_median": summary.model.m_eff_median,
        "output_dir": str(reel_outdir),
    }

    return summary_row


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    dynamic_df, static_df = prepare_data()
    selected = select_reels(dynamic_df, static_df)

    if selected.empty:
        raise ValueError("No eligible reels found after filtering.")

    selected.to_csv(OUTPUT_ROOT / "selected_reels.csv", index=False)

    print(f"Eligible reels selected for training: {len(selected)}")
    print(selected[["rank", "reels_shortcode", "kol_account", "post_time", "n_obs"]].to_string(index=False))

    merged = dynamic_df.merge(
        selected[["reels_shortcode", "kol_account", "post_time", "n_obs"]],
        on="reels_shortcode",
        how="inner",
    )

    training_rows = []
    error_rows = []

    for _, row in selected.iterrows():
        shortcode = str(row["reels_shortcode"])
        kol_account = row["kol_account"] if "kol_account" in row else None
        post_time = row["post_time"]

        reel_df = merged.loc[merged["reels_shortcode"] == shortcode].copy()
        reel_outdir = OUTPUT_ROOT / shortcode

        print(f"\n[TRAIN] {shortcode} | obs={len(reel_df)} | post_time={post_time}")

        try:
            summary_row = fit_one_reel(
                g=reel_df,
                reels_shortcode=shortcode,
                kol_account=kol_account,
                post_time=post_time,
                reel_outdir=reel_outdir,
            )
            training_rows.append(summary_row)
            print(
                f"[OK] {shortcode} | p={summary_row['p']:.4f} "
                f"| q={summary_row['q']:.4f} | M={summary_row['M_train']:.2f}"
            )

        except Exception as e:
            error_rows.append(
                {
                    "reels_shortcode": shortcode,
                    "kol_account": kol_account,
                    "post_time": post_time,
                    "error": str(e),
                }
            )
            print(f"[ERROR] {shortcode}: {e}")

    summary_df = pd.DataFrame(training_rows)
    errors_df = pd.DataFrame(error_rows)

    if not summary_df.empty:
        summary_df = summary_df.sort_values("n_obs", ascending=False).reset_index(drop=True)
        summary_df.to_csv(OUTPUT_ROOT / "training_summary.csv", index=False)

    if not errors_df.empty:
        errors_df.to_csv(OUTPUT_ROOT / "training_errors.csv", index=False)

    print("\nDone.")
    print(f"Output root: {OUTPUT_ROOT.resolve()}")
    print(f"Trained reels: {len(summary_df)}")
    print(f"Errors: {len(errors_df)}")


if __name__ == "__main__":
    main()
