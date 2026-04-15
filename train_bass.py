#!/usr/bin/env python3
"""
Batch fixed-M Bass training for multiple Instagram Reels.

Inputs
------
Dynamic:
  /Users/jimwu/Desktop/Bass_Model/Reels Data/reels_dynamic_info.csv
  Expected columns:
    reels_shortcode,views,plays,likes,comments,timestamp

Static:
  /Users/jimwu/Desktop/Bass_Model/Reels Data/reels_static_info.csv
  Expected columns:
    kol_account,reels_shortcode,post_time,duration,caption

Selection rules
---------------
- Skip rows with missing view data.
- Keep reels with static post_time on/after 2026-04-10.
- Rank by usable data points after cleaning.
- Train at most 20 reels.

Outputs
-------
For each reel:
  /Users/jimwu/Desktop/Bass_Model/Output/<reels_shortcode>/
    - observed_vs_fit.csv
    - 30d_projection.csv
    - effective_M.csv
    - summary.json
    - fixed_M_bass_fit_30d.png
    - effective_M.png
    - effective_M_log.png

Global outputs:
  /Users/jimwu/Desktop/Bass_Model/Output/
    - selected_reels.csv
    - training_summary.csv
    - training_errors.csv
"""

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
DYNAMIC_CSV = Path("/Users/jimwu/Desktop/Bass_Model/Reels Data/reels_dynamic_info.csv")
STATIC_CSV = Path("/Users/jimwu/Desktop/Bass_Model/Reels Data/reels_static_info.csv")
OUTPUT_ROOT = Path("/Users/jimwu/Desktop/Bass_Model/Output")

# Interpreted as posted on/after 2026-04-10.
# If you want strictly after 4/10, change ">=" logic below to ">".
POST_DATE_CUTOFF = pd.Timestamp("2026-04-10 00:00:00")

MAX_REELS = 20
MIN_OBS = 3  # minimum usable points after cleaning


@dataclass
class ModelFitSummary:
    objective: str
    p: float
    q: float
    M_train: float
    mape_pct: float
    rmse: float
    r2: float
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
    mape_model: ModelFitSummary
    rmse_model: ModelFitSummary


def bass_cumulative(t: np.ndarray, p: float, q: float, M: float) -> np.ndarray:
    """
    Standard closed-form Bass cumulative curve.
    """
    t = np.asarray(t, dtype=float)
    eps = 1e-12

    if M <= 0 or p < 0 or q < 0:
        return np.full_like(t, np.nan, dtype=float)

    if p == 0 and q == 0:
        return np.zeros_like(t, dtype=float)

    if q == 0:
        return M * (1.0 - np.exp(-p * t))

    p_safe = max(float(p), eps)
    e = np.exp(-(p + q) * t)
    return M * (1.0 - e) / (1.0 + (q / p_safe) * e)


def fixed_bass_objective(
    theta: np.ndarray, t: np.ndarray, y: np.ndarray, objective: str = "mape"
) -> float:
    """
    Objective for fixed-M Bass fit.
    """
    p, q, M = theta

    if p <= 0 or q <= 0 or p > 5 or q > 5:
        return 1e12
    if M <= np.max(y):
        return 1e12

    yhat = bass_cumulative(t, p, q, M)
    if np.any(~np.isfinite(yhat)):
        return 1e12

    if objective == "mape":
        eps = 1e-9
        return float(np.mean(np.abs(y - yhat) / np.maximum(np.abs(y), eps)))
    elif objective == "rmse":
        return float(np.sqrt(np.mean((y - yhat) ** 2)))
    else:
        raise ValueError(f"Unsupported objective: {objective}")


def fit_fixed_bass(
    t: np.ndarray, y: np.ndarray, objective: str = "mape"
) -> tuple[float, float, float]:
    """
    Fit scalar p, q, M by global + local optimization.
    """
    ymax = float(np.max(y))

    bounds = [
        (1e-8, 5.0),                 # p
        (1e-8, 5.0),                 # q
        (ymax * 1.001, ymax * 100.0) # M
    ]

    de = differential_evolution(
        fixed_bass_objective,
        bounds=bounds,
        args=(t, y, objective),
        seed=42,
        polish=False,
        maxiter=100,
        popsize=20,
        tol=1e-7,
    )

    local = minimize(
        fixed_bass_objective,
        x0=de.x,
        args=(t, y, objective),
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 3000},
    )

    p, q, M = local.x
    return float(p), float(q), float(M)


def implied_M_from_rate(
    N: np.ndarray, rate: np.ndarray, p: float, q: float
) -> np.ndarray:
    """
    Solve interval-level effective M from:
        rate = (p + q*N/M) * (M - N)

    Rearranged quadratic:
        p M^2 + ((q-p)N - rate) M - q N^2 = 0
    """
    eps = 1e-12
    p = max(float(p), eps)
    q = max(float(q), eps)

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


def metrics(y: np.ndarray, yhat: np.ndarray) -> tuple[float, float, float]:
    """
    Return MAPE (%), RMSE, R^2.
    """
    eps = 1e-9
    mape = 100.0 * np.mean(np.abs(y - yhat) / np.maximum(np.abs(y), eps))
    rmse = float(np.sqrt(np.mean((y - yhat) ** 2)))
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return float(mape), rmse, float(r2)


def build_model_fit_summary(
    objective: str,
    p: float,
    q: float,
    M_train: float,
    y: np.ndarray,
    yhat: np.ndarray,
    M_smooth: np.ndarray,
) -> ModelFitSummary:
    """
    Package fit metrics + effective M summary.
    """
    mape, rmse, r2 = metrics(y, yhat)
    return ModelFitSummary(
        objective=objective,
        p=p,
        q=q,
        M_train=M_train,
        mape_pct=mape,
        rmse=rmse,
        r2=r2,
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

    dynamic_df["views"] = pd.to_numeric(dynamic_df["views"], errors="coerce")
    dynamic_df["timestamp"] = pd.to_datetime(dynamic_df["timestamp"], errors="coerce")

    static_df["post_time"] = pd.to_datetime(static_df["post_time"], errors="coerce")

    # Skip rows without view data.
    dynamic_df = dynamic_df.dropna(subset=["views", "timestamp", "reels_shortcode"]).copy()
    static_df = static_df.dropna(subset=["reels_shortcode", "post_time"]).copy()

    dynamic_df["reels_shortcode"] = dynamic_df["reels_shortcode"].astype(str)
    static_df["reels_shortcode"] = static_df["reels_shortcode"].astype(str)

    return dynamic_df, static_df


def select_reels(dynamic_df: pd.DataFrame, static_df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter reels by post date and rank by usable data points.
    """
    eligible_static = static_df.loc[static_df["post_time"] >= POST_DATE_CUTOFF].copy()

    merged = dynamic_df.merge(
        eligible_static[["reels_shortcode", "kol_account", "post_time"]],
        on="reels_shortcode",
        how="inner",
    )

    # Deduplicate per reel/timestamp after merge.
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

    selected = counts.head(MAX_REELS).copy()
    return selected


def fit_one_reel(
    g: pd.DataFrame,
    reels_shortcode: str,
    kol_account: str | None,
    post_time: pd.Timestamp | None,
    reel_outdir: Path,
) -> dict:
    """
    Train one reel and write all outputs into its folder.
    Returns a flat summary dict for the global master summary.
    """
    g = (
        g.sort_values("timestamp")
         .drop_duplicates(subset=["timestamp"], keep="last")
         .dropna(subset=["timestamp", "views"])
         .reset_index(drop=True)
    )

    if len(g) < MIN_OBS:
        raise ValueError(f"Not enough usable observations: {len(g)}")

    first_obs_ts = g["timestamp"].iloc[0]
    last_obs_ts = g["timestamp"].iloc[-1]

    # Prefer true post_time if it exists and is not after first observation.
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
        raise ValueError("Negative times after fit_t0 anchor.")

    y_raw = g["views"].astype(float).to_numpy()
    y_mono = np.maximum.accumulate(y_raw)

    n_down = int(np.sum(np.diff(y_raw) < 0))
    down_mag = float(-np.sum(np.minimum(np.diff(y_raw), 0.0)))

    # Fit on true cumulative if true post_time exists.
    # Otherwise fit on growth since first observed snapshot.
    if used_true_post_time:
        y_fit = y_mono.copy()
    else:
        y_fit = y_mono - baseline_views

    if np.max(y_fit) <= 0:
        raise ValueError("Non-positive fitted target after preprocessing.")

    p_mape, q_mape, M_train_mape = fit_fixed_bass(t_days, y_fit, objective="mape")
    yhat_fit_mape = bass_cumulative(t_days, p_mape, q_mape, M_train_mape)

    p_rmse, q_rmse, M_train_rmse = fit_fixed_bass(t_days, y_fit, objective="rmse")
    yhat_fit_rmse = bass_cumulative(t_days, p_rmse, q_rmse, M_train_rmse)

    # Add baseline back only if fallback mode was used.
    yhat_mape = yhat_fit_mape + baseline_views
    yhat_rmse = yhat_fit_rmse + baseline_views

    # Effective M on the fitted scale.
    t_mid_days, N_left_fit, M_raw_mape = implied_M_series(t_days, y_fit, p_mape, q_mape)
    M_smooth_mape = smooth_log_series(M_raw_mape, window=9)

    _, _, M_raw_rmse = implied_M_series(t_days, y_fit, p_rmse, q_rmse)
    M_smooth_rmse = smooth_log_series(M_raw_rmse, window=9)

    N_left_display = N_left_fit + baseline_views

    duration_hours = float(t_hours[-1])
    duration_days = float(t_days[-1])

    horizon_days = max(30.0, duration_days)
    t_dense_days = np.linspace(0.0, horizon_days, 1500)

    y_dense_fit_mape = bass_cumulative(t_dense_days, p_mape, q_mape, M_train_mape)
    y_dense_fit_rmse = bass_cumulative(t_dense_days, p_rmse, q_rmse, M_train_rmse)

    y_dense_mape = y_dense_fit_mape + baseline_views
    y_dense_rmse = y_dense_fit_rmse + baseline_views

    ts_dense = fit_t0 + pd.to_timedelta(t_dense_days, unit="D")
    ts_mid = fit_t0 + pd.to_timedelta(t_mid_days, unit="D")

    summary = FitSummary(
        reels_shortcode=str(reels_shortcode),
        kol_account=None if pd.isna(kol_account) else str(kol_account),
        post_time=None if post_time is None or pd.isna(post_time) else str(post_time),
        fit_time_zero=str(fit_t0),
        used_true_post_time=used_true_post_time,
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
        mape_model=build_model_fit_summary(
            objective="mape",
            p=p_mape,
            q=q_mape,
            M_train=M_train_mape,
            y=y_mono,
            yhat=yhat_mape,
            M_smooth=M_smooth_mape,
        ),
        rmse_model=build_model_fit_summary(
            objective="rmse",
            p=p_rmse,
            q=q_rmse,
            M_train=M_train_rmse,
            y=y_mono,
            yhat=yhat_rmse,
            M_smooth=M_smooth_rmse,
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
    g_out["fixed_M_bass_fit_views_mape"] = yhat_mape
    g_out["fixed_M_bass_fit_views_rmse"] = yhat_rmse
    g_out.to_csv(reel_outdir / "observed_vs_fit.csv", index=False)

    dense_out = pd.DataFrame(
        {
            "timestamp": ts_dense,
            "t_days_since_fit_zero": t_dense_days,
            "fixed_M_bass_fit_views_mape": y_dense_mape,
            "fixed_M_bass_fit_views_rmse": y_dense_rmse,
        }
    )
    dense_out.to_csv(reel_outdir / "30d_projection.csv", index=False)

    m_eff_out = pd.DataFrame(
        {
            "timestamp_mid": ts_mid,
            "t_mid_days_since_fit_zero": t_mid_days,
            "views_left_display": N_left_display,
            "views_left_fit_scale": N_left_fit,
            "M_eff_raw_mape": M_raw_mape,
            "M_eff_smooth_mape": M_smooth_mape,
            "M_eff_raw_rmse": M_raw_rmse,
            "M_eff_smooth_rmse": M_smooth_rmse,
        }
    )
    m_eff_out.to_csv(reel_outdir / "effective_M.csv", index=False)

    with open(reel_outdir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(asdict(summary), f, indent=2)

    plt.figure(figsize=(10, 6))
    plt.plot(ts_dense, y_dense_mape, label="Fixed-M Bass projection (MAPE)")
    plt.plot(ts_dense, y_dense_rmse, label="Fixed-M Bass projection (RMSE)")
    plt.scatter(g["timestamp"], y_mono, s=18, label="Observed monotone views")
    if post_time is not None and pd.notna(post_time):
        plt.axvline(post_time, linestyle="--", alpha=0.7, label="post_time")
    plt.xlabel("timestamp")
    plt.ylabel("cumulative views")
    plt.title(f"Fixed-M Bass fit: {reels_shortcode}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(reel_outdir / "fixed_M_bass_fit_30d.png", dpi=180)
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(ts_mid, M_raw_mape, alpha=0.25, label="Effective M raw (MAPE)")
    plt.plot(ts_mid, M_smooth_mape, linewidth=2.0, label="Effective M smoothed (MAPE)")
    plt.plot(ts_mid, M_raw_rmse, alpha=0.25, label="Effective M raw (RMSE)")
    plt.plot(ts_mid, M_smooth_rmse, linewidth=2.0, label="Effective M smoothed (RMSE)")
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
    plt.plot(ts_mid, M_raw_mape, alpha=0.2, label="Effective M raw (MAPE)")
    plt.plot(ts_mid, M_smooth_mape, linewidth=2.0, label="Effective M smoothed (MAPE)")
    plt.plot(ts_mid, M_raw_rmse, alpha=0.2, label="Effective M raw (RMSE)")
    plt.plot(ts_mid, M_smooth_rmse, linewidth=2.0, label="Effective M smoothed (RMSE)")
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

    # Flat row for global summary CSV
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
        "mape_p": summary.mape_model.p,
        "mape_q": summary.mape_model.q,
        "mape_M_train": summary.mape_model.M_train,
        "mape_pct": summary.mape_model.mape_pct,
        "mape_rmse": summary.mape_model.rmse,
        "mape_r2": summary.mape_model.r2,
        "mape_m_eff_start": summary.mape_model.m_eff_start,
        "mape_m_eff_end": summary.mape_model.m_eff_end,
        "mape_m_eff_median": summary.mape_model.m_eff_median,
        "rmse_p": summary.rmse_model.p,
        "rmse_q": summary.rmse_model.q,
        "rmse_M_train": summary.rmse_model.M_train,
        "rmse_pct": summary.rmse_model.mape_pct,
        "rmse_rmse": summary.rmse_model.rmse,
        "rmse_r2": summary.rmse_model.r2,
        "rmse_m_eff_start": summary.rmse_model.m_eff_start,
        "rmse_m_eff_end": summary.rmse_model.m_eff_end,
        "rmse_m_eff_median": summary.rmse_model.m_eff_median,
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