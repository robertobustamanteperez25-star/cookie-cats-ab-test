"""Small, tested toolkit for analysing A/B tests with binary and continuous metrics.

Every function returns plain dicts/floats so results are easy to log, test and put in a report.
"""
from __future__ import annotations

import numpy as np
from scipy import stats


# ---------------------------------------------------------------- health checks
def srm_check(n_control: int, n_treatment: int, expected_ratio: float = 0.5) -> dict:
    """Sample Ratio Mismatch: chi-square test of the observed split vs the planned split.

    A p-value below ~0.01 means the randomisation or logging is probably broken and
    the results must be read with caution.
    """
    total = n_control + n_treatment
    expected = [total * (1 - expected_ratio), total * expected_ratio]
    chi2, p = stats.chisquare([n_control, n_treatment], f_exp=expected)
    return {"n_control": n_control, "n_treatment": n_treatment,
            "share_treatment": n_treatment / total, "chi2": float(chi2), "p_value": float(p)}


# ---------------------------------------------------------------- frequentist
def two_proportion_ztest(x_c: int, n_c: int, x_t: int, n_t: int, alpha: float = 0.05) -> dict:
    """Two-sided z-test for the difference in conversion rates (treatment - control).

    Uses the pooled standard error for the test statistic and the unpooled one for the
    confidence interval, which is the textbook (Wald) approach.
    """
    p_c, p_t = x_c / n_c, x_t / n_t
    diff = p_t - p_c
    p_pool = (x_c + x_t) / (n_c + n_t)
    se_pool = np.sqrt(p_pool * (1 - p_pool) * (1 / n_c + 1 / n_t))
    z = diff / se_pool
    p_value = 2 * stats.norm.sf(abs(z))
    se = np.sqrt(p_c * (1 - p_c) / n_c + p_t * (1 - p_t) / n_t)
    z_crit = stats.norm.ppf(1 - alpha / 2)
    return {"rate_control": p_c, "rate_treatment": p_t, "abs_diff": diff,
            "rel_diff": diff / p_c, "z": float(z), "p_value": float(p_value),
            "ci_low": diff - z_crit * se, "ci_high": diff + z_crit * se}


def holm_correction(p_values: dict[str, float], alpha: float = 0.05) -> dict[str, dict]:
    """Holm-Bonferroni step-down correction for several primary metrics."""
    ordered = sorted(p_values.items(), key=lambda kv: kv[1])
    m, out, still_rejecting = len(ordered), {}, True
    for i, (name, p) in enumerate(ordered):
        adj = min(1.0, p * (m - i))
        still_rejecting = still_rejecting and p <= alpha / (m - i)
        out[name] = {"p_value": p, "p_adjusted": adj, "significant": still_rejecting}
    # enforce monotone adjusted p-values
    running = 0.0
    for name, _ in ordered:
        running = max(running, out[name]["p_adjusted"])
        out[name]["p_adjusted"] = running
    return out


# ---------------------------------------------------------------- resampling / Bayesian
def bootstrap_diff(control: np.ndarray, treatment: np.ndarray, stat=np.mean,
                   n_boot: int = 5000, seed: int = 42) -> dict:
    """Percentile bootstrap CI for stat(treatment) - stat(control)."""
    rng = np.random.default_rng(seed)
    control, treatment = np.asarray(control), np.asarray(treatment)
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        diffs[b] = (stat(rng.choice(treatment, treatment.size)) -
                    stat(rng.choice(control, control.size)))
    return {"diff": float(stat(treatment) - stat(control)),
            "ci_low": float(np.percentile(diffs, 2.5)),
            "ci_high": float(np.percentile(diffs, 97.5)),
            "prob_negative": float((diffs < 0).mean()), "samples": diffs}


def bayesian_beta_binomial(x_c: int, n_c: int, x_t: int, n_t: int,
                           draws: int = 200_000, seed: int = 42) -> dict:
    """Beta(1,1) prior -> posterior for each rate; probability treatment beats control
    and expected loss (in absolute rate) of shipping each variant."""
    rng = np.random.default_rng(seed)
    pc = rng.beta(1 + x_c, 1 + n_c - x_c, draws)
    pt = rng.beta(1 + x_t, 1 + n_t - x_t, draws)
    return {"prob_treatment_better": float((pt > pc).mean()),
            "expected_loss_ship_treatment": float(np.maximum(pc - pt, 0).mean()),
            "expected_loss_ship_control": float(np.maximum(pt - pc, 0).mean()),
            "lift_ci_low": float(np.percentile(pt - pc, 2.5)),
            "lift_ci_high": float(np.percentile(pt - pc, 97.5))}


# ---------------------------------------------------------------- design
def sample_size_per_group(baseline: float, mde_abs: float, alpha: float = 0.05,
                          power: float = 0.8) -> int:
    """Users needed per group to detect an absolute change `mde_abs` (two-sided test)."""
    p1, p2 = baseline, baseline + mde_abs
    z_a, z_b = stats.norm.ppf(1 - alpha / 2), stats.norm.ppf(power)
    p_bar = (p1 + p2) / 2
    num = (z_a * np.sqrt(2 * p_bar * (1 - p_bar)) +
           z_b * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2
    return int(np.ceil(num / mde_abs ** 2))


def minimum_detectable_effect(baseline: float, n_per_group: int, alpha: float = 0.05,
                              power: float = 0.8) -> float:
    """Smallest absolute change detectable with `n_per_group` users per arm (approximation)."""
    z_a, z_b = stats.norm.ppf(1 - alpha / 2), stats.norm.ppf(power)
    return float((z_a + z_b) * np.sqrt(2 * baseline * (1 - baseline) / n_per_group))
