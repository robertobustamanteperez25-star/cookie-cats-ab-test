import sys
from pathlib import Path

import numpy as np
import pytest
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize, proportions_ztest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import abtest  # noqa: E402


def test_srm_balanced_split_passes():
    assert abtest.srm_check(50_000, 50_000)["p_value"] == pytest.approx(1.0)


def test_srm_detects_imbalance():
    assert abtest.srm_check(49_000, 51_000)["p_value"] < 1e-5


def test_ztest_matches_statsmodels():
    res = abtest.two_proportion_ztest(900, 5000, 1000, 5000)
    z, p = proportions_ztest([1000, 900], [5000, 5000])
    assert res["z"] == pytest.approx(z)
    assert res["p_value"] == pytest.approx(p)
    assert res["ci_low"] < res["abs_diff"] < res["ci_high"]


def test_holm_is_monotone_and_stricter():
    out = abtest.holm_correction({"a": 0.01, "b": 0.04, "c": 0.03})
    assert out["a"]["p_adjusted"] == pytest.approx(0.03)
    assert out["c"]["p_adjusted"] == pytest.approx(0.06)
    assert out["b"]["p_adjusted"] >= out["c"]["p_adjusted"]
    assert out["a"]["significant"] and not out["c"]["significant"]


def test_bootstrap_ci_contains_true_difference():
    rng = np.random.default_rng(0)
    c, t = rng.normal(10, 2, 4000), rng.normal(10.5, 2, 4000)
    res = abtest.bootstrap_diff(c, t, n_boot=1000)
    assert res["ci_low"] < 0.5 < res["ci_high"]


def test_bayesian_prefers_clearly_better_variant():
    res = abtest.bayesian_beta_binomial(400, 4000, 520, 4000)
    assert res["prob_treatment_better"] > 0.99
    assert res["expected_loss_ship_treatment"] < res["expected_loss_ship_control"]


def test_sample_size_close_to_statsmodels():
    n = abtest.sample_size_per_group(0.20, 0.01)
    ref = NormalIndPower().solve_power(proportion_effectsize(0.21, 0.20), alpha=0.05, power=0.8)
    assert abs(n - ref) / ref < 0.02


def test_mde_and_sample_size_are_consistent():
    mde = abtest.minimum_detectable_effect(0.19, 45_000)
    assert abtest.sample_size_per_group(0.19, mde) == pytest.approx(45_000, rel=0.05)
