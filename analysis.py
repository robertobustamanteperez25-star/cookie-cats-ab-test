"""Runs the full analysis: health checks, tests, sensitivity, charts and results/results.json."""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
import abtest  # noqa: E402

IMG, RES = ROOT / "images", ROOT / "results"
IMG.mkdir(exist_ok=True)
RES.mkdir(exist_ok=True)
C_COL, T_COL, GREY = "#2563eb", "#ea580c", "#6b7280"
plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titleweight": "bold", "axes.titlesize": 12, "font.size": 10})

df = pd.read_csv(ROOT / "data/raw/cookie_cats.csv")
assert df.userid.is_unique and df.notna().all().all()
ctrl, trt = df[df.version == "gate_30"], df[df.version == "gate_40"]
out = {"n_users": len(df)}

# 1. health check -------------------------------------------------------------
out["srm"] = abtest.srm_check(len(ctrl), len(trt))

# 2. primary metrics ------------------------------------------------------------
tests, pvals = {}, {}
for m in ["retention_1", "retention_7"]:
    args = (int(ctrl[m].sum()), len(ctrl), int(trt[m].sum()), len(trt))
    tests[m] = abtest.two_proportion_ztest(*args)
    tests[m]["bayes"] = abtest.bayesian_beta_binomial(*args)
    pvals[m] = tests[m]["p_value"]
out["tests"] = tests
out["holm"] = abtest.holm_correction(pvals)

# 3. guardrail: engagement (heavy-tailed -> winsorise + bootstrap) ---------------
cap = df.sum_gamerounds.quantile(0.999)
bs = abtest.bootstrap_diff(ctrl.sum_gamerounds.clip(upper=cap).values,
                           trt.sum_gamerounds.clip(upper=cap).values, n_boot=3000)
out["rounds"] = {k: v for k, v in bs.items() if k != "samples"} | {"cap_p999": float(cap),
                 "max_raw": int(df.sum_gamerounds.max())}

# 4. SRM sensitivity: worst case = the extra treatment users were low-engagement churners
excess = len(trt) - len(ctrl)
low = trt[(trt.sum_gamerounds < 30) & (~trt.retention_7)]
trt_wc = trt.drop(low.sample(excess, random_state=1).index)
out["srm_worst_case_r7"] = abtest.two_proportion_ztest(
    int(ctrl.retention_7.sum()), len(ctrl), int(trt_wc.retention_7.sum()), len(trt_wc))
out["srm_excess_users"] = int(excess)
out["excess_below_30_rounds"] = int((trt.sum_gamerounds < 30).sum() - (ctrl.sum_gamerounds < 30).sum())

# 5. design: power of this test and size of a confirmation test -----------------
base7 = tests["retention_7"]["rate_control"]
out["design"] = {"mde_r7_this_test": abtest.minimum_detectable_effect(base7, len(ctrl)),
                 "n_per_group_for_0_5pp": abtest.sample_size_per_group(base7, -0.005)}

json.dump(out, open(RES / "results.json", "w"), indent=2, default=float)

# ---------------------------------------------------------------- charts
# A. retention with 95% CI
fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
for ax, m, title in zip(axes, ["retention_1", "retention_7"], ["1-day retention", "7-day retention"]):
    rates, errs = [], []
    for g in (ctrl, trt):
        p = g[m].mean()
        rates.append(p * 100)
        errs.append(1.96 * np.sqrt(p * (1 - p) / len(g)) * 100)
    for i, (r, e, col) in enumerate(zip(rates, errs, [C_COL, T_COL])):
        ax.errorbar(i, r, yerr=e, fmt="o", color=col, capsize=8, ms=10, lw=2.5)
        ax.text(i + 0.12, r, f"{r:.1f}%", va="center", fontweight="bold", color=col)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Gate at level 30\n(control)", "Gate at level 40\n(treatment)"])
    ax.set_xlim(-0.5, 1.6)
    ax.set_ylim(min(rates) - 1.2, max(rates) + 1.2)
    ax.grid(axis="y", alpha=0.3)
    t = tests[m]
    ax.set_title(f"{title}: {t['abs_diff']*100:+.2f} pp (p = {t['p_value']:.3f})")
    ax.set_ylabel("% of players (95% CI)")
fig.suptitle("Moving the gate to level 40 lowers retention", fontweight="bold", y=1.02)
fig.tight_layout()
fig.savefig(IMG / "retention.png", bbox_inches="tight")
plt.close(fig)

# B. posterior of the 7-day difference
rng = np.random.default_rng(42)
xc, nc, xt, nt = int(ctrl.retention_7.sum()), len(ctrl), int(trt.retention_7.sum()), len(trt)
diff = (rng.beta(1 + xt, 1 + nt - xt, 200_000) - rng.beta(1 + xc, 1 + nc - xc, 200_000)) * 100
fig, ax = plt.subplots(figsize=(8, 3.4))
ax.hist(diff, bins=120, color=T_COL, alpha=0.8)
ax.axvline(0, color="black", lw=1)
ax.set_xlabel("7-day retention, gate 40 minus gate 30 (percentage points)")
ax.set_yticks([])
ax.set_title(f"Posterior: P(gate 40 is better) = {(diff > 0).mean()*100:.2f}%")
fig.tight_layout()
fig.savefig(IMG / "posterior_r7.png", bbox_inches="tight")
plt.close(fig)

# C. forest plot: robustness of the 7-day effect
rows = [("Main analysis", tests["retention_7"]),
        ("SRM worst case\n(extra users removed)", out["srm_worst_case_r7"])]
fig, ax = plt.subplots(figsize=(8, 2.6))
for i, (lab, r) in enumerate(rows):
    y = len(rows) - 1 - i
    ax.errorbar(r["abs_diff"] * 100, y, xerr=[[(r["abs_diff"] - r["ci_low"]) * 100],
                [(r["ci_high"] - r["abs_diff"]) * 100]], fmt="o", color=[T_COL, GREY][i],
                capsize=5, ms=8, lw=2)
    ax.text(r["ci_high"] * 100 + 0.08, y, f"{r['abs_diff']*100:+.2f} pp, p = {r['p_value']:.3f}",
            va="center")
ax.axvline(0, color="black", lw=1, ls="--")
ax.set_yticks(range(len(rows)))
ax.set_yticklabels([r[0] for r in rows][::-1])
ax.set_xlim(-1.6, 0.9)
ax.set_ylim(-0.6, 1.6)
ax.set_xlabel("Change in 7-day retention (pp, 95% CI)")
ax.set_title("Direction holds, but the sample ratio mismatch weakens the evidence")
fig.tight_layout()
fig.savefig(IMG / "robustness.png", bbox_inches="tight")
plt.close(fig)

# D. power curve
mdes = np.linspace(0.003, 0.02, 60)
ns = [abtest.sample_size_per_group(base7, -m) for m in mdes]
fig, ax = plt.subplots(figsize=(8, 3.4))
ax.plot(mdes * 100, np.array(ns) / 1000, color=C_COL, lw=2)
ax.axhline(len(ctrl) / 1000, color=GREY, ls="--")
ax.text(1.6, len(ctrl) / 1000 + 8, f"this test: {len(ctrl)/1000:.0f}K per group", color=GREY)
ax.scatter([0.5], [out["design"]["n_per_group_for_0_5pp"] / 1000], color=T_COL, zorder=3)
ax.annotate(f"confirm a 0.5 pp drop:\n{out['design']['n_per_group_for_0_5pp']/1000:.0f}K per group",
            (0.5, out["design"]["n_per_group_for_0_5pp"] / 1000), xytext=(0.75, 160),
            arrowprops=dict(arrowstyle="->", color=T_COL), color=T_COL)
ax.set_xlabel("Minimum detectable effect on 7-day retention (pp)")
ax.set_ylabel("Players per group (K)")
ax.set_ylim(0, 250)
ax.set_title("Sample size needed (alpha 0.05, power 80%)")
fig.tight_layout()
fig.savefig(IMG / "power_curve.png", bbox_inches="tight")
plt.close(fig)

print(json.dumps({k: out[k] for k in ["srm", "design", "srm_excess_users", "excess_below_30_rounds"]},
                 indent=1, default=float))
