"""
Step 2 Experiment:
Query-buffer detection + fake label + fake SHAP

This script reproduces the Step 2 evaluation:
- Baseline
- Detection only
- Detection + output distortion (second_best)
- Detection + output distortion (random_other)
"""

import numpy as np
import random
import importlib
import matplotlib.pyplot as plt
import utils

importlib.reload(utils)

# =========================================================
# Step 2 experiment:
# Query-buffer detection + fake label + fake SHAP
# =========================================================

# -----------------------------
# Common experiment config
# -----------------------------
budgets = [20, 40, 60, 80, 100, 150, 200, 300, 500, 750, 1000]
repeats = 5

wd = 5   # Mushroom dataset
wm = 4   # Random Forest
et = 1   # SHAP

sss = [1]
nfe = [3]
hms = 1


def reset_all_states():
    # -----------------------------
    # SIM state
    # -----------------------------
    try:
        st = utils._SIM_STATE
        st["n"] = 0
        st["history"] = []
        st["detected"] = False
        st["blocked"] = False
        st["detect_count"] = 0
        st["detect_qs"] = []
        st["first_detect_q"] = None
        st["last_hits"] = 0
        st["last_best_d"] = None
        if "recent_feat_idx" in st:
            st["recent_feat_idx"] = []
    except Exception:
        pass

    # -----------------------------
    # PRADA state
    # -----------------------------
    try:
        pst = utils._PRADA_STATE
        pst["n"] = 0
        pst["history"] = []
        pst["D"] = []
        pst["detected"] = False
        if "detect_count" in pst:
            pst["detect_count"] = 0
        if "detect_qs" in pst:
            pst["detect_qs"] = []
        if "first_detect_q" in pst:
            pst["first_detect_q"] = None
        if "last_pval" in pst:
            pst["last_pval"] = None
    except Exception:
        pass

    # -----------------------------
    # Query-cluster state
    # -----------------------------
    try:
        qst = utils._QUERY_CLUSTER_STATE
        qst["n"] = 0
        qst["buffer"] = []
        qst["detected"] = False
        qst["detect_count"] = 0
        qst["detect_qs"] = []
        qst["first_detect_q"] = None
        qst["last_hits"] = 0
        qst["last_best_d"] = None
    except Exception:
        pass


def run_setting(
    name,
    use_query_cluster_detector=False,
    flip_label_on_detect=False,
    fake_label_mode="second_best"
):
    # -----------------------------
    # Turn off unrelated defenses
    # -----------------------------
    utils.USE_PRADA = False
    utils.USE_VIEW_D = False
    utils.USE_VIEW_E = False

    # Keep SIM off if you want pure query-buffer detection only
    # Turn it on only if you want both detectors active
    utils.USE_SIM_CHECK = False

    # -----------------------------
    # Step 1 detector settings
    # -----------------------------
    utils.USE_QUERY_CLUSTER_DETECTOR = bool(use_query_cluster_detector)
    utils.QUERY_BUFFER_SIZE = 100
    utils.QUERY_CLUSTER_TAU = 0.35
    utils.QUERY_CLUSTER_MIN_HITS = 2
    utils.QUERY_CLUSTER_DELAY = 10

    # -----------------------------
    # Step 2 output distortion settings
    # -----------------------------
    utils.FLIP_LABEL_ON_DETECT = bool(flip_label_on_detect)
    utils.FAKE_LABEL_MODE = str(fake_label_mode)
    utils.USE_FAKE_SHAP_ON_DETECT = True

    # Turn these OFF so you isolate Step 2
    utils.SIM_CORRUPT_ON_DETECT = False
    utils.REVOKE_EXPLANATION_ON_DETECT = False
    utils.SIM_BLOCK_ON_DETECT = False
    utils.POST_DETECT_RANDOM_EXPANSION = False

    # Optional confidence gating OFF for this test
    utils.USE_CONF_GATING = False

    print(f"\n=== Running: {name} ===")
    print(
        "Settings:",
        f"USE_QUERY_CLUSTER_DETECTOR={utils.USE_QUERY_CLUSTER_DETECTOR}",
        f"QUERY_BUFFER_SIZE={utils.QUERY_BUFFER_SIZE}",
        f"QUERY_CLUSTER_TAU={utils.QUERY_CLUSTER_TAU}",
        f"QUERY_CLUSTER_MIN_HITS={utils.QUERY_CLUSTER_MIN_HITS}",
        f"QUERY_CLUSTER_DELAY={utils.QUERY_CLUSTER_DELAY}",
        f"FLIP_LABEL_ON_DETECT={utils.FLIP_LABEL_ON_DETECT}",
        f"FAKE_LABEL_MODE={utils.FAKE_LABEL_MODE}",
        f"USE_FAKE_SHAP_ON_DETECT={utils.USE_FAKE_SHAP_ON_DETECT}",
    )

    mean_sims, std_sims = [], []
    det_rates, first_detect_qs = [], []

    for q in budgets:
        sims, dets, fdqs = [], [], []

        for r in range(repeats):
            np.random.seed(r)
            random.seed(r)
            reset_all_states()

            accs, sims_mat, _, _ = utils.run_attack_auto(
                wd=wd,
                wm=wm,
                et=et,
                hms=hms,
                sss=sss,
                nfe=nfe,
                ql=[q],
                so=False
            )

            try:
                sim_val = float(sims_mat[0][0][0])
            except Exception:
                sim_val = float(sims_mat[0][0])

            sims.append(sim_val)

            detected_flag = bool(utils._QUERY_CLUSTER_STATE.get("detected", False))
            dets.append(1.0 if detected_flag else 0.0)

            fdq = utils._QUERY_CLUSTER_STATE.get("first_detect_q", None)
            fdqs.append(np.nan if fdq is None else float(fdq))

        mean_sims.append(float(np.mean(sims)))
        std_sims.append(float(np.std(sims)))
        det_rates.append(float(np.mean(dets)))
        first_detect_qs.append(float(np.nanmean(fdqs)) if np.any(~np.isnan(fdqs)) else np.nan)

    return {
        "name": name,
        "mean_sims": mean_sims,
        "std_sims": std_sims,
        "det_rates": det_rates,
        "first_detect_qs": first_detect_qs,
    }


# =========================================================
# Run experiments
# =========================================================

# 1) Baseline: no detector, no fake outputs
baseline = run_setting(
    name="Baseline",
    use_query_cluster_detector=False,
    flip_label_on_detect=False,
    fake_label_mode="second_best"
)

# 2) Step 1 only: detector ON, but no fake outputs yet
step1_only = run_setting(
    name="Step 1 only (detect only)",
    use_query_cluster_detector=True,
    flip_label_on_detect=False,
    fake_label_mode="second_best"
)

# 3) Step 2: detector ON + fake label + fake SHAP
step2_second_best = run_setting(
    name="Step 2 (detect + fake label/SHAP, second_best)",
    use_query_cluster_detector=True,
    flip_label_on_detect=True,
    fake_label_mode="second_best"
)

# 4) Optional stronger variant: random wrong label
step2_random_other = run_setting(
    name="Step 2 (detect + fake label/SHAP, random_other)",
    use_query_cluster_detector=True,
    flip_label_on_detect=True,
    fake_label_mode="random_other"
)


# =========================================================
# Plot 1: Extraction similarity
# =========================================================
plt.figure(figsize=(10, 5))
plt.errorbar(budgets, baseline["mean_sims"], yerr=baseline["std_sims"], marker='o', capsize=4, label=baseline["name"])
plt.errorbar(budgets, step1_only["mean_sims"], yerr=step1_only["std_sims"], marker='o', capsize=4, label=step1_only["name"])
plt.errorbar(budgets, step2_second_best["mean_sims"], yerr=step2_second_best["std_sims"], marker='o', capsize=4, label=step2_second_best["name"])
plt.errorbar(budgets, step2_random_other["mean_sims"], yerr=step2_random_other["std_sims"], marker='o', capsize=4, label=step2_random_other["name"])
plt.ylim(0, 1.05)
plt.title("Step 2: Extraction similarity vs Query budget")
plt.xlabel("Query budget")
plt.ylabel("rtest_sim (higher = attacker succeeds)")
plt.grid(True)
plt.legend()
plt.show()


# =========================================================
# Plot 2: Detection rate
# =========================================================
plt.figure(figsize=(10, 5))
plt.plot(budgets, step1_only["det_rates"], marker='o', label="Step 1 detection rate")
plt.plot(budgets, step2_second_best["det_rates"], marker='o', label="Step 2 second_best detection rate")
plt.plot(budgets, step2_random_other["det_rates"], marker='o', label="Step 2 random_other detection rate")
plt.ylim(0, 1.05)
plt.title("Step 2: Detection rate vs Query budget")
plt.xlabel("Query budget")
plt.ylabel("Detection rate")
plt.grid(True)
plt.legend()
plt.show()


# =========================================================
# Plot 3: Detection delay
# =========================================================
plt.figure(figsize=(10, 5))
plt.plot(budgets, step1_only["first_detect_qs"], marker='o', label="Step 1 first detect q")
plt.plot(budgets, step2_second_best["first_detect_qs"], marker='o', label="Step 2 second_best first detect q")
plt.plot(budgets, step2_random_other["first_detect_qs"], marker='o', label="Step 2 random_other first detect q")
plt.title("Step 2: Detection delay vs Query budget")
plt.xlabel("Query budget")
plt.ylabel("Mean first detect query")
plt.grid(True)
plt.legend()
plt.show()


# =========================================================
# Print summaries
# =========================================================
print("\nBaseline sims:", baseline["mean_sims"])
print("Step 1 sims:", step1_only["mean_sims"])
print("Step 1 detection rate:", step1_only["det_rates"])
print("Step 1 first detect q:", step1_only["first_detect_qs"])

print("\nStep 2 (second_best) sims:", step2_second_best["mean_sims"])
print("Step 2 (second_best) detection rate:", step2_second_best["det_rates"])
print("Step 2 (second_best) first detect q:", step2_second_best["first_detect_qs"])

print("\nStep 2 (random_other) sims:", step2_random_other["mean_sims"])
print("Step 2 (random_other) detection rate:", step2_random_other["det_rates"])
print("Step 2 (random_other) first detect q:", step2_random_other["first_detect_qs"])