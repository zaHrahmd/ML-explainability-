import lime
import lime.lime_tabular
import numpy as np
import pandas as pd
import random
import seaborn as sns
import shap
import sklearn
import warnings
import pickle
from scipy.stats import shapiro

from sklearn.tree import DecisionTreeClassifier
from matplotlib import pyplot as plt
import sklearn.tree as tree
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier as dt
from sklearn.linear_model import LogisticRegression as lr
from sklearn.naive_bayes import MultinomialNB as mnb
from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn.ensemble import RandomForestClassifier as rf
from sklearn.neural_network import MLPClassifier as mlp
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import os
import sys
from collections import defaultdict

# =========================================================
# GLOBAL CONFIGURATION
# Single source of truth: no duplicated / conflicting flags
# =========================================================

# ------------------------------
# PRADA
# ------------------------------
USE_PRADA = False
PRADA_WINDOW = 200
PRADA_DELAY = 10
PRADA_ALPHA = 0.05
PRADA_MIN_N = 30
PRADA_HISTORY = 400
PRADA_ACTION = "stealth_corrupt"  # "zero" | "shuffle" | "demote_promote" | "stealth_corrupt"

# ------------------------------
# SIM detector
# ------------------------------
USE_SIM_CHECK = False
SIM_HISTORY = 200
SIM_K = 5
SIM_TAU = 1.5
SIM_MIN_HITS = 1
SIM_DELAY_QUERIES = 12

# classical post-detect actions
SIM_BLOCK_ON_DETECT = False
SIM_CORRUPT_ON_DETECT = False
REVOKE_EXPLANATION_ON_DETECT = False

# traversal degradation after detect
POST_DETECT_RANDOM_EXPANSION = False
POST_DETECT_K = 1
POST_DETECT_AVOID_RECENT = False

# corruption parameters
SIM_CORRUPT_K = 8
SIM_CORRUPT_NOISE = 0.03
SIM_CORRUPT_PRESERVE_SUM = True
SIM_CORRUPT_KEEP_TOP1 = False
SIM_CORRUPT_FLIP_PROB = 0.35

# ------------------------------
# Step 1: query-buffer detector
# ------------------------------
USE_QUERY_CLUSTER_DETECTOR = False
QUERY_BUFFER_SIZE = 100
QUERY_CLUSTER_TAU = 0.35
QUERY_CLUSTER_MIN_HITS = 2
QUERY_CLUSTER_DELAY = 10

# ------------------------------
# Step 2: output distortion
# ------------------------------
FLIP_LABEL_ON_DETECT = False
FAKE_LABEL_MODE = "second_best"      # "second_best" | "random_other"
USE_FAKE_SHAP_ON_DETECT = False      # use SHAP(fake class)
RANDOMIZE_SHAP_ON_DETECT = False     # stronger: random explanation after detect

# ------------------------------
# confidence gating
# ------------------------------
USE_CONF_GATING = False
CONF_TAU = 0.4
CONF_FLIP_PROB = 0.5

# ------------------------------
# Optional other SHAP defenses
# ------------------------------
USE_VIEW_D = False
USE_VIEW_E = False

VIEWD_MID_M = 8
VIEWD_ALPHA = 0.25
VIEWD_SIGMA = 0.05
VIEWD_PRESERVE_SUM = True

VIEWE_TOPK = 5
VIEWE_CONC_TAU = 0.35
VIEWE_COVER_T = 3
VIEWE_MIN_QUERIES = 20
VIEWE_REVERSE_PROB = 1.0
VIEWE_DISRUPT_K = 8
VIEWE_MODE = "shuffle"
VIEWE_DELAY_QUERIES = 50
VIEWE_TOP_DEMOTE = 7
VIEWE_MID_PROMOTE = 7
VIEWE_DEMOTE_FACTOR = 0.1
VIEWE_PROMOTE_FACTOR = 3.0
VIEWE_LAMBDA = 1.0
VIEWE_CLIP_ALPHA = 1.0
VIEWE_PRESERVE_SUM = True
VIEWE_KEEP_TOP1_FIXED = False

# ------------------------------
# SHAP speed / budget
# ------------------------------
KERNEL_SHAP_NSAMPLES = 80

# SIM-defense params used by apply_similarity_defense
SIM_DEFENSE_MODE = "shuffle"
SIM_TOP_DEMOTE = 5
SIM_MID_PROMOTE = 5
SIM_DEMOTE_FACTOR = 0.2
SIM_PROMOTE_FACTOR = 2.5

# =========================================================
# RUNTIME STATES
# =========================================================
_PRADA_STATE = {
    "history": [],
    "D": [],
    "detected": False,
    "n": 0,
    "detect_count": 0,
    "first_detect_q": None,
    "detect_qs": [],
    "last_pval": None,
}

_SIM_STATE = {
    "n": 0,
    "history": [],
    "detected": False,
    "detect_count": 0,
    "detect_qs": [],
    "first_detect_q": None,
    "blocked": False,
    "last_hits": 0,
    "last_best_d": None,
    "recent_feat_idx": [],
}

_QUERY_CLUSTER_STATE = {
    "n": 0,
    "buffer": [],
    "detected": False,
    "detect_count": 0,
    "detect_qs": [],
    "first_detect_q": None,
    "last_hits": 0,
    "last_best_d": None,
}

_VIEWE_STATE = {
    "n_queries": 0,
    "patterns_seen": set(),
    "class_centroids": defaultdict(lambda: None),
    "class_counts": defaultdict(int),
}

# =========================================================
# RESET HELPERS
# =========================================================
def reset_prada_state():
    _PRADA_STATE["n"] = 0
    _PRADA_STATE["history"].clear()
    _PRADA_STATE["D"].clear()
    _PRADA_STATE["detected"] = False
    _PRADA_STATE["detect_count"] = 0
    _PRADA_STATE["first_detect_q"] = None
    _PRADA_STATE["detect_qs"].clear()
    _PRADA_STATE["last_pval"] = None


def reset_sim_state():
    _SIM_STATE["n"] = 0
    _SIM_STATE["history"].clear()
    _SIM_STATE["detected"] = False
    _SIM_STATE["detect_count"] = 0
    _SIM_STATE["detect_qs"].clear()
    _SIM_STATE["first_detect_q"] = None
    _SIM_STATE["blocked"] = False
    _SIM_STATE["last_hits"] = 0
    _SIM_STATE["last_best_d"] = None
    _SIM_STATE["recent_feat_idx"] = []


def reset_query_cluster_state():
    _QUERY_CLUSTER_STATE["n"] = 0
    _QUERY_CLUSTER_STATE["buffer"] = []
    _QUERY_CLUSTER_STATE["detected"] = False
    _QUERY_CLUSTER_STATE["detect_count"] = 0
    _QUERY_CLUSTER_STATE["detect_qs"] = []
    _QUERY_CLUSTER_STATE["first_detect_q"] = None
    _QUERY_CLUSTER_STATE["last_hits"] = 0
    _QUERY_CLUSTER_STATE["last_best_d"] = None

# =========================================================
# PRADA
# =========================================================
def _prada_dist_eps_l1(curr, prev, idx, eps):
    d = 0.0
    for j in idx:
        denom = float(eps[j]) + 1e-12
        d += abs(float(curr[j]) - float(prev[j])) / denom
    return float(d)


def prada_update(curr, detect_idx, epsilon_set):
    if not globals().get("USE_PRADA", False):
        return False

    st = _PRADA_STATE
    st["n"] += 1
    q = st["n"]
    curr = np.asarray(curr, dtype=float)

    hist = st["history"]
    if len(hist) > 0:
        dmin = min(_prada_dist_eps_l1(curr, prev, detect_idx, epsilon_set) for prev in hist)
        st["D"].append(float(dmin))
        W = int(globals().get("PRADA_WINDOW", 200))
        st["D"] = st["D"][-W:]

    H = int(globals().get("PRADA_HISTORY", 400))
    hist.append(curr.copy())
    st["history"] = hist[-H:]

    if q <= int(globals().get("PRADA_DELAY", 10)):
        return False

    if len(st["D"]) < int(globals().get("PRADA_MIN_N", 30)):
        return False

    if st["detected"]:
        return True

    alpha = float(globals().get("PRADA_ALPHA", 0.05))
    try:
        pval = shapiro(np.array(st["D"], dtype=float))[1]
        st["last_pval"] = float(pval)
        if pval < alpha:
            st["detected"] = True
            st["detect_count"] += 1
            st["detect_qs"].append(int(q))
            if st["first_detect_q"] is None:
                st["first_detect_q"] = int(q)
            return True
    except Exception:
        D = np.array(st["D"], dtype=float)
        mu = float(np.mean(D))
        sd = float(np.std(D))
        if mu > 1e-12 and sd < 0.10 * mu:
            st["detected"] = True
            st["detect_count"] += 1
            st["detect_qs"].append(int(q))
            if st["first_detect_q"] is None:
                st["first_detect_q"] = int(q)
            return True

    return False


def prada_apply_action(exp):
    mode = str(globals().get("PRADA_ACTION", "stealth_corrupt")).lower()
    exp = np.asarray(exp, dtype=float).copy()
    n = len(exp)

    if n == 0:
        return exp
    if mode == "zero":
        return np.zeros_like(exp)
    if mode == "shuffle":
        idx = np.argsort(-np.abs(exp))
        k = max(2, min(int(globals().get("SIM_K", 5)), n))
        topk = idx[:k]
        vals = exp[topk].copy()
        np.random.shuffle(vals)
        exp[topk] = vals
        return exp
    if mode == "demote_promote":
        try:
            return apply_similarity_defense(exp)
        except Exception:
            return np.zeros_like(exp)
    if mode == "stealth_corrupt":
        mu = float(exp.mean())
        sigma = float(exp.std()) + 1e-12
        idx_sorted = np.argsort(-np.abs(exp))
        k = min(5, n)
        top = idx_sorted[:k]
        mid = idx_sorted[k:2 * k] if 2 * k <= n else idx_sorted[k:]
        exp2 = exp.copy()
        if len(mid) > 0 and len(top) > 0:
            take = min(len(top), len(mid))
            perm_top = np.random.permutation(top)[:take]
            perm_mid = np.random.permutation(mid)[:take]
            tmp = exp2[perm_top].copy()
            exp2[perm_top] = exp2[perm_mid]
            exp2[perm_mid] = tmp
        z = (exp2 - mu) / sigma
        z = z + np.random.normal(0, 0.05, size=z.shape)
        return z * sigma + mu
    return exp

# =========================================================
# SIM / QUERY CLUSTER DISTANCES + DETECTORS
# =========================================================
def _sim_dist_eps(curr, prev, idx, eps):
    d = 0.0
    for j in idx:
        denom = float(eps[j]) + 1e-12
        d += abs(float(curr[j]) - float(prev[j])) / denom
    return float(d)


def _query_dist_eps_all(curr, prev, eps):
    curr = np.asarray(curr, dtype=float).reshape(-1)
    prev = np.asarray(prev, dtype=float).reshape(-1)
    eps = np.asarray(eps, dtype=float).reshape(-1)
    m = min(len(curr), len(prev), len(eps))
    if m == 0:
        return 0.0
    diffs = np.abs(curr[:m] - prev[:m]) / (eps[:m] + 1e-12)
    return float(np.mean(diffs))


def is_suspicious_query_cluster(curr, epsilon_set):
    if not globals().get("USE_QUERY_CLUSTER_DETECTOR", False):
        return (False, 0, None)

    st = _QUERY_CLUSTER_STATE
    st["n"] += 1
    q = st["n"]
    curr = np.asarray(curr, dtype=float).reshape(-1)

    if q <= int(globals().get("QUERY_CLUSTER_DELAY", 10)):
        st["buffer"].append(curr.copy())
        st["buffer"] = st["buffer"][-int(globals().get("QUERY_BUFFER_SIZE", 100)):]
        return (False, 0, None)

    buf = st["buffer"]
    if len(buf) == 0:
        st["buffer"].append(curr.copy())
        return (False, 0, None)

    hits = 0
    best_d = 1e18
    tau = float(globals().get("QUERY_CLUSTER_TAU", 0.35))

    for prev in buf:
        d = _query_dist_eps_all(curr, prev, epsilon_set)
        best_d = min(best_d, d)
        if d <= tau:
            hits += 1

    st["buffer"].append(curr.copy())
    st["buffer"] = st["buffer"][-int(globals().get("QUERY_BUFFER_SIZE", 100)):]
    st["last_hits"] = int(hits)
    st["last_best_d"] = float(best_d) if best_d is not None else None

    trigger = hits >= int(globals().get("QUERY_CLUSTER_MIN_HITS", 2))
    if trigger and not st["detected"]:
        st["detected"] = True
        st["detect_count"] += 1
        st["detect_qs"].append(int(q))
        if st["first_detect_q"] is None:
            st["first_detect_q"] = int(q)

    return (trigger, hits, best_d)


def is_autolycus_like(curr, top_idx, epsilon_set):
    if not USE_SIM_CHECK:
        return (False, 0, None)

    _SIM_STATE["n"] += 1
    q = _SIM_STATE["n"]

    if q <= SIM_DELAY_QUERIES:
        _SIM_STATE["history"].append(np.array(curr, dtype=float).copy())
        _SIM_STATE["history"] = _SIM_STATE["history"][-SIM_HISTORY:]
        _SIM_STATE["last_hits"] = 0
        _SIM_STATE["last_best_d"] = None
        return (False, 0, None)

    hist = _SIM_STATE["history"]
    if len(hist) == 0:
        hist.append(np.array(curr, dtype=float).copy())
        _SIM_STATE["last_hits"] = 0
        _SIM_STATE["last_best_d"] = None
        return (False, 0, None)

    hits = 0
    best_d = 1e18
    for prev in hist:
        d = _sim_dist_eps(curr, prev, top_idx, epsilon_set)
        best_d = min(best_d, d)
        if d <= SIM_TAU * len(top_idx):
            hits += 1

    hist.append(np.array(curr, dtype=float).copy())
    _SIM_STATE["history"] = hist[-SIM_HISTORY:]
    _SIM_STATE["last_hits"] = int(hits)
    _SIM_STATE["last_best_d"] = float(best_d) if best_d is not None else None

    trigger = hits >= SIM_MIN_HITS
    return (trigger, hits, best_d)

# =========================================================
# EXPLANATION / SHAP MANIPULATION HELPERS
# =========================================================
def apply_similarity_defense(exp):
    s = np.asarray(exp, dtype=float).copy()
    n = len(s)
    if n < 3:
        return s

    idx = np.argsort(-np.abs(s))
    base_sum = float(np.sum(s))
    mode = str(globals().get("SIM_DEFENSE_MODE", "demote_promote")).lower()

    if mode == "zero":
        return np.zeros_like(s)

    if mode == "shuffle":
        k = max(2, min(int(globals().get("SIM_K", 5)), n))
        topk = idx[:k]
        vals = s[topk].copy()
        np.random.shuffle(vals)
        s[topk] = vals
        return s

    k_top = max(1, min(int(globals().get("SIM_TOP_DEMOTE", 5)), n))
    k_mid = max(1, min(int(globals().get("SIM_MID_PROMOTE", 5)), max(0, n - k_top)))
    top = idx[:k_top]
    mid = idx[k_top:k_top + k_mid]

    dem = float(globals().get("SIM_DEMOTE_FACTOR", 0.2))
    pro = float(globals().get("SIM_PROMOTE_FACTOR", 2.5))

    s[top] *= dem
    if len(mid) > 0:
        s[mid] *= pro

    diff = float(np.sum(s) - base_sum)
    tail = idx[::-1]
    if len(tail) > 0:
        s[int(tail[0])] -= diff
    return s


def sim_corrupt_explanation(exp, n_f_e=None):
    s = np.asarray(exp, dtype=float).copy()
    n = len(s)
    if n < 4:
        return s

    base_sum = float(np.sum(s))
    idx_desc = np.argsort(-np.abs(s))

    k = int(globals().get("SIM_CORRUPT_K", 8))
    if n_f_e is not None:
        k = min(k, int(n_f_e))
    k = max(2, min(k, n // 2))

    m = min(2 * k, n)
    pool = idx_desc[:m]
    vals = s[pool].copy()
    np.random.shuffle(vals)

    flip_p = float(globals().get("SIM_CORRUPT_FLIP_PROB", 0.35))
    if flip_p > 0:
        flips = (np.random.rand(len(vals)) < flip_p)
        vals[flips] *= -1.0

    s[pool] = vals

    sigma = float(globals().get("SIM_CORRUPT_NOISE", 0.03))
    if sigma > 0:
        s = s + np.random.normal(0, sigma * (np.std(s) + 1e-12), size=s.shape)

    if globals().get("SIM_CORRUPT_PRESERVE_SUM", True):
        diff = float(np.sum(s) - base_sum)
        tail = [int(j) for j in idx_desc[::-1] if int(j) not in set(pool)]
        if tail:
            s[tail[0]] -= diff
        else:
            s[int(idx_desc[-1])] -= diff

    return s


def _shap_concentration_score(s):
    s = np.asarray(s, dtype=float)
    mass = np.sum(np.abs(s)) + 1e-12
    top = np.max(np.abs(s))
    return float(top / mass)


def _topk_pattern(s, k):
    s = np.asarray(s, dtype=float)
    idx = np.argsort(-np.abs(s))
    k = min(k, len(s))
    return tuple(idx[:k].tolist())


def _update_centroid(pred_class, s):
    s = np.asarray(s, dtype=float)
    mu = _VIEWE_STATE["class_centroids"][pred_class]
    cnt = _VIEWE_STATE["class_counts"][pred_class]

    if mu is None:
        mu = s.copy()
        cnt = 1
    else:
        cnt += 1
        mu = mu + (s - mu) / cnt

    _VIEWE_STATE["class_centroids"][pred_class] = mu
    _VIEWE_STATE["class_counts"][pred_class] = cnt


def _should_reverse_view_e(s, pred_class):
    conc = _shap_concentration_score(s)
    pattern = _topk_pattern(s, VIEWE_TOPK)

    if conc >= VIEWE_CONC_TAU:
        _VIEWE_STATE["patterns_seen"].add(pattern)

    coverage = len(_VIEWE_STATE["patterns_seen"])
    trigger = (coverage >= VIEWE_COVER_T)

    if _VIEWE_STATE["n_queries"] % 50 == 0:
        print("q=", _VIEWE_STATE["n_queries"],
              "conc=", round(conc, 3),
              "coverage=", coverage,
              "trigger=", trigger)
    return trigger


def view_e_adaptive_reverse_shap(s, pred_class=0):
    s = np.asarray(s, dtype=float).copy()
    n = len(s)
    if n < 3:
        return s

    base_sum = float(np.sum(s))
    idx = np.argsort(-np.abs(s))
    top_idx = int(idx[0])
    top_val = float(s[top_idx])

    _update_centroid(pred_class, s)
    _VIEWE_STATE["n_queries"] = _VIEWE_STATE.get("n_queries", 0) + 1
    q = int(_VIEWE_STATE["n_queries"])

    delay = int(globals().get("VIEWE_DELAY_QUERIES", 50))
    if q <= delay:
        return s

    do_defend = _should_reverse_view_e(s, pred_class)
    if not do_defend:
        return s

    p = float(globals().get("VIEWE_REVERSE_PROB", 1.0))
    if np.random.rand() >= p:
        return s

    k_top = int(globals().get("VIEWE_TOP_DEMOTE", 7))
    k_mid = int(globals().get("VIEWE_MID_PROMOTE", 7))
    demote_factor = float(globals().get("VIEWE_DEMOTE_FACTOR", 0.1))
    promote_factor = float(globals().get("VIEWE_PROMOTE_FACTOR", 3.0))

    k_top = max(1, min(k_top, n))
    k_mid = max(1, min(k_mid, max(0, n - k_top)))
    top = idx[:k_top]
    mid = idx[k_top:k_top + k_mid]

    s_tilde = s.copy()
    s_tilde[top] *= demote_factor
    if len(mid) > 0:
        s_tilde[mid] *= promote_factor

    lam = float(globals().get("VIEWE_LAMBDA", 1.0))
    lam = max(0.0, min(1.0, lam))
    s_tilde = (1.0 - lam) * s + lam * s_tilde

    if globals().get("VIEWE_KEEP_TOP1_FIXED", False):
        s_tilde[top_idx] = top_val

    alpha = float(globals().get("VIEWE_CLIP_ALPHA", 1.0))
    bounds = alpha * (np.abs(s) + 1e-12)
    delta = np.clip(s_tilde - s, -bounds, bounds)
    s_tilde = s + delta

    if globals().get("VIEWE_PRESERVE_SUM", True):
        diff = float(np.sum(s_tilde) - base_sum)
        tail_candidates = [int(j) for j in idx[::-1] if int(j) != top_idx]
        if tail_candidates:
            s_tilde[tail_candidates[0]] -= diff

    return s_tilde


def view_d_stability_limited(s):
    s = np.array(s, dtype=float).copy()
    if len(s) < 3:
        return s
    idx = np.argsort(-np.abs(s))
    top_idx = idx[0]
    base_sum = np.sum(s)
    top_value = s[top_idx]
    mid_m = min(VIEWD_MID_M, len(s) - 1)
    mid_idx = idx[1:1 + mid_m]
    if len(mid_idx) == 0:
        return s
    sigma = VIEWD_SIGMA * abs(top_value)
    noise = np.random.normal(0, sigma, size=len(mid_idx))
    noise -= np.mean(noise)
    alpha = VIEWD_ALPHA
    bounds = alpha * np.abs(s[mid_idx])
    noise = np.clip(noise, -bounds, bounds)
    s[mid_idx] += noise
    s[top_idx] = top_value
    if VIEWD_PRESERVE_SUM:
        diff = np.sum(s) - base_sum
        tail_idx = [i for i in range(len(s)) if i not in mid_idx and i != top_idx]
        if len(tail_idx) > 0:
            s[tail_idx[0]] -= diff
    return s

# =========================================================
# GENERAL UTILITIES
# =========================================================
class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


np.set_printoptions(suppress=True)
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)


def takeFourth(elem):
    return abs(elem[3])


def explanation_parser(expMap, expList, key, features):
    result = []
    for i in features:
        result = result + [[i, -1, -1, 0]]
    indices = []
    tmp = 0
    for i in expMap[key]:
        indices = indices + [i[0]]
    for i in expList:
        txt = i[0].split(' ')
        if len(txt) < 4:
            if (txt[-2] == '<=') or (txt[-2] == '<'):
                result[indices[tmp]][2] = float(txt[-1])
            else:
                result[indices[tmp]][1] = float(txt[-1])
        else:
            result[indices[tmp]][1] = float(txt[0])
            result[indices[tmp]][2] = float(txt[-1])
        result[indices[tmp]][3] = round(float(i[1]), 2)
        tmp = tmp + 1
    result.sort(key=takeFourth, reverse=True)
    return result


def extract_explanation_boundaries(model, explainer, n_ft):
    boundaries = np.zeros((n_ft, 3))
    sample = np.zeros(n_ft)
    for i in range(3):
        exp = explainer.explain_instance(sample, model.predict_proba, top_labels=1)
        exp_map = exp.as_map()
        key = list(exp_map.keys())[0]
        exp_list = exp.as_list(key)
        exp_parsed = explanation_parser(exp_map, exp_list, key, features)
        for j in range(n_ft):
            tmp_exp = exp_parsed[j]
            feature_index = [index for index, content in enumerate(features) if tmp_exp[0] in content][0]
            boundaries[feature_index, i] = tmp_exp[2]
            sample[feature_index] = tmp_exp[2] + 1
    return boundaries


def sample_set_generation(dataset, n_classes, n_samples_per_class):
    sample_set = []
    if isinstance(n_samples_per_class, list):
        lister = [int(x) for x in n_samples_per_class]
    else:
        lister = (np.ones((n_classes,), dtype=int) * int(n_samples_per_class)).tolist()

    for i in range(len(lister)):
        tmp = np.where(dataset[:, -1] == i)
        pop = tmp[0].tolist()
        if len(pop) == 0:
            continue
        need = int(lister[i])
        need = min(need, len(pop))
        if need <= 0:
            continue
        indices = random.sample(pop, need)
        for j in indices:
            sample_set += [dataset[j][:-1]]
    return sample_set


def _safe_shap_1d(explainer, curr, model, class_index=None, nsamples=None):
    if nsamples is None:
        nsamples = KERNEL_SHAP_NSAMPLES

    try:
        if hasattr(explainer, "expected_value") and explainer.__class__.__name__ == "KernelExplainer":
            shap_vals = explainer.shap_values(curr, nsamples=nsamples)
        else:
            shap_vals = explainer.shap_values(curr)
    except TypeError:
        shap_vals = explainer.shap_values(curr)

    if isinstance(shap_vals, list):
        if class_index is None:
            class_index = 0
        class_index = int(class_index)
        class_index = max(0, min(class_index, len(shap_vals) - 1))
        shap_vals = shap_vals[class_index]

    exp = np.asarray(shap_vals, dtype=float)
    exp = np.squeeze(exp)
    if exp.ndim == 2:
        exp = exp[0]
    return exp

# =========================================================
# TRAVERSAL
# =========================================================
def traverse_explanations_LIME(sample_set, explainer, model, n_visits_lb, n_visits_ub, upper_limit, n_f_e, args2):
    classes, features, n_classes, n_features, isCat, epsilon_set, canNegative, classPossibilities, dataset_name = args2
    if isinstance(n_visits_lb, int):
        n_visits_lb = np.ones(len(classes)) * n_visits_lb
        n_visits_ub = np.ones(len(classes)) * n_visits_ub
    n_visits = np.zeros(len(classes))
    samples = sample_set.copy()
    init_preds = model.predict_proba(samples)
    preds = []
    visited_samples = []
    k = n_f_e
    for i in init_preds:
        preds.append(np.argmax(i))
    for i in samples:
        visited_samples += [i]
    query = 1
    epsilon = 1
    isPassed = [n_visits[i] >= n_visits_lb[i] for i in range(len(n_visits_lb))]
    while len(samples) != 0 and not all(isPassed) and not query > upper_limit:
        curr = samples.pop(0)
        pred = model.predict_proba([curr])[0]
        pmax = float(np.max(pred))
        class_index = int(np.argmax(pred))

        if globals().get("USE_CONF_GATING", False) and pmax < float(globals().get("CONF_TAU", 0.0)):
            continue
        if n_visits[class_index] < n_visits_ub[class_index]:
            query += 1
            n_visits[class_index] += 1
            preds += [classes[class_index]]
            visited_samples += [curr]
            exp = explainer.explain_instance(curr, model.predict_proba)
            exp_map = exp.as_map()
            key = list(exp_map.keys())[0]
            exp_list = exp.as_list(key)
            exp_parsed = explanation_parser(exp_map, exp_list, key, features)

            tmp_exps, indices, cpys = [], [], []
            for i in range(k):
                tmp_exps += [exp_parsed[i]]
            for i in range(k):
                indices += [index for index, content in enumerate(features) if tmp_exps[i][0] in content]
            for i in range(2 * k):
                cpys += [np.copy(curr)]
            for i in range(k):
                cpys[2 * i][indices[i]] = tmp_exps[i][2]
                cpys[2 * i + 1][indices[i]] = tmp_exps[i][1]
            for i in range(2 * k):
                ind_i = int(i / 2)
                if (cpys[i][indices[ind_i]] >= 0):
                    if i % 2 == 0:
                        cpys[i][indices[ind_i]] += epsilon
                    else:
                        cpys[i][indices[ind_i]] -= epsilon
                tmp = (any((cpys[i] == x).all() for x in visited_samples) or
                       (any((cpys[i] == x).all() for x in samples)) or
                       (cpys[i][indices[ind_i]] < 0) or
                       (cpys[i][indices[ind_i]] >= classPossibilities[indices[ind_i]]))
                if not tmp:
                    samples += [cpys[i]]
    return visited_samples, preds, query


def traverse_explanations_SHAP(sample_set, explainer, model,
                               n_visits_lb, n_visits_ub,
                               upper_limit, n_f_e, args2, model_name):
    classes, features, n_classes, n_features, isCat, epsilon_set, canNegative, classPossibilities, dataset_name = args2

    if isinstance(n_visits_lb, int):
        n_v_lb = np.ones(len(classes)) * n_visits_lb
        n_v_ub = np.ones(len(classes)) * n_visits_ub
    else:
        n_v_lb = np.array(n_visits_lb, dtype=float)
        n_v_ub = np.array(n_visits_ub, dtype=float)

    n_visits = np.zeros(len(classes))
    samples = sample_set.copy()

    init_preds = model.predict_proba(samples)
    preds = [int(np.argmax(p)) for p in init_preds]

    visited_samples = []
    for s in samples:
        visited_samples.append(np.asarray(s, dtype=float).reshape(-1))

    query = 1
    isPassed = [n_visits[i] >= n_v_lb[i] for i in range(len(n_v_lb))]

    while len(samples) != 0 and not all(isPassed) and not query > upper_limit:
        query += 1
        curr = samples.pop(0)
        curr_np = np.asarray(curr, dtype=float).reshape(-1)
        feature_dim = len(curr_np)

        # Step 1: query-buffer detector
        cluster_triggered = False
        cluster_hits = 0
        cluster_best_d = None
        if globals().get("USE_QUERY_CLUSTER_DETECTOR", False):
            try:
                cluster_triggered, cluster_hits, cluster_best_d = is_suspicious_query_cluster(curr_np, epsilon_set)
            except Exception:
                cluster_triggered, cluster_hits, cluster_best_d = (False, 0, None)

        # prediction
        try:
            probs = model.predict_proba([curr_np])[0]
            real_class_index = int(np.argmax(probs))
            class_index = real_class_index
            pmax = float(np.max(probs))
        except Exception:
            real_class_index = int(model.predict([curr_np])[0])
            class_index = real_class_index
            probs = None
            pmax = 1.0

        if globals().get("USE_CONF_GATING", False) and pmax < float(globals().get("CONF_TAU", 0.0)):
            continue

        if query % 100 == 0:
            print(int(query / 100), end=" ")

        # SIM detection on REAL explanation only
        if globals().get("USE_SIM_CHECK", False):
            try:
                exp_detect = _safe_shap_1d(explainer, np.array([curr_np]), model, class_index=real_class_index)
                exp_detect = np.asarray(exp_detect, dtype=float).reshape(-1)
                k_detect = int(globals().get("SIM_K", 5))
                k_detect = max(2, min(k_detect, len(exp_detect)))
                top_idx = np.flip(np.argsort(np.abs(exp_detect)))[:k_detect]
                sim_triggered, hits, best_d = is_autolycus_like(curr_np, top_idx, epsilon_set)
                _SIM_STATE["last_hits"] = int(hits)
                _SIM_STATE["last_best_d"] = None if best_d is None else float(best_d)
                if sim_triggered and not _SIM_STATE["detected"]:
                    _SIM_STATE["detected"] = True
                    _SIM_STATE["detect_count"] += 1
                    _SIM_STATE["detect_qs"].append(int(query))
                    if _SIM_STATE["first_detect_q"] is None:
                        _SIM_STATE["first_detect_q"] = int(query)
            except Exception:
                pass

        # Query-cluster detector propagates into same detection flag
        if globals().get("USE_QUERY_CLUSTER_DETECTOR", False) and cluster_triggered:
            if not _QUERY_CLUSTER_STATE["detected"]:
                _QUERY_CLUSTER_STATE["detected"] = True
                _QUERY_CLUSTER_STATE["detect_count"] += 1
                _QUERY_CLUSTER_STATE["detect_qs"].append(int(query))
                if _QUERY_CLUSTER_STATE["first_detect_q"] is None:
                    _QUERY_CLUSTER_STATE["first_detect_q"] = int(query)

            if not _SIM_STATE["detected"]:
                _SIM_STATE["detected"] = True
                _SIM_STATE["detect_count"] += 1
                _SIM_STATE["detect_qs"].append(int(query))
                if _SIM_STATE["first_detect_q"] is None:
                    _SIM_STATE["first_detect_q"] = int(query)

        detected_now = bool(_SIM_STATE.get("detected", False))

        # Step 2: fake label after detection
        if detected_now and probs is not None and globals().get("FLIP_LABEL_ON_DETECT", False):
            mode = str(globals().get("FAKE_LABEL_MODE", "second_best")).lower()
            if len(probs) >= 2:
                if mode == "second_best":
                    ranked = np.argsort(probs)
                    class_index = int(ranked[-2])
                elif mode == "random_other":
                    other = [i for i in range(len(probs)) if i != real_class_index]
                    if len(other) > 0:
                        class_index = int(np.random.choice(other))

        if n_visits[class_index] < n_v_ub[class_index]:
            n_visits[class_index] += 1
            preds.append(classes[class_index])
            visited_samples.append(curr_np)

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")

                try:
                    if detected_now and globals().get("RANDOMIZE_SHAP_ON_DETECT", False):
                        exp = np.random.normal(0, 1, size=feature_dim)
                    else:
                        shap_class = class_index if (detected_now and globals().get("USE_FAKE_SHAP_ON_DETECT", False)) else real_class_index
                        exp = _safe_shap_1d(explainer, np.array([curr_np]), model, class_index=shap_class)
                        exp = np.asarray(exp, dtype=float).reshape(-1)
                except Exception:
                    exp = np.zeros(feature_dim, dtype=float)

                if len(exp) != feature_dim:
                    exp = np.asarray(exp, dtype=float).reshape(-1)
                    if len(exp) > feature_dim:
                        exp = exp[:feature_dim]
                    elif len(exp) < feature_dim:
                        pad = np.zeros(feature_dim - len(exp), dtype=float)
                        exp = np.concatenate([exp, pad])

                if detected_now:
                    if globals().get("SIM_CORRUPT_ON_DETECT", False):
                        try:
                            exp = sim_corrupt_explanation(exp, n_f_e=n_f_e)
                        except Exception:
                            pass
                    if globals().get("REVOKE_EXPLANATION_ON_DETECT", False):
                        exp = np.zeros_like(exp)
                    if globals().get("SIM_BLOCK_ON_DETECT", False):
                        _SIM_STATE["blocked"] = True

            if _SIM_STATE.get("blocked", False):
                if (query % 10) == 0:
                    print("[SIM-B1] blocked -> returning with", len(visited_samples), "samples at query", query)
                return visited_samples, preds, query

            if globals().get("USE_VIEW_D", False):
                exp = view_d_stability_limited(exp)
            if globals().get("USE_VIEW_E", False):
                exp = view_e_adaptive_reverse_shap(exp, pred_class=class_index)

            # Traversal ranking
            k = min(max(1, int(n_f_e)), feature_dim)
            if detected_now and globals().get("POST_DETECT_RANDOM_EXPANSION", False):
                k_post = int(globals().get("POST_DETECT_K", 1))
                k_post = max(1, min(k_post, feature_dim))
                all_idx = np.arange(feature_dim)
                if globals().get("POST_DETECT_AVOID_RECENT", False):
                    recent = set(_SIM_STATE.get("recent_feat_idx", []))
                    cand = np.array([i for i in all_idx if i not in recent], dtype=int)
                    if len(cand) < k_post:
                        cand = all_idx
                    sort_index = np.random.choice(cand, size=k_post, replace=False)
                    _SIM_STATE["recent_feat_idx"] = (_SIM_STATE.get("recent_feat_idx", []) + sort_index.tolist())[-30:]
                else:
                    sort_index = np.random.choice(all_idx, size=k_post, replace=False)
                k = len(sort_index)
            else:
                sort_index = np.flip(np.argsort(np.abs(exp)))[:k]
                sort_index = np.array([i for i in sort_index if i < feature_dim], dtype=int)
                k = len(sort_index)

            if k == 0:
                isPassed = [n_visits[i] >= n_v_lb[i] for i in range(len(n_v_lb))]
                continue

            cpys = []
            oldOption = False

            if oldOption:
                for i in range(2 * k):
                    cpys.append(np.copy(curr_np))
                for i in range(k):
                    cpys[2 * i][sort_index[i]] += epsilon_set[sort_index[i]]
                    cpys[2 * i + 1][sort_index[i]] -= epsilon_set[sort_index[i]]
                for i in range(2 * k):
                    ind_i = int(i / 2)
                    tmp = (
                        any((cpys[i] == x).all() for x in visited_samples) or
                        any((cpys[i] == x).all() for x in samples) or
                        (cpys[i][sort_index[ind_i]] < 0) or
                        (cpys[i][sort_index[ind_i]] >= classPossibilities[sort_index[ind_i]])
                    )
                    if not tmp:
                        samples.append(cpys[i])
            else:
                for i in range(2):
                    cpys.append(np.copy(curr_np))
                for i in range(k):
                    num = random.random()
                    tmp0 = cpys[0][sort_index[i]] + epsilon_set[sort_index[i]]
                    tmp1 = cpys[0][sort_index[i]] - epsilon_set[sort_index[i]]

                    if not isCat[sort_index[i]]:
                        cond1 = True
                    else:
                        cond1 = (tmp0 < classPossibilities[sort_index[i]])
                    cond2 = (tmp1 >= 0)

                    if num < 0.8:
                        if cond1:
                            cpys[0][sort_index[i]] += epsilon_set[sort_index[i]]
                            if cond2:
                                cpys[1][sort_index[i]] -= epsilon_set[sort_index[i]]
                        else:
                            cpys[0][sort_index[i]] -= epsilon_set[sort_index[i]]
                            if cond1:
                                cpys[1][sort_index[i]] += epsilon_set[sort_index[i]]
                    else:
                        if cond1:
                            cpys[1][sort_index[i]] += epsilon_set[sort_index[i]]
                            if cond2:
                                cpys[0][sort_index[i]] -= epsilon_set[sort_index[i]]
                        else:
                            cpys[1][sort_index[i]] -= epsilon_set[sort_index[i]]
                            if cond1:
                                cpys[0][sort_index[i]] += epsilon_set[sort_index[i]]

                # TRUE baseline vs defense behavior
                children_to_add = [0] if detected_now else [0, 1]
                for i in children_to_add:
                    tmp = (
                        any((cpys[i] == x).all() for x in visited_samples) or
                        any((cpys[i] == x).all() for x in samples)
                    )
                    if not tmp:
                        samples.append(cpys[i])

        isPassed = [n_visits[i] >= n_v_lb[i] for i in range(len(n_v_lb))]

    return visited_samples, preds, query

# =========================================================
# REST OF ORIGINAL PIPELINE
# =========================================================
def decode_pred(target, v_preds):
    v_pred_dec = np.zeros(len(v_preds))
    for i in range(len(v_preds)):
        for j in range(len(target)):
            if v_preds[i] == target[j]:
                v_pred_dec[i] = j
    return v_pred_dec


def mega_sample_generation(testx, testy, n, sizes, n_set):
    test_cpy = []
    for i in range(len(testy)):
        test_cpy += [np.append(testx[i], testy[i])]
    test_cpy = np.array(test_cpy)
    samples_mega = []
    for _ in range(n_set):
        sample_sets = []
        for j in range(len(sizes)):
            sample_set_sui = sample_set_generation(test_cpy, n, sizes[j])
            sample_sets += [sample_set_sui.copy()]
        samples_mega += [sample_sets]
    return samples_mega


def rtest_sim(shadow, target, test_data):
    count = 0
    shadow_result = shadow.predict_proba(test_data)
    target_result = target.predict_proba(test_data)
    for i in range(len(shadow_result)):
        if np.argmax(shadow_result[i]) == np.argmax(target_result[i]):
            count += 1
    return count / len(test_data)


def pickling(dataset, modelName, accuracies, rtest_sims, samples_mega):
    if explanation_tool == 1:
        tool = "SHAP"
    else:
        tool = "LIME"

    address_acc = tool + "/_models/" + modelName + "/" + modelName + "_accuracies_" + dataset
    address_sim = tool + "/_models/" + modelName + "/" + modelName + "_similarities_" + dataset
    address_smega = tool + "/_models/" + modelName + "/" + modelName + "_samples_mega_" + dataset

    acc_file = open(address_acc, 'wb')
    pickle.dump(accuracies, acc_file)
    acc_file.close()

    sim_file = open(address_sim, 'wb')
    pickle.dump(rtest_sims, sim_file)
    sim_file.close()

    smega_file = open(address_smega, 'wb')
    pickle.dump(samples_mega, smega_file)
    smega_file.close()


def unpickling(dataset, modelName):
    if explanation_tool == 1:
        tool = "SHAP"
    else:
        tool = "LIME"

    address_acc = tool + "/_models/" + modelName + "/" + modelName + "_accuracies_" + dataset
    address_sim = tool + "/_models/" + modelName + "/" + modelName + "_similarities_" + dataset
    address_smega = tool + "/_models/" + modelName + "/" + modelName + "_samples_mega_" + dataset

    acc_file = open(address_acc, 'rb')
    accs = pickle.load(acc_file)
    acc_file.close()

    sim_file = open(address_sim, 'rb')
    sims = pickle.load(sim_file)
    sim_file.close()

    smega_file = open(address_smega, 'rb')
    samples_mega = pickle.load(smega_file)
    smega_file.close()
    return accs, sims, samples_mega


def argmaxing(accs, rss, args4):
    how_many_sets, sample_set_sizes, nfe, query_limit = args4
    ql = len(query_limit)
    argmax_acc = accs.copy()
    argmax_sim = rss.copy()
    for i in range(1, ql):
        for j in range(max(len(nfe), len(sample_set_sizes))):
            idx = (ql * j) + i
            for k in range(how_many_sets):
                if argmax_sim[idx][k] < argmax_sim[idx-1][k]:
                    argmax_acc[idx][k] = argmax_acc[idx-1][k]
                    argmax_sim[idx][k] = argmax_sim[idx-1][k]
    return argmax_acc, argmax_sim


def load_dataset(which_dataset):
    if which_dataset == 0:
        X, y = shap.datasets.iris()
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )
        X_test_t, X_test_s, y_test_t, y_test_s = train_test_split(
            X_test, y_test, train_size=0.60, random_state=21, stratify=y_test
        )

        features = list(X.columns)
        classes = [0, 1, 2]
        n_features = len(features)
        n_classes = len(classes)
        dataset_name = 'iris'
        isCategorical = [False] * n_features
        canNegative = [False] * n_features
        epsilon_set = [0.828, 0.436, 1.765, 0.762]
        epsilon_set = [x / 4 for x in epsilon_set]

    elif which_dataset == 1:
        crop = pd.read_csv('data/crop/Crop_recommendation.csv')
        crop.drop(crop.index[1800:1900], inplace=True)
        crop.drop(crop.index[1400:1500], inplace=True)
        crop.drop(crop.index[1000:1100], inplace=True)
        crop.drop(crop.index[800:900], inplace=True)
        crop.drop(crop.index[200:300], inplace=True)
        features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall', 'label']
        label_encoder = LabelEncoder()
        for col in features:
            label_encoder.fit(crop[col])
            crop[col] = label_encoder.transform(crop[col])
        features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        X = crop[features]
        y = crop['label'].to_numpy()

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)
        X_test_t, X_test_s, y_test_t, y_test_s = train_test_split(X_test, y_test, train_size=0.60, stratify=y_test, random_state=21)

        n_features = len(features)
        classes = list(range(17))
        n_classes = len(classes)
        dataset_name = 'crop'
        isCategorical = [False] * n_features
        canNegative = [False] * n_features
        epsilon_set = [36.26, 34.17, 56.48, 5.34, 19.98, 0.79, 54.04]
        epsilon_set = [x // 4 for x in epsilon_set]

    elif which_dataset == 2:
        X, y = shap.datasets.adult()
        X.iloc[:, 2] -= 1
        X['Country'] = np.where(X['Country'] == 39, 1, 0)
        X['Capital Gain'] = X['Capital Gain'] - X['Capital Loss'] + 4356
        X = X.drop(['Capital Loss'], axis=1)
        X.rename(columns={'Capital Gain': 'Net Capital'}, inplace=True)
        y = y.astype(int)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
        X_test_t, X_test_s, y_test_t, y_test_s = train_test_split(X_test, y_test, train_size=0.60, random_state=21, stratify=y_test)

        classes = [0, 1]
        features = list(X.columns)
        n_features = len(features)
        n_classes = len(classes)
        epsilon_set = [20, 2, 4, 2, 4, 2, 2, 1, 3000, 5, 1]
        isCategorical = [False, True, True, True, True, True, True, True, False, False, True]
        canNegative = [False, False, False, False, False, False, False, False, True, False, False]
        dataset_name = 'adult'

    elif which_dataset == 3:
        from sklearn.datasets import load_breast_cancer
        bc = load_breast_cancer()
        X = pd.DataFrame(bc.data)
        y = bc.target
        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
            X, y, test_size=0.25, stratify=bc.target, random_state=42
        )
        X_test_t, X_test_s, y_test_t, y_test_s = sklearn.model_selection.train_test_split(
            X_test, y_test, train_size=0.60, stratify=y_test, random_state=21
        )
        features = bc.feature_names
        classes = [0, 1]
        n_features = len(features)
        n_classes = len(classes)
        isCategorical = [False] * n_features
        canNegative = [False] * n_features
        epsilon_set = list(X.std())
        dataset_name = 'breast'

    elif which_dataset == 4:
        nursery = pd.read_csv('data/nursery/nursery.csv')
        nursery[nursery == '?'] = np.nan
        features = list(nursery.columns)
        label_encoder = LabelEncoder()
        for col in features:
            label_encoder.fit(nursery[col])
            nursery[col] = label_encoder.transform(nursery[col])

        nursery.loc[nursery['final evaluation'] >= 2, 'final evaluation'] = 2
        features = nursery.columns[:-1]
        X = nursery[features]
        y = nursery['final evaluation'].to_numpy()

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
        X_test_t, X_test_s, y_test_t, y_test_s = train_test_split(X_test, y_test, train_size=0.60, random_state=21, stratify=y_test)
        classes = [0, 1, 2]
        n_features = len(features)
        n_classes = len(classes)
        isCategorical = [True] * n_features
        epsilon_set = [1] * n_features
        canNegative = [False] * n_features
        dataset_name = 'nursery'

    elif which_dataset == 5:
        mushroom = pd.read_csv('data/mushroom/mushroom_data.csv')
        mushroom[mushroom == '?'] = np.nan
        mushroom = mushroom.drop(mushroom.columns[16], axis=1)

        features = list(mushroom.columns)
        label_encoder = LabelEncoder()
        for col in features:
            label_encoder.fit(mushroom[col])
            mushroom[col] = label_encoder.transform(mushroom[col])

        features = mushroom.columns[1::]
        X = mushroom[features]
        y = mushroom['p'].to_numpy()

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
        X_test_t, X_test_s, y_test_t, y_test_s = train_test_split(X_test, y_test, train_size=0.60, random_state=21, stratify=y_test)
        classes = [0, 1]
        n_features = len(features)
        n_classes = len(classes)
        isCategorical = [True] * n_features
        epsilon_set = [1] * n_features
        canNegative = [False] * n_features
        dataset_name = 'mushroom'

    else:
        raise ValueError("Unsupported dataset index")

    classPossibilities = []
    for i in range(n_features):
        uniques, counts = np.unique(X_train.iloc[:, i], return_counts=True)
        classPossibilities.append(len(uniques))

    args1 = [X_train, X_test, y_train, y_test, X_test_t, X_test_s, y_test_t, y_test_s]
    args2 = [classes, features, n_classes, n_features, isCategorical, epsilon_set, canNegative, classPossibilities, dataset_name]
    return args1, args2


def load_model(which_model, X_train, y_train):
    if which_model == 0:
        depth = 15
        t_model = dt(max_depth=depth, random_state=101).fit(X_train.values, y_train)
        model_name = 'dt'
    elif which_model == 1:
        t_model = lr(random_state=101, max_iter=10000).fit(X_train.values, y_train)
        model_name = 'lr'
    elif which_model == 2:
        t_model = mnb().fit(X_train.values, y_train)
        model_name = 'nb'
    elif which_model == 3:
        n_classes = len(np.unique(y_train, return_counts=True)[0])
        t_model = knn(n_neighbors=n_classes).fit(X_train.values, y_train)
        model_name = 'knn'
    elif which_model == 4:
        depth = 15
        t_model = rf(max_depth=depth, random_state=101).fit(X_train.values, y_train)
        model_name = 'rdf'
    elif which_model == 5:
        t_model = mlp(activation='relu', solver='adam', max_iter=10000, random_state=101).fit(X_train.values, y_train)
        model_name = 'mlp'
    elif which_model == 6:
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.neural_network import MLPClassifier
        _hidden = (32, 32)
        t_model = Pipeline([
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("mlp", MLPClassifier(
                hidden_layer_sizes=_hidden,
                activation="relu",
                solver="adam",
                alpha=1e-4,
                learning_rate_init=1e-3,
                max_iter=400,
                random_state=0,
                verbose=False
            ))
        ])
        t_model.fit(X_train, y_train)
        model_name = f'Simple NN (MLP-{len(_hidden)}x{_hidden[0]})'
    else:
        raise ValueError("Unsupported model index")

    return t_model, model_name


def load_explainer(explanation_tool, t_model, model_name, X_train):
    if explanation_tool == 1:
        shap.initjs()

        if model_name in ("dt", "rdf"):
            return shap.TreeExplainer(t_model)

        if model_name == "lr":
            bg = X_train.sample(min(200, len(X_train)), random_state=0)
            try:
                return shap.LinearExplainer(
                    t_model,
                    bg,
                    feature_perturbation="interventional"
                )
            except TypeError:
                return shap.LinearExplainer(t_model, bg)

        f = lambda x: t_model.predict_proba(x)[:, 1]
        bg = X_train.sample(min(100, len(X_train)), random_state=0)
        return shap.KernelExplainer(f, bg, normalize=False)

    else:
        return lime.lime_tabular.LimeTabularExplainer(
            X_train.values,
            discretize_continuous=True
        )


def getModelInfo(t_model, X_train, y_train, X_test_t, y_test_t):
    predict_train = t_model.predict(X_train.values)
    predict_test = t_model.predict(X_test_t.values)
    print('Train results')
    print(confusion_matrix(y_train, predict_train))
    print(classification_report(y_train, predict_train))
    print('Test results')
    print(confusion_matrix(y_test_t, predict_test))
    print(classification_report(y_test_t, predict_test))
    t_accuracy = round(accuracy_score(y_test_t, t_model.predict(X_test_t.values)), 4)
    print('Model test accuracy: ', t_accuracy, '\n')
    return t_accuracy


def load_experiment_dicts():
    dataset_dict = {0: 'Iris', 1: 'Crop', 2: 'Adult Income', 3: 'Breast Cancer', 4: 'Nursery', 5: 'Mushroom'}
    model_dict = {0: 'Decision Tree', 1: 'Logistic Regression', 2: 'Multinomial Naive Bayes', 3: 'K Nearest Neighbor', 4: 'Random Forest', 5: 'Multilayer Perceptron', 6: 'Simple NN (MLP-2x32)'}
    exp_dict = {0: 'LIME', 1: 'SHAP'}
    return dataset_dict, model_dict, exp_dict


def run_attack_auto(wd, wm, et, hms, sss, nfe, ql, so):
    which_dataset = wd if isinstance(wd, int) else (lambda: (_ for _ in ()).throw(TypeError("Only integers are allowed")))()
    which_model = wm if isinstance(wm, int) else (lambda: (_ for _ in ()).throw(TypeError("Only integers are allowed")))()
    explanation_tool = et if isinstance(et, int) else (lambda: (_ for _ in ()).throw(TypeError("Only integers are allowed")))()
    how_many_sets = hms if isinstance(hms, int) else (lambda: (_ for _ in ()).throw(TypeError("Only integers are allowed")))()
    sample_set_sizes = sss if isinstance(sss, list) else (lambda: (_ for _ in ()).throw(TypeError("Only lists are allowed")))()
    nfe = nfe if isinstance(nfe, list) else (lambda: (_ for _ in ()).throw(TypeError("Only lists are allowed")))()
    query_limit = ql if isinstance(ql, list) else (lambda: (_ for _ in ()).throw(TypeError("Only lists are allowed")))()
    save_option = so if isinstance(so, bool) else (lambda: (_ for _ in ()).throw(TypeError("Only booleans are allowed")))()

    args1, args2 = load_dataset(which_dataset)
    X_train, X_test, y_train, y_test, X_test_t, X_test_s, y_test_t, y_test_s = args1
    classes, features, n_classes, n_features, isCategorical, epsilon_set, canNegative, classPossibilities, dataset_name = args2
    t_model, model_name = load_model(which_model, X_train, y_train)
    t_accuracy = getModelInfo(t_model, X_train, y_train, X_test_t, y_test_t)
    t_explainer = load_explainer(explanation_tool, t_model, model_name, X_train)
    dataset_dict, model_dict, exp_dict = load_experiment_dicts()
    print('Dataset:  ', dataset_dict.get(which_dataset))
    print('ML Model: ', model_dict.get(which_model))
    print(exp_dict.get(explanation_tool), 'is the explanation tool currently in use\n')

    samples_mega = mega_sample_generation(X_test_s.to_numpy(), y_test_s, n_classes, sample_set_sizes, how_many_sets)

    accuracies = []
    rtest_sims = []
    prioritizeSim = True

    if model_name in ('nb', 'mlp', 'lr', 'knn'):
        repetition = 1
    elif model_name == 'dt':
        repetition = 100
    else:
        repetition = 10

    relax_factor = 0.5
    lb_set = list(map(lambda x: int((x // n_classes) * (1 - relax_factor) + 1), query_limit))
    ub_set = list(map(lambda x: int((x // n_classes) * (n_classes + relax_factor) + 1), query_limit))
    depth = 15
    print('Lower bounds: ', lb_set, ' Upper bounds: ', ub_set, '(per class for both)\n')
    print('-----The attack starts here!-----\n')

    for f in nfe:
        print('Number of top features allowed to be explored (k):', f)
        for g in range(len(sample_set_sizes)):
            print('\nNumber of samples per class (n):', sample_set_sizes[g])
            max_sim = [False] * how_many_sets
            for h in range(len(lb_set)):
                lb, ub = lb_set[h], ub_set[h]
                real_accuracy = []
                sims = []
                for i in range(how_many_sets):
                    if max_sim[i]:
                        print('Sample set', i, " Max similarity reached, no need for traversal!")
                        sims += [1]
                        real_accuracy += [t_accuracy]
                    else:
                        if explanation_tool == 0:
                            v_samples_np, v_pred_dec, n_query = traverse_explanations_LIME(samples_mega[i][g], t_explainer, t_model, lb, ub, query_limit[h], f, args2)
                        elif explanation_tool == 1:
                            v_samples_np, v_pred_dec, n_query = traverse_explanations_SHAP(samples_mega[i][g], t_explainer, t_model, lb, ub, query_limit[h], f, args2, model_name)
                        else:
                            print('No valid explanation tool selected')
                            break
                        s_accuracy = []
                        sim = []
                        for k in range(repetition):
                            if model_name == 'dt':
                                s_model = dt(random_state=k, max_depth=depth)
                            elif model_name == 'lr':
                                s_model = lr(max_iter=1000, random_state=k)
                            elif model_name == 'nb':
                                s_model = mnb()
                            elif model_name == 'rdf':
                                s_model = rf(max_depth=depth, random_state=k)
                            elif model_name == 'knn':
                                s_model = knn(n_neighbors=n_classes)
                            elif model_name == 'mlp':
                                models = []
                                for layer in range(10):
                                    l = layer + 1
                                    models += [mlp(activation='tanh', hidden_layer_sizes=(10 * l), solver='adam', max_iter=10000)]
                                    models += [mlp(activation='relu', hidden_layer_sizes=(10 * l), solver='adam', max_iter=10000)]
                            else:
                                s_model = DecisionTreeClassifier(random_state=0)

                            if model_name == 'mlp':
                                for m in models:
                                    m.fit(v_samples_np, v_pred_dec)
                                    sim += [rtest_sim(m, t_model, X_test_t.values)]
                                    s_accuracy += [accuracy_score(y_test_t, m.predict(X_test_t.values))]
                            else:
                                s_model.fit(v_samples_np, v_pred_dec)
                                sim += [rtest_sim(s_model, t_model, X_test_t.values)]
                                s_accuracy += [accuracy_score(y_test_t, s_model.predict(X_test_t.values))]

                        tmp = np.argmax(sim) if prioritizeSim else np.argmax(s_accuracy)
                        m_sim = round(sim[tmp], 4)
                        sims += [m_sim]
                        real_accuracy += [round(s_accuracy[tmp], 4)]
                        if m_sim == 1:
                            max_sim[i] = True
                        print('Sample set', i, ', n_queries =', len(v_pred_dec), ', Top similarity =', m_sim)
                accuracies += [real_accuracy]
                rtest_sims += [sims]
                print("Accuracy: ", real_accuracy, "\nSimilarity: ", sims, "\n")

    args0 = [which_dataset, which_model, explanation_tool]
    args3 = [t_model, model_name, t_accuracy, t_explainer]
    args4 = [how_many_sets, sample_set_sizes, nfe, query_limit]
    other_args = [args0, args1, args2, args3, args4]

    accuracies, rtest_sims = argmaxing(accuracies, rtest_sims, args4)
    if save_option:
        save_results(dataset_name, model_name, accuracies, rtest_sims, samples_mega)

    return accuracies, rtest_sims, samples_mega, other_args


def run_attack_auto_v2(wd, wm, et, hms, sss, nfe, ql, so):
    which_dataset = wd if isinstance(wd, int) else (lambda: (_ for _ in ()).throw(TypeError("Only integers are allowed")))()
    which_model = wm if isinstance(wm, int) else (lambda: (_ for _ in ()).throw(TypeError("Only integers are allowed")))()
    explanation_tool = et if isinstance(et, int) else (lambda: (_ for _ in ()).throw(TypeError("Only integers are allowed")))()
    how_many_sets = hms if isinstance(hms, int) else (lambda: (_ for _ in ()).throw(TypeError("Only integers are allowed")))()
    sample_set_sizes = sss if isinstance(sss, list) else (lambda: (_ for _ in ()).throw(TypeError("Only lists are allowed")))()
    nfe = nfe if isinstance(nfe, list) else (lambda: (_ for _ in ()).throw(TypeError("Only lists are allowed")))()
    query_limit = ql if isinstance(ql, list) else (lambda: (_ for _ in ()).throw(TypeError("Only lists are allowed")))()
    save_option = so if isinstance(so, bool) else (lambda: (_ for _ in ()).throw(TypeError("Only booleans are allowed")))()

    args1, args2 = load_dataset(which_dataset)
    X_train, X_test, y_train, y_test, X_test_t, X_test_s, y_test_t, y_test_s = args1
    classes, features, n_classes, n_features, isCategorical, epsilon_set, canNegative, classPossibilities, dataset_name = args2
    t_model, model_name = load_model(which_model, X_train, y_train)
    t_accuracy = getModelInfo(t_model, X_train, y_train, X_test_t, y_test_t)
    t_explainer = load_explainer(explanation_tool, t_model, model_name, X_train)
    dataset_dict, model_dict, exp_dict = load_experiment_dicts()
    print('Dataset:  ', dataset_dict.get(which_dataset))
    print('ML Model: ', model_dict.get(which_model))
    print(exp_dict.get(explanation_tool), 'is the explanation tool currently in use\n')

    samples_mega = mega_sample_generation(X_test_s.to_numpy(), y_test_s, n_classes, sample_set_sizes, how_many_sets)

    rows, cols, dims = len(query_limit), how_many_sets, len(nfe)
    accuracies = [[[0 for _ in range(cols)] for _ in range(rows)] for _ in range(dims)]
    rtest_sims = [[[0 for _ in range(cols)] for _ in range(rows)] for _ in range(dims)]
    prioritizeSim = True

    if model_name in ('nb', 'mlp', 'lr', 'knn'):
        repetition = 1
    elif model_name == 'dt':
        repetition = 100
    else:
        repetition = 10

    relax_factor = 0.8
    lb_set = list(map(lambda x: int((x // n_classes) * (1 - relax_factor) + 1), query_limit))
    ub_set = list(map(lambda x: int((x // n_classes) * (n_classes + relax_factor) + 1), query_limit))
    depth = 15
    print('Lower bounds: ', lb_set, ' Upper bounds: ', ub_set, '\n')
    print('-----The attack starts here!-----\n')

    for f in nfe:
        f_idx = nfe.index(f)
        print(f_idx)
        print('Number of top features allowed to be explored (k):', f)
        for g in range(len(sample_set_sizes)):
            print('\nNumber of samples per class (n):', sample_set_sizes[g])

            max_sim = [False] * how_many_sets

            for i in range(how_many_sets):
                if explanation_tool == 0:
                    v_samples_np, v_pred_dec, n_query = traverse_explanations_LIME(samples_mega[i][g], t_explainer, t_model, lb_set[-1], ub_set[-1], query_limit[-1], f, args2)
                elif explanation_tool == 1:
                    v_samples_np, v_pred_dec, n_query = traverse_explanations_SHAP(samples_mega[i][g], t_explainer, t_model, lb_set[-1], ub_set[-1], query_limit[-1], f, args2, model_name)
                else:
                    print('No valid explanation tool selected')
                    break
                for h in range(len(query_limit)):
                    data_x = v_samples_np[:query_limit[h]]
                    data_y = v_pred_dec[:query_limit[h]]
                    s_accuracy = []
                    sim = []
                    for k in range(repetition):
                        if model_name == 'dt':
                            s_model = dt(random_state=k, max_depth=depth)
                        elif model_name == 'lr':
                            s_model = lr(max_iter=1000, random_state=k)
                        elif model_name == 'nb':
                            s_model = mnb()
                        elif model_name == 'rdf':
                            s_model = rf(max_depth=depth, random_state=k)
                        elif model_name == 'knn':
                            s_model = knn(n_neighbors=n_classes)
                        else:
                            s_model = DecisionTreeClassifier(random_state=0)

                        s_model.fit(data_x, data_y)
                        sim += [rtest_sim(s_model, t_model, X_test_t.values)]
                        s_accuracy += [accuracy_score(y_test_t, s_model.predict(X_test_t.values))]

                    tmp = np.argmax(sim) if prioritizeSim else np.argmax(s_accuracy)
                    m_sim = round(sim[tmp], 4)
                    if m_sim == 1:
                        max_sim[i] = True
                    accuracies[f_idx][h][i] = round(s_accuracy[tmp], 4)
                    rtest_sims[f_idx][h][i] = round(sim[tmp], 4)

                print('Sample set', i, ', n_queries =', len(v_pred_dec), ', Top similarity =', m_sim)
        print("Accuracy: ", accuracies, "\nSimilarity: ", rtest_sims, "\n")

    args0 = [which_dataset, which_model, explanation_tool]
    args3 = [t_model, model_name, t_accuracy, t_explainer]
    args4 = [how_many_sets, sample_set_sizes, nfe, query_limit]
    other_args = [args0, args1, args2, args3, args4]

    if save_option:
        save_results(dataset_name, model_name, accuracies, rtest_sims, samples_mega)

    return accuracies, rtest_sims, samples_mega, other_args


def run_attack_prepared(isFast):
    if isFast:
        which_dataset = 0
        which_model = 1
        explanation_tool = 0
        how_many_sets = 10
        sample_set_sizes = [1]
        nfe = [3]
        query_limit = [0, 10, 25, 50, 100]
    else:
        which_dataset = 2
        which_model = 4
        explanation_tool = 1
        how_many_sets = 10
        sample_set_sizes = [5]
        nfe = [3, 5, 7]
        query_limit = [0, 100, 250, 500, 1000]

    return run_attack_auto(which_dataset, which_model, explanation_tool, how_many_sets, sample_set_sizes, nfe, query_limit, False)


def save_results(dataset_name, model_name, acs, rsims, smegas):
    try:
        pickling(dataset_name, model_name, acs, rsims, smegas)
        print('Save operation successful!')
    except Exception:
        print('Save operation failed!')


def load_results(dataset_name, model_name):
    return unpickling(dataset_name, model_name)
