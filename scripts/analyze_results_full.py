"""
analyze_results_full.py — Comprehensive Phase 2 OOD analysis.

Produces a full multi-metric scientific analysis of all 112 runs:

  Section 1:  Complete metrics summary (MAE, RMSE, R² on train/val/test)
  Section 2:  Generalization gap analysis (overfitting characterization)
  Section 3:  RMSE/MAE ratio analysis (outlier vs uniform error structure)
  Section 4:  OOD degradation on all metrics (MAE ratio, RMSE ratio, ΔR²)
  Section 5:  Val-test asymmetry (full quantification)
  Section 6:  Error distribution from prediction CSVs (bias, percentiles)
  Section 7:  Architecture comparison — quantified (not just who wins)
  Section 8:  Split difficulty — statistics + bootstrap confidence intervals
  Section 9:  Correlation analyses (i.i.d. vs OOD, coGN vs coNGN agreement)
  Section 10: Quality classification (R² bins: excellent/good/moderate/poor/collapsed)

Run from the project root:
    python scripts/analyze_results_full.py

All outputs saved to: results/phase2/analysis/
"""

import json
import csv
import math
import random
from pathlib import Path
from collections import defaultdict

random.seed(42)

# ============================================================================
# CONFIG
# ============================================================================

RESULTS_DIR = Path("results/phase2")
OUTPUT_DIR  = RESULTS_DIR / "analysis"

DATASETS = [
    "jdft2d", "phonons", "log_gvrh", "log_kvrh",
    "perovskites", "mp_gap", "mp_e_form",
]
SPLITS = [
    "index", "chemsys", "composition", "crystalsys",
    "elements", "periodictablegroups", "pointgroup", "sgnum",
]
OOD_SPLITS = [s for s in SPLITS if s != "index"]
MODELS = ["coGN", "coNGN"]

DATASET_UNITS = {
    "jdft2d":      "meV/atom",
    "phonons":     "cm⁻¹",
    "log_gvrh":    "log(GPa)",
    "log_kvrh":    "log(GPa)",
    "perovskites": "eV/unit cell",
    "mp_gap":      "eV",
    "mp_e_form":   "eV/atom",
}

SMALL_DATASETS = {"jdft2d"}

# Max rows to load from prediction CSVs per run
# (large datasets have 100k+ rows — sample for distribution stats)
MAX_CSV_ROWS = 5000

# R² quality bins
R2_BINS = [
    ("excellent",  0.95,  1.01, "R² > 0.95"),
    ("good",       0.80,  0.95, "0.80 ≤ R² < 0.95"),
    ("moderate",   0.50,  0.80, "0.50 ≤ R² < 0.80"),
    ("poor",       0.00,  0.50, "0.00 ≤ R² < 0.50"),
    ("collapsed", -999,   0.00, "R² < 0.00  (worse than predicting mean)"),
]

# Bootstrap settings for CI estimation
N_BOOTSTRAP = 2000

# ============================================================================
# UTILITIES
# ============================================================================

def fmt(v, d=4):
    return f"{v:.{d}f}" if v is not None else "—"

def fmt_pct(v):
    return f"{v:+.1f}%" if v is not None else "—"

def fmt_ratio(v, d=2):
    return f"{v:.{d}f}×" if v is not None else "—"

def section(title, char="="):
    w = 78
    print(f"\n{char*w}\n  {title}\n{char*w}")

def subsection(title):
    print(f"\n  --- {title} ---")

def r2_bin(r2):
    for name, lo, hi, _ in R2_BINS:
        if lo <= r2 < hi:
            return name
    return "collapsed" if r2 < 0 else "excellent"

def bootstrap_ci(vals, statfn=None, n=N_BOOTSTRAP, alpha=0.05):
    """Return (mean, lower_ci, upper_ci) using bootstrap resampling."""
    if statfn is None:
        statfn = lambda x: sum(x) / len(x)
    if len(vals) < 2:
        m = statfn(vals)
        return m, m, m
    samples = []
    for _ in range(n):
        resample = [random.choice(vals) for _ in vals]
        samples.append(statfn(resample))
    samples.sort()
    lo = samples[int(alpha/2 * n)]
    hi = samples[int((1 - alpha/2) * n)]
    return statfn(vals), lo, hi

def mean(vals):
    return sum(vals) / len(vals) if vals else None

def median(vals):
    if not vals: return None
    s = sorted(vals)
    n = len(s)
    return s[n//2] if n % 2 else (s[n//2-1] + s[n//2]) / 2

def std(vals):
    if len(vals) < 2: return None
    m = mean(vals)
    return math.sqrt(sum((v-m)**2 for v in vals) / (len(vals)-1))

def corr_pearson(xs, ys):
    """Pearson r between two lists."""
    if len(xs) < 3: return None
    mx, my = mean(xs), mean(ys)
    num = sum((x-mx)*(y-my) for x,y in zip(xs,ys))
    sx  = math.sqrt(sum((x-mx)**2 for x in xs))
    sy  = math.sqrt(sum((y-my)**2 for y in ys))
    if sx == 0 or sy == 0: return None
    return num / (sx * sy)

def spearman_r(xs, ys):
    """Spearman rank correlation."""
    if len(xs) < 3: return None
    def rank(arr):
        s = sorted(range(len(arr)), key=lambda i: arr[i])
        r = [0]*len(arr)
        for rank_i, orig_i in enumerate(s):
            r[orig_i] = rank_i + 1
        return r
    rx, ry = rank(xs), rank(ys)
    return corr_pearson(rx, ry)

# ============================================================================
# DATA LOADING
# ============================================================================

def load_metrics():
    """Load all metrics.json. Returns (results_dict, missing_list)."""
    results, missing = {}, []
    for ds in DATASETS:
        for sp in SPLITS:
            for model in MODELS:
                tag = f"{ds}__{sp}__{model}"
                mf  = RESULTS_DIR / tag / "metrics.json"
                if mf.exists():
                    try:
                        results[tag] = json.loads(mf.read_text())
                    except Exception as e:
                        print(f"  ERROR {tag}: {e}")
                else:
                    missing.append(tag)
    return results, missing


def load_prediction_stats(results):
    """
    For each run, load prediction CSV (sampled if large) and compute:
      - mean_signed_error : mean(y_pred - y_true)  — signed bias
      - rmse_mae_ratio    : RMSE / MAE from CSV (sanity cross-check)
      - p90_abs_error     : 90th percentile |y_pred - y_true|
      - p95_abs_error     : 95th percentile |y_pred - y_true|
      - frac_within_2mae  : fraction of predictions with |error| <= 2*MAE
      - n_loaded          : how many rows were loaded

    Returns dict: tag -> {train: {...}, val: {...}, test: {...}}
    """
    pred_stats = {}
    print("  Loading prediction CSVs (sampling large datasets)...")

    for tag in sorted(results.keys()):
        run_dir = RESULTS_DIR / tag
        pred_stats[tag] = {}
        for split_name in ("train", "val", "test"):
            csv_path = run_dir / f"predictions_{split_name}.csv"
            if not csv_path.exists():
                continue

            errors = []
            try:
                with open(csv_path, newline="") as f:
                    reader = csv.DictReader(f)
                    rows = list(reader)

                # Sample if too large
                if len(rows) > MAX_CSV_ROWS:
                    rows = random.sample(rows, MAX_CSV_ROWS)

                for row in rows:
                    try:
                        yt = float(row.get("y_true", row.get("target", "nan")))
                        yp = float(row.get("y_pred", row.get("prediction", "nan")))
                        if math.isfinite(yt) and math.isfinite(yp):
                            errors.append(yp - yt)
                    except (ValueError, KeyError):
                        continue

            except Exception as e:
                print(f"    WARNING reading {tag}/{split_name}: {e}")
                continue

            if not errors:
                continue

            abs_errors = [abs(e) for e in errors]
            mae_csv    = mean(abs_errors)
            abs_sorted = sorted(abs_errors)
            n          = len(abs_sorted)

            pred_stats[tag][split_name] = {
                "mean_signed_error": mean(errors),
                "p90_abs_error":     abs_sorted[int(0.90 * n)],
                "p95_abs_error":     abs_sorted[int(0.95 * n)],
                "mae_csv":           mae_csv,
                "rmse_csv":          math.sqrt(mean([e**2 for e in errors])),
                "rmse_mae_ratio":    math.sqrt(mean([e**2 for e in errors])) / mae_csv if mae_csv else None,
                "frac_within_2mae":  sum(1 for e in abs_errors if e <= 2*mae_csv) / len(abs_errors),
                "n_loaded":          len(errors),
            }

    print(f"  Done. Loaded stats for {len(pred_stats)} runs.")
    return pred_stats


def build_master(results):
    """Build unified master dict: (ds,sp,model) -> all metrics."""
    table = {}
    for tag, m in results.items():
        ds, sp, model = tag.split("__")
        table[(ds, sp, model)] = {
            # MAE
            "train_mae": m["train"]["mae"],
            "val_mae":   m["val"]["mae"],
            "test_mae":  m["test"]["mae"],
            # RMSE
            "train_rmse": m["train"]["rmse"],
            "val_rmse":   m["val"]["rmse"],
            "test_rmse":  m["test"]["rmse"],
            # R²
            "train_r2": m["train"]["r2"],
            "val_r2":   m["val"]["r2"],
            "test_r2":  m["test"]["r2"],
            # Meta
            "n_train":       m["meta"].get("n_train", 0),
            "n_val":         m["meta"].get("n_val",   0),
            "n_test":        m["meta"].get("n_test",  0),
            "epochs":        m["meta"].get("epochs",  0),
            "train_time_h":  m["meta"].get("train_time_s", 0) / 3600,
            "dataset":       ds,
            "split":         sp,
            "model":         model,
        }
    return table

# ============================================================================
# SECTION 1: COMPLETE METRICS SUMMARY
# ============================================================================

def sec1_complete_summary(master):
    section("SECTION 1: COMPLETE METRICS SUMMARY (MAE · RMSE · R²)")
    print("  All three metrics on all three sets for every run.\n")

    for ds in DATASETS:
        units = DATASET_UNITS[ds]
        small = " ⚠️ small-n" if ds in SMALL_DATASETS else ""
        print(f"\n  [{ds}  ({units}){small}]")
        print(f"  {'Split':22s}  {'Model':6s}  "
              f"{'trainMAE':>9} {'valMAE':>9} {'testMAE':>9}  "
              f"{'trnRMSE':>8} {'valRMSE':>8} {'tstRMSE':>8}  "
              f"{'trnR²':>7} {'valR²':>7} {'tstR²':>7}")
        print("  " + "-" * 110)
        for sp in SPLITS:
            for model in MODELS:
                v = master.get((ds, sp, model))
                if not v: continue
                print(f"  {sp:22s}  {model:6s}  "
                      f"{fmt(v['train_mae'],4):>9} "
                      f"{fmt(v['val_mae'],4):>9} "
                      f"{fmt(v['test_mae'],4):>9}  "
                      f"{fmt(v['train_rmse'],4):>8} "
                      f"{fmt(v['val_rmse'],4):>8} "
                      f"{fmt(v['test_rmse'],4):>8}  "
                      f"{fmt(v['train_r2'],3):>7} "
                      f"{fmt(v['val_r2'],3):>7} "
                      f"{fmt(v['test_r2'],3):>7}")

# ============================================================================
# SECTION 2: GENERALIZATION GAP
# ============================================================================

def sec2_generalization_gap(master):
    section("SECTION 2: GENERALIZATION GAP (train vs test)")
    print("  overfitting_ratio = train_mae / test_mae")
    print("  Near 0 = model memorizes training set but fails on test (severe overfitting).")
    print("  Near 1 = train and test performance similar (good generalization).\n")

    print(f"  {'Dataset':14s} {'Split':22s} {'Model':6s}  "
          f"{'trainMAE':>9} {'testMAE':>9}  "
          f"{'trnR²':>7} {'tstR²':>7}  "
          f"{'ratio':>7}  note")
    print("  " + "-" * 100)

    severe_overfit = []
    for ds in DATASETS:
        for sp in SPLITS:
            for model in MODELS:
                v = master.get((ds, sp, model))
                if not v: continue
                ratio = v["train_mae"] / v["test_mae"] if v["test_mae"] > 0 else None
                note = ""
                if ratio is not None and ratio < 0.05:
                    note = "  ← severe overfit"
                    severe_overfit.append((ds, sp, model, ratio))
                print(f"  {ds:14s} {sp:22s} {model:6s}  "
                      f"{fmt(v['train_mae'],4):>9} "
                      f"{fmt(v['test_mae'],4):>9}  "
                      f"{fmt(v['train_r2'],3):>7} "
                      f"{fmt(v['test_r2'],3):>7}  "
                      f"{fmt(ratio,3):>7}{note}")

    print(f"\n  Severe overfitting (ratio < 0.05): {len(severe_overfit)} runs")
    for ds, sp, model, ratio in sorted(severe_overfit, key=lambda x: x[3]):
        print(f"    {ds}/{sp}/{model}: {fmt(ratio,4)}")

# ============================================================================
# SECTION 3: RMSE/MAE RATIO (ERROR STRUCTURE)
# ============================================================================

def sec3_rmse_mae_ratio(master):
    section("SECTION 3: RMSE/MAE RATIO (error distribution structure)")
    print("  RMSE/MAE ratio = 1.0: all errors equal (uniform distribution)")
    print("  RMSE/MAE ratio >> 1.0: a few large outlier errors dominate RMSE")
    print("  Practical threshold: ratio > 2.0 = outlier-dominated errors\n")

    ratios_by_split = defaultdict(list)
    ratios_by_ds    = defaultdict(list)

    print(f"  {'Dataset':14s} {'Split':22s} {'Model':6s}  "
          f"{'test_MAE':>9} {'test_RMSE':>10}  "
          f"{'RMSE/MAE':>9}  note")
    print("  " + "-" * 80)

    for ds in DATASETS:
        for sp in SPLITS:
            for model in MODELS:
                v = master.get((ds, sp, model))
                if not v: continue
                if v["test_mae"] == 0: continue
                ratio = v["test_rmse"] / v["test_mae"]
                note  = "  ← outlier-dominated" if ratio > 2.0 else ""
                ratios_by_split[sp].append(ratio)
                ratios_by_ds[ds].append(ratio)
                print(f"  {ds:14s} {sp:22s} {model:6s}  "
                      f"{fmt(v['test_mae'],4):>9} "
                      f"{fmt(v['test_rmse'],4):>10}  "
                      f"{fmt(ratio,3):>9}{note}")

    subsection("Average RMSE/MAE ratio by split type")
    print(f"  {'Split':22s}  {'avg ratio':>10}  {'median':>8}")
    print("  " + "-" * 50)
    for sp in SPLITS:
        vals = ratios_by_split[sp]
        if vals:
            print(f"  {sp:22s}  {fmt(mean(vals),3):>10}  "
                  f"{fmt(median(vals),3):>8}")

    subsection("Average RMSE/MAE ratio by dataset")
    print(f"  {'Dataset':14s}  {'avg ratio':>10}  {'median':>8}")
    print("  " + "-" * 40)
    for ds in DATASETS:
        vals = ratios_by_ds[ds]
        if vals:
            print(f"  {ds:14s}  {fmt(mean(vals),3):>10}  "
                  f"{fmt(median(vals),3):>8}")

# ============================================================================
# SECTION 4: OOD DEGRADATION — ALL METRICS
# ============================================================================

def sec4_ood_all_metrics(master):
    section("SECTION 4: OOD DEGRADATION ON ALL METRICS")
    print("  MAE_ratio  = test_mae(OOD)  / test_mae(i.i.d.)")
    print("  RMSE_ratio = test_rmse(OOD) / test_rmse(i.i.d.)")
    print("  ΔR²        = test_r2(i.i.d.) - test_r2(OOD)   (positive = degradation)\n")

    degradation = {}
    for ds in DATASETS:
        for model in MODELS:
            iid = master.get((ds, "index", model))
            if not iid: continue
            for sp in OOD_SPLITS:
                ood = master.get((ds, sp, model))
                if not ood: continue
                degradation[(ds, sp, model)] = {
                    "mae_ratio":  ood["test_mae"]  / iid["test_mae"]  if iid["test_mae"]  else None,
                    "rmse_ratio": ood["test_rmse"] / iid["test_rmse"] if iid["test_rmse"] else None,
                    "delta_r2":   iid["test_r2"]   - ood["test_r2"],
                    "iid_test_mae":  iid["test_mae"],
                    "iid_test_r2":   iid["test_r2"],
                    "ood_test_mae":  ood["test_mae"],
                    "ood_test_r2":   ood["test_r2"],
                    "ood_val_mae":   ood["val_mae"],
                    "val_ratio":     ood["val_mae"] / iid["test_mae"] if iid["test_mae"] else None,
                    "is_small":      ds in SMALL_DATASETS,
                }

    print(f"  {'Dataset':14s} {'Split':22s} {'Model':6s}  "
          f"{'MAE_ratio':>10} {'RMSE_ratio':>11} {'ΔR²':>8}")
    print("  " + "-" * 80)
    for ds in DATASETS:
        for sp in OOD_SPLITS:
            for model in MODELS:
                d = degradation.get((ds, sp, model))
                if not d: continue
                print(f"  {ds:14s} {sp:22s} {model:6s}  "
                      f"{fmt_ratio(d['mae_ratio']):>10} "
                      f"{fmt_ratio(d['rmse_ratio']):>11} "
                      f"{fmt(d['delta_r2'],3):>8}")

    subsection("Average degradation per split type (excluding jdft2d)")
    print(f"  {'Split':22s}  {'avg MAE_ratio':>14} {'avg RMSE_ratio':>15} {'avg ΔR²':>9}")
    print("  " + "-" * 70)
    for sp in OOD_SPLITS:
        vals = [(d["mae_ratio"], d["rmse_ratio"], d["delta_r2"])
                for (ds,s,m), d in degradation.items()
                if s == sp and not d["is_small"]
                and d["mae_ratio"] and d["rmse_ratio"]]
        if not vals: continue
        avg_mae  = mean([v[0] for v in vals])
        avg_rmse = mean([v[1] for v in vals])
        avg_dr2  = mean([v[2] for v in vals])
        print(f"  {sp:22s}  {fmt_ratio(avg_mae):>14} "
              f"{fmt_ratio(avg_rmse):>15} "
              f"{fmt(avg_dr2,3):>9}")

    return degradation

# ============================================================================
# SECTION 5: VAL-TEST ASYMMETRY (FULL)
# ============================================================================

def sec5_val_test_asymmetry(degradation):
    section("SECTION 5: VAL-TEST ASYMMETRY (full quantification)")
    print("  val_ratio  = val_mae(OOD)  / test_mae(i.i.d.)")
    print("  test_ratio = test_mae(OOD) / test_mae(i.i.d.)")
    print("  gap        = test_ratio - val_ratio")
    print("  Large gap: validation masked the OOD failure — reporting val would mislead.\n")

    rows = []
    for (ds, sp, model), d in sorted(degradation.items()):
        if d["is_small"]: continue
        if d["val_ratio"] is None or d["mae_ratio"] is None: continue
        gap = d["mae_ratio"] - d["val_ratio"]
        rows.append((ds, sp, model, d["val_ratio"], d["mae_ratio"], gap))

    rows_sorted = sorted(rows, key=lambda x: x[5], reverse=True)

    print(f"  {'Tag':45s} {'val/iid':>9} {'test/iid':>10} {'gap':>8}  assessment")
    print("  " + "-" * 90)
    for ds, sp, model, vr, tr, gap in rows_sorted:
        tag = f"{ds}/{sp}/{model}"
        if gap > 5.0:
            assessment = "SEVERE: val hides collapse"
        elif gap > 2.0:
            assessment = "LARGE: val misleading"
        elif gap > 0.5:
            assessment = "moderate gap"
        elif gap < -0.5:
            assessment = "test easier than val"
        else:
            assessment = "consistent"
        print(f"  {tag:45s} {fmt_ratio(vr):>9} {fmt_ratio(tr):>10} "
              f"{fmt(gap,2):>8}  {assessment}")

    subsection("Summary statistics of val-test gap")
    all_gaps = [r[5] for r in rows]
    print(f"  Mean gap:   {fmt(mean(all_gaps),3)}")
    print(f"  Median gap: {fmt(median(all_gaps),3)}")
    print(f"  Max gap:    {fmt(max(all_gaps),3)} — {rows_sorted[0][0]}/{rows_sorted[0][1]}/{rows_sorted[0][2]}")
    print(f"  Cases with gap > 2.0: {sum(1 for g in all_gaps if g > 2.0)}")
    print(f"  Cases with gap > 5.0: {sum(1 for g in all_gaps if g > 5.0)}")

# ============================================================================
# SECTION 6: ERROR DISTRIBUTION FROM PREDICTION CSVs
# ============================================================================

def sec6_error_distribution(pred_stats, master):
    section("SECTION 6: ERROR DISTRIBUTION (from prediction CSVs)")
    print("  signed_bias   = mean(y_pred - y_true)  positive=overestimate")
    print("  rmse_mae      = RMSE/MAE ratio from raw predictions (sanity check)")
    print("  p90_err       = 90th percentile of |y_pred - y_true|")
    print("  p95_err       = 95th percentile of |y_pred - y_true|")
    print("  pct_2mae      = % predictions within 2×MAE of true value\n")

    print(f"  {'Tag':45s} {'set':6s} {'bias':>9} {'RMSE/MAE':>9} "
          f"{'p90_err':>9} {'p95_err':>9} {'pct_2mae':>9}")
    print("  " + "-" * 105)

    # Compute systematic bias summary per split type (test set only)
    bias_by_split = defaultdict(list)

    for ds in DATASETS:
        for sp in ["index", "elements", "periodictablegroups"]:  # representative splits
            for model in MODELS:
                tag = f"{ds}__{sp}__{model}"
                stats = pred_stats.get(tag, {})
                for split_name in ("test",):
                    s = stats.get(split_name)
                    if not s: continue
                    # Get MAE from metrics.json for p90/p95 context
                    v = master.get((ds, sp, model), {})
                    bias_by_split[sp].append(s["mean_signed_error"])
                    print(f"  {ds}/{sp}/{model:6s}  {split_name:6s} "
                          f"{fmt(s['mean_signed_error'],4):>9} "
                          f"{fmt(s['rmse_mae_ratio'],3):>9} "
                          f"{fmt(s['p90_abs_error'],4):>9} "
                          f"{fmt(s['p95_abs_error'],4):>9} "
                          f"{fmt(s['frac_within_2mae']*100,1):>9}%")

    subsection("Systematic bias summary by split type (test set, all datasets/models)")
    for sp in OOD_SPLITS:
        vals = bias_by_split.get(sp, [])
        if not vals: continue
        pos = sum(1 for v in vals if v > 0)
        neg = sum(1 for v in vals if v < 0)
        print(f"  {sp:22s}: mean_bias={fmt(mean(vals),4):>9}  "
              f"overestimate {pos}/{len(vals)}  underestimate {neg}/{len(vals)}")

# ============================================================================
# SECTION 7: ARCHITECTURE COMPARISON — QUANTIFIED
# ============================================================================

def sec7_architecture_comparison(master, degradation):
    section("SECTION 7: ARCHITECTURE COMPARISON — QUANTIFIED")
    print("  Beyond win/loss: how large is the advantage, and where?\n")

    improvements = []   # (ds, sp, improvement_pct, winner)
    cogn_wins = []
    cognn_wins = []

    for ds in DATASETS:
        for sp in SPLITS:
            v1 = master.get((ds, sp, "coGN"))
            v2 = master.get((ds, sp, "coNGN"))
            if not v1 or not v2: continue
            m1 = v1["test_mae"]
            m2 = v2["test_mae"]
            # improvement_pct: positive = coNGN is better, negative = coGN is better
            imp = (m1 - m2) / m1 * 100  # positive when coNGN < coGN
            improvements.append((ds, sp, imp))
            if m2 < m1:
                cognn_wins.append(imp)
            else:
                cogn_wins.append(-imp)

    print(f"  coGN wins:  {len(cogn_wins):3d} cases  "
          f"avg advantage: {fmt(mean(cogn_wins),2):>6}%  "
          f"max: {fmt(max(cogn_wins) if cogn_wins else 0,2):>6}%")
    print(f"  coNGN wins: {len(cognn_wins):3d} cases  "
          f"avg advantage: {fmt(mean(cognn_wins),2):>6}%  "
          f"max: {fmt(max(cognn_wins) if cognn_wins else 0,2):>6}%")

    subsection("coNGN improvement over coGN by dataset (test MAE, all splits averaged)")
    for ds in DATASETS:
        ds_imps = [imp for d,s,imp in improvements if d==ds]
        if not ds_imps: continue
        avg_imp = mean(ds_imps)
        n_cognn = sum(1 for i in ds_imps if i > 0)
        print(f"  {ds:14s}: avg improvement {fmt_pct(avg_imp):>8}  "
              f"coNGN better in {n_cognn}/{len(ds_imps)} splits")

    subsection("coNGN improvement over coGN by split type (all datasets averaged)")
    for sp in SPLITS:
        sp_imps = [imp for d,s,imp in improvements if s==sp]
        if not sp_imps: continue
        avg_imp = mean(sp_imps)
        print(f"  {sp:22s}: avg improvement {fmt_pct(avg_imp):>8}")

    subsection("Where coNGN advantage is largest (top 10 cases by improvement %)")
    top_cognn = sorted([(d,s,i) for d,s,i in improvements if i > 0],
                       key=lambda x: x[2], reverse=True)[:10]
    for ds, sp, imp in top_cognn:
        v1 = master[(ds, sp, "coGN")]["test_mae"]
        v2 = master[(ds, sp, "coNGN")]["test_mae"]
        print(f"  {ds}/{sp}: coGN={fmt(v1)} → coNGN={fmt(v2)}  "
              f"({fmt_pct(imp)} improvement)")

    subsection("OOD vs i.i.d. architecture advantage (does coNGN help more under OOD?)")
    iid_imps = [imp for d,s,imp in improvements if s=="index"]
    ood_imps = [imp for d,s,imp in improvements if s!="index"]
    print(f"  Avg coNGN improvement under i.i.d. splits: {fmt_pct(mean(iid_imps))}")
    print(f"  Avg coNGN improvement under OOD splits:    {fmt_pct(mean(ood_imps))}")
    print(f"  Note: positive = coNGN better. Similar values suggest coNGN's")
    print(f"  advantage is structural (better i.i.d.) rather than OOD-specific.")

# ============================================================================
# SECTION 8: SPLIT DIFFICULTY — STATISTICS + BOOTSTRAP CI
# ============================================================================

def sec8_split_difficulty_stats(degradation):
    section("SECTION 8: SPLIT DIFFICULTY — STATISTICS + BOOTSTRAP CI")
    print("  95% bootstrap confidence intervals on mean degradation ratio.")
    print("  All: all datasets. Large: excludes jdft2d.\n")

    split_vals_all   = defaultdict(list)
    split_vals_large = defaultdict(list)

    for (ds, sp, model), d in degradation.items():
        if d["mae_ratio"] is None: continue
        split_vals_all[sp].append(d["mae_ratio"])
        if not d["is_small"]:
            split_vals_large[sp].append(d["mae_ratio"])

    print(f"  {'Split':22s}  "
          f"{'mean(all)':>10} {'CI':>22}  "
          f"{'mean(large)':>12} {'CI':>22}  "
          f"{'median':>8} {'std':>8}  N")
    print("  " + "-" * 120)

    ranked = sorted(OOD_SPLITS,
                    key=lambda s: mean(split_vals_large[s]) if split_vals_large[s] else 0,
                    reverse=True)

    for sp in ranked:
        va  = split_vals_all[sp]
        vl  = split_vals_large[sp]
        if not va: continue

        m_all,  lo_all,  hi_all  = bootstrap_ci(va)
        m_large = None
        if vl:
            m_large, lo_large, hi_large = bootstrap_ci(vl)

        ci_all   = f"[{fmt_ratio(lo_all)}, {fmt_ratio(hi_all)}]"
        ci_large = f"[{fmt_ratio(lo_large)}, {fmt_ratio(hi_large)}]" if vl else "—"
        print(f"  {sp:22s}  "
              f"{fmt_ratio(m_all):>10} {ci_all:>22}  "
              f"{fmt_ratio(m_large) if m_large else '—':>12} {ci_large:>22}  "
              f"{fmt_ratio(median(va)):>8} {fmt(std(va),3) if std(va) else '—':>8}  "
              f"{len(va)}")

# ============================================================================
# SECTION 9: CORRELATION ANALYSES
# ============================================================================

def sec9_correlations(master, degradation):
    section("SECTION 9: CORRELATION ANALYSES")

    subsection("9a: Does better i.i.d. performance predict better OOD performance?")
    print("  Pearson/Spearman r between i.i.d. test_mae and mean OOD test_mae")
    print("  across all split types (per dataset, per model).\n")
    print(f"  {'Dataset':14s} {'Model':6s}  {'Pearson r':>10}  {'Spearman r':>11}  n")
    print("  " + "-" * 55)

    for ds in DATASETS:
        for model in MODELS:
            iid = master.get((ds, "index", model))
            if not iid: continue
            iid_mae = iid["test_mae"]
            ood_maes = []
            for sp in OOD_SPLITS:
                ood = master.get((ds, sp, model))
                if ood:
                    ood_maes.append(ood["test_mae"])
            if len(ood_maes) < 3: continue
            iid_vals = [iid_mae] * len(ood_maes)
            pr = corr_pearson(iid_vals, ood_maes)
            sr = spearman_r(iid_vals, ood_maes)
            print(f"  {ds:14s} {model:6s}  {fmt(pr,3):>10}  "
                  f"{fmt(sr,3):>11}  {len(ood_maes)}")

    subsection("9b: Are coGN and coNGN degradation ratios correlated?")
    print("  Spearman r between coGN and coNGN degradation ratios per (dataset, split).")
    print("  High r: both models agree on which splits are hard.\n")

    for ds in DATASETS:
        cogn_rats  = []
        cognn_rats = []
        for sp in OOD_SPLITS:
            d1 = degradation.get((ds, sp, "coGN"))
            d2 = degradation.get((ds, sp, "coNGN"))
            if d1 and d2 and d1["mae_ratio"] and d2["mae_ratio"]:
                cogn_rats.append(d1["mae_ratio"])
                cognn_rats.append(d2["mae_ratio"])
        if len(cogn_rats) < 3: continue
        sr = spearman_r(cogn_rats, cognn_rats)
        print(f"  {ds:14s}: Spearman r = {fmt(sr,3):>6}  "
              f"(n={len(cogn_rats)} OOD splits)  "
              f"{'↑ strong agreement' if sr and sr > 0.8 else '← moderate agreement' if sr and sr > 0.5 else '← weak agreement'}")

    subsection("9c: Global coGN vs coNGN degradation agreement (all non-jdft2d)")
    all_cogn  = []
    all_cognn = []
    for (ds, sp, model), d in degradation.items():
        if model != "coGN" or d["is_small"]: continue
        d2 = degradation.get((ds, sp, "coNGN"))
        if d2 and d["mae_ratio"] and d2["mae_ratio"]:
            all_cogn.append(d["mae_ratio"])
            all_cognn.append(d2["mae_ratio"])
    sr_global = spearman_r(all_cogn, all_cognn)
    pr_global = corr_pearson(all_cogn, all_cognn)
    print(f"  Pearson r:  {fmt(pr_global,3)}  "
          f"Spearman r: {fmt(sr_global,3)}  "
          f"(n={len(all_cogn)} pairs)")
    print(f"  Interpretation: both models {'strongly agree' if sr_global and sr_global > 0.85 else 'moderately agree'} "
          f"on which (dataset, split) pairs are hard.")

# ============================================================================
# SECTION 10: QUALITY CLASSIFICATION
# ============================================================================

def sec10_quality_bins(master):
    section("SECTION 10: QUALITY CLASSIFICATION (R² bins)")
    print("  Bins: excellent (R²>0.95), good (0.80-0.95), moderate (0.50-0.80),")
    print("        poor (0.00-0.50), collapsed (R²<0.00)\n")

    # Count per (split_type, bin) for test R²
    counts = defaultdict(lambda: defaultdict(int))
    totals = defaultdict(int)

    for (ds, sp, model), v in master.items():
        r2    = v["test_r2"]
        bin_  = r2_bin(r2)
        counts[sp][bin_] += 1
        totals[sp]       += 1

    bin_names = [b[0] for b in R2_BINS]
    header = f"  {'Split':22s} " + " ".join(f"{b:>12}" for b in bin_names) + "  total"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for sp in SPLITS:
        row = f"  {sp:22s} "
        for bn in bin_names:
            n   = counts[sp][bn]
            tot = totals[sp]
            pct = f"{n}/{tot}" if tot else "0/0"
            row += f"{pct:>12}"
        row += f"  {totals[sp]}"
        print(row)

    subsection("Distribution across all 112 runs (test R²)")
    all_r2   = [v["test_r2"] for v in master.values()]
    bin_totals = defaultdict(int)
    for r2 in all_r2:
        bin_totals[r2_bin(r2)] += 1
    for bn, _, _, desc in R2_BINS:
        n = bin_totals[bn]
        print(f"  {bn:12s} ({desc}): {n:3d} / {len(all_r2)}  "
              f"({n/len(all_r2)*100:.1f}%)")

    subsection("Collapsed runs (R² < 0.0) — worse than predicting mean")
    collapsed = [(ds,sp,model,v["test_r2"])
                 for (ds,sp,model),v in master.items()
                 if v["test_r2"] < 0.0]
    if collapsed:
        print(f"  {len(collapsed)} runs have R² < 0:")
        for ds, sp, model, r2 in sorted(collapsed, key=lambda x: x[3]):
            print(f"    {ds}/{sp}/{model}: R²={fmt(r2,3)}")
    else:
        print("  No collapsed runs.")

# ============================================================================
# CSV OUTPUT
# ============================================================================

def save_all_csvs(master, degradation, pred_stats):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Comprehensive master table
    with open(OUTPUT_DIR / "full_master_table.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "dataset","split","model",
            "train_mae","train_rmse","train_r2",
            "val_mae","val_rmse","val_r2",
            "test_mae","test_rmse","test_r2",
            "test_rmse_mae_ratio",
            "train_test_mae_ratio",
            "n_train","n_val","n_test",
            "epochs","train_time_h","units","is_small_dataset",
        ])
        for (ds, sp, model), v in sorted(master.items()):
            rmse_mae = v["test_rmse"] / v["test_mae"] if v["test_mae"] else None
            tr_tst   = v["train_mae"] / v["test_mae"] if v["test_mae"] else None
            w.writerow([
                ds, sp, model,
                round(v["train_mae"],6), round(v["train_rmse"],6), round(v["train_r2"],6),
                round(v["val_mae"],6),   round(v["val_rmse"],6),   round(v["val_r2"],6),
                round(v["test_mae"],6),  round(v["test_rmse"],6),  round(v["test_r2"],6),
                round(rmse_mae,4) if rmse_mae else "",
                round(tr_tst,4)   if tr_tst   else "",
                v["n_train"], v["n_val"], v["n_test"],
                v["epochs"], round(v["train_time_h"],4),
                DATASET_UNITS[ds], int(ds in SMALL_DATASETS),
            ])

    # 2. Comprehensive degradation table
    with open(OUTPUT_DIR / "full_degradation_table.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "dataset","split","model",
            "iid_test_mae","iid_test_rmse","iid_test_r2",
            "ood_val_mae","ood_val_r2",
            "ood_test_mae","ood_test_rmse","ood_test_r2",
            "mae_ratio","rmse_ratio","delta_r2",
            "val_ratio","val_test_gap",
            "is_small_dataset","units",
        ])
        for (ds, sp, model), d in sorted(degradation.items()):
            iid_v = master.get((ds, "index", model), {})
            ood_v = master.get((ds, sp, model), {})
            val_test_gap = (d["mae_ratio"] - d["val_ratio"]) if d["val_ratio"] else None
            w.writerow([
                ds, sp, model,
                round(d["iid_test_mae"],6), round(iid_v.get("test_rmse",0),6), round(d["iid_test_r2"],6),
                round(d["ood_val_mae"],6),  round(ood_v.get("val_r2",0),6),
                round(d["ood_test_mae"],6), round(ood_v.get("test_rmse",0),6), round(d["ood_test_r2"],6),
                round(d["mae_ratio"],4),  round(d["rmse_ratio"],4) if d["rmse_ratio"] else "",
                round(d["delta_r2"],4),
                round(d["val_ratio"],4) if d["val_ratio"] else "",
                round(val_test_gap,4)   if val_test_gap else "",
                int(d["is_small"]), DATASET_UNITS[ds],
            ])

    # 3. Error distribution table
    with open(OUTPUT_DIR / "error_distribution_table.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "dataset","split","model","set",
            "mean_signed_error","rmse_mae_ratio",
            "p90_abs_error","p95_abs_error",
            "frac_within_2mae","n_loaded",
        ])
        for tag, sets in sorted(pred_stats.items()):
            ds, sp, model = tag.split("__")
            for set_name, s in sets.items():
                w.writerow([
                    ds, sp, model, set_name,
                    round(s["mean_signed_error"],6),
                    round(s["rmse_mae_ratio"],4) if s["rmse_mae_ratio"] else "",
                    round(s["p90_abs_error"],6),
                    round(s["p95_abs_error"],6),
                    round(s["frac_within_2mae"],4),
                    s["n_loaded"],
                ])

    print(f"\n  CSVs saved to {OUTPUT_DIR}/")
    print(f"    full_master_table.csv        — {len(master)} rows (all metrics + ratios)")
    print(f"    full_degradation_table.csv   — {len(degradation)} rows "
          f"(MAE+RMSE+R² degradation, val-test gap)")
    print(f"    error_distribution_table.csv — "
          f"{sum(len(s) for s in pred_stats.values())} rows "
          f"(bias, percentiles from prediction CSVs)")

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading metrics.json files...")
    results, missing = load_metrics()
    print(f"  Loaded: {len(results)}  Missing: {len(missing)}")
    if missing:
        for m in missing: print(f"    {m}")

    master = build_master(results)

    print("\nLoading prediction CSVs for error distribution analysis...")
    pred_stats = load_prediction_stats(results)

    print("\nBuilding degradation table...")
    degradation = {}
    for ds in DATASETS:
        for model in MODELS:
            iid = master.get((ds, "index", model))
            if not iid: continue
            for sp in OOD_SPLITS:
                ood = master.get((ds, sp, model))
                if not ood: continue
                degradation[(ds, sp, model)] = {
                    "mae_ratio":   ood["test_mae"]  / iid["test_mae"]  if iid["test_mae"]  else None,
                    "rmse_ratio":  ood["test_rmse"] / iid["test_rmse"] if iid["test_rmse"] else None,
                    "delta_r2":    iid["test_r2"]   - ood["test_r2"],
                    "iid_test_mae": iid["test_mae"],
                    "iid_test_r2":  iid["test_r2"],
                    "ood_test_mae": ood["test_mae"],
                    "ood_test_r2":  ood["test_r2"],
                    "ood_val_mae":  ood["val_mae"],
                    "val_ratio":    ood["val_mae"] / iid["test_mae"] if iid["test_mae"] else None,
                    "is_small":     ds in SMALL_DATASETS,
                }

    # Run all sections
    sec1_complete_summary(master)
    sec2_generalization_gap(master)
    sec3_rmse_mae_ratio(master)
    sec4_ood_all_metrics(master)
    sec5_val_test_asymmetry(degradation)
    sec6_error_distribution(pred_stats, master)
    sec7_architecture_comparison(master, degradation)
    sec8_split_difficulty_stats(degradation)
    sec9_correlations(master, degradation)
    sec10_quality_bins(master)
    save_all_csvs(master, degradation, pred_stats)

    print("\n  Full analysis complete.")
    if missing:
        print(f"  Re-run once {len(missing)} missing run(s) are available.")