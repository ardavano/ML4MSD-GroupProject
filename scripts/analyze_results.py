"""
analyze_results.py — Phase 2 OOD batch results analysis.

Reads all 112 metrics.json files and produces:
  1. Master results table (test MAE per run, with R²)
  2. OOD degradation table (test MAE ratio vs i.i.d. index split,
     plus val MAE and val/test ratio to capture val-test asymmetry)
  3. Architecture comparison (coGN vs coNGN)
  4. Split-type difficulty ranking
  5. Key findings summary

Run from the project root:
    python scripts/analyze_results.py

Outputs to: results/phase2/analysis/
  master_table.csv      — 112 rows: all metrics per run
  degradation_table.csv — 98 rows: OOD degradation ratios + val/test asymmetry
"""

import json
import csv
from pathlib import Path
from collections import defaultdict

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

# Datasets where n is small enough that OOD ratios < 1.0 reflect sampling
# variance rather than genuine robustness. Flagged in reports.
SMALL_DATASETS = {"jdft2d"}   # n=636

# ============================================================================
# LOAD DATA
# ============================================================================

def load_all_results():
    """
    Load all metrics.json files. Returns:
        results: dict {tag -> metrics dict}
        missing: list of tags with no metrics.json
    """
    results = {}
    missing = []
    for dataset in DATASETS:
        for split in SPLITS:
            for model in MODELS:
                tag = f"{dataset}__{split}__{model}"
                mf  = RESULTS_DIR / tag / "metrics.json"
                if mf.exists():
                    try:
                        results[tag] = json.loads(mf.read_text())
                    except Exception as e:
                        print(f"  ERROR loading {tag}: {e}")
                else:
                    missing.append(tag)
    return results, missing


# ============================================================================
# TABLE BUILDERS
# ============================================================================

def build_master_table(results):
    """
    Returns dict: (dataset, split, model) -> {
        train_mae, train_rmse, train_r2,
        val_mae,   val_rmse,   val_r2,
        test_mae,  test_rmse,  test_r2,
        train_time_h, n_train, n_val, n_test
    }
    """
    table = {}
    for tag, m in results.items():
        dataset, split, model = tag.split("__")
        table[(dataset, split, model)] = {
            # Train metrics
            "train_mae":    m["train"]["mae"],
            "train_rmse":   m["train"]["rmse"],
            "train_r2":     m["train"]["r2"],
            # Val metrics
            "val_mae":      m["val"]["mae"],
            "val_rmse":     m["val"]["rmse"],
            "val_r2":       m["val"]["r2"],
            # Test metrics
            "test_mae":     m["test"]["mae"],
            "test_rmse":    m["test"]["rmse"],
            "test_r2":      m["test"]["r2"],
            # Meta
            "train_time_h": m["meta"].get("train_time_s", 0) / 3600,
            "n_train":      m["meta"].get("n_train", 0),
            "n_val":        m["meta"].get("n_val", 0),
            "n_test":       m["meta"].get("n_test", 0),
            "epochs":       m["meta"].get("epochs", 0),
        }
    return table


def build_degradation_table(master_table):
    """
    For each (dataset, model, OOD_split) computes:
        degradation_ratio = test_mae(OOD) / test_mae(index)
        val_test_ratio    = val_mae(OOD) / test_mae(index)
            — captures the val-test asymmetry within OOD splits:
              val_mae(OOD) ≈ iid_test_mae indicates the validation fold is
              "easy OOD"; test_mae(OOD) >> iid_test_mae indicates the test
              fold is "hard OOD". Reporting both exposes this asymmetry.

    Returns dict: (dataset, split, model) -> {
        iid_test_mae, iid_test_r2,
        ood_val_mae,  ood_val_r2,
        ood_test_mae, ood_test_r2,
        degradation_ratio,
        val_degradation_ratio,
        val_test_ratio,
        is_small_dataset
    }
    """
    degradation = {}
    for dataset in DATASETS:
        for model in MODELS:
            index_key = (dataset, "index", model)
            if index_key not in master_table:
                continue
            iid_row = master_table[index_key]
            iid_test_mae = iid_row["test_mae"]
            iid_test_r2  = iid_row["test_r2"]

            for split in SPLITS:
                if split == "index":
                    continue
                key = (dataset, split, model)
                if key not in master_table:
                    continue
                ood_row = master_table[key]
                ood_val_mae  = ood_row["val_mae"]
                ood_val_r2   = ood_row["val_r2"]
                ood_test_mae = ood_row["test_mae"]
                ood_test_r2  = ood_row["test_r2"]

                degradation[(dataset, split, model)] = {
                    # i.i.d. reference (index split test)
                    "iid_test_mae":         iid_test_mae,
                    "iid_test_r2":          iid_test_r2,
                    # OOD validation performance
                    "ood_val_mae":          ood_val_mae,
                    "ood_val_r2":           ood_val_r2,
                    # OOD test performance
                    "ood_test_mae":         ood_test_mae,
                    "ood_test_r2":          ood_test_r2,
                    # Key ratios
                    "degradation_ratio":    ood_test_mae / iid_test_mae,
                    "val_degradation_ratio": ood_val_mae / iid_test_mae,
                    # Ratio of OOD val to OOD test — near 1.0 means val and
                    # test are equally hard; >> 1.0 means test is much harder
                    "val_test_ratio":       ood_val_mae / ood_test_mae,
                    # Flag for small-dataset caution
                    "is_small_dataset":     dataset in SMALL_DATASETS,
                }
    return degradation


# ============================================================================
# FORMATTING HELPERS
# ============================================================================

def fmt(val, decimals=4):
    if val is None:
        return "—"
    return f"{val:.{decimals}f}"

def fmt_ratio(val):
    if val is None:
        return "—"
    return f"{val:.2f}×"

def section(title):
    width = 78
    print()
    print("=" * width)
    print(f"  {title}")
    print("=" * width)


# ============================================================================
# REPORTS
# ============================================================================

def report_summary(results, missing):
    section("BATCH SUMMARY")
    total = len(DATASETS) * len(SPLITS) * len(MODELS)
    print(f"  Completed: {len(results)} / {total}")
    print(f"  Missing:   {len(missing)}")
    print()
    print(f"  {'Dataset':14s}  {'coGN':6s}  {'coNGN':6s}  {'Status'}")
    print(f"  {'-'*14}  {'-'*6}  {'-'*6}  {'-'*20}")
    for ds in DATASETS:
        n_cogn  = sum(1 for s in SPLITS if f"{ds}__{s}__coGN"  in results)
        n_cognn = sum(1 for s in SPLITS if f"{ds}__{s}__coNGN" in results)
        status = "✅ complete" if n_cogn == 8 and n_cognn == 8 else "⚠️  partial"
        print(f"  {ds:14s}  {n_cogn}/8    {n_cognn}/8    {status}")
    if missing:
        print()
        print("  Missing:")
        for m in missing:
            print(f"    - {m}")


def report_master_table(master_table):
    section("MASTER RESULTS TABLE — Test MAE per (dataset, split, model)")
    print("  Values are test-set MAE in dataset units.")
    print("  '—' = run not complete. '← i.i.d.' = random split baseline.")
    print()

    col_w = 11
    header = f"  {'Dataset':14s} {'Split':22s} " + \
             f"{'coGN':>{col_w}} {'coNGN':>{col_w}}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for ds in DATASETS:
        units = DATASET_UNITS[ds]
        small_note = "  ⚠️  small n — R² unreliable" if ds in SMALL_DATASETS else ""
        print(f"\n  [{ds}  ({units}){small_note}]")
        for sp in SPLITS:
            v1 = master_table.get((ds, sp, "coGN"),  {}).get("test_mae")
            v2 = master_table.get((ds, sp, "coNGN"), {}).get("test_mae")
            r1 = master_table.get((ds, sp, "coGN"),  {}).get("test_r2")
            r2 = master_table.get((ds, sp, "coNGN"), {}).get("test_r2")
            s1 = f"{fmt(v1)} (R²={fmt(r1,3)})" if v1 is not None else "—"
            s2 = f"{fmt(v2)} (R²={fmt(r2,3)})" if v2 is not None else "—"
            marker = "  ← i.i.d." if sp == "index" else ""
            print(f"    {sp:22s}  {s1:25s}  {s2:25s}{marker}")


def report_degradation(degradation, master_table):
    section("OOD DEGRADATION RATIOS — test MAE(OOD) / test MAE(i.i.d.)")
    print("  Ratio > 1.0: model performs worse under OOD conditions.")
    print("  Higher = more degradation. i.i.d. baseline = index split test MAE.")
    print()
    print("  ⚠️  jdft2d (n=636): ratios < 1.0 or > expected reflect sampling")
    print("  variance, not genuine OOD effects. Treat with caution.")

    for model in MODELS:
        print(f"\n  [{model}]")
        print(f"  {'Dataset':14s} {'chemsys':>9} {'composit':>9} "
              f"{'crystalsy':>9} {'elements':>9} {'PTgroups':>9} "
              f"{'pointgrp':>9} {'sgnum':>9}  avg")
        print("  " + "-" * 100)

        for ds in DATASETS:
            row_vals = []
            for sp in ["chemsys","composition","crystalsys","elements",
                       "periodictablegroups","pointgroup","sgnum"]:
                key = (ds, sp, model)
                if key in degradation:
                    row_vals.append(degradation[key]["degradation_ratio"])
                else:
                    row_vals.append(None)

            strs   = [fmt_ratio(v) for v in row_vals]
            valid  = [v for v in row_vals if v is not None]
            avg_str = f"{sum(valid)/len(valid):.2f}×" if valid else "—"
            small_flag = " *" if ds in SMALL_DATASETS else ""
            print(f"  {ds:14s} " +
                  " ".join(f"{s:>9}" for s in strs) +
                  f"  {avg_str}{small_flag}")

    print("\n  * Small dataset — ratios unreliable (see note above)")


def report_val_test_asymmetry(degradation):
    section("VAL-TEST ASYMMETRY WITHIN OOD SPLITS")
    print("  val_mae(OOD) / test_mae(i.i.d.) — near 1.0 means validation fold")
    print("  looks like i.i.d. even under OOD split, hiding true test degradation.")
    print()
    print("  Highlighted cases where val looks i.i.d.-like (ratio < 1.2×)")
    print("  but test degrades severely (degradation_ratio > 3.0×).")
    print()

    asymmetric = []
    for (ds, sp, model), vals in sorted(degradation.items()):
        if vals["val_degradation_ratio"] < 1.2 and vals["degradation_ratio"] > 3.0:
            asymmetric.append((ds, sp, model, vals))

    if asymmetric:
        print(f"  {'Tag':45s} {'val/iid':>9} {'test/iid':>10}  note")
        print("  " + "-" * 80)
        for ds, sp, model, vals in asymmetric:
            tag = f"{ds}/{sp}/{model}"
            print(f"  {tag:45s} {fmt_ratio(vals['val_degradation_ratio']):>9} "
                  f"{fmt_ratio(vals['degradation_ratio']):>10}  "
                  f"val≈i.i.d., test collapses")
    else:
        print("  No strongly asymmetric cases found.")


def report_architecture_comparison(master_table):
    section("ARCHITECTURE COMPARISON — coGN vs coNGN (test MAE)")
    print("  Ratio = coNGN / coGN. < 1.0 means coNGN is better (lower MAE).")
    print("  ⚠️  jdft2d ratios may reflect split sampling variance (n=636).")
    print()

    wins_cogn = wins_cognn = ties = missing_count = 0
    rows = []

    for ds in DATASETS:
        for sp in SPLITS:
            k1 = (ds, sp, "coGN")
            k2 = (ds, sp, "coNGN")
            if k1 not in master_table or k2 not in master_table:
                missing_count += 1
                continue
            r1 = master_table[k1]["test_mae"]
            r2 = master_table[k2]["test_mae"]
            ratio   = r2 / r1
            winner  = "coNGN" if r2 < r1 else ("coGN" if r1 < r2 else "tie")
            flag    = " *" if ds in SMALL_DATASETS else ""
            if winner == "coNGN":  wins_cognn  += 1
            elif winner == "coGN": wins_cogn   += 1
            else:                  ties        += 1
            rows.append((ds, sp, r1, r2, ratio, winner, flag))

    print(f"  {'Dataset':14s} {'Split':22s} {'coGN MAE':>12} "
          f"{'coNGN MAE':>12} {'Ratio':>8}  Winner")
    print("  " + "-" * 82)
    for ds, sp, r1, r2, ratio, winner, flag in rows:
        print(f"  {ds:14s} {sp:22s} {fmt(r1):>12} {fmt(r2):>12} "
              f"{fmt_ratio(ratio):>8}  {winner}{flag}")

    print()
    print(f"  Summary: coGN wins {wins_cogn}×, coNGN wins {wins_cognn}×, "
          f"ties {ties}×, missing {missing_count}×")
    print(f"  (out of {len(rows)} head-to-head comparisons)")
    print(f"  * Small dataset — result may reflect sampling variance")


def report_split_difficulty(degradation):
    section("SPLIT TYPE DIFFICULTY RANKING")
    print("  Average OOD degradation ratio across all datasets and models.")
    print("  Higher = harder OOD challenge on average.")
    print()

    split_scores_all   = defaultdict(list)
    split_scores_large = defaultdict(list)   # excluding small datasets

    for (ds, sp, model), vals in degradation.items():
        split_scores_all[sp].append(vals["degradation_ratio"])
        if ds not in SMALL_DATASETS:
            split_scores_large[sp].append(vals["degradation_ratio"])

    ranked = sorted(split_scores_all.items(),
                    key=lambda x: sum(x[1])/len(x[1]), reverse=True)

    print(f"  {'Split':22s} {'Avg(all)':>12} {'Avg(excl jdft2d)':>18} "
          f"{'Min':>8} {'Max':>8} {'N':>4}")
    print("  " + "-" * 80)
    for sp, vals in ranked:
        avg_all   = sum(vals) / len(vals)
        large     = split_scores_large[sp]
        avg_large = sum(large) / len(large) if large else None
        print(f"  {sp:22s} {fmt_ratio(avg_all):>12} "
              f"{fmt_ratio(avg_large):>18} "
              f"{fmt_ratio(min(vals)):>8} {fmt_ratio(max(vals)):>8} "
              f"{len(vals):>4}")

    print()
    print("  Note: jdft2d excluded from 'Avg(excl jdft2d)' column because")
    print("  its small n (636) produces unreliable OOD ratios.")


def report_key_findings(master_table, degradation):
    section("KEY FINDINGS")

    # Worst OOD cases — exclude small datasets from this ranking
    all_cases    = list(degradation.items())
    large_cases  = [(k,v) for k,v in all_cases if not v["is_small_dataset"]]
    small_cases  = [(k,v) for k,v in all_cases if v["is_small_dataset"]]

    worst5 = sorted(large_cases,
                    key=lambda x: x[1]["degradation_ratio"], reverse=True)[:5]
    best5  = sorted(large_cases,
                    key=lambda x: x[1]["degradation_ratio"])[:5]

    print("\n  Top 5 worst OOD degradations (large datasets only — excludes jdft2d):")
    for (ds, sp, model), vals in worst5:
        print(f"    {ds}/{sp}/{model}: "
              f"{vals['ood_test_mae']:.4f} vs i.i.d. {vals['iid_test_mae']:.4f} "
              f"→ {fmt_ratio(vals['degradation_ratio'])}")

    print("\n  Top 5 most robust (large datasets only — excludes jdft2d):")
    for (ds, sp, model), vals in best5:
        print(f"    {ds}/{sp}/{model}: "
              f"{vals['ood_test_mae']:.4f} vs i.i.d. {vals['iid_test_mae']:.4f} "
              f"→ {fmt_ratio(vals['degradation_ratio'])}")

    print()
    print("  ⚠️  jdft2d results excluded from above rankings because sampling")
    print("  variance dominates at n=636. jdft2d ratios for reference:")
    for (ds, sp, model), vals in sorted(small_cases,
                    key=lambda x: x[1]["degradation_ratio"], reverse=True):
        print(f"    {ds}/{sp}/{model}: {fmt_ratio(vals['degradation_ratio'])}")

    # i.i.d. baselines
    print("\n  i.i.d. baselines (index split, test MAE and R²):")
    for ds in DATASETS:
        for model in MODELS:
            key = (ds, "index", model)
            if key in master_table:
                v = master_table[key]
                r2_note = "  ⚠️  R² unreliable (small n)" if ds in SMALL_DATASETS else ""
                print(f"    {ds}/{model}: MAE={v['test_mae']:.4f} {DATASET_UNITS[ds]}"
                      f"  R²={v['test_r2']:.4f}{r2_note}")

    # Training time
    print("\n  Training time summary (GPU-hours, completed runs only):")
    total_h = sum(v["train_time_h"] for v in master_table.values())
    print(f"    Total: {total_h:.1f} GPU-hours "
          f"(training phase only; excludes graph conversion overhead)")


# ============================================================================
# CSV OUTPUT
# ============================================================================

def save_csv(master_table, degradation):
    """
    Save two CSVs to OUTPUT_DIR.

    master_table.csv columns:
        dataset, split, model,
        train_mae, train_rmse, train_r2,
        val_mae,   val_rmse,   val_r2,
        test_mae,  test_rmse,  test_r2,
        n_train, n_val, n_test, epochs, train_time_h, units

    degradation_table.csv columns:
        dataset, split, model,
        iid_test_mae, iid_test_r2,
        ood_val_mae, ood_val_r2,
        ood_test_mae, ood_test_r2,
        degradation_ratio,         -- ood_test_mae / iid_test_mae
        val_degradation_ratio,     -- ood_val_mae  / iid_test_mae
        val_test_ratio,            -- ood_val_mae  / ood_test_mae
        is_small_dataset, units
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- master_table.csv ---
    master_path = OUTPUT_DIR / "master_table.csv"
    with open(master_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "dataset", "split", "model",
            "train_mae", "train_rmse", "train_r2",
            "val_mae",   "val_rmse",   "val_r2",
            "test_mae",  "test_rmse",  "test_r2",
            "n_train", "n_val", "n_test",
            "epochs", "train_time_h", "units",
        ])
        for (ds, sp, model), v in sorted(master_table.items()):
            w.writerow([
                ds, sp, model,
                v["train_mae"], v["train_rmse"], v["train_r2"],
                v["val_mae"],   v["val_rmse"],   v["val_r2"],
                v["test_mae"],  v["test_rmse"],  v["test_r2"],
                v["n_train"], v["n_val"], v["n_test"],
                v["epochs"],  round(v["train_time_h"], 4),
                DATASET_UNITS[ds],
            ])

    # --- degradation_table.csv ---
    deg_path = OUTPUT_DIR / "degradation_table.csv"
    with open(deg_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "dataset", "split", "model",
            "iid_test_mae", "iid_test_r2",
            "ood_val_mae",  "ood_val_r2",
            "ood_test_mae", "ood_test_r2",
            "degradation_ratio",
            "val_degradation_ratio",
            "val_test_ratio",
            "is_small_dataset", "units",
        ])
        for (ds, sp, model), v in sorted(degradation.items()):
            w.writerow([
                ds, sp, model,
                round(v["iid_test_mae"], 6),  round(v["iid_test_r2"], 6),
                round(v["ood_val_mae"],  6),  round(v["ood_val_r2"],  6),
                round(v["ood_test_mae"], 6),  round(v["ood_test_r2"], 6),
                round(v["degradation_ratio"],    4),
                round(v["val_degradation_ratio"],4),
                round(v["val_test_ratio"],       4),
                int(v["is_small_dataset"]),
                DATASET_UNITS[ds],
            ])

    n_master = sum(1 for _ in master_table)
    n_deg    = sum(1 for _ in degradation)
    print(f"\n  CSVs saved to {OUTPUT_DIR}/")
    print(f"    master_table.csv      — {n_master} rows "
          f"(all 112 runs, train+val+test MAE/RMSE/R²)")
    print(f"    degradation_table.csv — {n_deg} rows "
          f"(OOD splits only, degradation + val-test asymmetry)")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("Loading results...")
    results, missing = load_all_results()
    print(f"  Loaded {len(results)} completed runs, {len(missing)} missing.")

    if missing:
        print("  Missing:")
        for m in missing:
            print(f"    {m}")

    master_table = build_master_table(results)
    degradation  = build_degradation_table(master_table)

    report_summary(results, missing)
    report_master_table(master_table)
    report_degradation(degradation, master_table)
    report_val_test_asymmetry(degradation)
    report_architecture_comparison(master_table)
    report_split_difficulty(degradation)
    report_key_findings(master_table, degradation)
    save_csv(master_table, degradation)

    print()
    print("  Done.")
    if missing:
        print(f"  Re-run once the {len(missing)} missing run(s) complete.")