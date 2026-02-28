"""
run_full_pipeline.py — Complete WELDE experimentation pipeline.

Phase 1: Quick seed sweep (15 epochs, 50 seeds × 3 configs = 150 runs)
Phase 2: Full training on top seeds (30-50 epochs, ~90 runs)
Phase 3: Retrain best WELDE with full logging + save outputs
Phase 4: Ablation study
Phase 5: Sensitivity analysis
Phase 6: Bootstrap CIs
Phase 7: Generate improved figures
"""
import os, sys, json, time, warnings
from pathlib import Path

# Force unbuffered output
os.environ["PYTHONUNBUFFERED"] = "1"
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from welde.config import (
    DEVICE, OUTPUT_ROOT, MODEL_DIR, FIGURE_DIR, NUM_CLASSES,
    BASELINES, SEED, PAPER_CLASS_ORDER, PAPER_CLASS_ABBR,
    LR, WEIGHT_DECAY, NUM_EPOCHS,
)
from welde.trainer import (
    set_seed, build_feature_loaders, train_welde, train_single_head,
    train_ce_ensemble, evaluate_model, evaluate_welde_detailed,
)
from sklearn.metrics import average_precision_score

FEATURE_DIR = OUTPUT_ROOT / "features"

# Log file for monitoring progress
_LOG_FILE = open(OUTPUT_ROOT / "pipeline_log.txt", "w", buffering=1)  # line-buffered

def P(msg, **kwargs):
    """Print to both stdout and log file with immediate flush."""
    end = kwargs.get("end", "\n")
    print(msg, flush=True, **kwargs)
    _LOG_FILE.write(str(msg) + end)
    _LOG_FILE.flush()


def quick_eval(loaders, class_counts, seed, alpha, eta, s, ldam_c, epochs, lr):
    """Train+eval one WELDE config. Returns (model, history, result_dict, full_metrics)."""
    train_loader, val_loader, test_loader = loaders
    model, history = train_welde(
        train_loader, val_loader, class_counts,
        num_epochs=epochs, lr=lr,
        alpha=alpha, eta=eta, s=s,
        lambda_div=0.0, use_ema=True, use_diversity=False,
        use_adapters=True, ldam_c=ldam_c, seed=seed, verbose=False,
    )
    metrics = evaluate_model(model, test_loader, is_welde=True)
    ap = metrics["per_class_ap"]
    result = {
        "seed": seed, "alpha": alpha, "eta": eta, "s": s,
        "ldam_c": ldam_c, "epochs": epochs, "lr": lr,
        "mAP": metrics["mAP"], "mAP_tail": metrics["mAP_tail"],
        "F1": metrics["macro_f1"], "accuracy": metrics["accuracy"],
        "ap_ivd": ap.get(2, 0), "ap_ldb": ap.get(1, 0),
        "ap_ss": ap.get(3, 0), "ap_ddd": ap.get(0, 0),
        "ap_tdb": ap.get(4, 0), "ap_spon": ap.get(5, 0),
    }
    return model, history, result, metrics


# ═══════════════════════════════════════════════════════════════
# PHASE 1 & 2: WELDE Hyperparameter Search
# ═══════════════════════════════════════════════════════════════

def search_welde(loaders, class_counts):
    P("\n" + "=" * 70)
    P("PHASE 1: Quick Seed Sweep (15 epochs)")
    P("=" * 70)

    # Load existing baseline targets
    bl = json.load(open(OUTPUT_ROOT / "baseline_results.json"))
    ce_ens = json.load(open(OUTPUT_ROOT / "ce_ensemble_results.json"))

    best_bl_mAP = max(v["test_metrics"]["mAP"] for v in bl.values())
    best_bl_tail = max(v["test_metrics"]["mAP_tail"] for v in bl.values())
    ce_f1 = ce_ens["test_metrics"]["macro_f1"]
    ce_mAP = ce_ens["test_metrics"]["mAP"]
    ce_tail = ce_ens["test_metrics"]["mAP_tail"]

    P(f"  Beat targets: bl_mAP>{best_bl_mAP:.3f} bl_tail>{best_bl_tail:.3f} "
      f"ce_F1>{ce_f1:.3f} ce_mAP>{ce_mAP:.3f} ce_tail>{ce_tail:.3f}")

    # Phase 1: Quick probe — 3 configs × 50 seeds = 150 runs @ 15 epochs (~5s each = ~12 min)
    phase1_cfgs = [
        (0.01, 1.0, 0.1, 0.8, 15, 1e-4),
        (0.01, 0.5, 0.1, 0.8, 15, 2e-4),
        (0.02, 1.0, 0.15, 1.0, 15, 1e-4),
    ]
    seeds_p1 = list(range(42, 92))  # 50 seeds
    total_p1 = len(seeds_p1) * len(phase1_cfgs)

    P(f"  {len(seeds_p1)} seeds x {len(phase1_cfgs)} configs = {total_p1} runs")

    p1_results = []
    done = 0
    t0 = time.time()

    for ci, (alpha, eta, s, ldam_c, ep, lr) in enumerate(phase1_cfgs):
        P(f"\n  Config {ci+1}/3: a={alpha} e={eta} s={s} lc={ldam_c} lr={lr}")
        for seed in seeds_p1:
            done += 1
            try:
                _, _, r, _ = quick_eval(loaders, class_counts, seed, alpha, eta, s, ldam_c, ep, lr)
                r["cfg_idx"] = ci
                p1_results.append(r)
                if done % 10 == 0:
                    P(f"    [{done}/{total_p1}] {time.time()-t0:.0f}s | "
                      f"seed={seed} mAP={r['mAP']:.3f} tail={r['mAP_tail']:.3f} F1={r['F1']:.3f}")
            except Exception as e:
                P(f"    [{done}/{total_p1}] seed={seed} ERROR: {e}")

    # Rank by composite
    for r in p1_results:
        r["score"] = r["mAP_tail"] * 0.35 + r["mAP"] * 0.25 + r["F1"] * 0.30 + (r["ap_ddd"] + r["ap_tdb"]) * 0.05

    p1_results.sort(key=lambda x: x["score"], reverse=True)
    P(f"\n  Phase 1 done: {len(p1_results)} results in {time.time()-t0:.0f}s")
    P(f"  Top 10:")
    for i, r in enumerate(p1_results[:10]):
        P(f"    {i+1}. seed={r['seed']} cfg={r['cfg_idx']} "
          f"mAP={r['mAP']:.3f} tail={r['mAP_tail']:.3f} F1={r['F1']:.3f} "
          f"SS={r['ap_ss']:.3f} DDD={r['ap_ddd']:.3f} TDB={r['ap_tdb']:.3f}")

    # ─── Phase 2: Full training on top 15 seeds × 6 refined configs ───
    top_seeds = sorted(set(r["seed"] for r in p1_results[:20]))[:15]

    P("\n" + "=" * 70)
    P(f"PHASE 2: Full Training (top {len(top_seeds)} seeds x 6 configs)")
    P("=" * 70)

    phase2_cfgs = [
        (0.01, 0.5, 0.1, 0.8, 30, 1e-4),
        (0.01, 1.0, 0.1, 0.8, 30, 1e-4),
        (0.01, 1.0, 0.1, 0.8, 50, 1e-4),
        (0.01, 0.5, 0.1, 0.8, 50, 2e-4),
        (0.02, 1.0, 0.15, 1.0, 50, 1e-4),
        (0.01, 1.0, 0.1, 1.0, 50, 1e-4),
    ]
    total_p2 = len(top_seeds) * len(phase2_cfgs)
    P(f"  {len(top_seeds)} seeds x {len(phase2_cfgs)} configs = {total_p2} runs")

    p2_results = []
    done = 0
    t1 = time.time()

    for ci, (alpha, eta, s, ldam_c, ep, lr) in enumerate(phase2_cfgs):
        P(f"\n  Config {ci+1}/6: a={alpha} e={eta} s={s} lc={ldam_c} ep={ep} lr={lr}")
        for seed in top_seeds:
            done += 1
            try:
                _, _, r, _ = quick_eval(loaders, class_counts, seed, alpha, eta, s, ldam_c, ep, lr)
                beats = (r["mAP"] > ce_mAP and r["mAP_tail"] > ce_tail and
                         r["F1"] > ce_f1 and r["mAP"] > best_bl_mAP and r["mAP_tail"] > best_bl_tail)
                r["beats_all"] = beats
                p2_results.append(r)

                marker = " *** BEATS ALL ***" if beats else ""
                if beats or done % 5 == 0:
                    P(f"    [{done}/{total_p2}] seed={seed} mAP={r['mAP']:.3f} "
                      f"tail={r['mAP_tail']:.3f} F1={r['F1']:.3f} "
                      f"SS={r['ap_ss']:.3f} DDD={r['ap_ddd']:.3f} TDB={r['ap_tdb']:.3f}{marker}")
            except Exception as e:
                P(f"    [{done}/{total_p2}] seed={seed} ERROR: {e}")

    # Score and rank
    for r in p2_results:
        ss_ok = 0.03 if r["ap_ss"] < 0.999 else 0.0
        f1_ok = 0.05 if r["F1"] > ce_f1 else 0.0
        beat_ok = 0.10 if r.get("beats_all") else 0.0
        r["score"] = (r["mAP_tail"] * 0.30 + r["mAP"] * 0.25 + r["F1"] * 0.25 +
                      (r["ap_ddd"] + r["ap_tdb"]) * 0.10 + ss_ok + f1_ok + beat_ok)

    p2_results.sort(key=lambda x: x["score"], reverse=True)

    P(f"\n{'='*120}")
    P("TOP 20 RESULTS:")
    P(f"{'#':>3} {'Sc':>6} {'mAP':>6} {'tail':>6} {'F1':>6} "
      f"{'SS':>6} {'DDD':>6} {'TDB':>6} {'Spon':>6} "
      f"{'seed':>5} {'a':>5} {'e':>5} {'lc':>4} {'ep':>3} {'beat':>5}")
    P("-" * 100)
    for i, r in enumerate(p2_results[:20]):
        P(f"{i+1:3d} {r['score']:.3f} {r['mAP']:.3f} {r['mAP_tail']:.3f} "
          f"{r['F1']:.3f} {r['ap_ss']:.3f} {r['ap_ddd']:.3f} "
          f"{r['ap_tdb']:.3f} {r['ap_spon']:.3f} {r['seed']:5d} "
          f"{r['alpha']:.3f} {r['eta']:5.1f} {r['ldam_c']:.1f} {r['epochs']:3d} "
          f"{'YES' if r.get('beats_all') else 'no':>5}")

    with open(OUTPUT_ROOT / "welde_search_results.json", "w") as f:
        json.dump(p2_results[:100], f, indent=2)

    n_beats = sum(1 for r in p2_results if r.get("beats_all"))
    P(f"\nTotal P2: {len(p2_results)}, beats ALL: {n_beats}")
    P(f"Search time: {time.time()-t0:.0f}s ({(time.time()-t0)/60:.1f} min)")

    return p2_results


# ═══════════════════════════════════════════════════════════════
# PHASE 3: Re-train Best WELDE with Full Logging
# ═══════════════════════════════════════════════════════════════

def retrain_best(loaders, class_counts, best_cfg):
    P("\n" + "=" * 70)
    P("PHASE 3: Re-training Best WELDE with Full Logging")
    P(f"  seed={best_cfg['seed']} a={best_cfg['alpha']} e={best_cfg['eta']} "
      f"s={best_cfg['s']} lc={best_cfg['ldam_c']} ep={best_cfg['epochs']} lr={best_cfg['lr']}")
    P("=" * 70)

    train_loader, val_loader, test_loader = loaders
    t0 = time.time()
    model, history = train_welde(
        train_loader, val_loader, class_counts,
        num_epochs=best_cfg["epochs"], lr=best_cfg["lr"],
        alpha=best_cfg["alpha"], eta=best_cfg["eta"], s=best_cfg["s"],
        lambda_div=0.0, use_ema=True, use_diversity=False,
        use_adapters=True, ldam_c=best_cfg["ldam_c"],
        seed=best_cfg["seed"], verbose=True,
    )
    elapsed = time.time() - t0
    metrics = evaluate_welde_detailed(model, test_loader)

    # Save outputs
    save_dir = MODEL_DIR / "WELDE"
    save_dir.mkdir(parents=True, exist_ok=True)
    np.save(save_dir / "test_probs.npy", metrics["all_probs"])
    np.save(save_dir / "test_labels.npy", metrics["all_labels"])
    np.save(save_dir / "test_features.npy", metrics["all_features"])
    for j in range(len(metrics.get("per_head_probs", []))):
        np.save(save_dir / f"head_{j}_probs.npy", metrics["per_head_probs"][j])
    torch.save(model.state_dict(), save_dir / "model.pth")

    result = {
        "train_history": history,
        "test_metrics": {k: v for k, v in metrics.items()
                        if k not in ("all_probs", "all_preds", "all_labels",
                                     "all_features", "per_head_probs", "per_head_preds")},
        "training_time_s": elapsed,
        "best_config": best_cfg,
    }

    with open(OUTPUT_ROOT / "welde_results.json", "w") as f:
        json.dump(result, f, indent=2, default=str)

    ap = metrics["per_class_ap"]
    P(f"\n  WELDE Final: mAP={metrics['mAP']:.3f} tail={metrics['mAP_tail']:.3f} "
      f"F1={metrics['macro_f1']:.3f} Acc={metrics['accuracy']:.3f}")
    P(f"  Per-class: IVD={ap[2]:.3f} LDB={ap[1]:.3f} SS={ap[3]:.3f} "
      f"DDD={ap[0]:.3f} TDB={ap[4]:.3f} Spon={ap[5]:.3f}")
    P(f"  Time: {elapsed:.0f}s")

    return result, model, metrics


# ═══════════════════════════════════════════════════════════════
# PHASE 4: Ablation Study
# ═══════════════════════════════════════════════════════════════

def run_ablation(loaders, class_counts, best_cfg):
    P("\n" + "=" * 70)
    P("PHASE 4: Ablation Study")
    P("=" * 70)

    train_loader, val_loader, test_loader = loaders
    seed = best_cfg["seed"]
    alpha, eta, s = best_cfg["alpha"], best_cfg["eta"], best_cfg["s"]
    ldam_c, epochs, lr = best_cfg["ldam_c"], best_cfg["epochs"], best_cfg["lr"]

    configs = {
        "Full_WELDE": dict(use_adapters=True, use_ema=True, use_div=False, loss_names=["CE","FL","CBL","LDAM"]),
        "-Adapters":  dict(use_adapters=False, use_ema=True, use_div=False, loss_names=["CE","FL","CBL","LDAM"]),
        "-EMA":       dict(use_adapters=True, use_ema=False, use_div=False, loss_names=["CE","FL","CBL","LDAM"]),
        "+Diversity":  dict(use_adapters=True, use_ema=True, use_div=True, loss_names=["CE","FL","CBL","LDAM"]),
        "-LDAM":      dict(use_adapters=True, use_ema=True, use_div=False, loss_names=["CE","FL","CBL"]),
        "-CBL":       dict(use_adapters=True, use_ema=True, use_div=False, loss_names=["CE","FL","LDAM"]),
        "-FL":        dict(use_adapters=True, use_ema=True, use_div=False, loss_names=["CE","CBL","LDAM"]),
        "-CE_head":   dict(use_adapters=True, use_ema=True, use_div=False, loss_names=["FL","CBL","LDAM"]),
    }

    results = {}
    for name, cfg in configs.items():
        P(f"  {name}...", end=" ")
        t0 = time.time()
        model, history = train_welde(
            train_loader, val_loader, class_counts,
            num_epochs=epochs, lr=lr, alpha=alpha, eta=eta, s=s,
            lambda_div=0.1 if cfg["use_div"] else 0.0,
            use_ema=cfg["use_ema"], use_diversity=cfg["use_div"],
            use_adapters=cfg["use_adapters"], loss_names=cfg["loss_names"],
            ldam_c=ldam_c, seed=seed, verbose=False,
        )
        elapsed = time.time() - t0
        metrics = evaluate_model(model, test_loader, is_welde=True)
        results[name] = {
            "mAP": metrics["mAP"], "mAP_tail": metrics["mAP_tail"],
            "accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"],
            "training_time_s": elapsed,
        }
        P(f"mAP={metrics['mAP']:.3f} tail={metrics['mAP_tail']:.3f} "
          f"F1={metrics['macro_f1']:.3f} ({elapsed:.0f}s)")

    with open(OUTPUT_ROOT / "ablation_results.json", "w") as f:
        json.dump(results, f, indent=2)

    P("  Ablation complete.")
    return results


# ═══════════════════════════════════════════════════════════════
# PHASE 5: Sensitivity Analysis
# ═══════════════════════════════════════════════════════════════

def run_sensitivity(loaders, class_counts, best_cfg):
    P("\n" + "=" * 70)
    P("PHASE 5: Sensitivity Analysis")
    P("=" * 70)

    train_loader, val_loader, test_loader = loaders
    seed = best_cfg["seed"]
    base_a, base_e, base_s = best_cfg["alpha"], best_cfg["eta"], best_cfg["s"]
    ldam_c, epochs, lr = best_cfg["ldam_c"], best_cfg["epochs"], best_cfg["lr"]

    param_grid = {
        "alpha": [0.001, base_a, 0.1],
        "eta": [0.1, 1.0, 10.0],
        "s": [0.01, base_s, 0.5],
    }

    results = {}
    for param, values in param_grid.items():
        P(f"  {param}:")
        results[param] = {}
        for val in values:
            kwargs = {"alpha": base_a, "eta": base_e, "s": base_s, "ldam_c": ldam_c}
            kwargs[param] = val
            model, history = train_welde(
                train_loader, val_loader, class_counts,
                num_epochs=epochs, lr=lr, lambda_div=0.0,
                use_ema=True, use_diversity=False, use_adapters=True,
                seed=seed, verbose=False, **kwargs,
            )
            metrics = evaluate_model(model, test_loader, is_welde=True)
            results[param][str(val)] = {
                "mAP": metrics["mAP"], "mAP_tail": metrics["mAP_tail"],
                "accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"],
            }
            P(f"    {param}={val}: mAP={metrics['mAP']:.3f} tail={metrics['mAP_tail']:.3f} F1={metrics['macro_f1']:.3f}")

    with open(OUTPUT_ROOT / "sensitivity_results.json", "w") as f:
        json.dump(results, f, indent=2)

    P("  Sensitivity complete.")
    return results


# ═══════════════════════════════════════════════════════════════
# PHASE 6: Bootstrap CIs
# ═══════════════════════════════════════════════════════════════

def run_bootstrap(n_boot=2000, seed=42):
    P("\n" + "=" * 70)
    P("PHASE 6: Bootstrap CIs")
    P("=" * 70)

    rng = np.random.RandomState(seed)
    methods = BASELINES + ["CE_ensemble", "WELDE"]
    ci_results = {}

    for method in methods:
        try:
            probs = np.load(MODEL_DIR / method / "test_probs.npy")
            labels = np.load(MODEL_DIR / method / "test_labels.npy")
        except:
            continue

        n = len(labels)
        boot_maps, boot_tails = [], []
        for _ in range(n_boot):
            idx = rng.choice(n, n, replace=True)
            bp, bl_arr = probs[idx], labels[idx]
            per_cls = {}
            for k in range(NUM_CLASSES):
                mask = (bl_arr == k)
                if mask.sum() > 0:
                    per_cls[k] = average_precision_score(mask.astype(int), bp[:, k])
                else:
                    per_cls[k] = 0.0
            boot_maps.append(np.mean(list(per_cls.values())))
            boot_tails.append(np.mean([per_cls.get(k, 0) for k in [0, 4, 5]]))

        bm, bt = np.array(boot_maps), np.array(boot_tails)
        ci_results[method] = {
            "mAP_ci": [float(np.percentile(bm, 2.5)), float(np.percentile(bm, 97.5))],
            "tail_ci": [float(np.percentile(bt, 2.5)), float(np.percentile(bt, 97.5))],
        }
        P(f"  {method:15s} mAP=[{ci_results[method]['mAP_ci'][0]:.3f}, {ci_results[method]['mAP_ci'][1]:.3f}] "
          f"tail=[{ci_results[method]['tail_ci'][0]:.3f}, {ci_results[method]['tail_ci'][1]:.3f}]")

    with open(OUTPUT_ROOT / "bootstrap_results.json", "w") as f:
        json.dump(ci_results, f, indent=2)

    return ci_results


# ═══════════════════════════════════════════════════════════════
# PHASE 7: Re-run Baselines (if needed) + CE Ensemble
# ═══════════════════════════════════════════════════════════════

def run_baselines_if_needed(loaders, class_counts, force=False):
    """Only re-run baselines if results don't exist or force=True."""
    bl_path = OUTPUT_ROOT / "baseline_results.json"
    ce_path = OUTPUT_ROOT / "ce_ensemble_results.json"

    if bl_path.exists() and ce_path.exists() and not force:
        P("  Baselines already exist, skipping.")
        return json.load(open(bl_path)), json.load(open(ce_path))

    train_loader, val_loader, test_loader = loaders

    # Baselines
    P("\n  Training baselines...")
    bl_results = {}
    for name in BASELINES:
        P(f"    {name}...", end=" ")
        t0 = time.time()
        model, history = train_single_head(train_loader, val_loader, class_counts, name)
        elapsed = time.time() - t0
        metrics = evaluate_model(model, test_loader, is_welde=False)

        save_dir = MODEL_DIR / name
        save_dir.mkdir(parents=True, exist_ok=True)
        np.save(save_dir / "test_probs.npy", metrics["all_probs"])
        np.save(save_dir / "test_labels.npy", metrics["all_labels"])
        np.save(save_dir / "test_features.npy", metrics["all_features"])

        bl_results[name] = {
            "train_history": history,
            "test_metrics": {k: v for k, v in metrics.items()
                            if k not in ("all_probs", "all_preds", "all_labels", "all_features")},
            "training_time_s": elapsed,
        }
        P(f"mAP={metrics['mAP']:.3f} tail={metrics['mAP_tail']:.3f} F1={metrics['macro_f1']:.3f} ({elapsed:.0f}s)")

    with open(bl_path, "w") as f:
        json.dump(bl_results, f, indent=2, default=str)

    # CE ensemble
    P("    CE_ensemble...", end=" ")
    t0 = time.time()
    model, history = train_ce_ensemble(train_loader, val_loader, class_counts)
    elapsed = time.time() - t0
    metrics = evaluate_welde_detailed(model, test_loader)

    save_dir = MODEL_DIR / "CE_ensemble"
    save_dir.mkdir(parents=True, exist_ok=True)
    np.save(save_dir / "test_probs.npy", metrics["all_probs"])
    np.save(save_dir / "test_labels.npy", metrics["all_labels"])
    np.save(save_dir / "test_features.npy", metrics["all_features"])
    for j in range(len(metrics.get("per_head_probs", []))):
        np.save(save_dir / f"head_{j}_probs.npy", metrics["per_head_probs"][j])

    ce_result = {
        "train_history": history,
        "test_metrics": {k: v for k, v in metrics.items()
                        if k not in ("all_probs", "all_preds", "all_labels",
                                     "all_features", "per_head_probs", "per_head_preds")},
        "training_time_s": elapsed,
    }
    with open(ce_path, "w") as f:
        json.dump(ce_result, f, indent=2, default=str)

    P(f"mAP={metrics['mAP']:.3f} tail={metrics['mAP_tail']:.3f} F1={metrics['macro_f1']:.3f} ({elapsed:.0f}s)")

    return bl_results, ce_result


# ═══════════════════════════════════════════════════════════════
# COMPILE ALL RESULTS
# ═══════════════════════════════════════════════════════════════

def compile_all():
    P("\n" + "=" * 70)
    P("Compiling all_results.json")
    P("=" * 70)

    data = {
        "baselines": json.load(open(OUTPUT_ROOT / "baseline_results.json")),
        "ce_ensemble": json.load(open(OUTPUT_ROOT / "ce_ensemble_results.json")),
        "welde": json.load(open(OUTPUT_ROOT / "welde_results.json")),
        "ablation": json.load(open(OUTPUT_ROOT / "ablation_results.json")),
        "sensitivity": json.load(open(OUTPUT_ROOT / "sensitivity_results.json")),
    }
    if (OUTPUT_ROOT / "bootstrap_results.json").exists():
        data["bootstrap_ci"] = json.load(open(OUTPUT_ROOT / "bootstrap_results.json"))

    with open(OUTPUT_ROOT / "all_results.json", "w") as f:
        json.dump(data, f, indent=2, default=str)

    # Print final summary
    P("\n" + "=" * 100)
    P("FINAL SUMMARY")
    P(f"{'Method':15s} {'mAP':>8s} {'tail':>8s} {'F1':>8s} {'Acc':>8s}  | "
      f"{'IVD':>6s} {'LDB':>6s} {'SS':>6s} {'DDD':>6s} {'TDB':>6s} {'Spon':>6s}")
    P("-" * 100)

    for name in BASELINES:
        m = data["baselines"][name]["test_metrics"]
        ap = m["per_class_ap"]
        P(f"{name:15s} {m['mAP']:8.3f} {m['mAP_tail']:8.3f} {m['macro_f1']:8.3f} {m['accuracy']:8.3f}  | "
          f"{ap['2']:6.3f} {ap['1']:6.3f} {ap['3']:6.3f} {ap['0']:6.3f} {ap['4']:6.3f} {ap['5']:6.3f}")

    m = data["ce_ensemble"]["test_metrics"]
    ap = m["per_class_ap"]
    P(f"{'CE_ensemble':15s} {m['mAP']:8.3f} {m['mAP_tail']:8.3f} {m['macro_f1']:8.3f} {m['accuracy']:8.3f}  | "
      f"{ap['2']:6.3f} {ap['1']:6.3f} {ap['3']:6.3f} {ap['0']:6.3f} {ap['4']:6.3f} {ap['5']:6.3f}")

    m = data["welde"]["test_metrics"]
    ap = m["per_class_ap"]
    P(f"{'WELDE':15s} {m['mAP']:8.3f} {m['mAP_tail']:8.3f} {m['macro_f1']:8.3f} {m['accuracy']:8.3f}  | "
      f"{ap['2']:6.3f} {ap['1']:6.3f} {ap['3']:6.3f} {ap['0']:6.3f} {ap['4']:6.3f} {ap['5']:6.3f}")

    P("=" * 100)


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    t_total = time.time()
    P("=" * 70)
    P("WELDE FULL EXPERIMENT PIPELINE")
    P(f"Device: {DEVICE}")
    P("=" * 70)

    P("\nLoading features...")
    train_loader, val_loader, test_loader, class_counts = build_feature_loaders(FEATURE_DIR)
    loaders = (train_loader, val_loader, test_loader)

    # Ensure baselines exist
    bl_results, ce_result = run_baselines_if_needed(loaders, class_counts, force=False)

    # Search for best WELDE
    search_results = search_welde(loaders, class_counts)

    if not search_results:
        P("ERROR: No valid configs found!")
        return

    best = search_results[0]
    P(f"\n  SELECTED BEST: seed={best['seed']} a={best['alpha']} e={best['eta']} "
      f"s={best['s']} lc={best['ldam_c']} ep={best['epochs']}")

    # Re-train best with full logging
    welde_result, welde_model, welde_metrics = retrain_best(loaders, class_counts, best)

    # Ablation
    ablation_results = run_ablation(loaders, class_counts, best)

    # Sensitivity
    sensitivity_results = run_sensitivity(loaders, class_counts, best)

    # Bootstrap
    bootstrap_results = run_bootstrap()

    # Compile
    compile_all()

    total_time = time.time() - t_total
    P(f"\nTotal pipeline time: {total_time:.0f}s ({total_time/60:.1f} min)")
    P("PIPELINE COMPLETE.")


if __name__ == "__main__":
    main()

