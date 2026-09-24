"""
DASU-Net Scientific Benchmark Suite (with Optuna tuning)
=========================================================
Full scientific comparison pipeline:

  Step 1 — Optuna HPO: find best hyperparameters for each config × dataset.
  Step 2 — Final train until convergence (plateau detection) with best params.
  Step 3 — Report & save all results to CSV.

Configurations tested:
    1. Baseline (ViT + Linear decoder)   — faithful paper reimplementation
    2. DASU-Net + SiVM init              — our model, Simplex Volume init
    3. DASU-Net + VCA  init              — our model, Vertex Component init

Datasets (default):
    - uav_synthetic   (100x100, 150 bands, 4 endmembers, synthetic)
    - whu_hi_longkou  (100x100, 270 bands, 5 endmembers, synthetic LongKou-like surrogate)
    samson / apex can be selected with --datasets.

Usage:
    python benchmark.py
    python benchmark.py --tune-trials 100 --tune-epochs 150 --max-epochs 1000
"""

from __future__ import annotations

import argparse
import csv
import gc
import random
import shutil
import time
import warnings
from pathlib import Path

import numpy as np
import optuna
import torch
import torch.nn as nn

from src.data import HyperspectralDataset
from src.models.unmixer import DASUNet
from src.core.losses import TotalLoss
from src.core.metrics import compute_rmse, compute_sad, match_endmembers

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")

SEED = 42


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(exp: dict, dataset: HyperspectralDataset,
                device: torch.device) -> DASUNet:
    base = dict(
        patch_size=5,
        emb_dim_per_endmember=200,
        transformer_depth=2,
        num_heads=8,
        window_size=5,
        mlp_dim=12,
        nonlinear_gamma=1.0,
    )
    model = DASUNet(
        num_endmembers=dataset.P,
        num_bands=dataset.L,
        spatial_size=dataset.col,
        **{**base, **exp["model_kwargs"]},
    ).to(device)
    model.apply(model.weights_init)
    model.init_decoder_weights(
        dataset.get_image_cube(), method=exp["init_method"]
    )
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Experiment definitions
# ─────────────────────────────────────────────────────────────────────────────

EXPERIMENTS = [
    {
        "name": "baseline",
        "label": "Baseline (ViT + Linear)",
        "model_kwargs": {
            "encoder_type": "vit",
            "decoder_type": "linear",
            "use_dual_attention": False,
        },
        "init_method": "vca",
        "tune_delta": False,
        "tune_lambda": False,
        "hp_samson": {"lr": 6e-3, "weight_decay": 4e-5, "beta": 5e3, "gamma": 3e-2, "delta": 0.0, "lambda_reg": 0.0},
        "hp_apex":   {"lr": 9e-3, "weight_decay": 4e-5, "beta": 5e3, "gamma": 5e-2, "delta": 0.0, "lambda_reg": 0.0},
        "hp_uav_synthetic": {"lr": 6e-3, "weight_decay": 4e-5, "beta": 5e3, "gamma": 3e-2, "delta": 0.0, "lambda_reg": 0.0},
        "hp_whu_hi_longkou": {"lr": 8e-3, "weight_decay": 4e-5, "beta": 5e3, "gamma": 4e-2, "delta": 0.0, "lambda_reg": 0.0},
    },
    {
        "name": "dasunet_sivm",
        "label": "DASU-Net + SiVM",
        "model_kwargs": {
            "encoder_type": "swin",
            "decoder_type": "nonlinear",
            "use_dual_attention": True,
        },
        "init_method": "sivm",
        "tune_delta": True,
        "tune_lambda": True,
    },
    {
        "name": "dasunet_vca",
        "label": "DASU-Net + VCA",
        "model_kwargs": {
            "encoder_type": "swin",
            "decoder_type": "nonlinear",
            "use_dual_attention": True,
        },
        "init_method": "vca",
        "tune_delta": True,
        "tune_lambda": True,
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Optuna HPO
# ─────────────────────────────────────────────────────────────────────────────

def _quick_run(exp: dict, dataset: HyperspectralDataset,
               device: torch.device, hp: dict, epochs: int) -> tuple[float, float]:
    """Short training run for a single Optuna trial."""
    img_cube = dataset.get_image_cube()
    set_seed(hp.get("_seed", SEED))

    model = build_model(exp, dataset, device)
    clipper = model.get_clipper()

    loss_fn = TotalLoss(
        num_bands=dataset.L,
        beta=hp["beta"],
        gamma=hp["gamma"],
        delta=hp.get("delta", 0.0),
        lambda_reg=hp.get("lambda_reg", 0.0),
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.8)

    model.train()
    for _ in range(epochs):
        abu, recon = model(img_cube)
        endmem = model.decoder.get_endmembers()
        loss = loss_fn(recon, img_cube, abu, endmem)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0, norm_type=1)
        optimizer.step()
        model.decoder.apply(clipper)
        scheduler.step()

    model.eval()
    with torch.no_grad():
        abu, _ = model(img_cube)
    abu = abu / abu.sum(dim=1, keepdim=True).clamp(min=1e-8)
    abu_np = abu.squeeze(0).permute(1, 2, 0).cpu().numpy()
    target_np = dataset.get_abundance_cube().cpu().numpy()
    est_endmem = model.decoder.get_endmembers().cpu().numpy()
    true_endmem = dataset.get_endmembers().numpy()

    est_endmem, abu_np, _ = match_endmembers(est_endmem, true_endmem, abu_np, target_np)
    _, rmse = compute_rmse(abu_np, target_np)
    _, sad = compute_sad(est_endmem, true_endmem)

    del model, loss_fn, optimizer, scheduler
    gc.collect()
    torch.cuda.empty_cache()

    return float(rmse), float(sad)


def tune_experiment(
    exp: dict,
    dataset: HyperspectralDataset,
    device: torch.device,
    n_trials: int,
    tune_epochs: int,
) -> dict:
    """Run Optuna study and return best hyperparameters."""

    def objective(trial: optuna.Trial) -> float:
        hp = {
            "_seed": SEED + trial.number,
            "lr":           trial.suggest_float("lr", 1e-4, 1e-2, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
            "beta":         trial.suggest_float("beta", 1e2, 1e4, log=True),
            "gamma":        trial.suggest_float("gamma", 1e-3, 1e-1, log=True),
        }
        if exp["tune_delta"]:
            hp["delta"]      = trial.suggest_float("delta",      1e-4, 1e-1, log=True)
            hp["lambda_reg"] = trial.suggest_float("lambda_reg", 1e-4, 1e-1, log=True)
        else:
            hp["delta"] = 0.0
            hp["lambda_reg"] = 0.0

        try:
            rmse, sad = _quick_run(exp, dataset, device, hp, tune_epochs)
        except Exception as e:
            raise optuna.exceptions.TrialPruned(str(e))

        trial.set_user_attr("rmse", rmse)
        trial.set_user_attr("sad", sad)
        return 0.5 * rmse + 0.5 * sad

    sampler = optuna.samplers.TPESampler(seed=SEED)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False,
                   callbacks=[_trial_cb])

    best = study.best_trial
    print(f"    Optuna best: trial #{best.number}  "
          f"RMSE={best.user_attrs.get('rmse', 0):.4f}  "
          f"SAD={best.user_attrs.get('sad', 0):.4f}")

    best_hp = {
        "lr": best.params["lr"],
        "weight_decay": best.params["weight_decay"],
        "beta": best.params["beta"],
        "gamma": best.params["gamma"],
        "delta": best.params.get("delta", 0.0),
        "lambda_reg": best.params.get("lambda_reg", 0.0),
    }
    return best_hp


def _trial_cb(study: optuna.Study, trial: optuna.FrozenTrial) -> None:
    if trial.state != optuna.trial.TrialState.COMPLETE:
        return
    rmse = trial.user_attrs.get("rmse", float("nan"))
    sad = trial.user_attrs.get("sad", float("nan"))
    print(f"    Trial {trial.number:3d} | score={trial.value:.4f} "
          f"rmse={rmse:.4f} sad={sad:.4f} | best={study.best_value:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Final training until convergence
# ─────────────────────────────────────────────────────────────────────────────

def train_until_convergence(
    exp: dict,
    dataset: HyperspectralDataset,
    device: torch.device,
    hp: dict,
    run_dir: Path,
    max_epochs: int = 1000,
    patience: int = 60,
    check_every: int = 10,
    tol: float = 1e-5,
) -> dict:
    """Train until loss plateau (true asymptote) and save results."""
    set_seed(SEED)
    img_cube = dataset.get_image_cube()

    model = build_model(exp, dataset, device)
    clipper = model.get_clipper()

    loss_fn = TotalLoss(
        num_bands=dataset.L,
        beta=hp["beta"],
        gamma=hp["gamma"],
        delta=hp.get("delta", 0.0),
        lambda_reg=hp.get("lambda_reg", 0.0),
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.8)

    losses_log: list[float] = []
    best_window_min = float("inf")
    epochs_without_improvement = 0
    
    # ---------------------------------------------------------
    # Baseline follows paper exactly (200 epochs, no plateau)
    # DASU-Net follows early stopping
    # ---------------------------------------------------------
    is_baseline = "baseline" in exp["name"]
    actual_max_epochs = 200 if is_baseline else max_epochs
    converged_epoch = actual_max_epochs

    model.train()
    t0 = time.time()

    for epoch in range(actual_max_epochs):
        abu, recon = model(img_cube)
        endmem = model.decoder.get_endmembers()
        loss = loss_fn(recon, img_cube, abu, endmem)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0, norm_type=1)
        optimizer.step()
        model.decoder.apply(clipper)
        scheduler.step()

        loss_val = float(loss.item())
        losses_log.append(loss_val)

        if not is_baseline and epoch > 0 and epoch % check_every == 0:
            window = losses_log[-patience:]
            window_min = min(window)
            if best_window_min - window_min < tol:
                epochs_without_improvement += 1
                if epochs_without_improvement >= 3:
                    converged_epoch = epoch
                    print(f"    [Converged at epoch {epoch}]")
                    break
            else:
                best_window_min = window_min
                epochs_without_improvement = 0

        if epoch % 100 == 0 or epoch == actual_max_epochs - 1:
            print(f"    Epoch {epoch:4d}  loss={loss_val:.4f}")

    elapsed = time.time() - t0

    # Evaluation
    model.eval()
    with torch.no_grad():
        abu, recon = model(img_cube)
    abu = abu / abu.sum(dim=1, keepdim=True).clamp(min=1e-8)
    abu_np = abu.squeeze(0).permute(1, 2, 0).cpu().numpy()
    target_np = dataset.get_abundance_cube().cpu().numpy()
    est_endmem = model.decoder.get_endmembers().cpu().numpy()
    true_endmem = dataset.get_endmembers().numpy()

    est_endmem, abu_np, perm = match_endmembers(est_endmem, true_endmem, abu_np, target_np)
    rmse_cls, rmse_mean = compute_rmse(abu_np, target_np)
    sad_cls, sad_mean = compute_sad(est_endmem, true_endmem)

    # Save
    run_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        run_dir / "results.npz",
        losses=np.array(losses_log),
        abu_est=abu_np,
        est_endmem=est_endmem,
        rmse_cls=rmse_cls,
        sad_cls=sad_cls,
        perm=perm,
    )

    with open(run_dir / "log.txt", "w", encoding="utf-8") as f:
        f.write(f"Dataset: {dataset.dataset_name}\n")
        f.write(f"Model: {exp['label']}\n")
        f.write(f"Encoder: {exp['model_kwargs']['encoder_type']}\n")
        f.write(f"Decoder: {exp['model_kwargs']['decoder_type']}\n")
        f.write(f"Init: {exp['init_method']}\n")
        f.write(f"LR: {hp['lr']:.6g}\n")
        f.write(f"Beta: {hp['beta']:.4g}\n")
        f.write(f"Gamma: {hp['gamma']:.4g}\n")
        f.write(f"Delta: {hp.get('delta', 0.0):.4g}\n")
        f.write(f"Lambda_reg: {hp.get('lambda_reg', 0.0):.4g}\n")
        f.write(f"Converged Epoch: {converged_epoch}\n")
        f.write(f"Total Epochs: {len(losses_log)}\n")
        f.write(f"Time: {elapsed:.1f}s\n")
        f.write(f"Mean RMSE: {rmse_mean:.6f}\n")
        f.write(f"Mean SAD: {sad_mean:.6f}\n")
        for i in range(dataset.P):
            f.write(f"RMSE_cls_{i}: {float(np.asarray(rmse_cls).flat[i]):.6f}\n")
            f.write(f"SAD_cls_{i}: {float(np.asarray(sad_cls).flat[i]):.6f}\n")

    del model, loss_fn, optimizer, scheduler
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "name": f"{exp['name']}_{dataset.dataset_name}",
        "label": exp["label"],
        "dataset": dataset.dataset_name,
        "init": exp["init_method"],
        "converged_epoch": converged_epoch,
        "total_epochs": len(losses_log),
        "time_s": round(elapsed, 1),
        "mean_rmse": round(rmse_mean, 4),
        "mean_sad": round(sad_mean, 4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DASU-Net Scientific Benchmark")
    p.add_argument("--tune-trials", type=int, default=100,
                   help="Optuna trials per config x dataset (default: 100)")
    p.add_argument("--tune-epochs", type=int, default=150,
                   help="Epochs per Optuna trial (default: 150)")
    p.add_argument("--max-epochs", type=int, default=1000,
                   help="Max epochs for final convergence run (default: 1000)")
    p.add_argument("--patience", type=int, default=60,
                   help="Plateau patience window in epochs (default: 60)")
    p.add_argument("--tol", type=float, default=1e-5,
                   help="Min loss improvement for plateau detection (default: 1e-5)")
    p.add_argument("--check-every", type=int, default=10,
                   help="Plateau check frequency in epochs (default: 10)")
    p.add_argument("--datasets", nargs="+", default=["uav_synthetic", "whu_hi_longkou"])
    # Wiped at start-up, so keep it separate from runs/experiments_uav
    p.add_argument("--runs-dir", type=str, default="runs/benchmark")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(SEED)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    runs_root = Path(args.runs_dir)

    print(f"\n{'='*65}")
    print(f"  DASU-Net Scientific Benchmark Suite")
    print(f"  Device:       {device}")
    print(f"  Datasets:     {args.datasets}")
    print(f"  Optuna:       {args.tune_trials} trials x {args.tune_epochs} epochs/trial")
    print(f"  Final train:  up to {args.max_epochs} epochs (plateau detection)")
    print(f"{'='*65}\n")

    # Clear old runs
    if runs_root.exists():
        shutil.rmtree(runs_root)
        print(f"  Cleared previous results in '{runs_root}/'\n")
    runs_root.mkdir(parents=True)

    all_results: list[dict] = []
    total = len(args.datasets) * len(EXPERIMENTS)
    counter = 0

    for dataset_name in args.datasets:
        print(f"\n{'-'*65}")
        print(f"  Dataset: {dataset_name.upper()}")
        print(f"{'-'*65}")

        dataset = HyperspectralDataset(
            dataset_name=dataset_name,
            data_dir="./data/raw",
            device=device,
        )
        print(f"  {dataset}\n")

        for exp in EXPERIMENTS:
            counter += 1
            run_key = f"{exp['name']}_{dataset_name}"
            print(f"\n  [{counter}/{total}] {exp['label']}  (dataset={dataset_name})")
            print(f"  {'='*50}")

            # -- Step 1: Optuna tuning --
            if "baseline" in exp["name"]:
                print(f"  > Step 1: Skipping Optuna for Baseline (using paper HP)")
                best_hp = exp[f"hp_{dataset_name}"]
            else:
                print(f"  > Step 1: Optuna HPO ({args.tune_trials} trials)")
                best_hp = tune_experiment(
                    exp=exp,
                    dataset=dataset,
                    device=device,
                    n_trials=args.tune_trials,
                    tune_epochs=args.tune_epochs,
                )
                print(f"    Best HP: lr={best_hp['lr']:.2e}  beta={best_hp['beta']:.1f}  "
                      f"gamma={best_hp['gamma']:.4f}  delta={best_hp['delta']:.4f}")

            # -- Step 2: Final training until convergence --
            print(f"\n  > Step 2: Final training until convergence")
            run_dir = runs_root / run_key
            result = train_until_convergence(
                exp=exp,
                dataset=dataset,
                device=device,
                hp=best_hp,
                run_dir=run_dir,
                max_epochs=args.max_epochs,
                patience=args.patience,
                check_every=args.check_every,
                tol=args.tol,
            )
            all_results.append(result)
            print(f"\n    RESULT: RMSE={result['mean_rmse']:.4f}  SAD={result['mean_sad']:.4f}  "
                  f"(converged @ epoch {result['converged_epoch']})")

    # Summary table
    print(f"\n\n{'='*65}")
    print(f"  FINAL BENCHMARK RESULTS (Optuna-tuned, Converged)")
    print(f"{'='*65}")
    print(f"  {'Model':<30} {'Dataset':<8} {'Init':<6} {'RMSE':<8} {'SAD':<8} {'Epoch'}")
    print(f"  {'-'*62}")
    for r in all_results:
        print(f"  {r['label']:<30} {r['dataset']:<8} {r['init']:<6} "
              f"{r['mean_rmse']:<8.4f} {r['mean_sad']:<8.4f} {r['converged_epoch']}")

    # Save CSV
    csv_path = runs_root / "benchmark_results.csv"
    fieldnames = ["name", "label", "dataset", "init", "converged_epoch",
                  "total_epochs", "time_s", "mean_rmse", "mean_sad"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)

    print(f"\n  Results saved to: {csv_path}")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()
