"""
DASU-Net — Hyperparameter tuning with Optuna.

Usage:
    python tune.py                          # 50 trials, Samson, Swin
    python tune.py --n-trials 20            # quick run
    python tune.py --dataset apex           # different dataset
    python tune.py --study-name my_study    # resume named study

Results are saved to:
    configs/best_<dataset>_<study>.yaml     — best hyperparameters
    optuna_<study>.db                       — SQLite study for resuming
"""

from __future__ import annotations

import argparse
import gc
import logging
import random
import time
import warnings
from pathlib import Path

import numpy as np
import optuna
import torch
import torch.nn as nn
import yaml

from src.data import HyperspectralDataset
from src.models.unmixer import DASUNet
from src.core.losses import TotalLoss
from src.core.metrics import compute_rmse, compute_sad, match_endmembers
from src.utils.config_parser import load_config

# Suppress verbose Optuna / PyTorch output
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("torch").setLevel(logging.ERROR)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _quick_train(
    cfg: dict,
    dataset: HyperspectralDataset,
    device: torch.device,
    epochs: int,
    hp: dict,
) -> tuple[float, float]:
    """
    Run a short training loop with given hyperparameters.

    Returns
    -------
    (mean_rmse, mean_sad) — evaluated on the training image (no GT
    split — standard for unsupervised HSU evaluation).
    """
    m_cfg = cfg["model"]
    img_cube = dataset.get_image_cube()   # (1, L, H, W)

    # ---- Model ---------------------------------------------------
    model = DASUNet(
        num_endmembers=dataset.P,
        num_bands=dataset.L,
        spatial_size=dataset.col,
        encoder_type="swin",
        decoder_type="nonlinear",
        use_dual_attention=True,
        patch_size=m_cfg["patch_size"],
        emb_dim_per_endmember=m_cfg["emb_dim_per_endmember"],
        transformer_depth=m_cfg["transformer_depth"],
        num_heads=m_cfg["num_heads"],
        window_size=m_cfg["window_size"],
        mlp_dim=m_cfg["mlp_dim"],
        nonlinear_gamma=m_cfg.get("nonlinear_gamma", 1.0),
    ).to(device)

    model.apply(model.weights_init)
    model.init_decoder_weights(img_cube, method=m_cfg.get("init_method", "sivm"))
    clipper = model.get_clipper()

    # ---- Loss & Optimizer ----------------------------------------
    loss_fn = TotalLoss(
        num_bands=dataset.L,
        beta=hp["beta"],
        gamma=hp["gamma"],
        delta=hp["delta"],
        lambda_reg=hp["lambda_reg"],
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=hp["learning_rate"],
        weight_decay=hp.get("weight_decay", 4e-5),
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=15, gamma=0.8
    )

    # ---- Training loop -------------------------------------------
    model.train()
    for _ in range(epochs):
        abu_est, recon = model(img_cube)
        endmem = model.decoder.get_endmembers()
        loss = loss_fn(recon, img_cube, abu_est, endmem)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0, norm_type=1)
        optimizer.step()
        model.decoder.apply(clipper)
        scheduler.step()

    # ---- Evaluation ----------------------------------------------
    model.eval()
    with torch.no_grad():
        abu_est, _ = model(img_cube)

    abu_est = abu_est / abu_est.sum(dim=1, keepdim=True).clamp(min=1e-8)
    abu_np = abu_est.squeeze(0).permute(1, 2, 0).cpu().numpy()
    target_np = dataset.get_abundance_cube().cpu().numpy()
    est_endmem = model.decoder.get_endmembers().cpu().numpy()
    true_endmem = dataset.get_endmembers().numpy()

    est_endmem, abu_np, _ = match_endmembers(
        est_endmem, true_endmem, abu_np, target_np
    )
    _, rmse = compute_rmse(abu_np, target_np)
    _, sad = compute_sad(est_endmem, true_endmem)

    # Free GPU memory
    del model, loss_fn, optimizer, scheduler
    gc.collect()
    torch.cuda.empty_cache()

    return float(rmse), float(sad)


# ------------------------------------------------------------------
# Objective
# ------------------------------------------------------------------

def build_objective(cfg: dict, dataset: HyperspectralDataset,
                    device: torch.device, epochs: int, seed: int):
    """Closure over shared data so objective is pickleable."""

    def objective(trial: optuna.Trial) -> float:
        """
        Optuna objective — minimise alpha*RMSE + (1-alpha)*SAD.

        Sampled hyperparameters
        -----------------------
        learning_rate : log-uniform in [1e-4, 1e-2]
        beta          : log-uniform in [1e2, 1e4]   (MSE weight)
        gamma         : log-uniform in [1e-3, 1e-1]  (SAD weight)
        delta         : log-uniform in [1e-4, 1e-1]  (MinVol weight)
        lambda_reg    : log-uniform in [1e-4, 1e-1]  (Sparsity weight)
        """
        set_seed(seed + trial.number)

        hp = {
            "learning_rate": trial.suggest_float(
                "learning_rate", 1e-4, 1e-2, log=True),
            "beta":          trial.suggest_float(
                "beta",          1e2,  1e4,  log=True),
            "gamma":         trial.suggest_float(
                "gamma",         1e-3, 1e-1, log=True),
            "delta":         trial.suggest_float(
                "delta",         1e-4, 1e-1, log=True),
            "lambda_reg":    trial.suggest_float(
                "lambda_reg",    1e-4, 1e-1, log=True),
        }

        try:
            rmse, sad = _quick_train(cfg, dataset, device, epochs, hp)
        except Exception as e:
            # Mark trial as failed without crashing the study
            raise optuna.exceptions.TrialPruned(f"Training failed: {e}")

        # Combined metric (equal weighting; adjust alpha to preference)
        alpha = 0.5
        score = alpha * rmse + (1.0 - alpha) * sad

        # Log individual metrics as user attributes for later analysis
        trial.set_user_attr("rmse", rmse)
        trial.set_user_attr("sad",  sad)

        return score

    return objective


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="DASU-Net — Optuna hyperparameter search"
    )
    p.add_argument("--config", type=str, default="configs/base.yaml")
    p.add_argument("--dataset", type=str, default=None,
                   help="Dataset override (samson | apex)")
    p.add_argument("--n-trials", type=int, default=50,
                   help="Number of Optuna trials (default: 50)")
    p.add_argument("--epochs", type=int, default=40,
                   help="Epochs per trial (default: 40)")
    p.add_argument("--study-name", type=str, default="hsu_search",
                   help="Optuna study name (default: hsu_search)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-db", action="store_true",
                   help="Use in-memory storage (no SQLite persistence)")
    return p.parse_args()


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    # CLI overrides
    if args.dataset:
        cfg["dataset"]["name"] = args.dataset

    set_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataset_name = cfg["dataset"]["name"]

    print(f"\n{'='*60}")
    print(f"  DASU-Net — Optuna Hyperparameter Search")
    print(f"  Device:   {device}")
    print(f"  Dataset:  {dataset_name}")
    print(f"  Trials:   {args.n_trials}")
    print(f"  Epochs/trial: {args.epochs}")
    print(f"{'='*60}\n")

    # Load dataset once (shared across all trials)
    dataset = HyperspectralDataset(
        dataset_name=dataset_name,
        data_dir=cfg["dataset"]["data_dir"],
        device=device,
    )
    print(f"  {dataset}\n")

    # ---- Create / resume study -----------------------------------
    study_name = f"{args.study_name}_{dataset_name}"
    if args.no_db:
        storage = None
    else:
        db_path = f"optuna_{study_name}.db"
        storage = f"sqlite:///{db_path}"
        print(f"  Storage: {db_path}")

    sampler = optuna.samplers.TPESampler(seed=args.seed)
    study = optuna.create_study(
        study_name=study_name,
        direction="minimize",
        sampler=sampler,
        storage=storage,
        load_if_exists=True,
    )

    # ---- Run optimisation ----------------------------------------
    objective = build_objective(cfg, dataset, device, args.epochs, args.seed)

    print(f"  Starting {args.n_trials} trials...\n")
    t0 = time.time()
    study.optimize(
        objective,
        n_trials=args.n_trials,
        show_progress_bar=True,
        callbacks=[_progress_callback],
    )
    elapsed = time.time() - t0

    # ---- Report --------------------------------------------------
    best = study.best_trial
    print(f"\n{'='*60}")
    print(f"  Optimisation complete in {elapsed:.1f}s")
    print(f"  Best trial:  #{best.number}")
    print(f"  Best score:  {best.value:.6f}")
    print(f"  Best RMSE:   {best.user_attrs.get('rmse', 'N/A'):.6f}")
    print(f"  Best SAD:    {best.user_attrs.get('sad', 'N/A'):.6f}")
    print(f"\n  Best hyperparameters:")
    for k, v in best.params.items():
        print(f"    {k:<20}: {v:.6g}")
    print(f"{'='*60}\n")

    # ---- Save best params as YAML override -----------------------
    out_path = Path(f"configs/best_{dataset_name}_{args.study_name}.yaml")
    best_cfg = {
        "training": {
            "learning_rate": float(best.params["learning_rate"]),
            "beta":          float(best.params["beta"]),
            "gamma":         float(best.params["gamma"]),
            "delta":         float(best.params["delta"]),
            "lambda_reg":    float(best.params["lambda_reg"]),
        },
        "model": {
            "encoder_type":      "swin",
            "decoder_type":      "nonlinear",
            "use_dual_attention": True,
        },
        "_optuna": {
            "study_name":  study_name,
            "n_trials":    args.n_trials,
            "best_trial":  int(best.number),
            "best_score":  float(best.value),
            "best_rmse":   float(best.user_attrs.get("rmse", -1)),
            "best_sad":    float(best.user_attrs.get("sad", -1)),
        }
    }
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.dump(best_cfg, f, default_flow_style=False, sort_keys=False)
    print(f"  Best config saved to: {out_path}")
    print(f"\n  To train with best params:")
    print(f"    python main.py --config configs/base.yaml --override {out_path}\n")

    # ---- Top-5 summary -------------------------------------------
    print(f"  Top-5 trials:")
    completed = [t for t in study.trials
                 if t.state == optuna.trial.TrialState.COMPLETE]
    top5 = sorted(completed, key=lambda t: t.value)[:5]
    header = (f"  {'#':>4}  {'score':>8}  {'rmse':>8}  {'sad':>8}"
              f"  {'lr':>10}  {'beta':>8}  {'gamma':>8}  {'delta':>8}  {'lambda':>8}")
    print(header)
    print("  " + "-" * (len(header) - 2))
    for t in top5:
        rmse = t.user_attrs.get("rmse", float("nan"))
        sad  = t.user_attrs.get("sad",  float("nan"))
        p = t.params
        print(f"  {t.number:>4}  {t.value:>8.4f}  {rmse:>8.4f}  {sad:>8.4f}"
              f"  {p['learning_rate']:>10.2e}  {p['beta']:>8.1f}"
              f"  {p['gamma']:>8.4f}  {p['delta']:>8.4f}  {p['lambda_reg']:>8.4f}")


def _progress_callback(study: optuna.Study,
                        trial: optuna.FrozenTrial) -> None:
    """Print a one-liner after each trial."""
    rmse = trial.user_attrs.get("rmse", float("nan"))
    sad  = trial.user_attrs.get("sad",  float("nan"))
    best = study.best_value
    print(f"  Trial {trial.number:3d} | score={trial.value:.4f} "
          f"rmse={rmse:.4f} sad={sad:.4f} | best={best:.4f}")


if __name__ == "__main__":
    main()
