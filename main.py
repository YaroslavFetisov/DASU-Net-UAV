"""
DeepUnmixing_v2 — Main training script.

Usage:
    python main.py --config configs/base.yaml
    python main.py --config configs/base.yaml --override configs/exp_swin.yaml
    python main.py --config configs/base.yaml --epochs 3   # quick test
"""

from __future__ import annotations

import argparse
import os
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from src.data import HyperspectralDataset
from src.models.unmixer import DeepUnmixer
from src.core.losses import TotalLoss
from src.core.metrics import compute_rmse, compute_sad, match_endmembers
from src.utils.config_parser import load_config, get


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DeepUnmixing_v2 Training")
    p.add_argument("--config", type=str, default="configs/base.yaml")
    p.add_argument("--override", type=str, default=None,
                   help="Experiment YAML that overrides base config")
    # Quick overrides for testing
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--dataset", type=str, default=None)
    p.add_argument("--seed", type=int, default=None)
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ------------------------------------------------------------------
# Training
# ------------------------------------------------------------------

def train(cfg: dict, device: torch.device, run_dir: Path) -> None:
    ds_cfg = cfg["dataset"]
    m_cfg = cfg["model"]
    t_cfg = cfg["training"]

    # ---- 1. Data -------------------------------------------------
    dataset = HyperspectralDataset(
        dataset_name=ds_cfg["name"],
        data_dir=ds_cfg["data_dir"],
        device=device,
    )
    print(f"  {dataset}")
    img_cube = dataset.get_image_cube()     # (1, L, H, W)

    # ---- 2. Model ------------------------------------------------
    model = DeepUnmixer(
        num_endmembers=dataset.P,
        num_bands=dataset.L,
        spatial_size=dataset.col,
        encoder_type=m_cfg["encoder_type"],
        decoder_type=m_cfg["decoder_type"],
        use_dual_attention=m_cfg["use_dual_attention"],
        patch_size=m_cfg["patch_size"],
        emb_dim_per_endmember=m_cfg["emb_dim_per_endmember"],
        transformer_depth=m_cfg["transformer_depth"],
        num_heads=m_cfg["num_heads"],
        window_size=m_cfg["window_size"],
        mlp_dim=m_cfg["mlp_dim"],
        nonlinear_gamma=m_cfg.get("nonlinear_gamma", 1.0),
    ).to(device)
    print(f"  {model}")

    # Kaiming init for conv layers
    model.apply(model.weights_init)

    # Initialize decoder endmembers from data
    init_method = m_cfg.get("init_method", "sivm")
    model.init_decoder_weights(img_cube, method=init_method)
    print(f"  Decoder initialized with: {init_method}")

    # ---- 3. Loss, optimizer, scheduler ---------------------------
    loss_fn = TotalLoss(
        num_bands=dataset.L,
        beta=t_cfg["beta"],
        gamma=t_cfg["gamma"],
        delta=t_cfg.get("delta", 1e-2),
        lambda_reg=t_cfg.get("lambda_reg", 1e-2),
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=t_cfg["learning_rate"],
        weight_decay=t_cfg["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=t_cfg["scheduler_step"],
        gamma=t_cfg["scheduler_gamma"],
    )
    clipper = model.get_clipper()

    # ---- 4. Training loop ----------------------------------------
    epochs = t_cfg["epochs"]
    print_every = cfg["logging"]["print_every"]
    losses_log = []

    print(f"\n  Training for {epochs} epochs...")
    t0 = time.time()
    model.train()

    for epoch in range(epochs):
        abu_est, recon = model(img_cube)
        endmem = model.decoder.get_endmembers()  # (L, P)
        loss = loss_fn(recon, img_cube, abu_est, endmem)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=t_cfg["grad_clip_norm"],
            norm_type=t_cfg["grad_clip_type"],
        )
        optimizer.step()

        # Clamp decoder weights to non-negative
        model.decoder.apply(clipper)

        scheduler.step()
        losses_log.append(float(loss.item()))

        if epoch % print_every == 0 or epoch == epochs - 1:
            lr = optimizer.param_groups[0]["lr"]
            bd = loss_fn.breakdown(recon, img_cube, abu_est, endmem)
            print(f"    Epoch {epoch:4d}/{epochs}  "
                  f"loss={loss.item():.4f}  "
                  f"mse={bd['mse']:.2f}  sad={bd['sad']:.4f}  "
                  f"vol={bd['vol']:.4f}  spar={bd['spar']:.4f}  "
                  f"lr={lr:.6f}")

    elapsed = time.time() - t0
    print(f"\n  Training completed in {elapsed:.1f}s")

    # ---- 5. Evaluation -------------------------------------------
    model.eval()
    with torch.no_grad():
        abu_est, recon = model(img_cube)

    # Normalize abundances
    abu_est = abu_est / abu_est.sum(dim=1, keepdim=True).clamp(min=1e-8)

    # Convert to numpy
    abu_np = abu_est.squeeze(0).permute(1, 2, 0).cpu().numpy()     # (H,W,P)
    target_np = dataset.get_abundance_cube().cpu().numpy()           # (H,W,P)
    est_endmem = model.decoder.get_endmembers().cpu().numpy()        # (L,P)
    true_endmem = dataset.get_endmembers().numpy()                   # (L,P)

    # Match endmember ordering via Hungarian algorithm
    est_endmem, abu_np, perm = match_endmembers(
        est_endmem, true_endmem, abu_np, target_np
    )
    print(f"  Endmember permutation: {perm}")

    # Compute metrics
    rmse_cls, rmse_mean = compute_rmse(abu_np, target_np)
    sad_cls, sad_mean = compute_sad(est_endmem, true_endmem)

    print(f"\n  === Results ===")
    for i in range(dataset.P):
        print(f"    Endmember {i+1}: RMSE={rmse_cls[i]:.4f}, SAD={sad_cls[i]:.4f}")
    print(f"    Mean RMSE: {rmse_mean:.4f}")
    print(f"    Mean SAD:  {sad_mean:.4f}")

    # ---- 6. Save -------------------------------------------------
    torch.save(model.state_dict(), run_dir / "model.pth")
    np.savez(
        run_dir / "results.npz",
        losses=np.array(losses_log),
        abu_est=abu_np,
        est_endmem=est_endmem,
        rmse_cls=rmse_cls,
        sad_cls=sad_cls,
        perm=perm,
    )

    # Save summary text
    with open(run_dir / "log.txt", "w") as f:
        f.write(f"Dataset: {ds_cfg['name']}\n")
        f.write(f"Encoder: {m_cfg['encoder_type']}\n")
        f.write(f"Decoder: {m_cfg['decoder_type']}\n")
        f.write(f"DualAttn: {m_cfg['use_dual_attention']}\n")
        f.write(f"Init: {init_method}\n")
        f.write(f"Epochs: {epochs}\n")
        f.write(f"Time: {elapsed:.1f}s\n")
        f.write(f"Mean RMSE: {rmse_mean:.6f}\n")
        f.write(f"Mean SAD: {sad_mean:.6f}\n")

    print(f"\n  Saved to: {run_dir}")


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, args.override)

    # CLI overrides
    if args.epochs is not None:
        cfg["training"]["epochs"] = args.epochs
    if args.dataset is not None:
        cfg["dataset"]["name"] = args.dataset
    if args.seed is not None:
        cfg["seed"] = args.seed

    set_seed(cfg["seed"])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Create unique run directory
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    ds = cfg["dataset"]["name"]
    enc = cfg["model"]["encoder_type"]
    run_name = f"{ds}_{enc}_{ts}"
    run_dir = Path(cfg["logging"]["save_dir"]) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  DeepUnmixing_v2")
    print(f"  Device:  {device}")
    print(f"  Run:     {run_dir}")
    print(f"{'='*60}\n")

    train(cfg, device, run_dir)
    print(f"\n{'='*60}")
    print(f"  Done!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
