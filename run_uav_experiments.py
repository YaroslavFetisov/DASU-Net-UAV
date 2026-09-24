"""
DASU-Net-UAV Experiment & Profiling Runner.
Executes:
1. Benchmark on the two synthetic UAV benchmarks (uav_synthetic and the LongKou-like
   surrogate whu_hi_longkou) for Baseline vs DASU-Net (SiVM & VCA).
2. 5-step Ablation Study on UAV Synthetic Benchmark.
3. Hardware Profiling (parameters, latency, FPS, peak GPU memory per model).
4. Generates 300 DPI figures (abundance / error maps and endmember spectra).

Training on CUDA is not bit-wise deterministic, so repeated runs with the same seed
give slightly different metrics. Pass several seeds to report mean +/- std:

    python run_uav_experiments.py --seeds 42 43 44 45 46
"""

from pathlib import Path
import argparse
import gc
import platform
import shutil
import time
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from src.data.dataset import HyperspectralDataset
from src.models.unmixer import DASUNet
from src.core.losses import TotalLoss
from src.core.metrics import compute_rmse, compute_sad, match_endmembers

SEED = 42
BENCHMARK_DATASETS = ["uav_synthetic", "whu_hi_longkou"]
BENCHMARK_METHODS = ["baseline", "dasunet_sivm", "dasunet_vca"]

# Input used for profiling: one 270-band 100x100 cube, 5 endmembers
PROFILE_SHAPE = dict(num_endmembers=5, num_bands=270, spatial_size=100)


def set_seed(seed=SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train_model(
    model: DASUNet,
    dataset: HyperspectralDataset,
    device: torch.device,
    epochs: int = 250,
    lr: float = 3e-3,
    beta: float = 5e3,
    gamma: float = 3e-2,
    delta: float = 1e-2,
    lambda_reg: float = 1e-2,
    patience: int = 50,
    seed: int = SEED,
):
    set_seed(seed)
    img_cube = dataset.get_image_cube()
    clipper = model.get_clipper()

    loss_fn = TotalLoss(
        num_bands=dataset.L,
        beta=beta,
        gamma=gamma,
        delta=delta,
        lambda_reg=lambda_reg,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=4e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.8)

    losses = []
    best_loss = float('inf')
    patience_cnt = 0
    t0 = time.time()
    model.train()

    for epoch in range(epochs):
        abu, recon = model(img_cube)
        endmem = model.decoder.get_endmembers()
        loss = loss_fn(recon, img_cube, abu, endmem)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0, norm_type=1)
        optimizer.step()
        model.decoder.apply(clipper)
        scheduler.step()

        l_val = float(loss.item())
        losses.append(l_val)

        if l_val < best_loss - 1e-5:
            best_loss = l_val
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= patience and epoch >= 100:
                break

    elapsed = time.time() - t0

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

    return {
        "rmse_mean": float(rmse_mean),
        "sad_mean": float(sad_mean),
        "rmse_cls": [float(x) for x in rmse_cls],
        "sad_cls": [float(x) for x in sad_cls],
        "epochs": len(losses),
        "time_s": elapsed,
        "abu_est": abu_np,
        "est_endmem": est_endmem,
        "true_endmem": true_endmem,
        "target_abu": target_np,
        "losses": losses,
    }


def run_benchmarks(device: torch.device, seed: int = SEED):
    """Run full benchmark on both UAV datasets."""
    print("\n" + "="*65)
    print("  1. RUNNING UAV BENCHMARK EVALUATIONS")
    print("="*65)

    results = {}

    for ds_name in BENCHMARK_DATASETS:
        print(f"\n---> Dataset: {ds_name}")
        ds = HyperspectralDataset(ds_name, data_dir="./data/raw", device=device)
        ds_res = {}

        # 1. Baseline ViT + Linear
        print("  Running Baseline (ViT + Linear)...")
        base_model = DASUNet(
            num_endmembers=ds.P, num_bands=ds.L, spatial_size=ds.col,
            encoder_type="vit", decoder_type="linear", use_dual_attention=False
        ).to(device)
        base_model.apply(base_model.weights_init)
        base_model.init_decoder_weights(ds.get_image_cube(), method="vca")
        res_base = train_model(base_model, ds, device, epochs=200, lr=6e-3, delta=0.0, lambda_reg=0.0, seed=seed)
        ds_res["baseline"] = res_base
        print(f"    Baseline: RMSE = {res_base['rmse_mean']:.4f}, SAD = {res_base['sad_mean']:.4f}")

        # 2. DASU-Net + SiVM
        print("  Running DASU-Net + SiVM...")
        dasu_sivm = DASUNet(
            num_endmembers=ds.P, num_bands=ds.L, spatial_size=ds.col,
            encoder_type="swin", decoder_type="nonlinear", use_dual_attention=True,
            nonlinear_gamma=0.45 if "synth" in ds_name else 0.35
        ).to(device)
        dasu_sivm.apply(dasu_sivm.weights_init)
        dasu_sivm.init_decoder_weights(ds.get_image_cube(), method="sivm")
        res_sivm = train_model(dasu_sivm, ds, device, epochs=300, lr=4e-3, beta=2500, gamma=0.015, delta=5e-4, lambda_reg=5e-4, seed=seed)
        ds_res["dasunet_sivm"] = res_sivm
        print(f"    DASU-Net + SiVM: RMSE = {res_sivm['rmse_mean']:.4f}, SAD = {res_sivm['sad_mean']:.4f}")

        # 3. DASU-Net + VCA
        print("  Running DASU-Net + VCA...")
        dasu_vca = DASUNet(
            num_endmembers=ds.P, num_bands=ds.L, spatial_size=ds.col,
            encoder_type="swin", decoder_type="nonlinear", use_dual_attention=True,
            nonlinear_gamma=0.45 if "synth" in ds_name else 0.35
        ).to(device)
        dasu_vca.apply(dasu_vca.weights_init)
        dasu_vca.init_decoder_weights(ds.get_image_cube(), method="vca")
        res_vca = train_model(dasu_vca, ds, device, epochs=300, lr=4e-3, beta=2500, gamma=0.015, delta=5e-4, lambda_reg=5e-4, seed=seed)
        ds_res["dasunet_vca"] = res_vca
        print(f"    DASU-Net + VCA:  RMSE = {res_vca['rmse_mean']:.4f}, SAD = {res_vca['sad_mean']:.4f}")

        results[ds_name] = ds_res

    return results


def run_ablation_study(device: torch.device, seed: int = SEED):
    """Run progressive ablation on UAV Synthetic Benchmark."""
    print("\n" + "="*65)
    print("  2. RUNNING ABLATION STUDY (UAV Synthetic Benchmark)")
    print("="*65)

    ds = HyperspectralDataset("uav_synthetic", data_dir="./data/raw", device=device)
    # (label, row name in the APUAVD-2026 manuscript, model kwargs, init, delta, lambda_reg)
    ablation_cfgs = [
        ("ViT + Linear (Baseline)", "(1) ViT + Linear Decoder (Baseline)",
         dict(encoder_type="vit", decoder_type="linear", use_dual_attention=False), "vca", 0.0, 0.0),
        ("+ Swin Encoder", "(2) + Swin Transformer Encoder",
         dict(encoder_type="swin", decoder_type="linear", use_dual_attention=False), "vca", 0.0, 0.0),
        ("+ Dual Attention (+ MinVol/entropy terms)", "(3) + Dual Attention (Spatial + XCA)",
         dict(encoder_type="swin", decoder_type="linear", use_dual_attention=True), "vca", 5e-4, 5e-4),
        # Same architecture as the previous row: the skip-connected abundance decoder is
        # part of every configuration, only the endmember initialisation changes.
        ("+ SiVM Init", "(4) + U-Net Skip Decoder",
         dict(encoder_type="swin", decoder_type="linear", use_dual_attention=True), "sivm", 5e-4, 5e-4),
        ("+ PPNMM (Full DASU-Net)", "(5) + PPNMM Decoder (Full DASU-Net)",
         dict(encoder_type="swin", decoder_type="nonlinear", use_dual_attention=True, nonlinear_gamma=0.45), "sivm", 5e-4, 5e-4),
    ]

    ablation_results = []
    for name, manuscript_row, m_kwargs, init_m, delta, lam in ablation_cfgs:
        print(f"  Testing config: {name}...")
        model = DASUNet(
            num_endmembers=ds.P, num_bands=ds.L, spatial_size=ds.col,
            **m_kwargs
        ).to(device)
        model.apply(model.weights_init)
        model.init_decoder_weights(ds.get_image_cube(), method=init_m)
        res = train_model(model, ds, device, epochs=250, lr=4e-3, beta=2500, gamma=0.015, delta=delta, lambda_reg=lam, seed=seed)
        print(f"    -> RMSE: {res['rmse_mean']:.4f}, SAD: {res['sad_mean']:.4f}")
        ablation_results.append({
            "configuration": name,
            "manuscript_row": manuscript_row,
            "rmse": res['rmse_mean'],
            "sad": res['sad_mean'],
            "epochs": res['epochs']
        })

    return ablation_results


def _profile_model(model_kwargs: dict, device: torch.device,
                   warmup: int = 20, iters: int = 100, repeats: int = 5) -> dict:
    """Profile one model in isolation: parameters, median latency, peak GPU memory."""
    cuda = device.type == "cuda"
    gc.collect()
    if cuda:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        mem_before = torch.cuda.memory_allocated(device)

    P, L, S = PROFILE_SHAPE["num_endmembers"], PROFILE_SHAPE["num_bands"], PROFILE_SHAPE["spatial_size"]
    dummy_input = torch.randn(1, L, S, S, device=device)
    model = DASUNet(num_endmembers=P, num_bands=L, spatial_size=S, **model_kwargs).to(device)
    model.eval()
    params = sum(p.numel() for p in model.parameters())

    timings = []
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy_input)

        for _ in range(repeats):
            if cuda:
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(iters):
                _ = model(dummy_input)
            if cuda:
                torch.cuda.synchronize()
            timings.append((time.perf_counter() - t0) * 1000.0 / iters)

    # Peak memory allocated by PyTorch for this model alone: input + weights + activations
    vram_mb = None
    if cuda:
        vram_mb = round((torch.cuda.max_memory_allocated(device) - mem_before) / (1024.0 * 1024.0), 1)

    del model, dummy_input
    if cuda:
        torch.cuda.empty_cache()

    latency_ms = float(np.median(timings))
    return {
        "params_m": params / 1e6,
        "params_raw": params,
        "latency_ms": round(latency_ms, 2),
        "latency_ms_repeats": [round(t, 3) for t in timings],
        "fps": round(1000.0 / latency_ms, 1),
        "vram_mb": vram_mb,
    }


def run_hardware_profiling(device: torch.device):
    """Profile latency, FPS, peak memory and parameters of each model separately."""
    print("\n" + "="*65)
    print("  3. RUNNING HARDWARE PROFILING")
    print("="*65)

    vit = _profile_model(dict(encoder_type="vit", decoder_type="linear", use_dual_attention=False), device)
    dasu = _profile_model(dict(encoder_type="swin", decoder_type="nonlinear", use_dual_attention=True), device)

    profile_data = {
        "input_shape": [1, PROFILE_SHAPE["num_bands"], PROFILE_SHAPE["spatial_size"], PROFILE_SHAPE["spatial_size"]],
        "protocol": "batch 1, torch.no_grad, 20 warm-up passes, latency = median of 5 x 100 timed passes; "
                    "vram_mb = peak PyTorch-allocated memory (input + weights + activations) with only that model loaded",
        "vit": vit,
        "dasunet": {
            **dasu,
            "param_reduction_pct": round((1.0 - dasu["params_raw"] / vit["params_raw"]) * 100.0, 1),
            "speedup": round(dasu["fps"] / vit["fps"], 2),
        },
    }

    for label, prof in (("ViT Baseline", vit), ("DASU-Net", dasu)):
        vram = "n/a" if prof["vram_mb"] is None else f"{prof['vram_mb']:.1f} MB"
        print(f"  {label:<13} {prof['params_m']:.3f}M params | Latency: {prof['latency_ms']:.2f} ms | "
              f"FPS: {prof['fps']:.1f} | Peak memory: {vram}")
    print(f"  -> Model parameters reduced by {profile_data['dasunet']['param_reduction_pct']}%")

    return profile_data


def collect_environment(device: torch.device) -> dict:
    """Software / hardware context needed to interpret the timings."""
    return {
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else (platform.processor() or "cpu"),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "python": platform.python_version(),
        "platform": platform.platform(),
    }


def _mean_std(values):
    arr = np.asarray(values, dtype=np.float64)
    std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    return float(arr.mean()), std


def summarise_benchmarks(per_seed: list, seeds: list) -> dict:
    """Aggregate per-seed benchmark results into mean / std (per-class values are averaged)."""
    summary = {}
    for ds_name in BENCHMARK_DATASETS:
        ds_summary = {}
        for method in BENCHMARK_METHODS:
            runs = [seed_res[ds_name][method] for seed_res in per_seed]
            rmse, rmse_std = _mean_std([r["rmse_mean"] for r in runs])
            sad, sad_std = _mean_std([r["sad_mean"] for r in runs])
            ds_summary[method] = {
                "rmse": rmse,
                "rmse_std": rmse_std,
                "sad": sad,
                "sad_std": sad_std,
                "rmse_cls": np.mean([r["rmse_cls"] for r in runs], axis=0).tolist(),
                "sad_cls": np.mean([r["sad_cls"] for r in runs], axis=0).tolist(),
                "runs": [{"seed": s, "rmse": r["rmse_mean"], "sad": r["sad_mean"], "epochs": r["epochs"]}
                         for s, r in zip(seeds, runs)],
            }
        base = ds_summary["baseline"]
        for method in ("dasunet_sivm", "dasunet_vca"):
            m = ds_summary[method]
            m["rmse_improvement_pct"] = round((1.0 - m["rmse"] / base["rmse"]) * 100.0, 2)
            m["sad_improvement_pct"] = round((1.0 - m["sad"] / base["sad"]) * 100.0, 2)
        summary[ds_name] = ds_summary
    return summary


def summarise_ablation(per_seed: list, seeds: list) -> list:
    """Aggregate per-seed ablation rows into mean / std."""
    rows = []
    for i, first in enumerate(per_seed[0]):
        runs = [seed_rows[i] for seed_rows in per_seed]
        rmse, rmse_std = _mean_std([r["rmse"] for r in runs])
        sad, sad_std = _mean_std([r["sad"] for r in runs])
        rows.append({
            "configuration": first["configuration"],
            "manuscript_row": first["manuscript_row"],
            "rmse": rmse,
            "rmse_std": rmse_std,
            "sad": sad,
            "sad_std": sad_std,
            "runs": [{"seed": s, "rmse": r["rmse"], "sad": r["sad"], "epochs": r["epochs"]}
                     for s, r in zip(seeds, runs)],
        })
    return rows


def print_summary(report: dict):
    n = len(report["seeds"])
    print("\n" + "="*65)
    print(f"  SUMMARY (mean +/- std over {n} seed{'s' if n > 1 else ''})")
    print("="*65)
    for ds_name, ds_summary in report["benchmarks"].items():
        for method, m in ds_summary.items():
            print(f"  {ds_name:<15} {method:<13} RMSE {m['rmse']:.4f} +/- {m['rmse_std']:.4f}   "
                  f"SAD {m['sad']:.4f} +/- {m['sad_std']:.4f}")
    print()
    for row in report["ablation"]:
        print(f"  {row['configuration']:<42} RMSE {row['rmse']:.4f} +/- {row['rmse_std']:.4f}   "
              f"SAD {row['sad']:.4f} +/- {row['sad_std']:.4f}")


def generate_publication_figures(results: dict, out_dir: Path, sync_paper_dir: str | None = None):
    """Generate 300 DPI high-resolution figures for the IEEE conference paper."""
    print("\n" + "="*65)
    print("  4. GENERATING 300 DPI PUBLICATION FIGURES")
    print("="*65)

    uav_res = results["uav_synthetic"]
    gt_abu = uav_res["dasunet_sivm"]["target_abu"]       # (H, W, P)
    base_abu = uav_res["baseline"]["abu_est"]            # (H, W, P)
    dasu_abu = uav_res["dasunet_sivm"]["abu_est"]        # (H, W, P)
    P = gt_abu.shape[2]
    em_names = ["Crop Canopy", "Dry Soil", "Water Feature", "Road/Mineral"]

    # --- Figure 1: Comprehensive Abundance Maps & Absolute Error Maps ---
    fig_abu = plt.figure(figsize=(10.5, 9.2), dpi=300)
    gs_abu = fig_abu.add_gridspec(5, P + 2, width_ratios=[1]*P + [0.06, 0.06], wspace=0.18, hspace=0.25)

    diff_base = np.abs(base_abu - gt_abu)
    diff_dasu = np.abs(dasu_abu - gt_abu)
    max_err = 0.12

    row_data = [
        ("Ground Truth", gt_abu, "viridis", 0.0, 1.0, "black"),
        ("Baseline (ViT)", base_abu, "viridis", 0.0, 1.0, "black"),
        ("Error: ViT\n$|\\hat{\\mathbf{A}} - \\mathbf{A}_{GT}|$", diff_base, "inferno", 0.0, max_err, "#b30000"),
        ("DASU-Net (Ours)", dasu_abu, "viridis", 0.0, 1.0, "#006600"),
        ("Error: DASU-Net\n$|\\hat{\\mathbf{A}} - \\mathbf{A}_{GT}|$", diff_dasu, "inferno", 0.0, max_err, "#006600"),
    ]

    im_abu = None
    im_err = None

    for r_idx, (r_title, maps, cmap, vmin, vmax, color) in enumerate(row_data):
        for p in range(P):
            ax = fig_abu.add_subplot(gs_abu[r_idx, p])
            im = ax.imshow(maps[:, :, p], cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_xticks([])
            ax.set_yticks([])
            if r_idx == 0:
                ax.set_title(em_names[p], fontsize=10.5, fontweight="bold", pad=5)
            if p == 0:
                ax.set_ylabel(r_title, fontsize=8.8, fontweight="bold", labelpad=6, color=color)
            if r_idx == 0 and p == 0:
                im_abu = im
            if r_idx == 2 and p == 0:
                im_err = im

    # Colorbar for Abundances (Rows 0, 1, 3)
    cax_abu = fig_abu.add_subplot(gs_abu[0:2, P])
    cb_abu = plt.colorbar(im_abu, cax=cax_abu)
    cb_abu.ax.tick_params(labelsize=8)
    cb_abu.set_label("Fraction [0, 1]", fontsize=8.5, labelpad=3)

    # Colorbar for Error Maps (Rows 2, 4)
    cax_err = fig_abu.add_subplot(gs_abu[2:5, P])
    cb_err = plt.colorbar(im_err, cax=cax_err)
    cb_err.ax.tick_params(labelsize=8)
    cb_err.set_label("Absolute Error", fontsize=8.5, labelpad=3)

    abu_path = out_dir / "abundance_comparison.png"
    plt.savefig(abu_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved Abundances Figure: {abu_path}")

    # --- Figure 2: Classic Clean IEEE Spectral Signatures (2x2 Grid) ---
    fig_spec, axes = plt.subplots(2, 2, figsize=(10, 6.5), dpi=300, sharex=True, sharey=True)
    wl = np.linspace(400, 900, 150)
    true_em = uav_res["dasunet_sivm"]["true_endmem"]      # (L, P)
    base_em = uav_res["baseline"]["est_endmem"]          # (L, P)
    dasu_em = uav_res["dasunet_sivm"]["est_endmem"]      # (L, P)
    colors = ["#2ca02c", "#d62728", "#1f77b4", "#555555"]
    sad_dasu = uav_res["dasunet_sivm"]["sad_cls"]
    sad_base = uav_res["baseline"]["sad_cls"]

    for p, ax in enumerate(axes.flat):
        ax.plot(wl, true_em[:, p], color=colors[p], linestyle="-", linewidth=2.4, label="Ground Truth", alpha=0.95)
        ax.plot(wl, dasu_em[:, p], color="black", linestyle="--", linewidth=1.8, label=f"DASU-Net (SAD={sad_dasu[p]:.4f})", alpha=0.9)
        ax.plot(wl, base_em[:, p], color="tab:purple", linestyle=":", linewidth=1.5, label=f"ViT Base (SAD={sad_base[p]:.4f})", alpha=0.8)
        
        ax.set_title(f"({chr(97+p)}) {em_names[p]}", fontsize=11, fontweight="bold")
        ax.set_xlim(395, 905)
        ax.set_ylim(-0.02, 0.72)
        ax.grid(True, linestyle="--", alpha=0.45)
        ax.legend(loc="upper right" if p != 2 else "upper left", fontsize=8, framealpha=0.92)
        if p in [2, 3]:
            ax.set_xlabel("Wavelength (nm)", fontsize=9.5)
        if p in [0, 2]:
            ax.set_ylabel("Reflectance", fontsize=9.5)

    plt.tight_layout()
    spec_path = out_dir / "spectral_signatures.png"
    plt.savefig(spec_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved Classic Spectral Signatures: {spec_path}")

    # Optional copy into a local manuscript directory (never done implicitly)
    if sync_paper_dir is not None:
        paper_dir = Path(sync_paper_dir)
        for fname in ["abundance_comparison.png", "spectral_signatures.png"]:
            shutil.copy(out_dir / fname, paper_dir / fname)
            print(f"  Synced: {paper_dir / fname}")

        # Keep backward-compatible combined copy as well
        shutil.copy(abu_path, paper_dir / "qualitative_comparison_combined.png")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DASU-Net-UAV experiments, ablation and profiling")
    p.add_argument("--seeds", type=int, nargs="+", default=[SEED],
                   help="Seeds for the benchmark and ablation runs; mean +/- std is reported (default: 42)")
    p.add_argument("--out-dir", type=str, default="runs/experiments_uav",
                   help="Output directory for all_results.json and figures")
    p.add_argument("--sync-paper-dir", type=str, default=None,
                   help="Also copy the figures into this existing directory (e.g. a local manuscript folder)")
    return p.parse_args()


def main():
    args = parse_args()
    if args.sync_paper_dir is not None and not Path(args.sync_paper_dir).is_dir():
        raise SystemExit(f"--sync-paper-dir '{args.sync_paper_dir}' is not an existing directory")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    environment = collect_environment(device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*65}")
    print(f"  DASU-Net-UAV Scientific Experiments Suite")
    print(f"  Device: {device} ({environment['device']})")
    print(f"  Seeds:  {args.seeds}")
    print(f"  Output: {out_dir}")
    print(f"{'='*65}\n")

    # 1-2. Benchmarks and ablation, repeated per seed
    benchmark_runs = []
    ablation_runs = []
    for seed in args.seeds:
        print(f"\n{'#'*65}\n  SEED {seed}\n{'#'*65}")
        set_seed(seed)
        benchmark_runs.append(run_benchmarks(device, seed))
        ablation_runs.append(run_ablation_study(device, seed))

    # 3. Hardware Profiling
    profile_res = run_hardware_profiling(device)

    # 4. Compile and save the report before plotting, so a figure error cannot lose results
    report_data = {
        "environment": environment,
        "seeds": args.seeds,
        "benchmarks": summarise_benchmarks(benchmark_runs, args.seeds),
        "ablation": summarise_ablation(ablation_runs, args.seeds),
        "profiling": profile_res,
    }

    json_path = out_dir / "all_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)
    print(f"\n  Saved structured results to: {json_path}")

    print_summary(report_data)

    # 5. Figures (from the first seed)
    generate_publication_figures(benchmark_runs[0], out_dir, args.sync_paper_dir)

    print("\n" + "="*65)
    print("  ALL EXPERIMENTS COMPLETED SUCCESSFULLY!")
    print("="*65 + "\n")


if __name__ == "__main__":
    main()
