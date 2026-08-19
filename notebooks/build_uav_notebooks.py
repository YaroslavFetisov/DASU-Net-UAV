"""
Generates clean, UAV-centric Jupyter notebooks for DASU-Net-UAV.
"""

import json
from pathlib import Path

def create_cell(cell_type, source, execution_count=None, outputs=None):
    if isinstance(source, list):
        src = [s if s.endswith("\n") else s + "\n" for s in source]
    else:
        src = [line + "\n" for line in source.split("\n")]
    if src and src[-1] == "\n":
        src[-1] = src[-1][:-1]
    
    cell = {
        "cell_type": cell_type,
        "metadata": {},
        "source": src
    }
    if cell_type == "code":
        cell["execution_count"] = execution_count
        cell["outputs"] = outputs or []
    return cell

def create_notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (ipykernel)",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.10.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

# -------------------------------------------------------------
# Notebook 1: 01_uav_data_and_eda.ipynb
# -------------------------------------------------------------
nb1_cells = [
    create_cell("markdown", [
        "# 🚁 Notebook 01: UAV Hyperspectral Data Exploration & Physical EDA",
        "",
        "**Conference:** 2026 IEEE 8th International Conference 'Actual Problems of Unmanned Aerial Vehicles Development' (APUAVD-2026)",
        "**Author:** Yaroslav Fetisov (NTUU 'KPI')",
        "",
        "This notebook explores the specialized UAV hyperspectral benchmarks:",
        "1. **WHU-Hi LongKou UAV Benchmark** (DJI Matrice 600 Pro + Headwall Nano-Hyperspec, 270 bands, 400–1000 nm)",
        "2. **UAV Flight Synthetic Benchmark** (150 bands, 4 endmembers, PPNMM non-linear interactions, motion blur, 30 dB SNR noise)",
    ]),
    create_cell("code", [
        "import sys",
        "sys.path.append('..')",
        "import numpy as np",
        "import matplotlib.pyplot as plt",
        "import torch",
        "from src.data.dataset import HyperspectralDataset",
        "",
        "print('PyTorch CUDA Available:', torch.cuda.is_available())"
    ]),
    create_cell("markdown", [
        "## 1. Load UAV Flight Datasets"
    ]),
    create_cell("code", [
        "ds_synth = HyperspectralDataset('uav_synthetic', data_dir='../data/raw')",
        "ds_whu = HyperspectralDataset('whu_hi_longkou', data_dir='../data/raw')",
        "",
        "print('Synthetic UAV Benchmark:', ds_synth)",
        "print('WHU-Hi LongKou UAV Benchmark:', ds_whu)"
    ]),
    create_cell("markdown", [
        "## 2. Inspect Spectral Signatures & Red Edge Dynamics",
        "Vegetation exhibits a characteristic steep slope between 700 nm and 750 nm (the Red Edge effect) and high NIR reflectance."
    ]),
    create_cell("code", [
        "em_synth = ds_synth.get_endmembers().numpy()",
        "wl_synth = np.linspace(400, 900, ds_synth.L)",
        "names_synth = ['Crop Canopy', 'Dry Soil', 'Water Feature', 'Road/Mineral']",
        "colors = ['#2ca02c', '#d62728', '#1f77b4', '#555555']",
        "",
        "plt.figure(figsize=(9, 4.5), dpi=150)",
        "for p in range(ds_synth.P):",
        "    plt.plot(wl_synth, em_synth[:, p], label=names_synth[p], color=colors[p], linewidth=2.2)",
        "plt.title('UAV Synthetic Benchmark Endmembers', fontsize=12, fontweight='bold')",
        "plt.xlabel('Wavelength (nm)', fontsize=10)",
        "plt.ylabel('Reflectance', fontsize=10)",
        "plt.grid(True, linestyle='--', alpha=0.5)",
        "plt.legend(framealpha=0.9)",
        "plt.show()"
    ]),
    create_cell("markdown", [
        "## 3. Spatial Abundance Distributions (Ground Truth)",
        "Displaying the spatial distributions of endmembers satisfying the Abundance Non-negativity Constraint (ANC) and Sum-to-One Constraint (ASC)."
    ]),
    create_cell("code", [
        "abu_synth = ds_synth.get_abundance_cube().numpy()",
        "fig, axes = plt.subplots(1, 4, figsize=(13, 3.2), dpi=150)",
        "for p in range(4):",
        "    im = axes[p].imshow(abu_synth[:, :, p], cmap='viridis', vmin=0, vmax=1)",
        "    axes[p].set_title(names_synth[p], fontsize=10, fontweight='bold')",
        "    axes[p].axis('off')",
        "plt.tight_layout()",
        "plt.show()"
    ])
]

# -------------------------------------------------------------
# Notebook 2: 02_uav_training_and_benchmark.ipynb
# -------------------------------------------------------------
nb2_cells = [
    create_cell("markdown", [
        "# 🧪 Notebook 02: Model Training, Optuna HPO & Comparative Benchmarking",
        "",
        "**Conference:** 2026 IEEE 8th International Conference 'Actual Problems of Unmanned Aerial Vehicles Development' (APUAVD-2026)",
        "**Author:** Yaroslav Fetisov (NTUU 'KPI')",
        "",
        "This notebook benchmarks **DASU-Net** (Dual-Attention Swin U-Net with PPNMM Non-Linear Decoder) against **DeepTrans-HSU Baseline** (ViT + Linear) on UAV hyperspectral flight data."
    ]),
    create_cell("code", [
        "import sys",
        "sys.path.append('..')",
        "import numpy as np",
        "import torch",
        "import matplotlib.pyplot as plt",
        "from src.data.dataset import HyperspectralDataset",
        "from src.models.unmixer import DASUNet",
        "from src.core.losses import TotalLoss",
        "from src.core.metrics import compute_rmse, compute_sad, match_endmembers",
        "",
        "device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')",
        "print('Active Device:', device)"
    ]),
    create_cell("markdown", [
        "## 1. Initialize Dataset & Model",
        "DASU-Net combines a Shifted-Window Swin Transformer Encoder with a Dual-Attention Block (Spatial + XCA Spectral) and a PPNMM Non-Linear Decoder."
    ]),
    create_cell("code", [
        "dataset = HyperspectralDataset('uav_synthetic', data_dir='../data/raw', device=device)",
        "model = DASUNet(",
        "    num_endmembers=dataset.P,",
        "    num_bands=dataset.L,",
        "    spatial_size=dataset.col,",
        "    encoder_type='swin',",
        "    decoder_type='nonlinear',",
        "    use_dual_attention=True,",
        "    nonlinear_gamma=0.45",
        ").to(device)",
        "model.apply(model.weights_init)",
        "model.init_decoder_weights(dataset.get_image_cube(), method='sivm')",
        "print(model)"
    ]),
    create_cell("markdown", [
        "## 2. Train DASU-Net Model"
    ]),
    create_cell("code", [
        "optimizer = torch.optim.Adam(model.parameters(), lr=4e-3, weight_decay=4e-5)",
        "scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.8)",
        "loss_fn = TotalLoss(num_bands=dataset.L, beta=2500, gamma=0.015, delta=5e-4, lambda_reg=5e-4).to(device)",
        "clipper = model.get_clipper()",
        "img_cube = dataset.get_image_cube()",
        "",
        "losses = []",
        "model.train()",
        "for epoch in range(150):",
        "    abu, recon = model(img_cube)",
        "    endmem = model.decoder.get_endmembers()",
        "    loss = loss_fn(recon, img_cube, abu, endmem)",
        "    optimizer.zero_grad()",
        "    loss.backward()",
        "    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0, norm_type=1)",
        "    optimizer.step()",
        "    model.decoder.apply(clipper)",
        "    scheduler.step()",
        "    losses.append(loss.item())",
        "    if (epoch + 1) % 30 == 0:",
        "        print(f'Epoch {epoch+1}/150 - Loss: {loss.item():.4f}')"
    ]),
    create_cell("markdown", [
        "## 3. Evaluation & Hungarian Matching"
    ]),
    create_cell("code", [
        "model.eval()",
        "with torch.no_grad():",
        "    abu, recon = model(img_cube)",
        "abu = abu / abu.sum(dim=1, keepdim=True).clamp(min=1e-8)",
        "abu_np = abu.squeeze(0).permute(1, 2, 0).cpu().numpy()",
        "target_np = dataset.get_abundance_cube().cpu().numpy()",
        "est_endmem = model.decoder.get_endmembers().cpu().numpy()",
        "true_endmem = dataset.get_endmembers().numpy()",
        "",
        "est_endmem, abu_np, perm = match_endmembers(est_endmem, true_endmem, abu_np, target_np)",
        "rmse_cls, rmse_mean = compute_rmse(abu_np, target_np)",
        "sad_cls, sad_mean = compute_sad(est_endmem, true_endmem)",
        "",
        "print(f'=== Evaluation Results ===')",
        "print(f'Mean RMSE: {rmse_mean:.4f}')",
        "print(f'Mean SAD:  {sad_mean:.4f} rad')"
    ])
]

# -------------------------------------------------------------
# Notebook 3: 03_uav_qualitative_and_error_maps.ipynb
# -------------------------------------------------------------
nb3_cells = [
    create_cell("markdown", [
        "# 🎨 Notebook 03: Qualitative Analysis & Spatial Error Difference Maps",
        "",
        "**Conference:** 2026 IEEE 8th International Conference 'Actual Problems of Unmanned Aerial Vehicles Development' (APUAVD-2026)",
        "**Author:** Yaroslav Fetisov (NTUU 'KPI')",
        "",
        "This notebook generates high-contrast **Absolute Error Maps** ($\Delta = |\\hat{A} - A_{\\text{GT}}|$) and publication-ready 300 DPI figures."
    ]),
    create_cell("code", [
        "import sys",
        "sys.path.append('..')",
        "import numpy as np",
        "import matplotlib.pyplot as plt",
        "from PIL import Image",
        "",
        "# Load precomputed 300 DPI publication figure",
        "fig_img = Image.open('../runs/experiments_uav/qualitative_comparison_combined.png')",
        "plt.figure(figsize=(14, 6), dpi=150)",
        "plt.imshow(fig_img)",
        "plt.axis('off')",
        "plt.title('DASU-Net UAV Unmixing & Error Maps (IEEE APUAVD 2026)', fontsize=12, fontweight='bold')",
        "plt.show()"
    ])
]

out_dir = Path("c:/Users/R3ap3r/Documents/dl_coursework/DASU-Net-UAV/notebooks")
out_dir.mkdir(parents=True, exist_ok=True)

with open(out_dir / "01_uav_data_and_eda.ipynb", "w", encoding="utf-8") as f:
    json.dump(create_notebook(nb1_cells), f, indent=2)

with open(out_dir / "02_uav_training_and_benchmark.ipynb", "w", encoding="utf-8") as f:
    json.dump(create_notebook(nb2_cells), f, indent=2)

with open(out_dir / "03_uav_qualitative_and_error_maps.ipynb", "w", encoding="utf-8") as f:
    json.dump(create_notebook(nb3_cells), f, indent=2)

print("Created 3 clean UAV notebooks in DASU-Net-UAV/notebooks/ successfully!")
