# DASU-Net-UAV 🛸🌾

[![IEEE APUAVD 2026](https://img.shields.io/badge/IEEE_APUAVD-2026_Submission-blue?style=for-the-badge&logo=ieee)](http://apuavd.ieee.org.ua/)
[![Python 3.10](https://img.shields.io/badge/Python-3.10-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Optuna](https://img.shields.io/badge/Optuna-HPO_Tuned-4169E1?style=for-the-badge&logo=optuna)](https://optuna.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)

**DASU-Net-UAV** (Dual-Attention Swin U-Net for UAV Hyperspectral Remote Sensing) is an autonomous, lightweight deep AutoEncoder architecture designed for **Blind Hyperspectral Unmixing (HSU)** on Unmanned Aerial Vehicle (UAV) platforms.

Developed for the **2026 IEEE 8th International Conference "Actual Problems of Unmanned Aerial Vehicles Development" (APUAVD-2026)**, Kyiv, Ukraine.

**Author:** Yaroslav Fetisov  
**Affiliation:** Department of Computer Science, National Technical University of Ukraine *"Igor Sikorsky Kyiv Polytechnic Institute"* (NTUU "KPI"), Kyiv, Ukraine  
**Email:** [yaroslavf222@gmail.com](mailto:yaroslavf222@gmail.com)  

---

## 📌 UAV Operational Context & Motivation

Hyperspectral imaging from low-altitude Unmanned Aerial Vehicles (UAVs / drones) enables centimeter-grade sub-pixel material identification for precision agriculture, environmental monitoring, emergency response, and infrastructure inspection. 

However, onboard processing of UAV hyperspectral cubes presents unique challenges:
1. **Micro-Scale Multi-Bounce Scattering:** Low-altitude imagery of crop canopies, forestry, and complex terrains exhibits strong non-linear bilinear photon scattering between vegetative layers and soil.
2. **Platform Dynamics & Motion Blur:** Angular platform vibrations and flight kinematics require models that preserve fine spatial boundaries without quadratic blurring.
3. **Severe Edge Computing Constraints:** UAV onboard embedded computing modules (e.g., NVIDIA Jetson AGX Orin / Xavier) operate under strict payload weight and power-budget (SWaP) limits, making standard Vision Transformers ($\mathcal{O}(N^2)$ complexity) impractical for real-time operation.

**DASU-Net** directly solves these challenges through a lightweight, linear-complexity attention mechanism coupled with a non-linear physical decoder and boundary-preserving skip connections.

---

## 🚀 Key Architectural Highlights

```
  ┌────────────────────────────────────────────────────────────────────────┐
  │                           DASU-Net Pipeline                            │
  └────────────────────────────────────────────────────────────────────────┘
    Input HSI Cube (L x H x W)
           │
           ▼
    ┌──────────────────────┐
    │  1x1 Conv Encoder    │ ───► Early CNN Features (Skip Connection)
    │  (L -> 128 -> 64-> C)│                     │
    └──────────┬───────────┘                     │
               │                                 │
               ▼                                 │
    ┌──────────────────────┐                     │
    │ Dual Attention Block │                     │
    │ Spatial + XCA Spectral                     │
    └──────────┬───────────┘                     │
               │                                 │
               ▼                                 │
    ┌──────────────────────┐                     │
    │ Swin Encoder (W-MSA) │                     │
    │ O(N * ws^2) Linear   │                     │
    └──────────┬───────────┘                     │
               │                                 │
               ▼                                 │
    ┌──────────────────────┐                     │
    │ Global Embedding (z) │                     │
    └──────────┬───────────┘                     │
               │                                 │
               ▼                                 ▼
    ┌──────────────────────────────────────────────┐
    │       U-Net Abundance Decoder                │
    │   (ConvTranspose2d + Skip Fusion + Softmax)  │
    └──────────────────────┬───────────────────────┘
                           │
                           ▼
                  Abundance Maps (A)
                           │
                           ▼
    ┌──────────────────────────────────────────────┐
    │        Non-Linear PPNMM Decoder              │
    │   Y_hat = EA + gamma * Sum(a_i a_j e_i o e_j)│
    └──────────────────────┬───────────────────────┘
                           │
                           ▼
               Reconstructed HSI Cube (Y_hat)
```

1. **Shifted-Window Swin Transformer Encoder:** Replaces global self-attention with windowed attention, reducing computational complexity from $\mathcal{O}(N^2)$ to $\mathcal{O}(N \cdot w_s^2)$ for efficient execution on drone compute units.
2. **Parallel Dual Attention (Spatial + Spectral XCA):** Computes local windowed spatial attention alongside Cross-Covariance Attention (XCA) across spectral channels in $\mathcal{O}(N \cdot C^2)$ time with a learnable gated residual.
3. **U-Net Abundance Decoder with Skip Connections:** Restores high-frequency spatial boundaries and subtle micro-textures degraded by UAV motion dynamics via `ConvTranspose2d` and multi-scale feature concatenation.
4. **PPNMM Polynomial Post-Nonlinear Decoder:** Faithfully models secondary photon reflections ($\gamma \sum_{i<j} a_i a_j (\mathbf{e}_i \odot \mathbf{e}_j)$) characteristic of low-altitude UAV canopy sensing.
5. **Physics-Informed Regularization & Geometric Warm-Start:**
   - **Simplex Volume Regularization ($\mathcal{L}_{\text{MinVol}}$):** Numerically stable Cholesky-based $\log\det(\mathbf{E}^\top \mathbf{E} + \varepsilon \mathbf{I})$ constraint preventing degenerate solutions.
   - **Entropy Sparsity ($\mathcal{L}_{\text{Ent}}$):** Physical sparsity enforcement compatible with ASC Softmax constraints.
   - **SiVM & VCA Warm Start:** Deterministic endmember initialization providing rapid convergence and avoidance of suboptimal local minima.

---

## 🛠️ Installation & Environment Setup

### 1. Clone the Repository
```bash
git clone https://github.com/YaroslavFetisov/DASU-Net-UAV.git
cd DASU-Net-UAV
```

### 2. Set Up Python Environment
```bash
# Create Conda environment
conda create -n dasunet_uav python=3.10 -y
conda activate dasunet_uav

# Install PyTorch (CUDA 11.8 / 12.1 recommended)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install dependencies
pip install -r requirements.txt
```

---

## 💻 Usage & Workflows

### 1. End-to-End Scientific Benchmark (`benchmark.py`)
Executes automated Bayesian hyperparameter optimization (Optuna TPE), followed by full convergence training with plateau early-stopping and Hungarian matching evaluation:
```bash
python benchmark.py --tune-trials 100 --tune-epochs 150 --max-epochs 1000 --patience 60
```

### 2. Standalone Hyperparameter Tuning (`tune.py`)
Performs isolated Optuna HPO study with SQLite persistence and exports optimized configuration YAMLs:
```bash
python tune.py --n-trials 100 --epochs 150 --dataset samson --study-name uav_hsu_search
```

### 3. Single-Run Training (`main.py`)
Trains DASU-Net using standard or tuned configuration files:
```bash
# Default configuration
python main.py --config configs/base.yaml

# Tuned experiment
python main.py --config configs/base.yaml --override configs/best_samson_hsu_search.yaml
```

---

## ⚡ On-Board UAV Inference & Latency Profiling

To benchmark model throughput and GPU memory footprint for onboard drone deployment (e.g., NVIDIA Jetson / edge platforms), run the built-in PyTorch verification suite:

```bash
# Verify all model tensor dimensions and components
python -m src.models.unmixer
python -m src.models.encoders
python -m src.models.decoders
python -m src.core.losses
python -m src.core.metrics
```

---

## 📂 Repository Structure

```text
DASU-Net-UAV/
├── configs/                  # Experiment YAML configuration files
│   ├── base.yaml             # Base training and model parameters
│   ├── exp_swin.yaml         # Swin encoder ablation config
│   ├── exp_nonlinear.yaml    # PPNMM non-linear decoder config
│   └── exp_full.yaml         # Full DASU-Net configuration
├── data/
│   ├── raw/                  # Hyperspectral cubes (*.mat, ignored by Git)
│   └── processed/            # Preprocessed datasets (*.npy / *.pt)
├── notebooks/                # Visual comparison and metric analysis tools
│   ├── 01_data_exploration.ipynb
│   ├── 02_comparison.ipynb
│   └── 03_ablation_initialization.ipynb
├── src/
│   ├── core/                 # Loss functions, initialization (SiVM/VCA), metrics
│   ├── data/                 # HSI Dataset loaders and preprocessing pipeline
│   ├── models/               # Encoders, Decoders, and DASUNet architecture
│   └── utils/                # YAML parser, logger, and visualization utilities
├── benchmark.py              # Automated HPO & benchmark orchestration script
├── main.py                   # Single-run model training script
├── tune.py                   # Optuna hyperparameter optimization script
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
```

---

## 📑 Citation & Academic Reference

If you use this codebase or model in your research, please cite our conference paper:

```bibtex
@inproceedings{fetisov2026dasunet,
  author    = {Fetisov, Yaroslav},
  title     = {{DASU-Net}: Dual-Attention Swin {U-Net} for Blind Hyperspectral Unmixing in {UAV} Remote Sensing},
  booktitle = {Proc. 2026 IEEE 8th International Conference "Actual Problems of Unmanned Aerial Vehicles Development" (APUAVD)},
  year      = {2026},
  pages     = {1--5},
  address   = {Kyiv, Ukraine},
  publisher = {IEEE}
}
```

---

## 📜 License
This project is open-source software licensed under the **MIT License**.
