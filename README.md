# DASU-Net-UAV 🛸🌾

[![IEEE APUAVD 2026](https://img.shields.io/badge/IEEE_APUAVD-2026_Submission-blue?style=for-the-badge&logo=ieee)](http://apuavd.ieee.org.ua/)
[![Python 3.10](https://img.shields.io/badge/Python-3.10-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Optuna](https://img.shields.io/badge/Optuna-HPO-4169E1?style=for-the-badge&logo=optuna)](https://optuna.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)

**DASU-Net-UAV** (Dual-Attention Swin U-Net for UAV Hyperspectral Remote Sensing) is an autonomous, lightweight deep AutoEncoder architecture designed for **Blind Hyperspectral Unmixing (HSU)** on Unmanned Aerial Vehicle (UAV) platforms.

Developed for the **2026 IEEE 8th International Conference "Actual Problems of Unmanned Aerial Vehicles Development" (APUAVD-2026)**, Kyiv, Ukraine.

**Author:** Yaroslav Fetisov  
**Affiliation:** Department of Computer Science, National Technical University of Ukraine *"Igor Sikorsky Kyiv Polytechnic Institute"* (NTUU "KPI"), Kyiv, Ukraine  
**Email:** [yaroslavf222@gmail.com](mailto:yaroslavf222@gmail.com)  

---

## 📌 UAV Operational Context & Motivation

Hyperspectral imaging from low-altitude Unmanned Aerial Vehicles (UAVs / drones) enables centimeter-grade sub-pixel material identification for precision agriculture, environmental monitoring, emergency response, and infrastructure inspection. 

However, onboard processing of UAV hyperspectral cubes presents unique physical and operational challenges:
1. **Micro-Scale Multi-Bounce Scattering:** Low-altitude imagery of vegetative crop canopies and soil exhibits non-linear radiative transfer and bilinear photon interactions that violate the classical Linear Mixing Model (LMM).
2. **Platform Dynamics & Motion Blur:** Angular platform vibrations and flight kinematics require models that preserve fine spatial field boundaries without quadratic blurring.
3. **Severe Edge Computing Constraints:** UAV onboard embedded computing modules (e.g., NVIDIA Jetson AGX Orin / Xavier) operate under strict payload weight and power-budget (SWaP) limits, making standard Vision Transformers ($\mathcal{O}(N^2)$ complexity) impractical for real-time operation.

**DASU-Net** directly solves these challenges through a lightweight, linear-complexity attention mechanism coupled with a physics-informed non-linear decoder and boundary-preserving skip connections.

---

## 🚀 Key Architectural Highlights

![DASU-Net Architecture](assets/architecture.png)

1. **Shifted-Window Swin Transformer Encoder:** Replaces global self-attention with windowed attention, reducing computational complexity from $\mathcal{O}(N^2)$ to $\mathcal{O}(N \cdot w_s^2)$ for efficient execution on drone compute units.
2. **Parallel Dual Attention (Spatial W-MSA + Spectral XCA):** Computes local windowed spatial attention alongside Cross-Covariance Attention (XCA) across spectral channels in $\mathcal{O}(N \cdot C^2)$ time with a learnable gated residual.
3. **U-Net Abundance Decoder with Skip Connections:** Restores high-frequency spatial boundaries and subtle parcel micro-textures degraded by UAV motion dynamics via `ConvTranspose2d` and multi-scale feature concatenation.
4. **Physics-Informed PPNMM Non-Linear Decoder:** Faithfully models secondary photon reflections ($\gamma \sum_{i<j} a_i a_j (\mathbf{e}_i \odot \mathbf{e}_j)$) characteristic of low-altitude UAV canopy sensing.
5. **Physics-Informed Regularization & Geometric Warm-Start:**
   - **Simplex Volume Regularization ($\mathcal{L}_{\text{MinVol}}$):** Numerically stable Cholesky-based $\log\det(\mathbf{E}^\top \mathbf{E} + \varepsilon \mathbf{I})$ constraint preventing degenerate solutions.
   - **Entropy Sparsity ($\mathcal{L}_{\text{Ent}}$):** Physical sparsity enforcement compatible with ASC Softmax constraints.
   - **SiVM & VCA Warm Start:** Deterministic endmember initialization providing rapid convergence and avoidance of suboptimal local minima.

---

## 🗂️ Benchmarks

Both benchmarks are simulated by [`src/data/generate_uav_datasets.py`](src/data/generate_uav_datasets.py), so abundance and endmember ground truth is known exactly. The generator is deterministic: missing `.mat` files are recreated bit-identically in `data/raw/` on first use.

| Key | Benchmark | Bands | Endmembers | Size | Mixing & degradations |
| :--- | :--- | :---: | :---: | :---: | :--- |
| `uav_synthetic` | UAV Flight Synthetic | 150 (400–900 nm) | 4: crop, soil, water, road | 100 × 100 | PPNMM (γ = 0.45), motion blur, 30 dB noise |
| `whu_hi_longkou` | LongKou-like synthetic | 270 (400–1000 nm) | 5: crop, canopy, soil, water, road | 100 × 100 | PPNMM (γ = 0.35), motion blur, 32 dB noise |

Endmember spectra are analytic reflectance models. The `whu_hi_longkou` benchmark (the *LongKou-configured benchmark* in the APUAVD-2026 paper) follows the band configuration of the WHU-Hi LongKou Headwall Nano-Hyperspec acquisition but is fully simulated and contains no pixels of the original image; the original WHU-Hi LongKou scene is distributed with a land-cover classification map rather than abundance ground truth.

---

## 📊 Experimental Results

All numbers were produced with a single command on an NVIDIA GeForce RTX 4070 Ti (PyTorch 2.5.1, CUDA 11.8):

```bash
python run_uav_experiments.py --seeds 42 43 44 45 46
```

The complete output, including every per-seed value, is stored in [`results/all_results.json`](results/all_results.json). Training on CUDA is not bit-wise deterministic, so repeated runs give slightly different numbers; each table therefore lists the run with the script's default seed (42) next to the mean ± std over the five seeds.

### 1. Quantitative Benchmark Comparison

| Benchmark | Method | Init | RMSE ↓ (seed 42) | SAD (rad) ↓ (seed 42) | RMSE ↓ (mean ± std) | SAD (rad) ↓ (mean ± std) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **UAV Synthetic** | ViT baseline | VCA | 0.0263 | 0.0174 | 0.0294 ± 0.0020 | 0.0171 ± 0.0011 |
| | DASU-Net | SiVM | 0.0136 | 0.0150 | 0.0127 ± 0.0013 | 0.0145 ± 0.0004 |
| | DASU-Net | VCA | 0.0115 | 0.0134 | 0.0147 ± 0.0037 | 0.0154 ± 0.0018 |
| **LongKou-like** | ViT baseline | VCA | 0.0805 | 0.0142 | 0.0940 ± 0.0092 | 0.0216 ± 0.0048 |
| | DASU-Net | SiVM | 0.0630 | 0.0130 | 0.0804 ± 0.0162 | 0.0145 ± 0.0038 |
| | DASU-Net | VCA | 0.0502 | 0.0114 | 0.0807 ± 0.0286 | 0.0166 ± 0.0070 |

On average over the five seeds DASU-Net lowers abundance RMSE by 57 % (SiVM) / 50 % (VCA) on UAV Synthetic and by 14 % (SiVM / VCA) on the LongKou-like benchmark, where the result varies strongly between seeds.

The ViT baseline re-implements the DeepTrans-HSU ViT encoder with a linear decoder (`encoder_type="vit"`, `decoder_type="linear"` in [`src/models/unmixer.py`](src/models/unmixer.py)) and uses the same skip-connected abundance decoder as DASU-Net.

### 2. Incremental Architectural Ablation Study

UAV Synthetic benchmark; all five configurations share the same optimiser settings and 250-epoch budget (see `run_ablation_study` in [`run_uav_experiments.py`](run_uav_experiments.py)).

| # | Configuration | Change in the code w.r.t. the previous row | RMSE ↓ (seed 42) | SAD (rad) ↓ (seed 42) | RMSE ↓ (mean ± std) | SAD (rad) ↓ (mean ± std) |
| :-: | :--- | :--- | :---: | :---: | :---: | :---: |
| (1) | ViT + Linear Decoder (Baseline) | ViT encoder, linear decoder, VCA init | 0.0257 | 0.0144 | 0.0297 ± 0.0030 | 0.0172 ± 0.0024 |
| (2) | + Swin Transformer Encoder | ViT → Swin encoder | 0.0303 | 0.0163 | 0.0298 ± 0.0029 | 0.0168 ± 0.0020 |
| (3) | + Dual Attention (Spatial + XCA) | dual-attention block on; MinVol / entropy weights 0 → 5e-4 | 0.0289 | 0.0153 | 0.0297 ± 0.0024 | 0.0172 ± 0.0019 |
| (4) | + U-Net Skip Decoder | VCA → SiVM init (the skip-connected abundance decoder is used in every row) | 0.0264 | 0.0150 | 0.0306 ± 0.0036 | 0.0181 ± 0.0022 |
| (5) | + PPNMM Decoder (**Full DASU-Net**) | linear → PPNMM non-linear decoder | 0.0123 | 0.0145 | 0.0153 ± 0.0040 | 0.0157 ± 0.0017 |

### 3. Hardware Profiling

One 270-band 100 × 100 cube (P = 5), batch size 1, `torch.no_grad()`, 20 warm-up passes; latency is the median of 5 × 100 timed passes, and peak memory is the PyTorch-allocated memory (input, weights, activations) with only that model loaded.

| Metric | ViT baseline | DASU-Net |
| :--- | :---: | :---: |
| **Parameters** | 25.54 M | **17.18 M** (−32.7 %) |
| **GFLOPs per cube** | 6.56 | **4.77** |
| **Inference Latency** | 1.75 ms | 3.24 ms |
| **Throughput (FPS)** | 570.8 | 309.0 |
| **Peak GPU Memory** | 136.6 MB | 122.8 MB |

Measured on a desktop RTX 4070 Ti; embedded UAV computers such as NVIDIA Jetson modules are slower. GFLOPs were counted on the CPU with `torch.utils.flop_counter.FlopCounterMode` for the same input (convolutions and matrix products); the PPNMM decoder accounts for 0.08 of DASU-Net's 4.77 GFLOPs.

---

## 🖼️ Qualitative Evaluation

Example output of `run_uav_experiments.py` on the UAV Synthetic benchmark (ground truth, ViT baseline and DASU-Net with SiVM initialization).

### Fractional Abundance Estimation
![Abundance Comparison](assets/abundance_comparison.png)

### Endmember Spectral Signature Extraction
![Spectral Signatures](assets/spectral_signatures.png)

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

## 💻 Usage & Reproduction Workflows

### 0. Experiment Suite (`run_uav_experiments.py`)
Trains the ViT baseline and DASU-Net (SiVM / VCA) on both benchmarks, runs the 5-step ablation study and the hardware profiling, then writes `runs/experiments_uav/all_results.json` and two 300 DPI figures (about 4 minutes for five seeds on an RTX 4070 Ti):
```bash
python run_uav_experiments.py --seeds 42 43 44 45 46
```
Hyperparameters of this script are fixed in the code. `--out-dir` changes the output folder and `--sync-paper-dir <dir>` additionally copies the figures into an existing directory. Missing benchmark cubes are generated automatically (see the Benchmarks section above).

### 1. End-to-End Automated Scientific Benchmark (`benchmark.py`)
Executes automated Bayesian hyperparameter optimization (Optuna TPE), followed by full convergence training with plateau early-stopping and Hungarian matching evaluation. Results are written to `runs/benchmark/`, which is cleared at start-up:
```bash
python benchmark.py --tune-trials 100 --tune-epochs 150 --max-epochs 1000 --patience 60
```

### 2. Standalone Hyperparameter Tuning (`tune.py`)
Performs isolated Optuna HPO study with SQLite persistence and exports optimized configuration YAMLs:
```bash
python tune.py --n-trials 100 --epochs 150 --dataset uav_synthetic --study-name uav_hsu_search
```

### 3. Single-Run Training (`main.py`)
Trains DASU-Net using standard or tuned configuration files:
```bash
# Default configuration
python main.py --config configs/base.yaml

# Tuned UAV experiment
python main.py --config configs/uav_synthetic_dasunet.yaml
```

### 4. Interactive Jupyter Notebooks
Explore the dataset pipelines, model architectures, and qualitative unmixing results interactively:
- [`notebooks/01_uav_data_and_eda.ipynb`](notebooks/01_uav_data_and_eda.ipynb): UAV dataset loading, spectral signature exploration, and SNR analysis.
- [`notebooks/02_uav_training_and_benchmark.ipynb`](notebooks/02_uav_training_and_benchmark.ipynb): Model training, ablation tracking, and metric logging.
- [`notebooks/03_uav_qualitative_and_error_maps.ipynb`](notebooks/03_uav_qualitative_and_error_maps.ipynb): High-resolution abundance visualization and pixel-wise error heatmaps.

### 5. On-Board Drone Tensor Verification
Verify tensor dimensions, receptive fields, and numerical gradients:
```bash
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
├── assets/                   # Publication figures, diagrams, and comparison plots
│   ├── architecture.png      # DASU-Net pipeline diagram
│   ├── abundance_comparison.png
│   └── spectral_signatures.png
├── configs/                  # Experiment YAML configuration files
│   ├── base.yaml             # Base training and model parameters
│   ├── uav_synthetic_dasunet.yaml
│   ├── whu_hi_longkou_dasunet.yaml
│   ├── exp_swin.yaml         # Swin encoder ablation config
│   ├── exp_nonlinear.yaml    # PPNMM non-linear decoder config
│   └── exp_full.yaml         # Full DASU-Net configuration
├── data/
│   ├── raw/                  # Synthetic benchmark cubes (*.mat, generated on first use)
│   └── processed/            # Preprocessed dataset caches
├── notebooks/                # Interactive UAV analysis and evaluation notebooks
│   ├── 01_uav_data_and_eda.ipynb
│   ├── 02_uav_training_and_benchmark.ipynb
│   └── 03_uav_qualitative_and_error_maps.ipynb
├── results/
│   └── all_results.json      # Output of the run reported above (all seeds)
├── src/
│   ├── core/                 # Loss functions, initialization (SiVM/VCA), metrics
│   ├── data/                 # HSI Dataset loaders and synthetic UAV benchmark generator
│   ├── models/               # Encoders, Decoders, and DASUNet architecture
│   └── utils/                # YAML parser, logger, and visualization utilities
├── benchmark.py              # Automated HPO & benchmark orchestration script
├── main.py                   # Single-run model training script
├── run_uav_experiments.py    # Benchmark, ablation and profiling suite
├── tune.py                   # Optuna hyperparameter optimization script
├── requirements.txt          # Python dependencies
├── LICENSE                   # MIT License
└── README.md                 # Project documentation
```

---

## 📑 Citation & Academic Reference

If you use this codebase or model in your research, please cite our conference paper:

```bibtex
@inproceedings{fetisov2026dasunet,
  author    = {Fetisov, Yaroslav},
  title     = {{DASU-Net}: Dual-Attention Swin {U-Net} for Non-Linear Hyperspectral Unmixing in {UAV} Remote Sensing},
  booktitle = {Proc. 2026 IEEE 8th International Conference "Actual Problems of Unmanned Aerial Vehicles Development" (APUAVD)},
  year      = {2026},
  address   = {Kyiv, Ukraine},
  note      = {Submitted}
}
```

---

## 📜 License
This project is open-source software licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.
