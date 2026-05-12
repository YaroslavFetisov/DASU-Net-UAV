# DASU-Net 🌌

![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch)
![Optuna](https://img.shields.io/badge/Optuna-Hyperparameter_Tuning-blue?style=for-the-badge&logo=optuna)
![License](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)

**DASU-Net** is a state-of-the-art deep AutoEncoder framework designed for **Blind Hyperspectral Unmixing (HSU)**. It estimates pure material spectra (endmembers) and their fractional compositions (abundances) from mixed hyperspectral pixels in an unsupervised manner.

Building upon standard transformer-based baselines, this `v2` release fundamentally re-architects the feature extraction and decoding pathways, reducing the reconstruction error (RMSE) by **over 70%** compared to standard Vision Transformer (ViT) approaches.

---

## 🚀 Key Innovations & Architecture

1. **Swin Transformer over ViT**: Replaced global self-attention $\mathcal{O}(N^2)$ with windowed attention $\mathcal{O}(N \cdot ws^2)$. This significantly reduces computational complexity and memory footprint while better capturing local spatial contexts.
2. **Dual Attention Block (Spatial + XCA Spectral)**: Implements parallel spatial and Cross-Covariance Attention (XCA). XCA operates across the channel dimension, efficiently modeling spectral correlations without the $N \times N$ bottleneck.
3. **U-Net Style Abundance Decoder**: Unlike traditional methods that flatten latent vectors into spatial maps via a single linear layer (destroying high-frequency details), v2 employs a `ConvTranspose2d` pathway equipped with **Skip-Connections**. It fuses high-level semantic embeddings with low-level CNN features to preserve precise geographical boundaries.
4. **Non-linear PPNMM Decoder**: Upgraded from the linear $Y = EA$ constraint to a Polynomial Post-Nonlinear Mixing Model (PPNMM), simulating secondary photon reflections and scattering interferences ($\gamma \cdot (e_i \odot e_j)$).
5. **Physics-Informed Regularization**:
    *   **Warm Start Initialization**: Decoder endmembers are initialized via geometric projections (SiVM/VCA) to avoid local minima.
    *   **MinVol Penalty**: Uses numerically stable Cholesky decomposition ($\log \det(E^T E + \epsilon I)$) to prevent degenerate, non-physical endmember simplices.
    *   **Entropy Sparsity**: Penalizes abundance distribution entropy ($-A \log A$) instead of standard $L_1$, forcing physical sparsity under sum-to-one Softmax constraints.

---

## 📊 Benchmark Results

Evaluated on the widely adopted **Samson** dataset (156 bands, $95 \times 95$ spatial, 3 endmembers). 

| Metric | Baseline (ViT + Linear) | **DASU-Net (Swin + U-Net + PPNMM)** | Improvement |
| :--- | :---: | :---: | :---: |
| **Mean RMSE** | 0.5474 | **0.1585** | **-71.0%** 🚀 |
| **Mean SAD** | 0.4091 | **0.1938** | **-52.6%** 🚀 |
| **Water SAD** | 1.0950 | **0.4492** | Solved extreme absorption failure |
| **Soil/Rock SAD** | ~0.0390 | **0.0125** | Near-perfect spectral recovery |

*Note: The U-Net decoder with skip-connections accounted for ~50% of the RMSE drop. Optuna hyperparameter tuning and mathematical regularizations contributed the remaining ~20%.*

---

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/DASU-Net.git
cd DASU-Net

# Create Conda environment
conda create -n hsu_env python=3.9 -y
conda activate hsu_env

# Install requirements
pip install -r requirements.txt
```

---

## 💻 Usage

### 1. Standard Training
To run the model using the default configurations (Samson dataset):
```bash
python main.py --config configs/base.yaml
```

### 2. Training with Optimized Architecture
Apply the SOTA architecture and best hyperparameters:
```bash
python main.py --config configs/base.yaml --override configs/best_samson_hsu_search.yaml
```

### 3. AutoML Hyperparameter Tuning (Optuna)
The framework includes a fully automated, SQLite-backed hyperparameter tuner using the Tree-structured Parzen Estimator (TPE) algorithm. 
```bash
python tune.py --n-trials 100 --epochs 200 --study-name my_hsu_search
```
Because of the Swin Transformer's linear complexity, evaluating 200 epochs on an RTX 4070 Ti takes only ~2-3 seconds, making massive search spaces computationally trivial.

---

## 📂 Project Structure

```text
DASU-Net/
├── configs/                  # YAML configurations (base & overrides)
├── data/raw/                 # .mat hyperspectral datasets
├── notebooks/
│   └── 02_comparison.ipynb   # Visual comparison & metric evaluation
├── src/
│   ├── core/                 # Losses, metrics, SiVM/VCA initialization
│   ├── data/                 # Dataset and DataLoader logic
│   ├── models/               # Encoders, Decoders, and DASUNet AutoEncoder
│   └── utils/                # Plotting tools and config parsers
├── main.py                   # Main training loop
└── tune.py                   # Optuna hyperparameter search
```

---

## 📜 Citation & License
This project is an open-source research implementation developed for academic coursework. Feel free to use, modify, and distribute under the MIT License.

*If you find this repository useful for your research, please consider leaving a ⭐.*
