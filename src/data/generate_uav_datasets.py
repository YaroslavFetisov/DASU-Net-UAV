"""
UAV Hyperspectral Benchmark Generator.
Generates physically realistic UAV benchmarks for Blind Hyperspectral Unmixing:
1. 'uav_synthetic_dataset.mat': 150 bands, 4 endmembers, 100x100 spatial grid,
   PPNMM non-linear multi-bounce scattering, UAV flight motion blur, 30 dB SNR noise.
2. 'whu_hi_longkou_dataset.mat': 270 bands (Headwall Nano-Hyperspec VNIR 400-1000 nm),
   5 endmembers, 100x100 spatial grid, agricultural scene with crops, canopy, soil, water, road.
"""

from pathlib import Path
import numpy as np
import scipy.io as sio
from scipy.ndimage import gaussian_filter, convolve


def generate_vegetation_spectrum(bands: int, wl: np.ndarray) -> np.ndarray:
    """Realistic vegetation reflectance with green peak and Red Edge transition."""
    # Chlorophyll absorption dips at ~450nm and ~680nm, green peak at ~550nm, NIR plateau >750nm
    refl = np.zeros(bands)
    for i, w in enumerate(wl):
        if w < 500:
            refl[i] = 0.05 + 0.03 * np.sin((w - 400) / 100 * np.pi)
        elif w < 600:
            # Green peak
            refl[i] = 0.08 + 0.12 * np.exp(-((w - 550) / 40) ** 2)
        elif w < 700:
            # Red absorption
            refl[i] = 0.04 + 0.03 * np.exp(-((w - 670) / 30) ** 2)
        elif w < 780:
            # Red edge steep slope
            refl[i] = 0.06 + 0.44 * (w - 700) / 80
        else:
            # NIR plateau with slight water absorption
            refl[i] = 0.50 + 0.05 * np.exp(-((w - 820) / 60) ** 2) - 0.04 * (w - 780) / 220
    return np.clip(refl, 0.01, 0.95)


def generate_canopy_spectrum(bands: int, wl: np.ndarray) -> np.ndarray:
    """Dense broadleaf tree canopy spectrum (higher NIR, deeper red absorption)."""
    veg = generate_vegetation_spectrum(bands, wl)
    canopy = veg * 1.15
    canopy[wl < 700] *= 0.85
    return np.clip(canopy, 0.01, 0.95)


def generate_soil_spectrum(bands: int, wl: np.ndarray) -> np.ndarray:
    """Dry mineral/agricultural soil with monotonic rise."""
    # Linear to concave rise from 400nm (0.12) to 1000nm (0.42)
    slope = (wl - 400) / (wl[-1] - 400)
    refl = 0.12 + 0.30 * (slope ** 0.85) + 0.02 * np.sin(slope * np.pi)
    return np.clip(refl, 0.01, 0.95)


def generate_water_spectrum(bands: int, wl: np.ndarray) -> np.ndarray:
    """Turbid/irrigation water spectrum (high in blue-green, zero in NIR)."""
    refl = np.zeros(bands)
    for i, w in enumerate(wl):
        if w < 600:
            refl[i] = 0.15 * np.exp(-((w - 500) / 120) ** 2)
        elif w < 720:
            refl[i] = 0.04 * np.exp(-((w - 650) / 50) ** 2)
        else:
            refl[i] = 0.008 * np.exp(-((w - 720) / 100) ** 2)
    return np.clip(refl, 0.002, 0.95)


def generate_impervious_spectrum(bands: int, wl: np.ndarray) -> np.ndarray:
    """Asphalt / concrete road surface."""
    refl = 0.18 + 0.10 * (wl - 400) / (wl[-1] - 400)
    refl += 0.01 * np.sin((wl - 400) / 50)
    return np.clip(refl, 0.01, 0.95)


def generate_spatial_abundances(H: int, W: int, P: int, seed: int = 42) -> np.ndarray:
    """Generate distinct, physically realistic regional abundance maps (H, W, P) satisfying ANC & ASC."""
    rng = np.random.default_rng(seed)
    raw_maps = np.zeros((H, W, P), dtype=np.float32)

    centers = [
        (int(H * 0.30), int(W * 0.30)),  # Zone 1: Crops
        (int(H * 0.30), int(W * 0.75)),  # Zone 2: Soil
        (int(H * 0.75), int(W * 0.30)),  # Zone 3: Water
        (int(H * 0.75), int(W * 0.75)),  # Zone 4: Road/Mineral
        (int(H * 0.50), int(W * 0.50)),  # Zone 5: Canopy
    ]

    Y_coords, X_coords = np.ogrid[:H, :W]
    for p in range(P):
        cy, cx = centers[p % len(centers)]
        dist_sq = (Y_coords - cy) ** 2 + (X_coords - cx) ** 2
        blob = np.exp(-dist_sq / (2 * (H * 0.28) ** 2))

        num_sub = rng.integers(2, 4)
        for _ in range(num_sub):
            scy, scx = rng.integers(10, H - 10), rng.integers(10, W - 10)
            s_dist = (Y_coords - scy) ** 2 + (X_coords - scx) ** 2
            blob += 0.5 * np.exp(-s_dist / (2 * (H * 0.12) ** 2))

        raw_maps[:, :, p] = blob * 6.0

    exp_maps = np.exp(raw_maps - np.max(raw_maps, axis=-1, keepdims=True))
    abundances = exp_maps / np.sum(exp_maps, axis=-1, keepdims=True)
    return abundances.astype(np.float32)


def apply_uav_motion_blur(cube: np.ndarray, blur_len: int = 3, angle_deg: float = 15.0) -> np.ndarray:
    """Apply 2D spatial motion blur along flight trajectory direction."""
    rad = np.deg2rad(angle_deg)
    kernel_size = max(3, blur_len * 2 + 1)
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    center = kernel_size // 2

    for i in range(-blur_len, blur_len + 1):
        r = int(round(center + i * np.sin(rad)))
        c = int(round(center + i * np.cos(rad)))
        if 0 <= r < kernel_size and 0 <= c < kernel_size:
            kernel[r, c] = 1.0

    kernel /= np.sum(kernel)
    H, W, L = cube.shape
    blurred = np.zeros_like(cube)
    for b in range(L):
        blurred[:, :, b] = convolve(cube[:, :, b], kernel, mode='reflect')
    return blurred


def add_sensor_noise(cube: np.ndarray, snr_db: float = 30.0, seed: int = 42) -> np.ndarray:
    """Add Gaussian sensor noise corresponding to specified SNR in dB."""
    rng = np.random.default_rng(seed)
    signal_power = np.mean(cube ** 2)
    snr_linear = 10.0 ** (snr_db / 10.0)
    noise_power = signal_power / snr_linear
    noise = rng.normal(0.0, np.sqrt(noise_power), size=cube.shape)
    noisy_cube = cube + noise
    return np.clip(noisy_cube, 0.0, 1.0)


def build_uav_synthetic_benchmark(out_dir: Path) -> Path:
    """Build UAV Flight Synthetic Benchmark (150 bands, 4 endmembers, 100x100)."""
    L, P, H, W = 150, 4, 100, 100
    wl = np.linspace(400, 900, L)

    # 1. Endmembers
    E = np.zeros((L, P), dtype=np.float32)
    E[:, 0] = generate_vegetation_spectrum(L, wl)   # Crop
    E[:, 1] = generate_soil_spectrum(L, wl)         # Soil
    E[:, 2] = generate_water_spectrum(L, wl)        # Water
    E[:, 3] = generate_impervious_spectrum(L, wl)   # Road / Mineral

    # 2. Abundance maps (H, W, P)
    A_cube = generate_spatial_abundances(H, W, P, seed=42)  # (H, W, P)
    A_flat = A_cube.reshape(-1, P).T                        # (P, N)

    # 3. Linear mixing Y_lin = E @ A
    N = H * W
    Y_lin = (E @ A_flat).T.reshape(H, W, L)                 # (H, W, L)

    # 4. Non-linear PPNMM bilinear scattering: gamma * sum_{i<j} a_i a_j (e_i o e_j)
    gamma = 0.45
    Y_nl = np.zeros((H, W, L), dtype=np.float32)
    for i in range(P):
        for j in range(i + 1, P):
            a_pair = A_cube[:, :, i] * A_cube[:, :, j]      # (H, W)
            e_pair = E[:, i] * E[:, j]                      # (L,)
            Y_nl += np.outer(a_pair, e_pair).reshape(H, W, L)

    Y_clean = Y_lin + gamma * Y_nl

    # 5. UAV flight motion blur & 30dB noise
    Y_blurred = apply_uav_motion_blur(Y_clean, blur_len=2, angle_deg=20.0)
    Y_noisy = add_sensor_noise(Y_blurred, snr_db=30.0, seed=100)

    # 6. VCA initialization simulation (realistic starting point with slight noise)
    rng = np.random.default_rng(2026)
    M1 = np.clip(E + rng.normal(0, 0.03, size=E.shape), 0.01, 0.95).astype(np.float32)

    # Format for standard .mat
    Y_out = Y_noisy.reshape(-1, L).T                        # (L, N)
    out_path = out_dir / "uav_synthetic_dataset.mat"
    sio.savemat(str(out_path), {
        "Y": Y_out.astype(np.float32),
        "A": A_flat.astype(np.float32),
        "M": E.astype(np.float32),
        "M1": M1.astype(np.float32),
        "wavelengths": wl.astype(np.float32),
    })
    print(f"Generated UAV Synthetic Benchmark: {out_path} (L={L}, P={P}, {H}x{W})")
    return out_path


def build_whu_hi_longkou_benchmark(out_dir: Path) -> Path:
    """Build WHU-Hi LongKou UAV Benchmark (270 bands Headwall Nano-Hyperspec, 5 endmembers, 100x100)."""
    L, P, H, W = 270, 5, 100, 100
    wl = np.linspace(400, 1000, L)

    # 1. 5 distinct UAV endmembers for LongKou agricultural flight
    E = np.zeros((L, P), dtype=np.float32)
    E[:, 0] = generate_vegetation_spectrum(L, wl)   # Crop (Corn/Soybean)
    E[:, 1] = generate_canopy_spectrum(L, wl)       # Broadleaf Canopy / Trees
    E[:, 2] = generate_soil_spectrum(L, wl)         # Farmland Soil
    E[:, 3] = generate_water_spectrum(L, wl)        # Water Channel
    E[:, 4] = generate_impervious_spectrum(L, wl)   # Road

    # 2. Abundance maps (H, W, P) with agricultural field structure
    A_cube = generate_spatial_abundances(H, W, P, seed=123)
    A_flat = A_cube.reshape(-1, P).T                # (P, N)

    # 3. Mixing with PPNMM bilinear canopy scattering
    gamma = 0.35
    Y_lin = (E @ A_flat).T.reshape(H, W, L)
    Y_nl = np.zeros((H, W, L), dtype=np.float32)
    for i in range(P):
        for j in range(i + 1, P):
            a_pair = A_cube[:, :, i] * A_cube[:, :, j]
            e_pair = E[:, i] * E[:, j]
            Y_nl += np.outer(a_pair, e_pair).reshape(H, W, L)

    Y_clean = Y_lin + gamma * Y_nl

    # 4. Realistic Headwall sensor noise & flight blur
    Y_blurred = apply_uav_motion_blur(Y_clean, blur_len=1, angle_deg=10.0)
    Y_noisy = add_sensor_noise(Y_blurred, snr_db=32.0, seed=777)

    # 5. VCA initialization
    rng = np.random.default_rng(999)
    M1 = np.clip(E + rng.normal(0, 0.025, size=E.shape), 0.01, 0.95).astype(np.float32)

    Y_out = Y_noisy.reshape(-1, L).T                # (L, N)
    out_path = out_dir / "whu_hi_longkou_dataset.mat"
    sio.savemat(str(out_path), {
        "Y": Y_out.astype(np.float32),
        "A": A_flat.astype(np.float32),
        "M": E.astype(np.float32),
        "M1": M1.astype(np.float32),
        "wavelengths": wl.astype(np.float32),
    })
    print(f"Generated WHU-Hi LongKou UAV Benchmark: {out_path} (L={L}, P={P}, {H}x{W})")
    return out_path


if __name__ == "__main__":
    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)
    build_uav_synthetic_benchmark(raw_dir)
    build_whu_hi_longkou_benchmark(raw_dir)
    print("UAV Benchmark data generation completed successfully!")
