"""
Script: Ablation Study — WGAN-GP Configuration vs Cardinalidad de Class

Trains a WGAN-GP for 3 representative classes from CIC-IDS2017
(one LARGE, one SMALL, and one VERY_SMALL) using the 3
architecture configurations (LARGE, SMALL, VERY_SMALL). Total: 3 × 3 = 9 trainings.

For each class, compares the 3 resulting generators by measuring:
  - Empirical Wasserstein distance (1D per feature, aggregated)
  - Multivariate MMD² (RBF kernel, median heuristic)

This validates whether the adaptive configuration assignment based on
class cardinality is effectively optimal.

Usage:
    python 9_ablation.py --parallel --gpus 1,2,3   # PARALLEL! 3 GPUs (recommended)
    python 9_ablation.py --gpu 1                   # Sequential on GPU 1
    python 9_ablation.py --wgan-scale 0.5          # 50% epochs (default)
    python 9_ablation.py --wgan-scale 0.1          # 10% epochs (quick test)
    python 9_ablation.py --gen-samples 5000        # Synthetic samples
"""

import os
import sys
import json
import time
import argparse
import subprocess
import numpy as np
import pandas as pd
import polars as pl
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.stats import wasserstein_distance
from sklearn.preprocessing import MinMaxScaler

# ── Configure GPU ANTES de importar TensorFlow ─────────────────────────────
# TensorFlow fija las GPUs visibles en el momento del import, así que
# CUDA_VISIBLE_DEVICES debe estar definido antes de `import tensorflow`.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
_is_parallel = '--parallel' in sys.argv
_is_worker = '--_worker' in sys.argv
_gpu_arg = '0'
for i, arg in enumerate(sys.argv):
    if arg == '--gpu' and i + 1 < len(sys.argv):
        _gpu_arg = sys.argv[i + 1]
        break
if _is_parallel and not _is_worker:
    # El orquestador no necesita GPU — los workers son procesos separados
    os.environ["CUDA_VISIBLE_DEVICES"] = ''
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = _gpu_arg

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = '<PATH_TO_CICIDS2017_CSV>'
LATENT_DIM = 100
RANDOM_STATE = 42
SMALL_CLASS_THRESHOLD = 15000

# Representative classes (automatically selected if not overridden)
# LARGE:      Benign (~2.3M)  → >15 000 samples
# SMALL:      Brute Force (~13k)  → 5 000–15 000 samples
# VERY_SMALL: Bot (~1966)     → <5 000 samples
DEFAULT_REPRESENTATIVES = {
    'large':      'Benign',
    'small':      'Brute Force',
    'very_small': 'Bot',
}

FEATURES_BASE = [
    'Source IP', 'Destination IP',
    'Source Port', 'Destination Port', 'Protocol',
    'Total Fwd Packets', 'Total Backward Packets',
    'Total Length of Fwd Packets', 'Total Length of Bwd Packets',
    'Flow Duration', 'Flow IAT Mean', 'Flow IAT Std',
    'Fwd IAT Mean', 'Bwd IAT Mean',
    'Fwd Packet Length Mean', 'Bwd Packet Length Mean',
    'Packet Length Std', 'Max Packet Length',
    'SYN Flag Count', 'ACK Flag Count', 'FIN Flag Count',
    'RST Flag Count', 'PSH Flag Count',
]
LABEL_COLUMN = 'Attack Type'
LOG_COLUMNS = [
    'Total Fwd Packets', 'Total Backward Packets',
    'Total Length of Fwd Packets', 'Total Length of Bwd Packets',
    'Flow Duration', 'Flow IAT Mean', 'Flow IAT Std',
    'Fwd IAT Mean', 'Bwd IAT Mean',
    'Fwd Packet Length Mean', 'Bwd Packet Length Mean',
    'Packet Length Std', 'Max Packet Length',
]

# ── Adaptive configurations (idénticas a 1_gan_wgan.py) ─────────────────
CONFIG_LARGE = {
    'name': 'LARGE',
    'batch_size': 128,
    'epochs': 15000,
    'n_critic': 5,
    'lambda_gp': 10.0,
    'generator_layers': [256, 512, 256],
    'critic_layers': [512, 256, 128],
    'learning_rate': 1e-4,
    'oversample_factor': 1,
    'noise_std': 0.0,
}
CONFIG_SMALL = {
    'name': 'SMALL',
    'batch_size': 32,
    'epochs': 25000,
    'n_critic': 3,
    'lambda_gp': 15.0,
    'generator_layers': [128, 256, 128],
    'critic_layers': [256, 128, 64],
    'learning_rate': 5e-5,
    'oversample_factor': 10,
    'noise_std': 0.02,
}
CONFIG_VERY_SMALL = {
    'name': 'VERY_SMALL',
    'batch_size': 16,
    'epochs': 30000,
    'n_critic': 2,
    'lambda_gp': 20.0,
    'generator_layers': [64, 128, 64],
    'critic_layers': [128, 64, 32],
    'learning_rate': 2e-5,
    'oversample_factor': 20,
    'noise_std': 0.03,
}

ALL_CONFIGS = {
    'LARGE': CONFIG_LARGE,
    'SMALL': CONFIG_SMALL,
    'VERY_SMALL': CONFIG_VERY_SMALL,
}


# ═══════════════════════════════════════════════════════════════════════════════
# LOADING AND PREPROCESSING (identical to 1_gan_wgan.py / 8_load_test.py)
# ═══════════════════════════════════════════════════════════════════════════════

def load_and_preprocess():
    """
    Loads CIC-IDS2017 and applies the same preprocessing as the
    proposed approach: IP expansion, log-transform, MinMax scaling to [-1, 1].
    """
    print("\n[DATA] Loading CIC-IDS2017...")
    t0 = time.time()
    df = pl.read_csv(DATASET_PATH, low_memory=False).to_pandas()
    df.columns = df.columns.str.strip()
    print(f"  Read in {time.time() - t0:.1f}s — {len(df):,} rows")

    df = df[FEATURES_BASE + [LABEL_COLUMN]].copy()

    # Expand IPs to octets
    for prefix, col_ip in [('Src_IP', 'Source IP'), ('Dst_IP', 'Destination IP')]:
        octetos = df[col_ip].astype(str).str.split('.', expand=True)
        for i in range(4):
            df[f'{prefix}_{i+1}'] = pd.to_numeric(octetos[i], errors='coerce').fillna(0).astype(int)
    df.drop(columns=['Source IP', 'Destination IP'], inplace=True)

    labels = df[LABEL_COLUMN].values
    features_df = df.drop(columns=[LABEL_COLUMN])
    feature_names = features_df.columns.tolist()

    # Log-transform
    features_proc = features_df.copy()
    for col in LOG_COLUMNS:
        if col in features_proc.columns:
            features_proc[col] = np.log1p(features_proc[col].clip(lower=0))
            features_proc[col] = features_proc[col].clip(lower=-20, upper=20)
    features_proc.replace([np.inf, -np.inf], np.nan, inplace=True)
    features_proc.fillna(0, inplace=True)

    scaler = MinMaxScaler(feature_range=(-1, 1))
    X = scaler.fit_transform(features_proc)

    # Class distribution
    unique, counts = np.unique(labels, return_counts=True)
    class_dist = dict(zip(unique, counts.astype(int)))
    print(f"  Classes: {len(unique)}")
    for c in sorted(class_dist, key=class_dist.get, reverse=True):
        tier = ("LARGE" if class_dist[c] >= SMALL_CLASS_THRESHOLD
                else "SMALL" if class_dist[c] >= 5000 else "VERY_SMALL")
        print(f"    [{tier:<10}] {c:<15}: {class_dist[c]:>10,}")

    return X, labels, feature_names, scaler, class_dist


def get_native_config_name(n_samples):
    """Returns the native configuration name based on cardinality."""
    if n_samples < 5000:
        return 'VERY_SMALL'
    elif n_samples < SMALL_CLASS_THRESHOLD:
        return 'SMALL'
    else:
        return 'LARGE'


# ═══════════════════════════════════════════════════════════════════════════════
# DATA AUGMENTATION (identical to 1_gan_wgan.py)
# ═══════════════════════════════════════════════════════════════════════════════

def oversample_with_noise(X, factor=10, noise_std=0.02):
    """Oversampling with Gaussian noise to densify the distribution."""
    if factor <= 1 and noise_std == 0:
        return X
    X_augmented = [X]
    for _ in range(factor - 1):
        noise = np.random.normal(0, noise_std, X.shape)
        X_noisy = np.clip(X + noise, -1, 1)
        X_augmented.append(X_noisy)
    X_final = np.vstack(X_augmented)
    np.random.shuffle(X_final)
    return X_final


# ═══════════════════════════════════════════════════════════════════════════════
# WGAN-GP MODELS (identical to 1_gan_wgan.py / 8_load_test.py)
# ═══════════════════════════════════════════════════════════════════════════════

def build_wgan_generator(latent_dim, output_dim, layer_sizes):
    """WGAN-GP Generator with BatchNorm and LeakyReLU."""
    inp = layers.Input(shape=(latent_dim,))
    x = inp
    for i, u in enumerate(layer_sizes):
        x = layers.Dense(u)(x)
        x = layers.LeakyReLU(0.2)(x)
        if i < len(layer_sizes) - 1:
            x = layers.BatchNormalization(momentum=0.8)(x)
    out = layers.Dense(output_dim, activation='tanh')(x)
    return models.Model(inp, out, name='WGAN_Generator')


def build_wgan_critic(input_dim, layer_sizes):
    """WGAN-GP Critic (no sigmoid, linear output)."""
    inp = layers.Input(shape=(input_dim,))
    x = inp
    for u in layer_sizes:
        x = layers.Dense(u)(x)
        x = layers.LeakyReLU(0.2)(x)
    out = layers.Dense(1)(x)
    return models.Model(inp, out, name='WGAN_Critic')


def compute_gradient_penalty(critic, real_samples, fake_samples):
    """Gradient Penalty for WGAN-GP (identical to 1_gan_wgan.py)."""
    batch_size = tf.shape(real_samples)[0]
    alpha = tf.random.uniform([batch_size, 1], 0.0, 1.0)
    alpha = tf.broadcast_to(alpha, tf.shape(real_samples))
    interpolated = alpha * real_samples + (1 - alpha) * fake_samples
    with tf.GradientTape() as tape:
        tape.watch(interpolated)
        validity = critic(interpolated)
    gradients = tape.gradient(validity, interpolated)
    gradients = tf.reshape(gradients, [batch_size, -1])
    gp = tf.reduce_mean((tf.norm(gradients, axis=1) - 1.0) ** 2)
    return gp


def train_wgan_gp(generator, critic, X_train, epochs, config, print_every=500):
    """
    WGAN-GP Training — identical to 1_gan_wgan.py.
    """
    batch_size = min(config['batch_size'], len(X_train))
    n_critic   = config['n_critic']
    lambda_gp  = config['lambda_gp']
    lr         = config['learning_rate']

    gen_optimizer    = optimizers.Adam(learning_rate=lr, beta_1=0.0, beta_2=0.9)
    critic_optimizer = optimizers.Adam(learning_rate=lr, beta_1=0.0, beta_2=0.9)

    X_tf = tf.convert_to_tensor(X_train, dtype=tf.float32)
    n_samples = X_tf.shape[0]

    g_losses = []
    c_losses = []

    t0 = time.time()

    for epoch in range(1, epochs + 1):
        # Train Critic
        for _ in range(n_critic):
            idx = np.random.randint(0, n_samples, batch_size)
            real_samples = tf.gather(X_tf, idx)
            noise = tf.random.normal((batch_size, LATENT_DIM))
            fake_samples = generator(noise, training=True)

            with tf.GradientTape() as tape:
                real_validity = critic(real_samples, training=True)
                fake_validity = critic(fake_samples, training=True)
                gp = compute_gradient_penalty(critic, real_samples, fake_samples)
                critic_loss = (tf.reduce_mean(fake_validity)
                               - tf.reduce_mean(real_validity)
                               + lambda_gp * gp)
            grads = tape.gradient(critic_loss, critic.trainable_variables)
            critic_optimizer.apply_gradients(zip(grads, critic.trainable_variables))

        # Train Generator
        noise = tf.random.normal((batch_size, LATENT_DIM))
        with tf.GradientTape() as tape:
            fake_samples = generator(noise, training=True)
            fake_validity = critic(fake_samples, training=True)
            generator_loss = -tf.reduce_mean(fake_validity)
        grads = tape.gradient(generator_loss, generator.trainable_variables)
        gen_optimizer.apply_gradients(zip(grads, generator.trainable_variables))

        g_losses.append(float(generator_loss.numpy()))
        c_losses.append(float(critic_loss.numpy()))

        if epoch % print_every == 0 or epoch == 1:
            elapsed = time.time() - t0
            eta = elapsed / epoch * (epochs - epoch)
            eta_str = time.strftime('%H:%M:%S', time.gmtime(eta))
            print(f"      [Epoch {epoch:>6}/{epochs}] "
                  f"C_loss={critic_loss.numpy():>8.4f}  "
                  f"G_loss={generator_loss.numpy():>8.4f}  "
                  f"ETA: {eta_str}")

    elapsed_total = time.time() - t0
    print(f"      Training completed in "
          f"{time.strftime('%H:%M:%S', time.gmtime(elapsed_total))}")

    return g_losses, c_losses, elapsed_total


# ═══════════════════════════════════════════════════════════════════════════════
# MATHEMATICAL METRICS (identical to 7_math_analysis.py)
# ═══════════════════════════════════════════════════════════════════════════════

def wasserstein_per_feature(X_real, X_synth):
    """
    Empirical W₁ per feature (1D).
    W₁(P, Q) = ∫₀¹ |F_P⁻¹(q) − F_Q⁻¹(q)| dq
    """
    n_features = X_real.shape[1]
    distances = np.zeros(n_features)
    for j in range(n_features):
        distances[j] = wasserstein_distance(X_real[:, j], X_synth[:, j])
    return distances


def mmd_rbf(X, Y, gamma=None):
    """
    Multivariate MMD² with RBF kernel.
    MMD²(P, Q) = E[k(x,x')] − 2·E[k(x,y)] + E[k(y,y')]
    k(a, b) = exp(−γ · ||a − b||²)
    γ estimated with median heuristic.
    """
    MAX_SAMPLES = 5000
    rng = np.random.RandomState(RANDOM_STATE)
    if len(X) > MAX_SAMPLES:
        X = X[rng.choice(len(X), MAX_SAMPLES, replace=False)]
    if len(Y) > MAX_SAMPLES:
        Y = Y[rng.choice(len(Y), MAX_SAMPLES, replace=False)]

    if gamma is None:
        n_sub = min(1000, len(X), len(Y))
        X_sub = X[rng.choice(len(X), n_sub, replace=False)]
        Y_sub = Y[rng.choice(len(Y), n_sub, replace=False)]
        XY = np.vstack([X_sub, Y_sub])
        dists_sq = np.sum((XY[:, None, :] - XY[None, :, :]) ** 2, axis=-1)
        median_dist_sq = np.median(dists_sq[dists_sq > 0])
        if median_dist_sq == 0:
            median_dist_sq = 1.0
        gamma = 1.0 / (2.0 * median_dist_sq)

    def rbf_kernel_mean(A, B, gamma):
        BLOCK = 1000
        total = 0.0
        count = 0
        for i in range(0, len(A), BLOCK):
            A_block = A[i:i + BLOCK]
            for j in range(0, len(B), BLOCK):
                B_block = B[j:j + BLOCK]
                sq_A = np.sum(A_block ** 2, axis=1, keepdims=True)
                sq_B = np.sum(B_block ** 2, axis=1, keepdims=True)
                D_sq = sq_A + sq_B.T - 2.0 * A_block @ B_block.T
                K = np.exp(-gamma * D_sq)
                total += K.sum()
                count += K.size
        return total / count

    Kxx = rbf_kernel_mean(X, X, gamma)
    Kxy = rbf_kernel_mean(X, Y, gamma)
    Kyy = rbf_kernel_mean(Y, Y, gamma)
    mmd_sq = Kxx - 2.0 * Kxy + Kyy
    return float(max(mmd_sq, 0.0))


# ═══════════════════════════════════════════════════════════════════════════════
# REPRESENTATIVE SELECTION
# ═══════════════════════════════════════════════════════════════════════════════

def select_representatives(class_dist):
    """
    Automatically selects one representative per tier.
    Escoge la class con más samples dentro de cada tier
    (to have sufficient data), except VERY_SMALL where
    se escoge la que más samples tenga dentro del tier.
    """
    tiers = {'large': [], 'small': [], 'very_small': []}
    for cls, n in class_dist.items():
        if n >= SMALL_CLASS_THRESHOLD:
            tiers['large'].append((cls, n))
        elif n >= 5000:
            tiers['small'].append((cls, n))
        else:
            tiers['very_small'].append((cls, n))

    # Within each tier, try to use the default, or the one with the highest cardinality
    reps = {}
    for tier_name, candidates in tiers.items():
        if not candidates:
            print(f"  [WARN] No classes in tier {tier_name}")
            continue
        default = DEFAULT_REPRESENTATIVES.get(tier_name)
        found = [c for c, n in candidates if c == default]
        if found:
            reps[tier_name] = default
        else:
            # Choose the one with the highest cardinality in that tier
            candidates.sort(key=lambda x: x[1], reverse=True)
            reps[tier_name] = candidates[0][0]

    return reps


# ═══════════════════════════════════════════════════════════════════════════════
# SAMPLE GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

def generate_samples(generator, n_samples):
    """Generate synthetic samples from a trained generator."""
    noise = np.random.normal(0, 1, (n_samples, LATENT_DIM))
    return generator.predict(noise, verbose=0)


# ═══════════════════════════════════════════════════════════════════════════════
# PLOTS
# ═══════════════════════════════════════════════════════════════════════════════

def plot_training_curves(g_losses, c_losses, title, output_path):
    """Plot training curves."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    window = max(1, min(100, len(g_losses) // 10))

    for ax, losses, label in [(axes[0], g_losses, 'Generator'),
                               (axes[1], c_losses, 'Critic')]:
        ax.plot(losses, alpha=0.2, color='blue')
        if window > 1:
            smooth = pd.Series(losses).rolling(window).mean()
            ax.plot(smooth, color='red', linewidth=1.5, label='Smoothed')
        ax.set_title(f'{label} Loss')
        ax.set_xlabel('Epoch')
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig.suptitle(title, fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_ablation_heatmaps(results_df, output_dir):
    """
    Generates heatmaps of W₁ mean and MMD² (classes × configs).
    """
    classes = results_df['class'].unique()
    configs = ['LARGE', 'SMALL', 'VERY_SMALL']

    for metric, label, fmt in [('w1_mean', 'W₁ Mean', '.6f'),
                                ('mmd2', 'MMD²', '.8f')]:
        fig, ax = plt.subplots(figsize=(8, 5))
        matrix = np.full((len(classes), len(configs)), np.nan)
        native_configs = {}

        for i, cls in enumerate(classes):
            cls_rows = results_df[results_df['class'] == cls]
            native_configs[cls] = cls_rows['native_config'].iloc[0]
            for j, cfg in enumerate(configs):
                row = cls_rows[cls_rows['config'] == cfg]
                if len(row) > 0:
                    matrix[i, j] = row[metric].values[0]

        im = ax.imshow(matrix, cmap='RdYlGn_r', aspect='auto')
        ax.set_xticks(range(len(configs)))
        ax.set_xticklabels(configs, fontsize=10)
        ax.set_yticks(range(len(classes)))
        ylabels = [f"{cls}\n(nativa: {native_configs[cls]})" for cls in classes]
        ax.set_yticklabels(ylabels, fontsize=9)

        # Anotar valores y marcar la config nativa con borde
        for i in range(len(classes)):
            for j in range(len(configs)):
                val = matrix[i, j]
                if not np.isnan(val):
                    text = format(val, fmt)
                    # Negrita si es la config nativa
                    is_native = (configs[j] == native_configs[classes[i]])
                    weight = 'bold' if is_native else 'normal'
                    color = 'white' if val > (np.nanmax(matrix) * 0.6) else 'black'
                    ax.text(j, i, text, ha='center', va='center',
                            fontsize=8, fontweight=weight, color=color)
                    if is_native:
                        rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                             linewidth=3, edgecolor='blue',
                                             facecolor='none')
                        ax.add_patch(rect)

        plt.colorbar(im, ax=ax, label=label)
        ax.set_xlabel('WGAN-GP Configuration', fontsize=11)
        ax.set_ylabel('Class (native tier)', fontsize=11)
        ax.set_title(f'Ablación: {label} — Real vs Synthetic\n'
                     f'(blue border = assigned native config)',
                     fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'ablation_heatmap_{metric}.png'),
                    dpi=200, bbox_inches='tight')
        plt.close()
        print(f"  Heatmap saved: ablation_heatmap_{metric}.png")


def plot_ablation_bars(results_df, output_dir):
    """Grouped bar chart by class, one bar per config."""
    classes = results_df['class'].unique()
    configs = ['LARGE', 'SMALL', 'VERY_SMALL']
    colors = {'LARGE': '#2196F3', 'SMALL': '#FF9800', 'VERY_SMALL': '#4CAF50'}

    for metric, label in [('w1_mean', 'W₁ Mean'), ('mmd2', 'MMD²')]:
        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(classes))
        width = 0.25

        for k, cfg in enumerate(configs):
            vals = []
            for cls in classes:
                row = results_df[(results_df['class'] == cls) &
                                 (results_df['config'] == cfg)]
                vals.append(row[metric].values[0] if len(row) > 0 else 0)
            bars = ax.bar(x + k * width, vals, width, label=cfg,
                          color=colors[cfg], alpha=0.85)

        # Marcar la barra de la config nativa
        for i, cls in enumerate(classes):
            native = results_df[results_df['class'] == cls]['native_config'].iloc[0]
            j = configs.index(native)
            ax.bar(x[i] + j * width, 0, width, edgecolor='red',
                   linewidth=0)  # invisible; we add a star instead
            row = results_df[(results_df['class'] == cls) &
                             (results_df['config'] == native)]
            if len(row) > 0:
                val = row[metric].values[0]
                ax.annotate('★', (x[i] + j * width, val),
                            ha='center', va='bottom', fontsize=14, color='red')

        ax.set_xticks(x + width)
        ax.set_xticklabels(classes, fontsize=10)
        ax.set_ylabel(label, fontsize=11)
        ax.set_title(f'Ablación: {label} by Class and Configuration\n'
                     f'(★ = assigned native configuration)', fontsize=12)
        ax.legend(title='Config')
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'ablation_bars_{metric}.png'),
                    dpi=200, bbox_inches='tight')
        plt.close()
        print(f"  Bars saved: ablation_bars_{metric}.png")


# ═══════════════════════════════════════════════════════════════════════════════
# TEXT SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════

def save_summary_txt(results_df, training_log, config_info, output_dir):
    """Generates complete summary in plain text."""
    lines = []
    lines.append("=" * 100)
    lines.append(" " * 15 + "ABLATION STUDY — WGAN-GP CONFIGURATION vs CARDINALITY")
    lines.append(" " * 30 + f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 100)

    lines.append("\nOBJECTIVE:")
    lines.append("  Evaluate whether the adaptive configuration assignment (LARGE / SMALL / VERY_SMALL)")
    lines.append("  based on each class cardinality is effectively optimal.")
    lines.append("  To do so, we train each representative class with ALL configurations")
    lines.append("  and compare generator quality using W₁ and MMD².\n")

    lines.append("METHODOLOGY:")
    lines.append("  - Dataset: CIC-IDS2017")
    lines.append(f"  - WGAN-GP scale: {config_info['wgan_scale']}")
    lines.append(f"  - Synthetic samples: {config_info['gen_samples']:,}")
    lines.append(f"  - GPU: {config_info.get('gpu_name', 'N/A')}")
    lines.append("  - Metrics:")
    lines.append("      W₁ Mean: Empirical average Wasserstein distance per feature (1D)")
    lines.append("      MMD²:    Multivariate Maximum Mean Discrepancy (RBF kernel)\n")

    lines.append("─" * 100)
    lines.append("REPRESENTATIVE CLASSES")
    lines.append("─" * 100)
    for tier, cls in config_info['representatives'].items():
        n = config_info['class_samples'][cls]
        native = get_native_config_name(n)
        lines.append(f"  {tier.upper():<12} → {cls:<15} ({n:>10,} samples) "
                      f"[config nativa: {native}]")

    lines.append("\n" + "─" * 100)
    lines.append("CONFIGURATIONS")
    lines.append("─" * 100)
    for cfg_name, cfg in ALL_CONFIGS.items():
        lines.append(f"\n  {cfg_name}:")
        for k, v in cfg.items():
            if k != 'name':
                lines.append(f"    {k:<22}: {v}")

    # ── Results por class ──
    lines.append("\n\n" + "=" * 100)
    lines.append("ABLATION STUDY RESULTS")
    lines.append("=" * 100)

    classes = results_df['class'].unique()
    configs = ['LARGE', 'SMALL', 'VERY_SMALL']

    for cls in classes:
        cls_df = results_df[results_df['class'] == cls].copy()
        native = cls_df['native_config'].iloc[0]
        n_real = cls_df['n_real'].iloc[0]

        lines.append(f"\n{'─' * 80}")
        lines.append(f"  CLASE: {cls}  ({n_real:,} samples)  |  Config nativa: {native}")
        lines.append(f"{'─' * 80}")
        lines.append(f"  {'Config':<14} {'Epochs':>8} {'Time':>10} "
                      f"{'W₁ Mean':>12} {'W₁ Median':>12} {'W₁ Max':>10} {'MMD²':>14}  Nativa?")
        lines.append(f"  {'─'*14} {'─'*8} {'─'*10} {'─'*12} {'─'*12} {'─'*10} {'─'*14}  {'─'*7}")

        for _, row in cls_df.iterrows():
            is_native = ' ◄── ★' if row['config'] == native else ''
            t_str = time.strftime('%H:%M:%S', time.gmtime(row['train_time']))
            lines.append(
                f"  {row['config']:<14} {int(row['epochs']):>8} {t_str:>10} "
                f"{row['w1_mean']:>12.6f} {row['w1_median']:>12.6f} "
                f"{row['w1_max']:>10.6f} {row['mmd2']:>14.8f}{is_native}"
            )

        # Determine the best config for this class
        best_w1 = cls_df.loc[cls_df['w1_mean'].idxmin()]
        best_mmd = cls_df.loc[cls_df['mmd2'].idxmin()]

        lines.append(f"\n  → Best W₁ Mean:  {best_w1['config']} ({best_w1['w1_mean']:.6f})"
                      f"{'  ✓ matches native' if best_w1['config'] == native else '  ✗ does NOT match'}")
        lines.append(f"  → Best MMD²:     {best_mmd['config']} ({best_mmd['mmd2']:.8f})"
                      f"{'  ✓ matches native' if best_mmd['config'] == native else '  ✗ does NOT match'}")

    # ── Table summary ──
    lines.append("\n\n" + "=" * 100)
    lines.append("COMPARATIVE SUMMARY TABLE")
    lines.append("=" * 100)
    lines.append(f"\n  {'Class':<16} {'Nativa':<14} {'Mejor W₁':<14} "
                  f"{'Mejor MMD²':<14} {'W₁ Match':<10} {'MMD² Match':<10}")
    lines.append(f"  {'─'*16} {'─'*14} {'─'*14} {'─'*14} {'─'*10} {'─'*10}")

    n_match_w1 = 0
    n_match_mmd = 0
    for cls in classes:
        cls_df = results_df[results_df['class'] == cls]
        native = cls_df['native_config'].iloc[0]
        best_w1 = cls_df.loc[cls_df['w1_mean'].idxmin(), 'config']
        best_mmd = cls_df.loc[cls_df['mmd2'].idxmin(), 'config']
        match_w1 = '✓' if best_w1 == native else '✗'
        match_mmd = '✓' if best_mmd == native else '✗'
        if best_w1 == native:
            n_match_w1 += 1
        if best_mmd == native:
            n_match_mmd += 1
        lines.append(f"  {cls:<16} {native:<14} {best_w1:<14} {best_mmd:<14} "
                      f"{match_w1:<10} {match_mmd:<10}")

    total_classes = len(classes)
    lines.append(f"\n  Native config = best match:")
    lines.append(f"    W₁ Mean: {n_match_w1}/{total_classes} "
                  f"({100*n_match_w1/total_classes:.0f}%)")
    lines.append(f"    MMD²:    {n_match_mmd}/{total_classes} "
                  f"({100*n_match_mmd/total_classes:.0f}%)")

    # ── Automatic conclusion ──
    lines.append("\n\n" + "=" * 100)
    lines.append("CONCLUSION")
    lines.append("=" * 100)
    if n_match_w1 == total_classes and n_match_mmd == total_classes:
        lines.append("  The cardinality-based adaptive configuration is OPTIMAL:")
        lines.append("  in all cases the native configuration produces the best quality.")
    elif (n_match_w1 + n_match_mmd) >= total_classes:
        lines.append("  The adaptive configuration is MOSTLY CORRECT:")
        lines.append(f"  the native config is the best in {n_match_w1}/{total_classes} (W₁) "
                      f"y {n_match_mmd}/{total_classes} (MMD²) of the cases.")
    else:
        lines.append("  Results suggest the current adaptive assignment")
        lines.append("  is NOT optimal in all cases. It is recommended to review the")
        lines.append("  configurations for classes where the native was not the best.")

    # Execution details
    lines.append("\n\n" + "─" * 100)
    lines.append("EXECUTION DETAILS")
    lines.append("─" * 100)
    total_time = sum(r['train_time'] for _, r in results_df.iterrows())
    lines.append(f"  Total training time: "
                  f"{time.strftime('%H:%M:%S', time.gmtime(total_time))}")
    lines.append(f"  Trainings performed: {len(results_df)}")
    lines.append(f"  Timestamp: {config_info['timestamp']}")

    lines.append("\n" + "=" * 100)

    text = "\n".join(lines)
    path = os.path.join(output_dir, 'RESUMEN_ABLATION.txt')
    with open(path, 'w', encoding='utf-8') as f:
        f.write(text)

    print(f"\n  Summary saved: {path}")
    print("\n" + text)
    return text


# ═══════════════════════════════════════════════════════════════════════════════
# WORKER (subproceso: entrena 3 configs para 1 class en 1 GPU)
# ═══════════════════════════════════════════════════════════════════════════════

def worker_main():
    """Subprocess worker: trains 3 configs for 1 class on 1 GPU."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--_worker', action='store_true')
    parser.add_argument('--gpu', type=str, required=True)
    parser.add_argument('--class-name', type=str, required=True)
    parser.add_argument('--class-tier', type=str, required=True)
    parser.add_argument('--data-npz', type=str, required=True)
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument('--wgan-scale', type=float, default=0.5)
    parser.add_argument('--gen-samples', type=int, default=5000)
    args = parser.parse_args()

    # GPU memory growth
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for g in gpus:
            try:
                tf.config.experimental.set_memory_growth(g, True)
            except RuntimeError:
                pass

    # Load data compartidos
    data = np.load(args.data_npz, allow_pickle=True)
    X_all = data['X']
    labels = data['labels']
    feature_names = list(data['feature_names'])

    cls = args.class_name
    tier = args.class_tier
    X_cls = X_all[labels == cls]
    n_cls = len(X_cls)
    native_cfg = get_native_config_name(n_cls)
    data_dim = X_all.shape[1]

    print(f"\n{'━' * 80}")
    print(f"  [WORKER GPU {args.gpu}] Class: {cls} ({n_cls:,} samples) "
          f"| Config nativa: {native_cfg}")
    print(f"{'━' * 80}")

    training_log = []
    results = []

    # Pre-fit scaler for evaluation (once per class)
    scaler_eval = MinMaxScaler()
    X_real_scaled = scaler_eval.fit_transform(X_cls)

    for cfg_name, cfg in ALL_CONFIGS.items():
        print(f"\n  ── [GPU {args.gpu}] {cls} × {cfg_name} ──")

        base_epochs = cfg['epochs']
        actual_epochs = max(1, int(base_epochs * args.wgan_scale))

        X_train = oversample_with_noise(
            X_cls, factor=cfg['oversample_factor'], noise_std=cfg['noise_std'])

        print(f"      Data: {len(X_train):,}  "
              f"(oversampling x{cfg['oversample_factor']}, noise={cfg['noise_std']})")
        print(f"      G: {cfg['generator_layers']}  C: {cfg['critic_layers']}")
        print(f"      Epochs: {actual_epochs} (base {base_epochs} × {args.wgan_scale})")
        print(f"      batch={cfg['batch_size']}  n_critic={cfg['n_critic']}  "
              f"λ_gp={cfg['lambda_gp']}  lr={cfg['learning_rate']}")

        gen = build_wgan_generator(LATENT_DIM, data_dim, cfg['generator_layers'])
        cri = build_wgan_critic(data_dim, cfg['critic_layers'])

        gen_params = int(np.sum([np.prod(v.shape)
                                 for v in gen.trainable_variables]))
        cri_params = int(np.sum([np.prod(v.shape)
                                 for v in cri.trainable_variables]))
        print(f"      Params G: {gen_params:,}  C: {cri_params:,}")

        g_losses, c_losses, train_time = train_wgan_gp(
            gen, cri, X_train,
            epochs=actual_epochs,
            config=cfg,
            print_every=max(500, actual_epochs // 10)
        )

        # Save training curve
        safe_cls = cls.replace(' ', '_').lower()
        plot_training_curves(
            g_losses, c_losses,
            title=f'{cls} × {cfg_name} (GPU {args.gpu})',
            output_path=os.path.join(
                args.output_dir, f'curves_{safe_cls}_{cfg_name.lower()}.png'))

        training_log.append({
            'class': cls, 'config': cfg_name, 'native_config': native_cfg,
            'epochs': actual_epochs, 'train_time': train_time,
            'gen_params': gen_params, 'cri_params': cri_params,
            'final_g_loss': g_losses[-1], 'final_c_loss': c_losses[-1],
        })

        # ── W₁ + MMD² Evaluation ──
        X_synth = generate_samples(gen, args.gen_samples)
        X_synth_scaled = scaler_eval.transform(X_synth)

        w1_feats = wasserstein_per_feature(X_real_scaled, X_synth_scaled)
        w1_mean = float(np.mean(w1_feats))
        w1_median = float(np.median(w1_feats))
        w1_max = float(np.max(w1_feats))
        w1_std = float(np.std(w1_feats))
        mmd2 = mmd_rbf(X_real_scaled, X_synth_scaled)

        is_native = ' ★' if cfg_name == native_cfg else ''
        print(f"      → W₁={w1_mean:.6f}  MMD²={mmd2:.8f}{is_native}")

        results.append({
            'class': cls, 'tier': tier, 'config': cfg_name,
            'native_config': native_cfg, 'n_real': n_cls,
            'n_synth': args.gen_samples, 'epochs': actual_epochs,
            'train_time': train_time,
            'w1_mean': w1_mean, 'w1_median': w1_median,
            'w1_max': w1_max, 'w1_std': w1_std, 'mmd2': mmd2,
            **{f'w1_{feat}': float(w1_feats[i])
               for i, feat in enumerate(feature_names)},
        })

        # Free models
        del cri, gen
        tf.keras.backend.clear_session()

    # Save results del worker como JSON
    safe_cls = cls.replace(' ', '_').lower()
    output = {'results': results, 'training_log': training_log}
    json_path = os.path.join(args.output_dir, f'_worker_{safe_cls}.json')
    with open(json_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\n[WORKER GPU {args.gpu}] {cls} completed — {json_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# PARALLEL ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════════

def parallel_main(args):
    """
    Orquestador paralelo: 1 class por GPU.
    Lanza 3 subprocesos (workers) y agrega los results al final.
    """
    gpu_list = [g.strip() for g in args.gpus.split(',')]

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(BASE_DIR, 'results_ablation', f'ablation_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)

    print("╔" + "═" * 78 + "╗")
    print("║" + " ESTUDIO DE ABLACIÓN — MODO PARALELO (3 GPUs) ".center(78) + "║")
    print("╠" + "═" * 78 + "╣")
    print(f"║  GPUs                    : {args.gpus:<50}║")
    print(f"║  WGAN-GP scale           : {args.wgan_scale:<50}║")
    print(f"║  Synthetic samples     : {args.gen_samples:<50}║")
    print(f"║  Output                  : {output_dir[-50:]:<50}║")
    print("╚" + "═" * 78 + "╝")

    # ── [1] Cargar data ──
    X, labels, feature_names, scaler, class_dist = load_and_preprocess()

    # ── [2] Seleccionar representantes ──
    reps = select_representatives(class_dist)
    class_list = list(reps.items())  # [(tier, class_name), ...]

    print(f"\n[PARALLEL] Representantes:")
    for tier, cls in class_list:
        n = class_dist[cls]
        native = get_native_config_name(n)
        print(f"  {tier.upper():<12} → {cls:<15} ({n:,} samples, nativa: {native})")

    if len(class_list) < 3:
        print("[ERROR] Se necesitan al menos 3 tiers. Abortando.")
        sys.exit(1)

    if len(gpu_list) < len(class_list):
        print(f"[WARN] {len(class_list)} classs pero solo {len(gpu_list)} GPUs. "
              f"Se asignan cíclicamente.")

    # ── [3] Guardar data compartidos para workers ──
    data_npz = os.path.join(output_dir, '_shared_data.npz')
    np.savez(data_npz, X=X, labels=labels,
             feature_names=np.array(feature_names, dtype=object))
    data_size_mb = os.path.getsize(data_npz) / (1024 * 1024)
    print(f"\n[PARALLEL] Data compartidos: {data_npz} ({data_size_mb:.1f} MB)")

    # ── Config info ──
    config_info = {
        'wgan_scale': args.wgan_scale,
        'gen_samples': args.gen_samples,
        'gpu': args.gpus,
        'gpu_name': f"Paralelo: GPUs {args.gpus}",
        'timestamp': timestamp,
        'representatives': dict(reps),
        'class_samples': {cls: int(class_dist[cls]) for _, cls in class_list},
        'configs': {name: {k: v for k, v in cfg.items() if k != 'name'}
                    for name, cfg in ALL_CONFIGS.items()},
    }
    with open(os.path.join(output_dir, 'experiment_config.json'), 'w') as f:
        json.dump(config_info, f, indent=2, default=str)

    # ── [4] Lanzar workers ──
    print(f"\n{'=' * 80}")
    print(f"  LANZANDO {len(class_list)} WORKERS EN PARALELO")
    print(f"{'=' * 80}")

    processes = []
    for i, (tier, cls) in enumerate(class_list):
        gpu_id = gpu_list[i % len(gpu_list)]
        safe_cls = cls.replace(' ', '_').lower()
        log_path = os.path.join(output_dir, f'worker_{safe_cls}_gpu{gpu_id}.log')

        cmd = [
            sys.executable, os.path.abspath(__file__),
            '--_worker',
            '--gpu', gpu_id,
            '--class-name', cls,
            '--class-tier', tier,
            '--data-npz', data_npz,
            '--output-dir', output_dir,
            '--wgan-scale', str(args.wgan_scale),
            '--gen-samples', str(args.gen_samples),
        ]

        log_f = open(log_path, 'w')
        p = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT)
        processes.append((p, cls, gpu_id, log_f, log_path))

        print(f"  GPU {gpu_id} → {cls:<15} (PID {p.pid}) "
              f"| Log: {os.path.basename(log_path)}")

    # ── [5] Esperar a todos los workers ──
    print(f"\n[PARALLEL] Esperando a {len(processes)} workers...")
    print(f"  (logs en tiempo real: tail -f {output_dir}/worker_*.log)\n")

    t0 = time.time()
    while True:
        all_done = True
        for p, cls, gpu_id, _, _ in processes:
            if p.poll() is None:
                all_done = False
        if all_done:
            break
        time.sleep(30)
        elapsed = time.time() - t0
        status_parts = []
        for p, cls, gpu_id, _, _ in processes:
            if p.poll() is not None:
                code = p.returncode
                s = "✓" if code == 0 else f"✗({code})"
            else:
                s = "⏳"
            status_parts.append(f"GPU{gpu_id}[{cls[:8]}]:{s}")
        print(f"  [{time.strftime('%H:%M:%S', time.gmtime(elapsed))}] "
              f"{' | '.join(status_parts)}")

    # Cerrar logs
    for _, _, _, log_f, _ in processes:
        log_f.close()

    total_time = time.time() - t0
    print(f"\n[PARALLEL] Workers completeds en "
          f"{time.strftime('%H:%M:%S', time.gmtime(total_time))}")

    # Verificar errores
    any_error = False
    for p, cls, gpu_id, _, log_path in processes:
        if p.returncode != 0:
            any_error = True
            print(f"  [ERROR] Worker GPU {gpu_id} ({cls}) "
                  f"falló con código {p.returncode}")
            print(f"          Ver log: {log_path}")
    if any_error:
        print("\n[WARN] Algunos workers fallaron. Se agregan los que completaron.")

    # ── [6] Agregar results ──
    print(f"\n{'=' * 80}")
    print(f"  AGREGANDO RESULTADOS")
    print(f"{'=' * 80}")

    all_results = []
    all_training_log = []
    for tier, cls in class_list:
        safe_cls = cls.replace(' ', '_').lower()
        json_path = os.path.join(output_dir, f'_worker_{safe_cls}.json')
        if os.path.exists(json_path):
            with open(json_path) as f:
                worker_data = json.load(f)
            all_results.extend(worker_data['results'])
            all_training_log.extend(worker_data['training_log'])
            print(f"  ✓ {cls}: {len(worker_data['results'])} results")
        else:
            print(f"  ✗ {cls}: sin results (worker falló)")

    if not all_results:
        print("[ERROR] No hay results. Todos los workers fallaron.")
        sys.exit(1)

    results_df = pd.DataFrame(all_results)

    # CSV summary
    base_cols = ['class', 'tier', 'config', 'native_config', 'n_real', 'n_synth',
                 'epochs', 'train_time', 'w1_mean', 'w1_median', 'w1_max',
                 'w1_std', 'mmd2']
    results_df[base_cols].to_csv(
        os.path.join(output_dir, 'ablation_results_summary.csv'), index=False)
    print(f"  CSV summary: ablation_results_summary.csv")

    # CSV detallado
    results_df.to_csv(
        os.path.join(output_dir, 'ablation_results_detailed.csv'), index=False)
    print(f"  CSV detallado: ablation_results_detailed.csv")

    # Plots
    plot_ablation_heatmaps(results_df, output_dir)
    plot_ablation_bars(results_df, output_dir)

    # Summary TXT
    save_summary_txt(results_df, all_training_log, config_info, output_dir)

    # JSON completo
    full_output = {
        'config': config_info,
        'training_log': all_training_log,
        'results': all_results,
    }
    with open(os.path.join(output_dir, 'ablation_full_results.json'), 'w') as f:
        json.dump(full_output, f, indent=2, default=str)
    print(f"  JSON completo: ablation_full_results.json")

    # Limpiar ficheros temporales
    if os.path.exists(data_npz):
        os.remove(data_npz)
    for tier, cls in class_list:
        safe_cls = cls.replace(' ', '_').lower()
        json_path = os.path.join(output_dir, f'_worker_{safe_cls}.json')
        if os.path.exists(json_path):
            os.remove(json_path)

    print(f"\n  Todos los artefactos en: {output_dir}")
    print("\n" + "=" * 80)
    print("  ESTUDIO DE ABLACIÓN PARALELO COMPLETADO")
    print("=" * 80)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Ablation Study — WGAN-GP Configuration vs Cardinalidad',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  python 9_ablation.py --gpu 1                 # Sequential on GPU 1
  python 9_ablation.py --parallel --gpus 1,2,3 # ¡Paralelo! 1 class/GPU
  python 9_ablation.py --wgan-scale 0.1        # Test rápido (10%% epochs)
  python 9_ablation.py --wgan-scale 1.0        # Epochs completos
  python 9_ablation.py --gen-samples 10000     # 10k samples synthetic
        """
    )
    parser.add_argument('--wgan-scale', type=float, default=0.5,
                        help='Factor multiplicador de epochs (default: 0.5)')
    parser.add_argument('--gen-samples', type=int, default=5000,
                        help='Synthetic samples a generar por combinación (default: 5000)')
    parser.add_argument('--gpu', type=str, default='0',
                        help='GPU a utilizar en modo secuencial (default: 0)')
    parser.add_argument('--parallel', action='store_true',
                        help='Modo paralelo: 1 class por GPU (3 GPUs)')
    parser.add_argument('--gpus', type=str, default='1,2,3',
                        help='GPUs para modo paralelo, separadas por coma (default: 1,2,3)')

    args = parser.parse_args()

    # ── Dispatch: modo paralelo o secuencial ──
    if args.parallel:
        parallel_main(args)
        return

    # ── GPU (ya configurada antes del import de TF) ──
    gpus = tf.config.list_physical_devices('GPU')
    gpu_name = "N/A (CPU)"
    if gpus:
        try:
            for g in gpus:
                tf.config.experimental.set_memory_growth(g, True)
            gpu_name = str(gpus[0])
        except RuntimeError:
            pass

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(BASE_DIR, 'results_ablation', f'ablation_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)

    print("╔" + "═" * 78 + "╗")
    print("║" + " ESTUDIO DE ABLACIÓN — WGAN-GP × CARDINALIDAD ".center(78) + "║")
    print("╠" + "═" * 78 + "╣")
    print(f"║  WGAN-GP scale           : {args.wgan_scale:<50}║")
    print(f"║  Synthetic samples     : {args.gen_samples:<50}║")
    print(f"║  GPU                     : {str(gpu_name):<50}║")
    print(f"║  Output                  : {output_dir[-50:]:<50}║")
    print("╚" + "═" * 78 + "╝")

    # ── [1] Cargar data ──
    X, labels, feature_names, scaler, class_dist = load_and_preprocess()

    # ── [2] Seleccionar representantes ──
    reps = select_representatives(class_dist)
    print(f"\n[ABLATION] Representantes seleccionados:")
    for tier, cls in reps.items():
        n = class_dist[cls]
        native = get_native_config_name(n)
        print(f"  {tier.upper():<12} → {cls:<15} ({n:,} samples, nativa: {native})")

    if len(reps) < 3:
        print("[ERROR] Se necesitan al menos 3 tiers con classs. Abortando.")
        sys.exit(1)

    # ── Config info ──
    config_info = {
        'wgan_scale': args.wgan_scale,
        'gen_samples': args.gen_samples,
        'gpu': args.gpu,
        'gpu_name': str(gpu_name),
        'timestamp': timestamp,
        'representatives': reps,
        'class_samples': {cls: int(class_dist[cls]) for cls in reps.values()},
        'configs': {name: {k: v for k, v in cfg.items() if k != 'name'}
                    for name, cfg in ALL_CONFIGS.items()},
    }
    with open(os.path.join(output_dir, 'experiment_config.json'), 'w') as f:
        json.dump(config_info, f, indent=2, default=str)

    # ── [3] Train 3 classes × 3 configs = 9 WGAN-GP ──
    print("\n" + "=" * 80)
    print("  PHASE 1: TRAINING (3 classes × 3 configs = 9 WGAN-GP)")
    print("=" * 80)

    data_dim = X.shape[1]
    all_results = []
    training_log = []
    generators = {}  # (cls, cfg_name) → generator

    combo_idx = 0
    total_combos = len(reps) * len(ALL_CONFIGS)

    for tier, cls in reps.items():
        X_cls = X[labels == cls]
        n_cls = len(X_cls)
        native_cfg = get_native_config_name(n_cls)

        print(f"\n{'━' * 80}")
        print(f"  CLASE: {cls}  ({n_cls:,} samples)  |  Config nativa: {native_cfg}")
        print(f"{'━' * 80}")

        for cfg_name, cfg in ALL_CONFIGS.items():
            combo_idx += 1
            print(f"\n    ── [{combo_idx}/{total_combos}] {cls} × {cfg_name} ──")

            # Epochs escalados
            base_epochs = cfg['epochs']
            actual_epochs = max(1, int(base_epochs * args.wgan_scale))

            # Oversampling (siempre según la config actual, no la nativa)
            X_train = oversample_with_noise(
                X_cls,
                factor=cfg['oversample_factor'],
                noise_std=cfg['noise_std']
            )

            print(f"      Training data: {len(X_train):,} "
                  f"(oversampling x{cfg['oversample_factor']}, "
                  f"noise={cfg['noise_std']})")
            print(f"      Arquitectura G: {cfg['generator_layers']}  "
                  f"C: {cfg['critic_layers']}")
            print(f"      Epochs: {actual_epochs} (base {base_epochs} × "
                  f"{args.wgan_scale})")
            print(f"      batch_size={cfg['batch_size']}  n_critic={cfg['n_critic']}  "
                  f"λ_gp={cfg['lambda_gp']}  lr={cfg['learning_rate']}")

            # Build models
            gen = build_wgan_generator(LATENT_DIM, data_dim, cfg['generator_layers'])
            cri = build_wgan_critic(data_dim, cfg['critic_layers'])

            gen_params = int(np.sum([np.prod(v.shape)
                                     for v in gen.trainable_variables]))
            cri_params = int(np.sum([np.prod(v.shape)
                                     for v in cri.trainable_variables]))
            print(f"      Params G: {gen_params:,}  C: {cri_params:,}")

            # Train
            g_losses, c_losses, train_time = train_wgan_gp(
                gen, cri, X_train,
                epochs=actual_epochs,
                config=cfg,
                print_every=max(500, actual_epochs // 10)
            )

            # Save generator
            generators[(cls, cfg_name)] = gen

            # Save curva
            safe_cls = cls.replace(' ', '_').lower()
            plot_training_curves(
                g_losses, c_losses,
                title=f'{cls} × {cfg_name}',
                output_path=os.path.join(
                    output_dir, f'curves_{safe_cls}_{cfg_name.lower()}.png')
            )

            training_log.append({
                'class': cls,
                'config': cfg_name,
                'native_config': native_cfg,
                'epochs': actual_epochs,
                'train_time': train_time,
                'gen_params': gen_params,
                'cri_params': cri_params,
                'final_g_loss': g_losses[-1],
                'final_c_loss': c_losses[-1],
            })

            # Free critic
            del cri
            tf.keras.backend.clear_session()

    # ── [4] Evaluar con metrics W₁ + MMD² ──
    print("\n\n" + "=" * 80)
    print("  PHASE 2: EVALUATION — WASSERSTEIN + MMD²")
    print("=" * 80)

    for tier, cls in reps.items():
        X_cls_real = X[labels == cls]
        n_cls = len(X_cls_real)
        native_cfg = get_native_config_name(n_cls)

        print(f"\n{'─' * 70}")
        print(f"  Evaluating class: {cls}  ({n_cls:,} samples)")
        print(f"{'─' * 70}")

        # Normalizar real (para metrics comparables)
        scaler_eval = MinMaxScaler()
        X_real_scaled = scaler_eval.fit_transform(X_cls_real)

        for cfg_name in ALL_CONFIGS:
            gen = generators[(cls, cfg_name)]
            X_synth = generate_samples(gen, args.gen_samples)

            # Scale synthetic with the same scaler
            X_synth_scaled = scaler_eval.transform(X_synth)

            # W₁ por feature
            w1_feats = wasserstein_per_feature(X_real_scaled, X_synth_scaled)
            w1_mean = float(np.mean(w1_feats))
            w1_median = float(np.median(w1_feats))
            w1_max = float(np.max(w1_feats))
            w1_std = float(np.std(w1_feats))

            # MMD²
            mmd2 = mmd_rbf(X_real_scaled, X_synth_scaled)

            is_native = ' ★' if cfg_name == native_cfg else ''
            print(f"    {cfg_name:<14} W₁ mean={w1_mean:.6f}  "
                  f"W₁ med={w1_median:.6f}  W₁ max={w1_max:.6f}  "
                  f"MMD²={mmd2:.8f}{is_native}")

            all_results.append({
                'class': cls,
                'tier': tier,
                'config': cfg_name,
                'native_config': native_cfg,
                'n_real': n_cls,
                'n_synth': args.gen_samples,
                'epochs': [t['epochs'] for t in training_log
                           if t['class'] == cls and t['config'] == cfg_name][0],
                'train_time': [t['train_time'] for t in training_log
                               if t['class'] == cls and t['config'] == cfg_name][0],
                'w1_mean': w1_mean,
                'w1_median': w1_median,
                'w1_max': w1_max,
                'w1_std': w1_std,
                'mmd2': mmd2,
                # Detail por feature
                **{f'w1_{feat}': float(w1_feats[i])
                   for i, feat in enumerate(feature_names)},
            })

    # ── [5] Guardar results ──
    print("\n\n" + "=" * 80)
    print("  FASE 3: GUARDADO DE RESULTADOS")
    print("=" * 80)

    results_df = pd.DataFrame(all_results)

    # CSV summary
    base_cols = ['class', 'tier', 'config', 'native_config', 'n_real', 'n_synth',
                 'epochs', 'train_time', 'w1_mean', 'w1_median', 'w1_max',
                 'w1_std', 'mmd2']
    results_df[base_cols].to_csv(
        os.path.join(output_dir, 'ablation_results_summary.csv'), index=False)
    print(f"  CSV summary: ablation_results_summary.csv")

    # CSV detallado (con W₁ por feature)
    results_df.to_csv(
        os.path.join(output_dir, 'ablation_results_detailed.csv'), index=False)
    print(f"  CSV detallado: ablation_results_detailed.csv")

    # Heatmaps y barras
    plot_ablation_heatmaps(results_df, output_dir)
    plot_ablation_bars(results_df, output_dir)

    # Summary TXT
    save_summary_txt(results_df, training_log, config_info, output_dir)

    # JSON completo
    full_output = {
        'config': config_info,
        'training_log': training_log,
        'results': all_results,
    }
    with open(os.path.join(output_dir, 'ablation_full_results.json'), 'w') as f:
        json.dump(full_output, f, indent=2, default=str)
    print(f"  JSON completo: ablation_full_results.json")

    # Free generators
    for gen in generators.values():
        del gen
    tf.keras.backend.clear_session()

    print(f"\n  Todos los artefactos en: {output_dir}")
    print("\n" + "=" * 80)
    print("  ESTUDIO DE ABLACIÓN COMPLETADO")
    print("=" * 80)


if __name__ == "__main__":
    if '--_worker' in sys.argv:
        worker_main()
    else:
        main()
