"""
Script: WGAN-GP Training for UNSW-NB15

Trains a WGAN-GP for each class of the UNSW-NB15 dataset with adaptive
configuration based on each class size.

CLASSES USED:
  - Benign (~3.39M samples)
  - Exploits (~31K)
  - Fuzzers (~30K)
  - Reconnaissance (~17K)
  - Generic (~4.6K)
  - DoS (~4.5K)
  - Shellcode (~2.1K)
  (Backdoor, Analysis, and Worms are discarded due to insufficient samples)

STRATEGIES FOR MINORITY CLASSES:
  1. Oversampling with Gaussian noise
  2. Adaptive batch size
  3. Adaptive architecture (lighter for small classes)
  4. More training epochs
  5. Higher Lambda GP (stronger regularization)

Usage:
    python gan_wgan_unsw.py --all                        # All classes
    python gan_wgan_unsw.py --classes Exploits Fuzzers    # Specific classes
    python gan_wgan_unsw.py --classes Shellcode DoS        # Small classes
    python gan_wgan_unsw.py --dry-run --all                # Only show config
"""

import os
import sys
import math
import json
import argparse
import pickle
import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers

# Reduce TensorFlow verbosity
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ----------------------------
# Configuration
# ----------------------------
DATASET_PATH = '<PATH_TO_UNSWNB15_CSV>'
OUTPUT_DIR = '<PATH_TO_WGAN_UNSW_MODELS_DIR>'
LATENT_DIM = 100
LABEL_COLUMN = 'Label'

# Classes to use (Backdoor, Analysis, Worms discarded due to insufficient data)
VALID_CLASSES = ['Benign', 'Exploits', 'Fuzzers', 'Reconnaissance', 'Generic', 'DoS', 'Shellcode']

# Classes considered "minority"
MINORITY_CLASSES = ['Generic', 'DoS', 'Shellcode']

# Sample threshold to consider a class as "small"
SMALL_CLASS_THRESHOLD = 15000

# Class name to folder mapping
CLASS_TO_FOLDER = {
    'Benign': 'benign',
    'Exploits': 'exploits',
    'Fuzzers': 'fuzzers',
    'Reconnaissance': 'reconnaissance',
    'Generic': 'generic',
    'DoS': 'dos',
    'Shellcode': 'shellcode',
}

# Base features used for the GAN (equivalent to CIC-IDS2017)
FEATURES_BASE = [
    'Src Port', 'Dst Port', 'Protocol',
    'Total Fwd Packet', 'Total Bwd packets',
    'Total Length of Fwd Packet', 'Total Length of Bwd Packet',
    'Flow Duration', 'Flow IAT Mean', 'Flow IAT Std',
    'Fwd IAT Mean', 'Bwd IAT Mean',
    'Fwd Packet Length Mean', 'Bwd Packet Length Mean',
    'Packet Length Std', 'Packet Length Max',
    'SYN Flag Count', 'ACK Flag Count', 'FIN Flag Count',
    'RST Flag Count', 'PSH Flag Count'
]

# Features with expanded IPs (final feature order for models)
FEATURE_NAMES = FEATURES_BASE + [
    'Src_IP_1', 'Src_IP_2', 'Src_IP_3', 'Src_IP_4',
    'Dst_IP_1', 'Dst_IP_2', 'Dst_IP_3', 'Dst_IP_4'
]

# Columns requiring logarithmic transformation
LOG_COLUMNS = [
    'Total Fwd Packet', 'Total Bwd packets',
    'Total Length of Fwd Packet', 'Total Length of Bwd Packet',
    'Flow Duration', 'Flow IAT Mean', 'Flow IAT Std',
    'Fwd IAT Mean', 'Bwd IAT Mean',
    'Fwd Packet Length Mean', 'Bwd Packet Length Mean',
    'Packet Length Std', 'Packet Length Max'
]

# Configuration por defecto para classs GRANDES (>15000 samples)
CONFIG_LARGE = {
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

# Configuration para classs PEQUEÑAS (<15000 samples)
CONFIG_SMALL = {
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

# Configuration para classs MUY PEQUEÑAS (<5000 samples)
CONFIG_VERY_SMALL = {
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


# ----------------------------
# Preprocessing
# ----------------------------
class UNSWPreprocessor:
    """Preprocessor specific to UNSW-NB15."""

    def __init__(self, dataset_path: str):
        self.path = dataset_path
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        self.feature_columns = None

    def load(self) -> pd.DataFrame:
        file_size = os.path.getsize(self.path)
        print(f"  Dataset size: {file_size / (1024 * 1024):.2f} MB")

        start_time = datetime.now()
        print("  Reading data with Polars...")
        df_pl = pl.read_csv(self.path, low_memory=False)
        print(f"  Data read in {datetime.now() - start_time}")

        df = df_pl.to_pandas()
        df.columns = df.columns.str.strip()
        return df

    @staticmethod
    def prepare_base_df(df: pd.DataFrame) -> pd.DataFrame:
        """Prepares the base DataFrame with expanded IPs and class filtering."""
        df = df.copy()
        df.columns = df.columns.str.strip()

        # Filter only valid classes
        df = df[df[LABEL_COLUMN].isin(VALID_CLASSES)].copy()

        # Select features + IPs + label
        cols_needed = ['Src IP', 'Dst IP'] + FEATURES_BASE + [LABEL_COLUMN]
        cols_available = [c for c in cols_needed if c in df.columns]
        df = df[cols_available].copy()

        # Expand IPs to octets
        if 'Src IP' in df.columns:
            octetos = df['Src IP'].astype(str).str.split('.', expand=True)
            for i in range(4):
                df[f'Src_IP_{i+1}'] = pd.to_numeric(octetos[i], errors='coerce').fillna(0).astype(int)
            df.drop(columns=['Src IP'], inplace=True)

        if 'Dst IP' in df.columns:
            octetos = df['Dst IP'].astype(str).str.split('.', expand=True)
            for i in range(4):
                df[f'Dst_IP_{i+1}'] = pd.to_numeric(octetos[i], errors='coerce').fillna(0).astype(int)
            df.drop(columns=['Dst IP'], inplace=True)

        return df

    def gan_subset_preparation(self, df_subset: pd.DataFrame):
        """Prepares a subset for GAN training: log-transform + scale."""
        features = df_subset.drop(columns=[LABEL_COLUMN])
        labels = df_subset[LABEL_COLUMN]
        self.feature_columns = features.columns.tolist()

        features_proc = features.copy()

        for col in LOG_COLUMNS:
            if col in features_proc.columns:
                features_proc[col] = np.log1p(features_proc[col].clip(lower=0))
                features_proc[col] = features_proc[col].clip(lower=-20, upper=20)

        features_proc.replace([np.inf, -np.inf], np.nan, inplace=True)
        features_proc.fillna(0, inplace=True)

        X = self.scaler.fit_transform(features_proc)
        return X, labels


# ----------------------------
# Data Augmentation for Small Classes
# ----------------------------
def oversample_with_noise(X, factor=10, noise_std=0.02):
    """Oversampling with Gaussian noise to densify the distribution."""
    if factor <= 1 and noise_std == 0:
        return X

    print(f"  Applying oversampling x{factor} with noise_std={noise_std}")

    X_augmented = [X]
    for _ in range(factor - 1):
        noise = np.random.normal(0, noise_std, X.shape)
        X_noisy = np.clip(X + noise, -1, 1)
        X_augmented.append(X_noisy)

    X_final = np.vstack(X_augmented)
    np.random.shuffle(X_final)
    print(f"  Augmented data: {len(X)} -> {len(X_final)}")
    return X_final


# ----------------------------
# WGAN-GP Model
# ----------------------------
def build_generator(latent_dim, output_dim, layer_sizes=[256, 512, 256]):
    noise = layers.Input(shape=(latent_dim,))
    x = noise
    for i, units in enumerate(layer_sizes):
        x = layers.Dense(units)(x)
        x = layers.LeakyReLU(0.2)(x)
        if i < len(layer_sizes) - 1:
            x = layers.BatchNormalization(momentum=0.8)(x)
    output = layers.Dense(output_dim, activation='tanh')(x)
    return models.Model(noise, output, name="Generator")


def build_critic(input_dim, layer_sizes=[512, 256, 128], dropout_rate=0.0):
    inp = layers.Input(shape=(input_dim,))
    x = inp
    for units in layer_sizes:
        x = layers.Dense(units)(x)
        x = layers.LeakyReLU(0.2)(x)
        if dropout_rate > 0:
            x = layers.Dropout(dropout_rate)(x)
    output = layers.Dense(1)(x)
    return models.Model(inp, output, name="Critic")


def gradient_penalty(critic, real_samples, fake_samples):
    batch_size = tf.shape(real_samples)[0]
    alpha = tf.random.uniform([batch_size, 1], 0.0, 1.0)
    alpha = tf.broadcast_to(alpha, tf.shape(real_samples))
    interpolated = alpha * real_samples + (1 - alpha) * fake_samples

    with tf.GradientTape() as tape:
        tape.watch(interpolated)
        validity = critic(interpolated)

    gradients = tape.gradient(validity, interpolated)
    gradients = tf.reshape(gradients, [batch_size, -1])
    return tf.reduce_mean((tf.norm(gradients, axis=1) - 1.0) ** 2)


def train_wgan_gp(generator, critic, X_train, config, print_interval=1000):
    """WGAN-GP training with custom configuration."""
    latent_dim = LATENT_DIM
    batch_size = config['batch_size']
    epochs = config['epochs']
    n_critic = config['n_critic']
    lambda_gp = config['lambda_gp']
    lr = config['learning_rate']

    gen_optimizer = optimizers.Adam(learning_rate=lr, beta_1=0.0, beta_2=0.9)
    critic_optimizer = optimizers.Adam(learning_rate=lr, beta_1=0.0, beta_2=0.9)

    X_train = tf.convert_to_tensor(X_train, dtype=tf.float32)
    n_samples = X_train.shape[0]

    g_losses, c_losses = [], []

    print(f"\n  Starting WGAN-GP training:")
    print(f"    - Batch size: {batch_size}")
    print(f"    - Epochs: {epochs}")
    print(f"    - N_critic: {n_critic}")
    print(f"    - Lambda GP: {lambda_gp}")
    print(f"    - Learning rate: {lr}")
    print(f"    - Training data: {n_samples}")

    start_time = datetime.now()

    for epoch in range(1, epochs + 1):
        # Train critic
        for _ in range(n_critic):
            idx = np.random.randint(0, n_samples, batch_size)
            real_samples = tf.gather(X_train, idx)
            noise = tf.random.normal((batch_size, latent_dim))
            fake_samples = generator(noise, training=True)

            with tf.GradientTape() as tape:
                real_validity = critic(real_samples, training=True)
                fake_validity = critic(fake_samples, training=True)
                gp = gradient_penalty(critic, real_samples, fake_samples)
                critic_loss = tf.reduce_mean(fake_validity) - tf.reduce_mean(real_validity) + lambda_gp * gp

            grads = tape.gradient(critic_loss, critic.trainable_variables)
            critic_optimizer.apply_gradients(zip(grads, critic.trainable_variables))

        # Train generator
        noise = tf.random.normal((batch_size, latent_dim))
        with tf.GradientTape() as tape:
            fake_samples = generator(noise, training=True)
            fake_validity = critic(fake_samples, training=True)
            generator_loss = -tf.reduce_mean(fake_validity)

        grads = tape.gradient(generator_loss, generator.trainable_variables)
        gen_optimizer.apply_gradients(zip(grads, generator.trainable_variables))

        g_losses.append(generator_loss.numpy())
        c_losses.append(critic_loss.numpy())

        if epoch % print_interval == 0 or epoch == 1:
            elapsed = datetime.now() - start_time
            eta = elapsed / epoch * (epochs - epoch)
            print(f"    [Epoch {epoch:>6}/{epochs}] C_loss: {critic_loss.numpy():>8.4f} | "
                  f"G_loss: {generator_loss.numpy():>8.4f} | ETA: {str(eta).split('.')[0]}")

    total_time = datetime.now() - start_time
    print(f"\n  Training completed in {total_time}")
    return g_losses, c_losses


# ----------------------------
# Utilities
# ----------------------------
def generate_samples(generator, n_samples, latent_dim):
    noise = np.random.normal(0, 1, (n_samples, latent_dim))
    return generator.predict(noise, verbose=0)


def reconstruct_features(X_synthetic, scaler, feature_names):
    """Reconstructs features to original scale."""
    X_inv = scaler.inverse_transform(X_synthetic)
    df_rec = pd.DataFrame(X_inv, columns=feature_names)

    for col in LOG_COLUMNS:
        if col in df_rec.columns:
            df_rec[col] = np.expm1(df_rec[col])

    for col in df_rec.columns:
        if col.startswith('Src_IP_') or col.startswith('Dst_IP_'):
            df_rec[col] = df_rec[col].round().clip(0, 255).astype(int)

    df_rec['Src Port'] = df_rec['Src Port'].round().clip(1, 65535).astype(int)
    df_rec['Dst Port'] = df_rec['Dst Port'].round().clip(1, 65535).astype(int)
    df_rec['Protocol'] = df_rec['Protocol'].round().clip(0, 255).astype(int)

    for col in LOG_COLUMNS + ['SYN Flag Count', 'ACK Flag Count',
                                'FIN Flag Count', 'RST Flag Count', 'PSH Flag Count']:
        if col in df_rec.columns:
            df_rec[col] = df_rec[col].clip(lower=0)

    integer_columns = ['Total Fwd Packet', 'Total Bwd packets',
                        'SYN Flag Count', 'ACK Flag Count',
                        'FIN Flag Count', 'RST Flag Count', 'PSH Flag Count']
    for col in integer_columns:
        if col in df_rec.columns:
            df_rec[col] = df_rec[col].round().astype(int)

    return df_rec


def plot_training_curves(g_losses, c_losses, output_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    window = min(100, len(g_losses) // 10)
    if window > 1:
        g_smooth = pd.Series(g_losses).rolling(window).mean()
        c_smooth = pd.Series(c_losses).rolling(window).mean()
    else:
        g_smooth, c_smooth = g_losses, c_losses

    axes[0].plot(g_losses, alpha=0.3, label='Raw')
    axes[0].plot(g_smooth, label='Smoothed')
    axes[0].set_title('Generator Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(c_losses, alpha=0.3, label='Raw')
    axes[1].plot(c_smooth, label='Smoothed')
    axes[1].set_title('Critic Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_kde_comparison(X_real, X_synth, feature_names, scaler, output_path, sample_size=5000):
    n_features = X_real.shape[1]
    cols_per_row = 4
    rows = math.ceil(n_features / cols_per_row)

    idx_real = np.random.choice(len(X_real), min(sample_size, len(X_real)), replace=False)
    idx_synth = np.random.choice(len(X_synth), min(sample_size, len(X_synth)), replace=False)
    Xr, Xs = X_real[idx_real], X_synth[idx_synth]

    plt.figure(figsize=(cols_per_row * 4, rows * 3))

    for i in range(n_features):
        r = Xr[:, i].reshape(-1, 1)
        s = Xs[:, i].reshape(-1, 1)
        r = scaler.inverse_transform(
            np.pad(r, ((0, 0), (i, scaler.n_features_in_ - i - 1)), mode='constant'))[:, i]
        s = scaler.inverse_transform(
            np.pad(s, ((0, 0), (i, scaler.n_features_in_ - i - 1)), mode='constant'))[:, i]
        r, s = r[np.isfinite(r)], s[np.isfinite(s)]

        if np.std(r) < 1e-6:
            continue
        try:
            kde_r = gaussian_kde(r)
            kde_s = gaussian_kde(s)
            xmin = min(r.min(), s.min())
            xmax = max(r.max(), s.max())
            x = np.linspace(xmin, xmax, 300)

            plt.subplot(rows, cols_per_row, i + 1)
            plt.plot(x, kde_r(x), label="Real", linewidth=1.5)
            plt.plot(x, kde_s(x), '--', label="Synthetic", linewidth=1.5)
            plt.title(feature_names[i], fontsize=9)
            plt.grid(alpha=0.3)
            if i == 0:
                plt.legend()
        except Exception:
            continue

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def get_config_for_class(n_samples):
    if n_samples < 5000:
        print(f"  Using CONFIG_VERY_SMALL (n={n_samples} < 5000)")
        return CONFIG_VERY_SMALL.copy()
    elif n_samples < SMALL_CLASS_THRESHOLD:
        print(f"  Using CONFIG_SMALL (n={n_samples} < {SMALL_CLASS_THRESHOLD})")
        return CONFIG_SMALL.copy()
    else:
        print(f"  Using CONFIG_LARGE (n={n_samples} >= {SMALL_CLASS_THRESHOLD})")
        return CONFIG_LARGE.copy()


# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser(
        description='WGAN-GP Training for UNSW-NB15',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--classes', nargs='+', default=None,
                        help='Specific classes a entrenar')
    parser.add_argument('--all', action='store_true',
                        help='Train ALL classes with adaptive configuration')
    parser.add_argument('--gpu', type=str, default='0',
                        help='GPU to use (default: 0)')
    parser.add_argument('--samples', type=int, default=10000,
                        help='Número de samples synthetic a generar por class')
    parser.add_argument('--dry-run', action='store_true',
                        help='Only show configuration sin entrenar')

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    print("=" * 80)
    print("WGAN-GP TRAINING - UNSW-NB15")
    print("=" * 80)

    # Load data
    print("\n[1] Loading UNSW-NB15 dataset...")
    prep_global = UNSWPreprocessor(DATASET_PATH)
    df_raw = prep_global.load()
    df_base = UNSWPreprocessor.prepare_base_df(df_raw)

    # Show distribution
    print("\n[2] Class distribution in the dataset:")
    print("-" * 50)
    class_counts = df_base[LABEL_COLUMN].value_counts()
    for class, count in class_counts.items():
        marker = "!! " if count < SMALL_CLASS_THRESHOLD else "OK "
        print(f"  {marker}{class:<20}: {count:>10,} samples")
    print("-" * 50)

    # Determine classes to train
    if args.all:
        classes_to_train = [c for c in VALID_CLASSES if c in class_counts.index]
    elif args.classes:
        classes_to_train = args.classes
    else:
        classes_to_train = MINORITY_CLASSES

    print(f"\n[3] Classes to train: {classes_to_train}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    label_encoder = LabelEncoder().fit(df_base[LABEL_COLUMN])

    for class_name in classes_to_train:
        print("\n" + "=" * 80)
        print(f"TRAINING CLASS: {class_name}")
        print("=" * 80)

        df_cls = df_base[df_base[LABEL_COLUMN] == class_name].copy()

        if len(df_cls) < 100:
            print(f"  [SKIP] Muy pocas samples ({len(df_cls)})")
            continue

        n_original = len(df_cls)
        print(f"\n  Original samples: {n_original:,}")

        # Para classs muy grandes (Benign), submuestrear for GAN training
        MAX_TRAIN_SAMPLES = 100000
        if n_original > MAX_TRAIN_SAMPLES:
            print(f"  Subsampling from {n_original:,} a {MAX_TRAIN_SAMPLES:,} for GAN training")
            df_cls = df_cls.sample(n=MAX_TRAIN_SAMPLES, random_state=42)

        config = get_config_for_class(len(df_cls))

        if args.dry_run:
            print(f"\n  [DRY-RUN] Configuration that would be used:")
            for k, v in config.items():
                print(f"    {k}: {v}")
            continue

        # Preprocess
        prep_cls = UNSWPreprocessor(DATASET_PATH)
        X_cls, _ = prep_cls.gan_subset_preparation(df_cls)

        # Oversampling with noise
        X_train = oversample_with_noise(
            X_cls,
            factor=config['oversample_factor'],
            noise_std=config['noise_std']
        )

        # Build models
        generator = build_generator(
            LATENT_DIM, X_cls.shape[1],
            layer_sizes=config['generator_layers']
        )
        critic = build_critic(
            X_cls.shape[1],
            layer_sizes=config['critic_layers']
        )

        print(f"\n  Generator architecture: {config['generator_layers']}")
        print(f"  Critic architecture: {config['critic_layers']}")

        # Train
        g_losses, c_losses = train_wgan_gp(
            generator, critic, X_train, config,
            print_interval=max(1000, config['epochs'] // 20)
        )

        # Save
        safe_name = CLASS_TO_FOLDER[class_name]
        cls_dir = os.path.join(OUTPUT_DIR, safe_name)
        os.makedirs(cls_dir, exist_ok=True)

        generator.save(os.path.join(cls_dir, f'generator_{safe_name}.h5'))
        critic.save(os.path.join(cls_dir, f'critic_{safe_name}.h5'))

        with open(os.path.join(cls_dir, 'scaler.pkl'), 'wb') as f:
            pickle.dump(prep_cls.scaler, f)

        with open(os.path.join(cls_dir, 'training_config.json'), 'w') as f:
            json.dump({
                'class_name': class_name,
                'dataset': 'UNSW-NB15',
                'original_samples': n_original,
                'train_samples': len(df_cls),
                'augmented_samples': len(X_train),
                'n_features': X_cls.shape[1],
                'feature_names': prep_cls.feature_columns,
                'config': config,
                'timestamp': datetime.now().isoformat()
            }, f, indent=2)

        # Generar samples synthetic
        print(f"\n  Generating {args.samples:,} samples synthetic...")
        X_synth = generate_samples(generator, args.samples, LATENT_DIM)
        np.save(os.path.join(cls_dir, f'synthetic_scaled_{safe_name}.npy'), X_synth)

        df_synth = reconstruct_features(X_synth, prep_cls.scaler, prep_cls.feature_columns)
        df_synth['Label'] = class_name
        df_synth.to_csv(os.path.join(cls_dir, f'synthetic_reconstructed_{safe_name}.csv'), index=False)

        # Plots
        print("  Generating plots...")
        plot_training_curves(g_losses, c_losses,
                             os.path.join(cls_dir, f'training_curves_{safe_name}.png'))
        plot_kde_comparison(X_cls, X_synth, prep_cls.feature_columns, prep_cls.scaler,
                            os.path.join(cls_dir, f'kde_comparison_{safe_name}.png'))

        print(f"\n  [OK] Results saved to: {cls_dir}")

        del generator, critic
        tf.keras.backend.clear_session()

    print("\n" + "=" * 80)
    print("TRAINING COMPLETED")
    print("=" * 80)
    print(f"\nResults saved to: {OUTPUT_DIR}")
    print("\nNext steps:")
    print("  1. Generate synthetic dataset:")
    print("     python generate_synthetic_dataset_unsw.py --balanced 10000")
    print("  2. Evaluate with TSTR:")
    print("     python train_synthetic_test_real_unsw.py")


if __name__ == "__main__":
    main()
