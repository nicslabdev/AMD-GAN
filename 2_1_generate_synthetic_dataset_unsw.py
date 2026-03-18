"""
Script: Generator de Dataset Sintético UNSW-NB15

Genera un dataset synthetic del UNSW-NB15 especificando el número de samples
por class, utilizando los models WGAN-GP entrenados previamente.

Usage:
    python generate_synthetic_dataset_unsw.py --interactive
    python generate_synthetic_dataset_unsw.py --balanced 10000
    python generate_synthetic_dataset_unsw.py --config config.json
    python generate_synthetic_dataset_unsw.py --benign 50000 --exploits 10000
"""

import os
import sys
import json
import argparse
import pickle
import numpy as np
import pandas as pd
import polars as pl
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import load_model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Redirigir archivos temporales a /home para evitar llenar /tmp
_tmpdir = os.path.join(os.path.expanduser('~'), '.cache', 'tmp')
os.makedirs(_tmpdir, exist_ok=True)
os.environ['TMPDIR'] = _tmpdir
os.environ['TEMP'] = _tmpdir
os.environ['TMP'] = _tmpdir

# ----------------------------
# Constantes y Configuration
# ----------------------------
MODELS_DIR = '<PATH_TO_WGAN_UNSW_MODELS_DIR>'
OUTPUT_DIR = '<PATH_TO_GENERATED_DATASETS_UNSW>'
DATASET_PATH = '<PATH_TO_UNSWNB15_CSV>'
LATENT_DIM = 100
LABEL_COLUMN = 'Label'

# Class name to folder mapping de models
CLASS_TO_FOLDER = {
    'Benign': 'benign',
    'Exploits': 'exploits',
    'Fuzzers': 'fuzzers',
    'Reconnaissance': 'reconnaissance',
    'Generic': 'generic',
    'DoS': 'dos',
    'Shellcode': 'shellcode',
}

FOLDER_TO_CLASS = {v: k for k, v in CLASS_TO_FOLDER.items()}

CLASS_TO_LABEL = {
    'Benign': 0,
    'DoS': 1,
    'Exploits': 2,
    'Fuzzers': 3,
    'Generic': 4,
    'Reconnaissance': 5,
    'Shellcode': 6,
}

# Features base (mismo orden que en training GAN)
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

FEATURE_NAMES = FEATURES_BASE + [
    'Src_IP_1', 'Src_IP_2', 'Src_IP_3', 'Src_IP_4',
    'Dst_IP_1', 'Dst_IP_2', 'Dst_IP_3', 'Dst_IP_4'
]

LOG_COLUMNS = [
    'Total Fwd Packet', 'Total Bwd packets',
    'Total Length of Fwd Packet', 'Total Length of Bwd Packet',
    'Flow Duration', 'Flow IAT Mean', 'Flow IAT Std',
    'Fwd IAT Mean', 'Bwd IAT Mean',
    'Fwd Packet Length Mean', 'Bwd Packet Length Mean',
    'Packet Length Std', 'Packet Length Max'
]


# ----------------------------
# Funciones de Utilidad
# ----------------------------
def print_header():
    print("=" * 70)
    print("  GENERADOR DE DATASET SINTÉTICO UNSW-NB15 - WGAN-GP")
    print("  AMD-GAN - Intrusion Detection System")
    print("=" * 70)


def get_available_classes():
    """Obtiene las classs disponibles verificando models existentes."""
    available = {}
    for class_name, folder in CLASS_TO_FOLDER.items():
        model_path = os.path.join(MODELS_DIR, folder, f'generator_{folder}.h5')
        if os.path.exists(model_path):
            available[class_name] = {
                'folder': folder,
                'model_path': model_path,
                'source_dir': MODELS_DIR,
            }
    return available


def print_available_classes(available_classes):
    print("\nClasss disponibles:")
    print("-" * 60)
    for i, (class_name, info) in enumerate(available_classes.items(), 1):
        print(f"  {i}. {class_name:<20} -> {info['folder']}")
    print("-" * 60)


def load_generator(class_name, available_classes):
    if class_name not in available_classes:
        raise ValueError(f"Class '{class_name}' no disponible")
    model_path = available_classes[class_name]['model_path']
    print(f"  Loading model: {os.path.basename(model_path)}")
    return load_model(model_path, compile=False)


def load_scaler(class_name, source_dir=None):
    """Carga el scaler para una class. Si no existe, lo recrea desde data."""
    folder = CLASS_TO_FOLDER[class_name]
    search_dir = source_dir or MODELS_DIR

    scaler_path = os.path.join(search_dir, folder, 'scaler.pkl')
    if os.path.exists(scaler_path):
        print(f"  Loading scaler desde: {os.path.dirname(scaler_path)}")
        with open(scaler_path, 'rb') as f:
            return pickle.load(f)

    # Recrear desde data originales
    print(f"  Recreando scaler para {class_name} desde data originales...")

    df_pl = pl.read_csv(DATASET_PATH, low_memory=False)
    df = df_pl.to_pandas()
    df.columns = df.columns.str.strip()

    df_cls = df[df[LABEL_COLUMN] == class_name].copy()

    features = df_cls[FEATURES_BASE].copy()

    # IPs
    if 'Src IP' in df_cls.columns:
        octetos = df_cls['Src IP'].astype(str).str.split('.', expand=True)
        for i in range(4):
            features[f'Src_IP_{i+1}'] = pd.to_numeric(octetos[i], errors='coerce').fillna(0).astype(int)

    if 'Dst IP' in df_cls.columns:
        octetos = df_cls['Dst IP'].astype(str).str.split('.', expand=True)
        for i in range(4):
            features[f'Dst_IP_{i+1}'] = pd.to_numeric(octetos[i], errors='coerce').fillna(0).astype(int)

    # Transformación logarítmica
    for col in LOG_COLUMNS:
        if col in features.columns:
            features[col] = np.log1p(features[col].clip(lower=0))
            features[col] = features[col].clip(lower=-20, upper=20)

    features.replace([np.inf, -np.inf], np.nan, inplace=True)
    features.fillna(0, inplace=True)

    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler.fit(features[FEATURE_NAMES])

    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)

    return scaler


def generate_samples(generator, n_samples, latent_dim=LATENT_DIM, batch_size=1024):
    all_samples = []
    remaining = n_samples
    while remaining > 0:
        current_batch = min(batch_size, remaining)
        noise = np.random.normal(0, 1, (current_batch, latent_dim))
        samples = generator.predict(noise, verbose=0)
        all_samples.append(samples)
        remaining -= current_batch
    return np.vstack(all_samples)


def reconstruct_features(X_synthetic, scaler):
    """Reconstructs features to original scale."""
    X_inv = scaler.inverse_transform(X_synthetic)
    df_rec = pd.DataFrame(X_inv, columns=FEATURE_NAMES)

    # Transformación inversa logarítmica
    for col in LOG_COLUMNS:
        if col in df_rec.columns:
            df_rec[col] = np.expm1(df_rec[col])

    # IPs
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


def generate_dataset(samples_per_class, output_name=None, include_scaled=False):
    """Genera el dataset synthetic completo."""
    print_header()

    available_classes = get_available_classes()
    print_available_classes(available_classes)

    for class_name in samples_per_class:
        if class_name not in available_classes:
            print(f"\n[WARNING] Class '{class_name}' no disponible, se omitirá.")

    valid_samples = {k: v for k, v in samples_per_class.items()
                     if k in available_classes and v > 0}

    if not valid_samples:
        print("\n[ERROR] No hay classs válidas para generar.")
        return None

    print("\n" + "=" * 70)
    print("CONFIGURACIÓN DE GENERACIÓN")
    print("=" * 70)
    total_samples = sum(valid_samples.values())
    for class_name, n in valid_samples.items():
        pct = (n / total_samples) * 100
        print(f"  {class_name:<20}: {n:>10,} samples ({pct:>5.1f}%)")
    print("-" * 70)
    print(f"  {'TOTAL':<20}: {total_samples:>10,} samples")
    print("=" * 70)

    synthetic_parts = []
    scaled_parts = []

    print("\nGenerating data synthetics...")
    start_time = datetime.now()

    for class_name, n_samples_cls in valid_samples.items():
        print(f"\n[{class_name}] Generating {n_samples_cls:,} samples...")

        generator = load_generator(class_name, available_classes)
        source_dir = available_classes[class_name].get('source_dir')
        scaler = load_scaler(class_name, source_dir=source_dir)

        print(f"  Generating con WGAN-GP...")
        X_synth = generate_samples(generator, n_samples_cls)

        if include_scaled:
            scaled_parts.append({'class': class_name, 'data': X_synth})

        print(f"  Reconstruyendo features...")
        df_synth = reconstruct_features(X_synth, scaler)

        df_synth['Label'] = class_name
        synthetic_parts.append(df_synth)
        print(f"  [OK] {len(df_synth):,} samples generadas")

        del generator
        tf.keras.backend.clear_session()

    print("\nCombinando dataset...")
    df_all = pd.concat(synthetic_parts, ignore_index=True)
    df_all = df_all.sample(frac=1, random_state=42).reset_index(drop=True)

    elapsed = datetime.now() - start_time
    print(f"\nTiempo de generación: {elapsed}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if output_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_name = f"unsw_synthetic_{timestamp}"

    csv_path = os.path.join(OUTPUT_DIR, f"{output_name}.csv")
    df_all.to_csv(csv_path, index=False)
    print(f"\n[SAVED] Dataset: {csv_path}")

    config = {
        'generated_at': datetime.now().isoformat(),
        'dataset': 'UNSW-NB15',
        'samples_per_class': valid_samples,
        'total_samples': len(df_all),
        'models_dir': MODELS_DIR,
    }
    config_path = os.path.join(OUTPUT_DIR, f"{output_name}_config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"[SAVED] Config: {config_path}")

    if include_scaled:
        scaled_path = os.path.join(OUTPUT_DIR, f"{output_name}_scaled.npz")
        np.savez(scaled_path, **{p['class']: p['data'] for p in scaled_parts})
        print(f"[SAVED] Scaled: {scaled_path}")

    # Summary
    print("\n" + "=" * 70)
    print("RESUMEN DEL DATASET GENERADO")
    print("=" * 70)
    print(f"\nClass distribution:")
    for class_name in valid_samples:
        count = len(df_all[df_all['Label'] == class_name])
        pct = (count / len(df_all)) * 100
        print(f"  {class_name:<20}: {count:>10,} ({pct:>5.1f}%)")
    print(f"\n  Total: {len(df_all):,} samples")
    print(f"  Archivo: {csv_path}")
    print("=" * 70)

    return df_all


def interactive_mode():
    """Modo interactivo para especificar samples."""
    print_header()

    available_classes = get_available_classes()
    print_available_classes(available_classes)

    if not available_classes:
        print("\n[ERROR] No se encontraron models entrenados en:", MODELS_DIR)
        print("  Entrena los models primero con: python gan_wgan_unsw.py --all")
        return

    print("\nModo interactivo - Especifica el número de samples por class")
    print("(Ingresa 0 o deja vacío para omitir una class)\n")

    samples_per_class = {}
    for class_name in available_classes:
        while True:
            try:
                inp = input(f"  {class_name:<20}: ")
                if inp.strip() == '':
                    n = 0
                else:
                    n = int(inp)
                if n < 0:
                    print("    [!] Debe ser >= 0")
                    continue
                if n > 0:
                    samples_per_class[class_name] = n
                break
            except ValueError:
                print("    [!] Ingresa un número válido")

    if not samples_per_class:
        print("\n[!] No se especificaron samples. Saliendo.")
        return

    output_name = input("\nNombre del dataset (Enter para auto): ").strip() or None
    generate_dataset(samples_per_class, output_name)


def main():
    parser = argparse.ArgumentParser(
        description='Genera datasets synthetics UNSW-NB15 usando models WGAN-GP',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  python generate_synthetic_dataset_unsw.py --interactive
  python generate_synthetic_dataset_unsw.py --balanced 10000
  python generate_synthetic_dataset_unsw.py --benign 50000 --exploits 10000
  python generate_synthetic_dataset_unsw.py --config config.json
        """
    )

    parser.add_argument('--interactive', '-i', action='store_true')
    parser.add_argument('--config', '-c', type=str, help='Archivo JSON de configuration')
    parser.add_argument('--balanced', '-b', type=int, help='N samples de cada class')

    parser.add_argument('--benign', type=int, default=0)
    parser.add_argument('--exploits', type=int, default=0)
    parser.add_argument('--fuzzers', type=int, default=0)
    parser.add_argument('--reconnaissance', type=int, default=0)
    parser.add_argument('--generic', type=int, default=0)
    parser.add_argument('--dos', type=int, default=0)
    parser.add_argument('--shellcode', type=int, default=0)

    parser.add_argument('--output', '-o', type=str, help='Nombre base para salida')
    parser.add_argument('--include-scaled', action='store_true')
    parser.add_argument('--list-classes', '-l', action='store_true')

    args = parser.parse_args()

    if args.list_classes:
        print_header()
        available = get_available_classes()
        print_available_classes(available)
        return

    if args.interactive:
        interactive_mode()
        return

    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
        samples = config.get('samples_per_class', {})
        generate_dataset(samples, config.get('output_name', args.output), args.include_scaled)
        return

    if args.balanced:
        available = get_available_classes()
        samples = {cls: args.balanced for cls in available}
        generate_dataset(samples, args.output, args.include_scaled)
        return

    # Desde argumentos individuales
    arg_mapping = {
        'Benign': args.benign,
        'Exploits': args.exploits,
        'Fuzzers': args.fuzzers,
        'Reconnaissance': args.reconnaissance,
        'Generic': args.generic,
        'DoS': args.dos,
        'Shellcode': args.shellcode,
    }

    samples_per_class = {k: v for k, v in arg_mapping.items() if v > 0}

    if samples_per_class:
        generate_dataset(samples_per_class, args.output, args.include_scaled)
    else:
        print_header()
        print("\nNo se especificaron samples a generar.")
        print("Usa --help para ver las opciones disponibles.")
        print("\nEjemplos rápidos:")
        print("  python generate_synthetic_dataset_unsw.py --interactive")
        print("  python generate_synthetic_dataset_unsw.py --balanced 10000")
        print("  python generate_synthetic_dataset_unsw.py --benign 50000 --exploits 10000")


if __name__ == "__main__":
    main()
