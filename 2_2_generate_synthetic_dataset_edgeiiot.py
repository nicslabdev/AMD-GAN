"""
Script: Generator de Dataset Sintético Edge-IIoT 2022

Genera un dataset synthetic del Edge-IIoT especificando el número de samples
por class, utilizando los models WGAN-GP entrenados previamente.

Usage:
    python 2_2_generate_synthetic_dataset_edgeiiot.py --interactive
    python 2_2_generate_synthetic_dataset_edgeiiot.py --balanced 10000
    python 2_2_generate_synthetic_dataset_edgeiiot.py --config config.json
    python 2_2_generate_synthetic_dataset_edgeiiot.py --normal 50000 --ddos_udp 10000
"""

import os
import sys
import json
import argparse
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import load_model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Redirigir archivos temporales
_tmpdir = os.path.join(os.path.expanduser('~'), '.cache', 'tmp')
os.makedirs(_tmpdir, exist_ok=True)
os.environ['TMPDIR'] = _tmpdir
os.environ['TEMP'] = _tmpdir
os.environ['TMP'] = _tmpdir

# ----------------------------
# Constantes y Configuration
# ----------------------------
MODELS_DIR = '<PATH_TO_WGAN_EDGEIIOT_MODELS_DIR>'
OUTPUT_DIR = '<PATH_TO_GENERATED_DATASETS_EDGEIIOT>'
DATASET_PATH = '<PATH_TO_EDGEIIOT_CSV>'
LATENT_DIM = 100
LABEL_COLUMN = 'Attack_type'

# Class name to folder mapping de models
VALID_CLASSES = [
    'Normal', 'DDoS_UDP', 'DDoS_ICMP', 'SQL_injection', 'Password',
    'Vulnerability_scanner', 'DDoS_TCP', 'DDoS_HTTP', 'Uploading',
    'Backdoor', 'Port_Scanning', 'XSS', 'Ransomware', 'MITM', 'Fingerprinting'
]

CLASS_TO_FOLDER = {c: c.lower() for c in VALID_CLASSES}
FOLDER_TO_CLASS = {v: k for k, v in CLASS_TO_FOLDER.items()}

CLASS_TO_LABEL = {c: i for i, c in enumerate(sorted(VALID_CLASSES))}

# Features base numéricas (50 features, mismo orden que training GAN)
FEATURES_BASE = [
    'arp.opcode', 'arp.hw.size',
    'icmp.checksum', 'icmp.seq_le', 'icmp.transmit_timestamp', 'icmp.unused',
    'http.file_data', 'http.content_length', 'http.request.uri.query',
    'http.request.method', 'http.referer', 'http.request.full_uri',
    'http.request.version', 'http.response', 'http.tls_port',
    'tcp.ack', 'tcp.ack_raw', 'tcp.checksum',
    'tcp.connection.fin', 'tcp.connection.rst', 'tcp.connection.syn',
    'tcp.connection.synack', 'tcp.dstport', 'tcp.flags', 'tcp.flags.ack',
    'tcp.len', 'tcp.seq', 'tcp.srcport',
    'udp.port', 'udp.stream', 'udp.time_delta',
    'dns.qry.name', 'dns.qry.name.len', 'dns.qry.qu', 'dns.qry.type',
    'dns.retransmission', 'dns.retransmit_request', 'dns.retransmit_request_in',
    'mqtt.conflag.cleansess', 'mqtt.conflags', 'mqtt.hdrflags',
    'mqtt.len', 'mqtt.msg_decoded_as', 'mqtt.msgtype',
    'mqtt.proto_len', 'mqtt.topic_len', 'mqtt.ver',
    'mbtcp.len', 'mbtcp.trans_id', 'mbtcp.unit_id',
]

# Features con IPs expandidas a octetos (50 base + 8 octetos IP = 58 features)
FEATURE_NAMES = FEATURES_BASE + [
    'Src_IP_1', 'Src_IP_2', 'Src_IP_3', 'Src_IP_4',
    'Dst_IP_1', 'Dst_IP_2', 'Dst_IP_3', 'Dst_IP_4',
]

LOG_COLUMNS = [
    'tcp.ack_raw', 'tcp.checksum', 'tcp.seq', 'tcp.srcport', 'tcp.dstport',
    'tcp.len', 'udp.port', 'http.content_length', 'http.file_data',
    'icmp.checksum',
]

# Columnas a descartar del raw dataset
# NOTA: ip.src_host e ip.dst_host se expanden a octetos, no se descartan
DROP_COLUMNS = [
    'frame.time',
    'arp.dst.proto_ipv4', 'arp.src.proto_ipv4',
    'tcp.options', 'tcp.payload',
    'mqtt.conack.flags', 'mqtt.msg', 'mqtt.protoname', 'mqtt.topic',
    'Attack_label', 'Attack_type'
]


# ----------------------------
# Funciones de Utilidad
# ----------------------------
def print_header():
    print("=" * 70)
    print("  GENERADOR DE DATASET SINTÉTICO EDGE-IIoT 2022 - WGAN-GP")
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
        print(f"  {i}. {class_name:<25} -> {info['folder']}")
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

    df = pd.read_csv(DATASET_PATH, low_memory=False)
    df.columns = df.columns.str.strip()

    df_cls = df[df[LABEL_COLUMN] == class_name].copy()

    # Seleccionar features base numéricas
    features_available = [f for f in FEATURES_BASE if f in df_cls.columns]
    features = df_cls[features_available].copy()

    for col in features.columns:
        features[col] = pd.to_numeric(features[col], errors='coerce').fillna(0)

    # Expand IPs to octets
    if 'ip.src_host' in df_cls.columns:
        octetos = df_cls['ip.src_host'].astype(str).str.split('.', expand=True)
        for i in range(4):
            features[f'Src_IP_{i+1}'] = pd.to_numeric(octetos[i], errors='coerce').fillna(0).astype(int)
    else:
        for i in range(4):
            features[f'Src_IP_{i+1}'] = 0

    if 'ip.dst_host' in df_cls.columns:
        octetos = df_cls['ip.dst_host'].astype(str).str.split('.', expand=True)
        for i in range(4):
            features[f'Dst_IP_{i+1}'] = pd.to_numeric(octetos[i], errors='coerce').fillna(0).astype(int)
    else:
        for i in range(4):
            features[f'Dst_IP_{i+1}'] = 0

    # Transformación logarítmica
    for col in LOG_COLUMNS:
        if col in features.columns:
            features[col] = np.log1p(features[col].clip(lower=0))
            features[col] = features[col].clip(lower=-20, upper=20)

    features.replace([np.inf, -np.inf], np.nan, inplace=True)
    features.fillna(0, inplace=True)

    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler.fit(features)

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
    # Get feature names from scaler
    n_features = scaler.n_features_in_
    feature_names = FEATURE_NAMES[:n_features]

    X_inv = scaler.inverse_transform(X_synthetic)
    df_rec = pd.DataFrame(X_inv, columns=feature_names)

    # Transformación inversa logarítmica
    for col in LOG_COLUMNS:
        if col in df_rec.columns:
            df_rec[col] = np.expm1(df_rec[col])

    # Clamp ports
    port_cols = ['tcp.srcport', 'tcp.dstport', 'udp.port']
    for col in port_cols:
        if col in df_rec.columns:
            df_rec[col] = df_rec[col].round().clip(0, 65535).astype(int)

    # Clamp IP octets to 0-255
    for col in df_rec.columns:
        if col.startswith('Src_IP_') or col.startswith('Dst_IP_'):
            df_rec[col] = df_rec[col].round().clip(0, 255).astype(int)

    # Clamp flag/binary columns
    flag_cols = [c for c in df_rec.columns if 'connection.' in c or 'flags.ack' in c
                 or 'retransmission' in c or 'retransmit' in c]
    for col in flag_cols:
        if col in df_rec.columns:
            df_rec[col] = df_rec[col].round().clip(0, 1).astype(int)

    # Non-negative
    for col in df_rec.columns:
        df_rec[col] = df_rec[col].clip(lower=0)

    # Integer columns
    int_cols = ['arp.opcode', 'arp.hw.size', 'tcp.flags', 'tcp.len',
                'dns.qry.name.len', 'dns.qry.type', 'mqtt.msgtype',
                'mqtt.proto_len', 'mqtt.topic_len', 'mqtt.ver',
                'mbtcp.len', 'mbtcp.trans_id', 'mbtcp.unit_id']
    for col in int_cols:
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
        print(f"  {class_name:<25}: {n:>10,} samples ({pct:>5.1f}%)")
    print("-" * 70)
    print(f"  {'TOTAL':<25}: {total_samples:>10,} samples")
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

        df_synth[LABEL_COLUMN] = class_name
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
        output_name = f"edgeiiot_synthetic_{timestamp}"

    csv_path = os.path.join(OUTPUT_DIR, f"{output_name}.csv")
    df_all.to_csv(csv_path, index=False)
    print(f"\n[SAVED] Dataset: {csv_path}")

    config = {
        'generated_at': datetime.now().isoformat(),
        'dataset': 'Edge-IIoT-2022',
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
        count = len(df_all[df_all[LABEL_COLUMN] == class_name])
        pct = (count / len(df_all)) * 100
        print(f"  {class_name:<25}: {count:>10,} ({pct:>5.1f}%)")
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
        print("  Entrena los models primero con: python 1_2_gan_wgan_edgeiiot.py --all")
        return

    print("\nModo interactivo - Especifica el número de samples por class")
    print("(Ingresa 0 o deja vacío para omitir una class)\n")

    samples_per_class = {}
    for class_name in available_classes:
        while True:
            try:
                inp = input(f"  {class_name:<25}: ")
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
        description='Genera datasets synthetics Edge-IIoT 2022 usando models WGAN-GP',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  python 2_2_generate_synthetic_dataset_edgeiiot.py --interactive
  python 2_2_generate_synthetic_dataset_edgeiiot.py --balanced 10000
  python 2_2_generate_synthetic_dataset_edgeiiot.py --normal 50000 --ddos_udp 10000
  python 2_2_generate_synthetic_dataset_edgeiiot.py --config config.json
        """
    )

    parser.add_argument('--interactive', '-i', action='store_true')
    parser.add_argument('--config', '-c', type=str, help='Archivo JSON de configuration')
    parser.add_argument('--balanced', '-b', type=int, help='N samples de cada class')

    # Individual class arguments
    for cls in VALID_CLASSES:
        arg_name = cls.lower().replace(' ', '_')
        parser.add_argument(f'--{arg_name}', type=int, default=0,
                            help=f'Muestras de {cls}')

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
    arg_mapping = {}
    for cls in VALID_CLASSES:
        arg_name = cls.lower().replace(' ', '_')
        val = getattr(args, arg_name, 0)
        if val > 0:
            arg_mapping[cls] = val

    if arg_mapping:
        generate_dataset(arg_mapping, args.output, args.include_scaled)
    else:
        print_header()
        print("\nNo se especificaron samples a generar.")
        print("Usa --help para ver las opciones disponibles.")
        print("\nEjemplos rápidos:")
        print("  python 2_2_generate_synthetic_dataset_edgeiiot.py --interactive")
        print("  python 2_2_generate_synthetic_dataset_edgeiiot.py --balanced 10000")
        print("  python 2_2_generate_synthetic_dataset_edgeiiot.py --normal 50000 --ddos_udp 5000")


if __name__ == "__main__":
    main()
