"""
Script: Comparación de Técnicas de Oversampling para Edge-IIoT 2022

Compara diferentes técnicas de oversampling para balancear el dataset Edge-IIoT:
  1. Original (sin oversampling) - baseline
  2. GAN (WGAN-GP entrenada en Edge-IIoT)
  3. SMOTE
  4. ADASYN
  5. BorderlineSMOTE
  6. SMOTE-ENN (oversampling + limpieza)
  7. Random Oversampling

Para cada técnica:
  - Se balancea el dataset de training
  - Se entrena el mismo model (Logistic Regression)
  - Se evalúa en el mismo conjunto de test (sin modificar)

Usage:
    python 3_2experiment_oversampling_edgeiiot.py
    python 3_2experiment_oversampling_edgeiiot.py --skip-gan
    python 3_2experiment_oversampling_edgeiiot.py --max-samples 50000
"""

import os
import sys
import pickle
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from collections import Counter

# ML
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score, precision_score, recall_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from lightgbm import LGBMClassifier

# Oversampling
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE, RandomOverSampler
from imblearn.combine import SMOTEENN

# TensorFlow para cargar generatores GAN
import tensorflow as tf
from tensorflow.keras.models import load_model

import warnings
warnings.filterwarnings('ignore')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Redirigir archivos temporales
_tmpdir = os.path.join(os.path.expanduser('~'), '.cache', 'tmp')
os.makedirs(_tmpdir, exist_ok=True)
os.environ['TMPDIR'] = _tmpdir
os.environ['TEMP'] = _tmpdir
os.environ['TMP'] = _tmpdir

# ----------------------------
# Configuration
# ----------------------------
REAL_DATA_PATH = '<PATH_TO_EDGEIIOT_CSV>'
MODELS_DIR = '<PATH_TO_WGAN_EDGEIIOT_MODELS_DIR>'
OUTPUT_DIR = '<PATH_TO_RESULTS_OVERSAMPLING_EDGEIIOT>'
RANDOM_STATE = 42
LATENT_DIM = 100
LABEL_COLUMN = 'Attack_type'

# AMD-GAN Adaptive Training Engine thresholds
# d = 58 features (50 base + 8 IP octets) → θ_high = ⌈ 2.5 × 58²⌈ = 8410, θ_low = ⌈0.8 × 58²⌈ = 2692
D_FEATURES = 58
RHO_HIGH = 2.5
RHO_LOW = 0.8
THETA_HIGH = 8410   # ⌈ρ_high · d²⌉
THETA_LOW = 2692     # ⌈ρ_low · d²⌉

# Classs válidas
VALID_CLASSES = [
    'Normal', 'DDoS_UDP', 'DDoS_ICMP', 'SQL_injection', 'Password',
    'Vulnerability_scanner', 'DDoS_TCP', 'DDoS_HTTP', 'Uploading',
    'Backdoor', 'Port_Scanning', 'XSS', 'Ransomware', 'MITM', 'Fingerprinting'
]

CLASS_TO_FOLDER = {c: c.lower() for c in VALID_CLASSES}

# Features base numéricas (50 features)
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

# Columnas no numéricas a descartar del raw
# NOTA: ip.src_host e ip.dst_host se expanden a octetos, no se descartan
DROP_COLUMNS = [
    'frame.time',
    'arp.dst.proto_ipv4', 'arp.src.proto_ipv4',
    'tcp.options', 'tcp.payload',
    'mqtt.conack.flags', 'mqtt.msg', 'mqtt.protoname', 'mqtt.topic',
    'Attack_label', 'Attack_type'
]


# ----------------------------
# Funciones de Carga de Data
# ----------------------------
def print_header():
    print("=" * 100)
    print("  EXPERIMENTO: COMPARACIÓN DE TÉCNICAS DE OVERSAMPLING - EDGE-IIoT 2022")
    print("  GAN (WGAN-GP) vs SMOTE vs ADASYN vs BorderlineSMOTE vs SMOTE-ENN")
    print("  AMD-GAN Adaptive Training Engine | θ_high={}, θ_low={}".format(THETA_HIGH, THETA_LOW))
    print("=" * 100)


def cargar_data_reales():
    """Carga el dataset Edge-IIoT 2022."""
    print("\n[1] CARGANDO DATOS REALES (Edge-IIoT 2022)")
    print("-" * 50)

    file_size = os.path.getsize(REAL_DATA_PATH)
    print(f"  Archivo: {REAL_DATA_PATH}")
    print(f"  Size: {file_size / (1024 * 1024):.2f} MB")

    start_time = datetime.now()
    df = pd.read_csv(REAL_DATA_PATH, low_memory=False)
    df.columns = df.columns.str.strip()
    print(f"  Cargado en {datetime.now() - start_time}")
    print(f"  Total registros: {len(df):,}")

    return df


def preparar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Prepara features para el model, incluyendo expansión de IPs a octetos."""
    df = df.copy()

    df['Label_Class'] = df[LABEL_COLUMN]

    # Filter only valid classes
    df = df[df['Label_Class'].isin(VALID_CLASSES)].copy()

    # Seleccionar features base numéricas disponibles
    features_available = [f for f in FEATURES_BASE if f in df.columns]
    features_df = df[features_available].copy()

    # Convertir a numérico
    for col in features_df.columns:
        features_df[col] = pd.to_numeric(features_df[col], errors='coerce').fillna(0)

    # Expand IPs to octets (como en CICIDS/UNSW)
    if 'ip.src_host' in df.columns:
        octetos = df['ip.src_host'].astype(str).str.split('.', expand=True)
        for i in range(4):
            features_df[f'Src_IP_{i+1}'] = pd.to_numeric(octetos[i], errors='coerce').fillna(0).astype(int)
    else:
        for i in range(4):
            features_df[f'Src_IP_{i+1}'] = 0

    if 'ip.dst_host' in df.columns:
        octetos = df['ip.dst_host'].astype(str).str.split('.', expand=True)
        for i in range(4):
            features_df[f'Dst_IP_{i+1}'] = pd.to_numeric(octetos[i], errors='coerce').fillna(0).astype(int)
    else:
        for i in range(4):
            features_df[f'Dst_IP_{i+1}'] = 0

    features_df['Label_Class'] = df['Label_Class'].values

    features_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    features_df.fillna(0, inplace=True)

    return features_df


def mostrar_distribucion(y, label_encoder, titulo="Distribution"):
    print(f"\n  {titulo}:")
    counter = Counter(y)
    total = len(y)
    for label_idx in sorted(counter.keys()):
        class = label_encoder.inverse_transform([label_idx])[0]
        count = counter[label_idx]
        pct = (count / total) * 100
        print(f"    {class:<25}: {count:>10,} ({pct:>5.1f}%)")
    print(f"    {'TOTAL':<25}: {total:>10,}")


# ----------------------------
# Funciones de Oversampling
# ----------------------------
def oversample_gan(X_train, y_train, label_encoder, target_count):
    """Usa los generatores GAN para hacer oversampling de classs minoritarias."""
    print("\n  [GAN] Aplicando oversampling con WGAN-GP Edge-IIoT...")

    X_resampled = X_train.copy()
    y_resampled = y_train.copy()

    for class_idx, class in enumerate(label_encoder.classes_):
        current_count = np.sum(y_train == class_idx)

        if current_count >= target_count:
            print(f"    {class}: {current_count:,} >= {target_count:,} (sin cambios)")
            continue

        n_generate = target_count - current_count
        print(f"    {class}: {current_count:,} -> {target_count:,} (generating {n_generate:,})")

        folder = CLASS_TO_FOLDER.get(class)
        if not folder:
            print(f"      [WARN] No hay model GAN para {class}")
            continue

        generator_path = os.path.join(MODELS_DIR, folder, f'generator_{folder}.h5')
        scaler_path = os.path.join(MODELS_DIR, folder, 'scaler.pkl')

        if not os.path.exists(generator_path):
            print(f"      [WARN] No se encontró generator para {class}")
            continue

        generator = load_model(generator_path, compile=False)

        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
        else:
            print(f"      [WARN] No se encontró scaler para {class}")
            continue

        noise = np.random.normal(0, 1, (n_generate, LATENT_DIM))
        X_synthetic_scaled = generator.predict(noise, verbose=0)

        X_synthetic = scaler.inverse_transform(X_synthetic_scaled)

        # Invertir transformación logarítmica
        for i, col in enumerate(FEATURE_NAMES):
            if col in LOG_COLUMNS and i < X_synthetic.shape[1]:
                X_synthetic[:, i] = np.expm1(X_synthetic[:, i])

        X_synthetic = np.clip(X_synthetic, 0, None)

        X_resampled = np.vstack([X_resampled, X_synthetic])
        y_resampled = np.concatenate([y_resampled, np.full(n_generate, class_idx)])

        del generator
        tf.keras.backend.clear_session()

    return X_resampled, y_resampled


def oversample_smote(X_train, y_train, label_encoder, target_count):
    print("\n  [SMOTE] Aplicando SMOTE...")
    counter = Counter(y_train)
    sampling_strategy = {}
    for k in counter:
        if counter[k] < target_count:
            # Para classs muy pequeñas, mínimo k_neighbors=2
            sampling_strategy[k] = target_count

    if not sampling_strategy:
        print("    No se requiere oversampling")
        return X_train, y_train

    # Ajustar k_neighbors si alguna class es muy pequeña
    min_count = min(counter[k] for k in sampling_strategy)
    k_neighbors = min(5, min_count - 1) if min_count > 1 else 1

    smote = SMOTE(sampling_strategy=sampling_strategy, random_state=RANDOM_STATE,
                  k_neighbors=max(1, k_neighbors))
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    for class_idx, class in enumerate(label_encoder.classes_):
        old_count = counter.get(class_idx, 0)
        new_count = np.sum(y_resampled == class_idx)
        if new_count > old_count:
            print(f"    {class}: {old_count:,} -> {new_count:,}")

    return X_resampled, y_resampled


def oversample_adasyn(X_train, y_train, label_encoder, target_count):
    print("\n  [ADASYN] Aplicando ADASYN...")
    counter = Counter(y_train)
    sampling_strategy = {k: target_count for k in counter if counter[k] < target_count}
    if not sampling_strategy:
        return X_train, y_train

    min_count = min(counter[k] for k in sampling_strategy)
    n_neighbors = min(5, min_count - 1) if min_count > 1 else 1

    try:
        adasyn = ADASYN(sampling_strategy=sampling_strategy, random_state=RANDOM_STATE,
                        n_neighbors=max(1, n_neighbors))
        X_resampled, y_resampled = adasyn.fit_resample(X_train, y_train)
        for class_idx, class in enumerate(label_encoder.classes_):
            old_count = counter.get(class_idx, 0)
            new_count = np.sum(y_resampled == class_idx)
            if new_count > old_count:
                print(f"    {class}: {old_count:,} -> {new_count:,}")
        return X_resampled, y_resampled
    except Exception as e:
        print(f"    [ERROR] ADASYN falló: {e}")
        return oversample_smote(X_train, y_train, label_encoder, target_count)


def oversample_borderline(X_train, y_train, label_encoder, target_count):
    print("\n  [BorderlineSMOTE] Aplicando BorderlineSMOTE...")
    counter = Counter(y_train)
    sampling_strategy = {k: target_count for k in counter if counter[k] < target_count}
    if not sampling_strategy:
        return X_train, y_train

    min_count = min(counter[k] for k in sampling_strategy)
    k_neighbors = min(5, min_count - 1) if min_count > 1 else 1

    try:
        borderline = BorderlineSMOTE(sampling_strategy=sampling_strategy,
                                     random_state=RANDOM_STATE,
                                     k_neighbors=max(1, k_neighbors))
        X_resampled, y_resampled = borderline.fit_resample(X_train, y_train)
        for class_idx, class in enumerate(label_encoder.classes_):
            old_count = counter.get(class_idx, 0)
            new_count = np.sum(y_resampled == class_idx)
            if new_count > old_count:
                print(f"    {class}: {old_count:,} -> {new_count:,}")
        return X_resampled, y_resampled
    except Exception as e:
        print(f"    [ERROR] BorderlineSMOTE falló: {e}")
        return oversample_smote(X_train, y_train, label_encoder, target_count)


def oversample_smoteenn(X_train, y_train, label_encoder, target_count):
    print("\n  [SMOTE-ENN] Aplicando SMOTE + ENN cleaning...")
    counter = Counter(y_train)
    sampling_strategy = {k: target_count for k in counter if counter[k] < target_count}
    if not sampling_strategy:
        return X_train, y_train

    min_count = min(counter[k] for k in sampling_strategy)
    k_neighbors = min(5, min_count - 1) if min_count > 1 else 1

    try:
        smote = SMOTE(sampling_strategy=sampling_strategy, random_state=RANDOM_STATE,
                      k_neighbors=max(1, k_neighbors))
        smoteenn = SMOTEENN(smote=smote, random_state=RANDOM_STATE)
        X_resampled, y_resampled = smoteenn.fit_resample(X_train, y_train)
        print(f"    Muestras antes: {len(y_train):,}")
        print(f"    Muestras después: {len(y_resampled):,}")
        for class_idx, class in enumerate(label_encoder.classes_):
            old_count = counter.get(class_idx, 0)
            new_count = np.sum(y_resampled == class_idx)
            print(f"    {class}: {old_count:,} -> {new_count:,}")
        return X_resampled, y_resampled
    except Exception as e:
        print(f"    [ERROR] SMOTE-ENN falló: {e}")
        return oversample_smote(X_train, y_train, label_encoder, target_count)


def oversample_random(X_train, y_train, label_encoder, target_count):
    print("\n  [RandomOverSampler] Aplicando oversampling aleatorio...")
    counter = Counter(y_train)
    sampling_strategy = {k: target_count for k in counter if counter[k] < target_count}
    if not sampling_strategy:
        return X_train, y_train

    ros = RandomOverSampler(sampling_strategy=sampling_strategy, random_state=RANDOM_STATE)
    X_resampled, y_resampled = ros.fit_resample(X_train, y_train)
    for class_idx, class in enumerate(label_encoder.classes_):
        old_count = counter.get(class_idx, 0)
        new_count = np.sum(y_resampled == class_idx)
        if new_count > old_count:
            print(f"    {class}: {old_count:,} -> {new_count:,}")
    return X_resampled, y_resampled


# ----------------------------
# Training y Evaluation
# ----------------------------
def crear_model(model_tipo='logistic'):
    if model_tipo == 'logistic':
        return LogisticRegression(
            max_iter=500, random_state=RANDOM_STATE, n_jobs=-1,
            class_weight='balanced', solver='lbfgs'
        )
    elif model_tipo == 'tree':
        return DecisionTreeClassifier(
            max_depth=10, random_state=RANDOM_STATE, class_weight='balanced'
        )
    else:
        return LGBMClassifier(
            n_estimators=200, max_depth=15, learning_rate=0.1,
            num_leaves=63, random_state=RANDOM_STATE, n_jobs=-1,
            verbose=-1, class_weight='balanced'
        )


MODELO_TIPO = 'logistic'


def entrenar_y_evaluar(X_train, y_train, X_test, y_test, label_encoder,
                       nombre_tecnica, output_dir):
    print(f"\n  Entrenando model con {nombre_tecnica}...")
    print(f"    Train: {len(X_train):,} samples")

    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_train_scaled = np.nan_to_num(X_train_scaled, nan=0, posinf=0, neginf=0)
    X_test_scaled = np.nan_to_num(X_test_scaled, nan=0, posinf=0, neginf=0)

    model = crear_model(MODELO_TIPO)
    start_time = datetime.now()
    model.fit(X_train_scaled, y_train)
    train_time = datetime.now() - start_time

    y_pred = model.predict(X_test_scaled)

    accuracy = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
    f1_weighted = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    precision_macro = precision_score(y_test, y_pred, average='macro', zero_division=0)
    recall_macro = recall_score(y_test, y_pred, average='macro', zero_division=0)

    print(f"    Accuracy: {accuracy:.4f}")
    print(f"    F1-Macro: {f1_macro:.4f}")
    print(f"    F1-Weighted: {f1_weighted:.4f}")
    print(f"    Tiempo training: {train_time}")

    report = classification_report(y_test, y_pred, target_names=label_encoder.classes_,
                                   output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(os.path.join(output_dir, f'{nombre_tecnica}_classification_report.csv'))

    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm, index=label_encoder.classes_, columns=label_encoder.classes_)
    cm_df.to_csv(os.path.join(output_dir, f'{nombre_tecnica}_confusion_matrix.csv'))

    return {
        'Tecnica': nombre_tecnica,
        'Train_Samples': len(X_train),
        'Test_Samples': len(X_test),
        'Accuracy': accuracy,
        'F1_Macro': f1_macro,
        'F1_Weighted': f1_weighted,
        'Precision_Macro': precision_macro,
        'Recall_Macro': recall_macro,
        'Train_Time': str(train_time),
        'Report': report,
        'Confusion_Matrix': cm
    }


def generar_summary(results, label_encoder, output_dir):
    """Genera summary comparativo de todas las técnicas."""
    summary = []
    summary.append("=" * 120)
    summary.append(" " * 15 + "COMPARACIÓN DE TÉCNICAS DE OVERSAMPLING - EDGE-IIoT 2022")
    summary.append(" " * 25 + "GAN vs Métodos Tradicionales")
    summary.append(" " * 20 + "AMD-GAN Adaptive Training Engine | θ_high={}, θ_low={}".format(THETA_HIGH, THETA_LOW))
    summary.append(" " * 30 + f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    summary.append("=" * 120)

    results_sorted = sorted(results, key=lambda x: x['F1_Macro'], reverse=True)

    summary.append("\n" + "=" * 120)
    summary.append("1. RANKING DE TÉCNICAS (ordenado por F1-Macro)")
    summary.append("=" * 120)

    summary.append("\n" + "-" * 120)
    header = f"{'Pos':<5} {'Técnica':<20} | {'Train':>12} | {'Accuracy':>10} {'F1-Macro':>10} {'F1-Weight':>10} | {'Prec':>8} {'Recall':>8}"
    summary.append(header)
    summary.append("-" * 120)

    mejor = results_sorted[0]
    for i, r in enumerate(results_sorted, 1):
        delta = r['F1_Macro'] - mejor['F1_Macro']
        delta_str = f"({delta:+.4f})" if i > 1 else "(best)"
        linea = (f"{i:<5} {r['Tecnica']:<20} | {r['Train_Samples']:>12,} | "
                 f"{r['Accuracy']:>10.4f} {r['F1_Macro']:>10.4f} {r['F1_Weighted']:>10.4f} | "
                 f"{r['Precision_Macro']:>8.4f} {r['Recall_Macro']:>8.4f} {delta_str}")
        summary.append(linea)

    # GAN vs Others
    summary.append("\n\n" + "=" * 120)
    summary.append("2. COMPARACIÓN GAN vs OTROS MÉTODOS")
    summary.append("=" * 120)

    gan_result = next((r for r in results if r['Tecnica'] == 'GAN'), None)
    baseline_result = next((r for r in results if r['Tecnica'] == 'Original'), None)
    smote_result = next((r for r in results if r['Tecnica'] == 'SMOTE'), None)

    if gan_result and baseline_result:
        summary.append(f"\n  GAN vs Original:")
        summary.append(f"    Delta F1-Macro: {gan_result['F1_Macro'] - baseline_result['F1_Macro']:+.4f}")
        summary.append(f"    Delta Accuracy: {gan_result['Accuracy'] - baseline_result['Accuracy']:+.4f}")

    if gan_result and smote_result:
        summary.append(f"\n  GAN vs SMOTE:")
        summary.append(f"    Delta F1-Macro: {gan_result['F1_Macro'] - smote_result['F1_Macro']:+.4f}")
        summary.append(f"    Delta Accuracy: {gan_result['Accuracy'] - smote_result['Accuracy']:+.4f}")

    # F1 por class
    summary.append("\n\n" + "=" * 120)
    summary.append("3. F1-SCORE POR CLASE PARA CADA TÉCNICA")
    summary.append("=" * 120)

    tecnicas_nombres = [r['Tecnica'][:12] for r in results]
    header_line = f"{'Class':<25} | " + " | ".join([f"{t:>12}" for t in tecnicas_nombres])
    summary.append("\n" + "-" * len(header_line))
    summary.append(header_line)
    summary.append("-" * len(header_line))

    for class in label_encoder.classes_:
        valores = []
        for r in results:
            if class in r['Report']:
                valores.append(f"{r['Report'][class]['f1-score']:>12.4f}")
            else:
                valores.append(f"{'N/A':>12}")
        summary.append(f"{class:<25} | " + " | ".join(valores))

    # AMD-GAN Threshold Analysis
    summary.append("\n\n" + "=" * 120)
    summary.append("4. ANÁLISIS POR CATEGORÍA AMD-GAN (θ_high={}, θ_low={})".format(THETA_HIGH, THETA_LOW))
    summary.append("=" * 120)

    for categoria, desc in [
        ('VERY_SMALL', f'< {THETA_LOW} samples (MITM, Fingerprinting)'),
        ('LARGE', f'>= {THETA_HIGH} samples (resto)')
    ]:
        summary.append(f"\n  {categoria}: {desc}")
        if gan_result:
            for class in label_encoder.classes_:
                if class in gan_result['Report']:
                    f1_val = gan_result['Report'][class]['f1-score']
                    summary.append(f"    {class}: F1={f1_val:.4f}")

    # Conclusión
    summary.append("\n\n" + "=" * 120)
    summary.append("5. CONCLUSIONES")
    summary.append("=" * 120)
    summary.append(f"\n  MEJOR TÉCNICA: {mejor['Tecnica']} (F1-Macro = {mejor['F1_Macro']:.4f})")

    if gan_result:
        gan_pos = next(i for i, r in enumerate(results_sorted, 1) if r['Tecnica'] == 'GAN')
        if gan_pos == 1:
            summary.append(f"  GAN supera a todas las técnicas tradicionales de oversampling")
        elif gan_pos <= 3:
            summary.append(f"  GAN en posición {gan_pos} - Competitivo con métodos tradicionales")
        else:
            summary.append(f"  GAN en posición {gan_pos} - Los métodos tradicionales funcionan mejor")

    summary_texto = "\n".join(summary)

    with open(os.path.join(output_dir, 'RESUMEN_COMPARATIVO.txt'), 'w') as f:
        f.write(summary_texto)

    print("\n" + summary_texto)

    # CSV comparativo
    comparative = [{
        'Tecnica': r['Tecnica'],
        'Train_Samples': r['Train_Samples'],
        'Accuracy': r['Accuracy'],
        'F1_Macro': r['F1_Macro'],
        'F1_Weighted': r['F1_Weighted'],
        'Precision_Macro': r['Precision_Macro'],
        'Recall_Macro': r['Recall_Macro'],
        'Train_Time': r['Train_Time']
    } for r in results]

    pd.DataFrame(comparative).to_csv(os.path.join(output_dir, 'comparative_tecnicas.csv'), index=False)

    return summary_texto


# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser(
        description='Oversampling technique comparison para Edge-IIoT 2022'
    )
    parser.add_argument('--max-samples', type=int, default=100000,
                        help='Máximo samples por class para balanceo (default: 100000)')
    parser.add_argument('--test-size', type=float, default=0.3)
    parser.add_argument('--skip-gan', action='store_true',
                        help='Omitir técnica GAN')
    parser.add_argument('--output', type=str, default=None)

    args = parser.parse_args()

    print_header()

    if args.output:
        output_dir = args.output
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(OUTPUT_DIR, f'comparison_{timestamp}')

    os.makedirs(output_dir, exist_ok=True)
    print(f"\n  Results se guardarán en: {output_dir}")

    # Cargar y preparar data
    df = cargar_data_reales()
    df_features = preparar_features(df)

    # Determinar classs para el experimento
    # Solo incluir classs con suficientes samples en el train set
    class_counts = df_features['Label_Class'].value_counts()
    classs_experimento = [c for c in VALID_CLASSES if c in class_counts.index]

    print(f"\n[2] DISTRIBUCIÓN DE CLASES")
    print("-" * 50)
    print(f"  Classes: {classs_experimento}")
    print(f"  Total samples: {len(df_features):,}")
    for class in classs_experimento:
        count = class_counts.get(class, 0)
        pct = (count / len(df_features)) * 100
        # Clasificar según AMD-GAN thresholds
        if count < THETA_LOW:
            cat = "VERY_SMALL"
        elif count < THETA_HIGH:
            cat = "SMALL"
        else:
            cat = "LARGE"
        print(f"    {class:<25}: {count:>10,} ({pct:>5.1f}%) [{cat}]")

    feature_cols = [c for c in df_features.columns if c != 'Label_Class']
    X = df_features[feature_cols].values

    label_encoder = LabelEncoder()
    label_encoder.fit(classs_experimento)
    y = label_encoder.transform(df_features['Label_Class'])

    print(f"\n  Features: {len(feature_cols)}")
    print(f"  Classs codificadas: {list(label_encoder.classes_)}")

    # Split
    print(f"\n[3] DIVIDIENDO TRAIN/TEST")
    print("-" * 50)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=RANDOM_STATE, stratify=y
    )

    print(f"  Train: {len(X_train):,} samples")
    print(f"  Test: {len(X_test):,} samples")

    mostrar_distribucion(y_train, label_encoder, "Distribution Train (original)")
    mostrar_distribucion(y_test, label_encoder, "Distribution Test")

    counter = Counter(y_train)
    max_class_count = max(counter.values())
    target_count = min(max_class_count, args.max_samples)
    print(f"\n  Target para balanceo: {target_count:,} samples por class")

    # Aplicar técnicas
    print(f"\n[4] APLICANDO TÉCNICAS DE OVERSAMPLING")
    print("=" * 70)

    tecnicas = {}

    # 1. Original
    print("\n" + "-" * 70)
    print("  [Original] No oversampling (baseline)")
    tecnicas['Original'] = (X_train.copy(), y_train.copy())
    mostrar_distribucion(y_train, label_encoder, "Distribution")

    # 2. GAN
    if not args.skip_gan:
        print("\n" + "-" * 70)
        X_gan, y_gan = oversample_gan(X_train, y_train, label_encoder, target_count)
        tecnicas['GAN'] = (X_gan, y_gan)
        mostrar_distribucion(y_gan, label_encoder, "Distribution GAN")

    # 3. SMOTE
    print("\n" + "-" * 70)
    X_smote, y_smote = oversample_smote(X_train, y_train, label_encoder, target_count)
    tecnicas['SMOTE'] = (X_smote, y_smote)
    mostrar_distribucion(y_smote, label_encoder, "Distribution SMOTE")

    # 4. ADASYN
    print("\n" + "-" * 70)
    X_adasyn, y_adasyn = oversample_adasyn(X_train, y_train, label_encoder, target_count)
    tecnicas['ADASYN'] = (X_adasyn, y_adasyn)
    mostrar_distribucion(y_adasyn, label_encoder, "Distribution ADASYN")

    # 5. BorderlineSMOTE
    print("\n" + "-" * 70)
    X_bl, y_bl = oversample_borderline(X_train, y_train, label_encoder, target_count)
    tecnicas['BorderlineSMOTE'] = (X_bl, y_bl)
    mostrar_distribucion(y_bl, label_encoder, "Distribution BorderlineSMOTE")

    # 6. SMOTE-ENN
    print("\n" + "-" * 70)
    X_senn, y_senn = oversample_smoteenn(X_train, y_train, label_encoder, target_count)
    tecnicas['SMOTE_ENN'] = (X_senn, y_senn)
    mostrar_distribucion(y_senn, label_encoder, "Distribution SMOTE-ENN")

    # 7. Random
    print("\n" + "-" * 70)
    X_rnd, y_rnd = oversample_random(X_train, y_train, label_encoder, target_count)
    tecnicas['RandomOverSampler'] = (X_rnd, y_rnd)
    mostrar_distribucion(y_rnd, label_encoder, "Distribution Random")

    # Train y evaluar
    print(f"\n\n[5] ENTRENAMIENTO Y EVALUACIÓN")
    print("=" * 70)
    print(f"  Model: Logistic Regression (multinomial, balanced)")
    print(f"  Test set: {len(X_test):,} samples")

    results = []
    for nombre, (X_t, y_t) in tecnicas.items():
        print(f"\n" + "-" * 70)
        result = entrenar_y_evaluar(X_t, y_t, X_test, y_test, label_encoder, nombre, output_dir)
        results.append(result)

    # Summary
    print(f"\n\n[6] GENERANDO RESUMEN COMPARATIVO")
    print("=" * 70)
    generar_summary(results, label_encoder, output_dir)

    print(f"\n\n{'='*100}")
    print(f"  EXPERIMENTO COMPLETADO")
    print(f"  Results saved to: {output_dir}")
    print(f"{'='*100}")


if __name__ == "__main__":
    main()
