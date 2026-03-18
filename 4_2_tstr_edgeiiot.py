"""
Script: Train on Synthetic, Test on Real - Edge-IIoT 2022 (TSTR)

Compara CUATRO escenarios:
  1. TSTR Multiclass: Entrena en synthetics, testea en reales (15 classs)
  2. TRTR Multiclass: Entrena en reales, testea en reales (15 classs) - baseline
  3. TSTR Binario: Entrena en synthetics, testea en reales (Normal vs ATTACK)
  4. TRTR Binario: Entrena en reales, testea en reales (Normal vs ATTACK) - baseline

Usa los datasets synthetics generados por 2_2_generate_synthetic_dataset_edgeiiot.py

Usage:
    python 4_2_tstr_edgeiiot.py                     # Interactivo
    python 4_2_tstr_edgeiiot.py --dataset nombre     # Dataset específico
    python 4_2_tstr_edgeiiot.py --list               # Listar datasets
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier,
    VotingClassifier
)
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

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
SYNTHETIC_DATASETS_DIR = '<PATH_TO_GENERATED_DATASETS_EDGEIIOT>'
OUTPUT_BASE_DIR = '<PATH_TO_RESULTS_TSTR_EDGEIIOT>'
RANDOM_STATE = 42
LABEL_COLUMN = 'Attack_type'

# AMD-GAN Adaptive Training Engine thresholds (d=58: 50 base + 8 IP octets)
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

# Features base numéricas (50 features, mismo orden que GAN training)
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
# Utilities para manejo de datasets
# ----------------------------
def listar_datasets_disponibles():
    """Lista todos los datasets synthetics Edge-IIoT disponibles."""
    if not os.path.exists(SYNTHETIC_DATASETS_DIR):
        print(f"[ERROR] No existe el directorio: {SYNTHETIC_DATASETS_DIR}")
        return []

    datasets = []
    for f in os.listdir(SYNTHETIC_DATASETS_DIR):
        if f.endswith('.csv') and not f.endswith('_config.json'):
            csv_path = os.path.join(SYNTHETIC_DATASETS_DIR, f)
            config_path = csv_path.replace('.csv', '_config.json')

            try:
                n_rows = sum(1 for _ in open(csv_path)) - 1
                config = {}
                if os.path.exists(config_path):
                    import json
                    with open(config_path, 'r') as cf:
                        config = json.load(cf)

                datasets.append({
                    'nombre': f.replace('.csv', ''),
                    'archivo': f,
                    'path': csv_path,
                    'rows': n_rows,
                    'config': config,
                    'classs': config.get('samples_per_class', {})
                })
            except Exception as e:
                print(f"  [WARN] Error leyendo {f}: {e}")

    return datasets


def mostrar_datasets_disponibles(datasets):
    print("\n" + "=" * 80)
    print("DATASETS SINTÉTICOS EDGE-IIoT 2022 DISPONIBLES")
    print("=" * 80)

    if not datasets:
        print("  No hay datasets disponibles en:", SYNTHETIC_DATASETS_DIR)
        print("  Genera uno con: python 2_2_generate_synthetic_dataset_edgeiiot.py --interactive")
        return

    for i, ds in enumerate(datasets, 1):
        print(f"\n  [{i}] {ds['nombre']}")
        print(f"      Archivo: {ds['archivo']}")
        print(f"      Total samples: {ds['rows']:,}")
        if ds['classs']:
            for class, n in ds['classs'].items():
                print(f"        - {class}: {n:,}")

    print("\n" + "=" * 80)


def seleccionar_dataset_interactivo(datasets):
    mostrar_datasets_disponibles(datasets)
    if not datasets:
        return None

    while True:
        try:
            inp = input("\nSelecciona un dataset (número o nombre): ").strip()
            if inp.isdigit():
                idx = int(inp) - 1
                if 0 <= idx < len(datasets):
                    return datasets[idx]
                print(f"  [!] Número debe estar entre 1 y {len(datasets)}")
                continue
            for ds in datasets:
                if ds['nombre'].lower() == inp.lower():
                    return ds
            print(f"  [!] Dataset '{inp}' no encontrado")
        except KeyboardInterrupt:
            print("\n\nCancelado.")
            return None


def cargar_dataset_sintetico(dataset_info) -> pd.DataFrame:
    print("\n" + "=" * 70)
    print("CARGANDO DATASET SINTÉTICO EDGE-IIoT")
    print("=" * 70)

    print(f"  Archivo: {dataset_info['archivo']}")
    df = pd.read_csv(dataset_info['path'])
    print(f"  Total samples: {len(df):,}")

    # Detectar columna de class
    if LABEL_COLUMN in df.columns:
        label_col = LABEL_COLUMN
    elif 'Label_Class' in df.columns:
        label_col = 'Label_Class'
    elif 'Label' in df.columns:
        label_col = 'Label'
    else:
        raise ValueError(f"No se encontró columna de class ({LABEL_COLUMN}, Label_Class o Label)")

    df['Label_Class'] = df[label_col]

    print(f"\n  Class distribution:")
    for class, count in df['Label_Class'].value_counts().items():
        pct = (count / len(df)) * 100
        print(f"    {class}: {count:,} ({pct:.1f}%)")

    return df


def cargar_data_reales() -> pd.DataFrame:
    print("\n" + "=" * 70)
    print("CARGANDO DATOS REALES (Edge-IIoT 2022)")
    print("=" * 70)

    file_size = os.path.getsize(REAL_DATA_PATH)
    print(f"  Size del archivo: {file_size / (1024 * 1024):.2f} MB")

    start_time = datetime.now()
    df = pd.read_csv(REAL_DATA_PATH, low_memory=False)
    df.columns = df.columns.str.strip()
    print(f"  Data cargados en {datetime.now() - start_time}")

    return df


def preparar_features_reales(df: pd.DataFrame) -> pd.DataFrame:
    """Prepara el dataset real con la misma estructura de features que el synthetic, incluyendo IPs expandidas."""
    df = df.copy()
    df.columns = df.columns.str.strip()

    df['Label_Class'] = df[LABEL_COLUMN]

    # Seleccionar solo classs válidas
    df = df[df['Label_Class'].isin(VALID_CLASSES)].copy()

    # Seleccionar features base numéricas disponibles
    features_available = [f for f in FEATURES_BASE if f in df.columns]
    result = df[features_available].copy()

    # Convertir a numérico
    for col in result.columns:
        result[col] = pd.to_numeric(result[col], errors='coerce').fillna(0)

    # Expand IPs to octets (como en CICIDS/UNSW)
    if 'ip.src_host' in df.columns:
        octetos = df['ip.src_host'].astype(str).str.split('.', expand=True)
        for i in range(4):
            result[f'Src_IP_{i+1}'] = pd.to_numeric(octetos[i], errors='coerce').fillna(0).astype(int)
    else:
        for i in range(4):
            result[f'Src_IP_{i+1}'] = 0

    if 'ip.dst_host' in df.columns:
        octetos = df['ip.dst_host'].astype(str).str.split('.', expand=True)
        for i in range(4):
            result[f'Dst_IP_{i+1}'] = pd.to_numeric(octetos[i], errors='coerce').fillna(0).astype(int)
    else:
        for i in range(4):
            result[f'Dst_IP_{i+1}'] = 0

    result['Label_Class'] = df['Label_Class'].values

    result.replace([np.inf, -np.inf], np.nan, inplace=True)
    result.fillna(0, inplace=True)

    return result


def filtrar_classs(df: pd.DataFrame, classs_objetivo: list) -> pd.DataFrame:
    df_filtrado = df[df['Label_Class'].isin(classs_objetivo)].copy()
    print(f"\n  Filas después de filtrar classs objetivo: {len(df_filtrado):,} de {len(df):,}")
    return df_filtrado


def preparar_data_para_training(df_synth, df_real, label_encoder):
    cols_synth = set(df_synth.columns) - {'Label', 'Label_Class', LABEL_COLUMN}
    cols_real = set(df_real.columns) - {'Label', 'Label_Class', LABEL_COLUMN}
    feature_cols = sorted(list(cols_synth.intersection(cols_real)))

    print(f"\n  Features comunes para training: {len(feature_cols)}")

    X_synth = df_synth[feature_cols].values
    y_synth = label_encoder.transform(df_synth['Label_Class'])

    X_real = df_real[feature_cols].values
    y_real = label_encoder.transform(df_real['Label_Class'])

    X_synth = np.nan_to_num(X_synth, nan=0, posinf=0, neginf=0)
    X_real = np.nan_to_num(X_real, nan=0, posinf=0, neginf=0)

    return X_synth, y_synth, X_real, y_real, feature_cols


def entrenar_y_evaluar_model(model, nombre, X_train, y_train, X_test, y_test,
                               label_encoder, output_dir, prefijo=""):
    print(f"\n  Entrenando {nombre}...", end=" ")

    start_time = datetime.now()
    model.fit(X_train, y_train)
    train_time = datetime.now() - start_time

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
    f1_weighted = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    print(f"Acc: {accuracy:.4f}, F1-Macro: {f1_macro:.4f}, F1-Weighted: {f1_weighted:.4f}")

    report = classification_report(y_test, y_pred, target_names=label_encoder.classes_,
                                   output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(os.path.join(output_dir, f'{prefijo}{nombre}_classification_report.csv'))

    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm, index=label_encoder.classes_, columns=label_encoder.classes_)
    cm_df.to_csv(os.path.join(output_dir, f'{prefijo}{nombre}_confusion_matrix.csv'))

    return {
        'Model': nombre,
        'Accuracy': accuracy,
        'F1_Macro': f1_macro,
        'F1_Weighted': f1_weighted,
        'Train_Time': str(train_time),
        'Train_Samples': len(y_train),
        'Test_Samples': len(y_test),
        'Report': report,
        'Confusion_Matrix': cm
    }


def ejecutar_experimento(X_train, y_train, X_test, y_test, label_encoder,
                         output_dir, prefijo, descripcion):
    print(f"\n{'='*70}")
    print(f"{descripcion}")
    print(f"  Train: {len(X_train):,} samples | Test: {len(X_test):,} samples")
    print('='*70)

    models = {
        'Dummy': DummyClassifier(strategy='most_frequent', random_state=RANDOM_STATE),
        'LogisticReg': LogisticRegression(max_iter=1000,
                                          n_jobs=-1, random_state=RANDOM_STATE, solver='saga'),
        'RandomForest': RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE,
                                                n_jobs=-1, class_weight='balanced_subsample'),
        'KNeighbors': KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
        'LightGBM': LGBMClassifier(n_estimators=100, random_state=RANDOM_STATE,
                                    n_jobs=-1, verbose=-1),
        'XGBoost': XGBClassifier(n_estimators=100, random_state=RANDOM_STATE,
                                  n_jobs=-1, verbosity=0, use_label_encoder=False),
        'MLP': MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, random_state=RANDOM_STATE, early_stopping=True, validation_fraction=0.1),
        'Voting': VotingClassifier(
            estimators=[
                ('rf', RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)),
                ('et', ExtraTreesClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)),
                ('lgbm', LGBMClassifier(n_estimators=50, random_state=RANDOM_STATE, n_jobs=-1, verbose=-1))
            ],
            voting='soft', n_jobs=-1
        ),
    }

    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results = []
    for nombre, model in models.items():
        result = entrenar_y_evaluar_model(
            model, nombre, X_train_scaled, y_train, X_test_scaled, y_test,
            label_encoder, output_dir, prefijo
        )
        result['Experimento'] = descripcion
        results.append(result)

    return results


def generar_summary_global(results_tstr_multi, results_trtr_multi,
                           results_tstr_binary, results_trtr_binary,
                           label_encoder_multi, label_encoder_binary,
                           output_dir, dataset_name):
    """Genera un summary global combinando los 4 experimentos."""

    summary = []
    summary.append("=" * 120)
    summary.append(" " * 25 + "RESUMEN GLOBAL: TSTR vs TRTR - EDGE-IIoT 2022")
    summary.append(" " * 15 + "Clasificación Multiclass y Binaria")
    summary.append(" " * 15 + "AMD-GAN Adaptive Training Engine | θ_high={}, θ_low={}".format(THETA_HIGH, THETA_LOW))
    summary.append(f" " * 30 + f"Dataset: {dataset_name}")
    summary.append(f" " * 40 + f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    summary.append("=" * 120)

    # Summary ejecutivo
    mejor_tstr_multi = max(results_tstr_multi, key=lambda x: x['F1_Macro'])
    mejor_trtr_multi = max(results_trtr_multi, key=lambda x: x['F1_Macro'])
    mejor_tstr_binary = max(results_tstr_binary, key=lambda x: x['F1_Macro'])
    mejor_trtr_binary = max(results_trtr_binary, key=lambda x: x['F1_Macro'])

    gap_multi = mejor_tstr_multi['F1_Macro'] - mejor_trtr_multi['F1_Macro']
    gap_binary = mejor_tstr_binary['F1_Macro'] - mejor_trtr_binary['F1_Macro']

    summary.append(f"\nClasses: {', '.join(label_encoder_multi.classes_)}")

    summary.append("\n" + "=" * 120)
    summary.append("1. RESUMEN EJECUTIVO")
    summary.append("=" * 120)
    summary.append(f"\n  MULTICLASE ({len(label_encoder_multi.classes_)} classs):")
    summary.append(f"    TSTR: {mejor_tstr_multi['Model']:<15} F1-Macro = {mejor_tstr_multi['F1_Macro']:.4f}  Accuracy = {mejor_tstr_multi['Accuracy']:.4f}")
    summary.append(f"    TRTR: {mejor_trtr_multi['Model']:<15} F1-Macro = {mejor_trtr_multi['F1_Macro']:.4f}  Accuracy = {mejor_trtr_multi['Accuracy']:.4f}  (baseline)")
    summary.append(f"    Gap TSTR-TRTR: {gap_multi:+.4f}")
    summary.append(f"\n  BINARIA (Normal vs ATTACK):")
    summary.append(f"    TSTR: {mejor_tstr_binary['Model']:<15} F1-Macro = {mejor_tstr_binary['F1_Macro']:.4f}  Accuracy = {mejor_tstr_binary['Accuracy']:.4f}")
    summary.append(f"    TRTR: {mejor_trtr_binary['Model']:<15} F1-Macro = {mejor_trtr_binary['F1_Macro']:.4f}  Accuracy = {mejor_trtr_binary['Accuracy']:.4f}  (baseline)")
    summary.append(f"    Gap TSTR-TRTR: {gap_binary:+.4f}")

    # Table multiclass
    summary.append("\n\n" + "=" * 120)
    summary.append("2. COMPARATIVA MULTICLASE")
    summary.append("=" * 120)

    summary.append("\n" + "-" * 120)
    header = (f"{'Model':<20} | {'TSTR Acc':>10} {'TSTR F1-M':>10} {'TSTR F1-W':>10} | "
              f"{'TRTR Acc':>10} {'TRTR F1-M':>10} {'TRTR F1-W':>10} | {'dF1-M':>8} {'dAcc':>8}")
    summary.append(header)
    summary.append("-" * 120)

    for r_tstr in results_tstr_multi:
        model = r_tstr['Model']
        r_trtr = next((r for r in results_trtr_multi if r['Model'] == model), None)
        if r_trtr:
            delta_f1 = r_tstr['F1_Macro'] - r_trtr['F1_Macro']
            delta_acc = r_tstr['Accuracy'] - r_trtr['Accuracy']
            linea = (f"{model:<20} | {r_tstr['Accuracy']:>10.4f} {r_tstr['F1_Macro']:>10.4f} "
                     f"{r_tstr['F1_Weighted']:>10.4f} | {r_trtr['Accuracy']:>10.4f} "
                     f"{r_trtr['F1_Macro']:>10.4f} {r_trtr['F1_Weighted']:>10.4f} | "
                     f"{delta_f1:>+8.4f} {delta_acc:>+8.4f}")
            summary.append(linea)

    # Table binaria
    summary.append("\n\n" + "=" * 120)
    summary.append("3. COMPARATIVA BINARIA (Normal vs ATTACK)")
    summary.append("=" * 120)

    summary.append("\n" + "-" * 120)
    summary.append(header)
    summary.append("-" * 120)

    for r_tstr in results_tstr_binary:
        model = r_tstr['Model']
        r_trtr = next((r for r in results_trtr_binary if r['Model'] == model), None)
        if r_trtr:
            delta_f1 = r_tstr['F1_Macro'] - r_trtr['F1_Macro']
            delta_acc = r_tstr['Accuracy'] - r_trtr['Accuracy']
            linea = (f"{model:<20} | {r_tstr['Accuracy']:>10.4f} {r_tstr['F1_Macro']:>10.4f} "
                     f"{r_tstr['F1_Weighted']:>10.4f} | {r_trtr['Accuracy']:>10.4f} "
                     f"{r_trtr['F1_Macro']:>10.4f} {r_trtr['F1_Weighted']:>10.4f} | "
                     f"{delta_f1:>+8.4f} {delta_acc:>+8.4f}")
            summary.append(linea)

    # Análisis por class
    summary.append("\n\n" + "=" * 120)
    summary.append(f"4. ANÁLISIS POR CLASE - MEJOR TSTR: {mejor_tstr_multi['Model']}")
    summary.append("=" * 120)

    report_tstr = mejor_tstr_multi['Report']
    report_trtr = mejor_trtr_multi['Report']

    summary.append("\n" + "-" * 100)
    summary.append(f"{'Class':<25} | {'TSTR Prec':>10} {'TSTR Rec':>10} {'TSTR F1':>10} | {'TRTR F1':>10} | {'dF1':>8}")
    summary.append("-" * 100)

    for class in label_encoder_multi.classes_:
        if class in report_tstr and class in report_trtr:
            tstr_r = report_tstr[class]
            trtr_r = report_trtr[class]
            delta = tstr_r['f1-score'] - trtr_r['f1-score']
            summary.append(f"{class:<25} | {tstr_r['precision']:>10.4f} {tstr_r['recall']:>10.4f} "
                           f"{tstr_r['f1-score']:>10.4f} | {trtr_r['f1-score']:>10.4f} | {delta:>+8.4f}")

    # AMD-GAN category analysis
    summary.append("\n\n" + "=" * 120)
    summary.append("5. ANÁLISIS POR CATEGORÍA AMD-GAN (θ_high={}, θ_low={})".format(THETA_HIGH, THETA_LOW))
    summary.append("=" * 120)

    summary.append(f"\n  VERY_SMALL (< {THETA_LOW} samples): Classs con samples insuficientes")
    summary.append(f"    MITM, Fingerprinting")
    summary.append(f"  LARGE (>= {THETA_HIGH} samples): Classs con data abundantes")
    summary.append(f"    {', '.join([c for c in VALID_CLASSES if c not in ['MITM', 'Fingerprinting']])}")

    if report_tstr:
        summary.append(f"\n  F1 TSTR por categoría:")
        for cat, classs in [
            ('VERY_SMALL', ['MITM', 'Fingerprinting']),
            ('LARGE', [c for c in VALID_CLASSES if c not in ['MITM', 'Fingerprinting']])
        ]:
            f1_vals = [report_tstr[c]['f1-score'] for c in classs if c in report_tstr]
            if f1_vals:
                avg_f1 = np.mean(f1_vals)
                summary.append(f"    {cat}: F1-Macro avg = {avg_f1:.4f}")

    # Insights
    summary.append("\n\n" + "=" * 120)
    summary.append("6. INSIGHTS")
    summary.append("=" * 120)

    if mejor_tstr_binary['F1_Macro'] > 0.95:
        summary.append(f"\n  EXCELENTE: TSTR Binario F1-Macro = {mejor_tstr_binary['F1_Macro']:.4f} (>0.95)")
    elif mejor_tstr_binary['F1_Macro'] > 0.90:
        summary.append(f"\n  MUY BUENO: TSTR Binario F1-Macro = {mejor_tstr_binary['F1_Macro']:.4f} (>0.90)")
    else:
        summary.append(f"\n  ACEPTABLE: TSTR Binario F1-Macro = {mejor_tstr_binary['F1_Macro']:.4f}")

    if mejor_tstr_multi['F1_Macro'] > 0.85:
        summary.append(f"  ÉXITO: TSTR Multiclass F1-Macro = {mejor_tstr_multi['F1_Macro']:.4f} (>0.85)")
    else:
        summary.append(f"  TSTR Multiclass F1-Macro = {mejor_tstr_multi['F1_Macro']:.4f} (<0.85) - Hay margen de mejora")

    # IoT-specific insight
    summary.append(f"\n  Edge-IIoT 2022 - Observaciones IoT:")
    summary.append(f"    Dataset IoT/IIoT con 15 tipos de ataque")
    summary.append(f"    Incluye ataques específicos IoT: MQTT, Modbus, DNS")
    summary.append(f"    Classs muy minoritarias: MITM ({THETA_LOW}), Fingerprinting (< {THETA_LOW})")

    # Metrics finales
    summary.append("\n\n" + "=" * 120)
    summary.append("7. MÉTRICAS CLAVE")
    summary.append("=" * 120)
    summary.append(f"\n  BINARIA (Normal vs ATTACK):")
    summary.append(f"    TSTR F1-Macro: {mejor_tstr_binary['F1_Macro']:.4f}    TRTR F1-Macro: {mejor_trtr_binary['F1_Macro']:.4f}    Gap: {gap_binary:+.4f}")
    summary.append(f"  MULTICLASE ({len(label_encoder_multi.classes_)} classs):")
    summary.append(f"    TSTR F1-Macro: {mejor_tstr_multi['F1_Macro']:.4f}    TRTR F1-Macro: {mejor_trtr_multi['F1_Macro']:.4f}    Gap: {gap_multi:+.4f}")

    summary_texto = "\n".join(summary)

    with open(os.path.join(output_dir, 'RESUMEN_GLOBAL.txt'), 'w') as f:
        f.write(summary_texto)

    print("\n" + summary_texto)

    # CSV global
    comparative_global = []
    for exp_name, results in [
        ('TSTR_Multi', results_tstr_multi),
        ('TRTR_Multi', results_trtr_multi),
        ('TSTR_Binary', results_tstr_binary),
        ('TRTR_Binary', results_trtr_binary)
    ]:
        for r in results:
            comparative_global.append({
                'Experimento': exp_name,
                'Model': r['Model'],
                'Accuracy': r['Accuracy'],
                'F1_Macro': r['F1_Macro'],
                'F1_Weighted': r['F1_Weighted']
            })

    pd.DataFrame(comparative_global).to_csv(os.path.join(output_dir, 'comparative_global.csv'), index=False)

    return summary_texto


# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser(
        description='TSTR evaluation vs TRTR para Edge-IIoT 2022',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--dataset', '-d', type=str, help='Nombre del dataset synthetic')
    parser.add_argument('--list', '-l', action='store_true', help='Listar datasets disponibles')
    parser.add_argument('--max-train', type=int, default=100000)
    parser.add_argument('--max-test', type=int, default=100000)

    args = parser.parse_args()

    datasets = listar_datasets_disponibles()

    if args.list:
        mostrar_datasets_disponibles(datasets)
        return

    print("=" * 100)
    print("EXPERIMENTO TSTR vs TRTR - EDGE-IIoT 2022")
    print("AMD-GAN Adaptive Training Engine | θ_high={}, θ_low={}".format(THETA_HIGH, THETA_LOW))
    print("=" * 100)

    # Seleccionar dataset
    if args.dataset:
        dataset_info = None
        for ds in datasets:
            if ds['nombre'].lower() == args.dataset.lower():
                dataset_info = ds
                break
        if not dataset_info:
            print(f"\n[ERROR] Dataset '{args.dataset}' no encontrado.")
            mostrar_datasets_disponibles(datasets)
            return
    else:
        dataset_info = seleccionar_dataset_interactivo(datasets)
        if not dataset_info:
            return

    output_dir = os.path.join(OUTPUT_BASE_DIR, dataset_info['nombre'])
    os.makedirs(output_dir, exist_ok=True)

    # Load data synthetics
    df_synth = cargar_dataset_sintetico(dataset_info)

    classs_sinteticas = sorted(df_synth['Label_Class'].unique().tolist())
    print(f"\n  Classs en dataset synthetic: {classs_sinteticas}")

    # Load data reales
    df_real_raw = cargar_data_reales()
    df_real = preparar_features_reales(df_real_raw)
    df_real = filtrar_classs(df_real, classs_sinteticas)

    print(f"\n  Distribution de classs reales (filtradas):")
    print(df_real['Label_Class'].value_counts())

    # Label encoders
    label_encoder_multi = LabelEncoder()
    label_encoder_multi.fit(classs_sinteticas)
    print(f"\n  Classs multiclass: {label_encoder_multi.classes_}")

    label_encoder_binary = LabelEncoder()
    label_encoder_binary.fit(['ATTACK', 'Normal'])
    print(f"  Classs binarias: {label_encoder_binary.classes_}")

    # Preparar data
    X_synth, y_synth_multi, X_real, y_real_multi, feature_cols = preparar_data_para_training(
        df_synth, df_real, label_encoder_multi
    )

    # Binario: Normal = 1, ATTACK = 0
    normal_idx = np.where(label_encoder_multi.classes_ == 'Normal')[0]
    if len(normal_idx) == 0:
        print("\n[WARNING] No se encontró class 'Normal'.")
        normal_idx = 0
    else:
        normal_idx = normal_idx[0]

    # Normal → label 1 (Normal), todo lo demás → label 0 (ATTACK)
    y_synth_binary = np.where(y_synth_multi == normal_idx, 1, 0)
    y_real_binary = np.where(y_real_multi == normal_idx, 1, 0)

    # Split data reales
    np.random.seed(RANDOM_STATE)

    X_real_train, X_real_test, y_real_train_multi, y_real_test_multi = train_test_split(
        X_real, y_real_multi, test_size=0.3, random_state=RANDOM_STATE, stratify=y_real_multi
    )

    y_real_train_binary = np.where(y_real_train_multi == normal_idx, 1, 0)
    y_real_test_binary = np.where(y_real_test_multi == normal_idx, 1, 0)

    # Limitar size
    max_train, max_test = args.max_train, args.max_test

    if len(X_real_train) > max_train:
        idx_train = []
        for class in range(len(label_encoder_multi.classes_)):
            idx_class = np.where(y_real_train_multi == class)[0]
            n_sample = min(len(idx_class), max_train // len(label_encoder_multi.classes_))
            if n_sample > 0:
                idx_train.extend(np.random.choice(idx_class, n_sample, replace=False))
        idx_train = np.array(idx_train)
        np.random.shuffle(idx_train)
        X_real_train = X_real_train[idx_train]
        y_real_train_multi = y_real_train_multi[idx_train]
        y_real_train_binary = y_real_train_binary[idx_train]

    if len(X_real_test) > max_test:
        idx_test = np.random.choice(len(X_real_test), max_test, replace=False)
        X_real_test = X_real_test[idx_test]
        y_real_test_multi = y_real_test_multi[idx_test]
        y_real_test_binary = y_real_test_binary[idx_test]

    # Summary
    print(f"\n{'='*100}")
    print("RESUMEN DE DATOS")
    print('='*100)
    print(f"  Dataset synthetic: {dataset_info['nombre']}")
    print(f"  Data Sintéticos (Train TSTR): {len(X_synth):,}")
    print(f"  Data Reales Train (TRTR): {len(X_real_train):,}")
    print(f"  Data Reales Test: {len(X_real_test):,}")
    print(f"  Features: {len(feature_cols)}")

    print(f"\n  MULTICLASE en Train Sintético:")
    for i, class in enumerate(label_encoder_multi.classes_):
        print(f"    {class}: {np.sum(y_synth_multi == i):,}")

    print(f"\n  BINARIO en Train Sintético:")
    print(f"    Normal: {np.sum(y_synth_binary == 1):,}")
    print(f"    ATTACK: {np.sum(y_synth_binary == 0):,}")

    # Experimentos MULTICLASE
    print("\n\n" + "#"*100)
    print("#" + " "*35 + "CLASIFICACIÓN MULTICLASE" + " "*37 + "#")
    print("#"*100)

    results_tstr_multi = ejecutar_experimento(
        X_synth, y_synth_multi, X_real_test, y_real_test_multi,
        label_encoder_multi, output_dir, "TSTR_MULTI_",
        "TSTR MULTICLASE (Train Synthetic, Test Real)"
    )

    results_trtr_multi = ejecutar_experimento(
        X_real_train, y_real_train_multi, X_real_test, y_real_test_multi,
        label_encoder_multi, output_dir, "TRTR_MULTI_",
        "TRTR MULTICLASE (Train Real, Test Real) - BASELINE"
    )

    # Experimentos BINARIOS
    print("\n\n" + "#"*100)
    print("#" + " "*37 + "CLASIFICACIÓN BINARIA" + " "*38 + "#")
    print("#"*100)

    results_tstr_binary = ejecutar_experimento(
        X_synth, y_synth_binary, X_real_test, y_real_test_binary,
        label_encoder_binary, output_dir, "TSTR_BINARY_",
        "TSTR BINARIO (Train Synthetic, Test Real)"
    )

    results_trtr_binary = ejecutar_experimento(
        X_real_train, y_real_train_binary, X_real_test, y_real_test_binary,
        label_encoder_binary, output_dir, "TRTR_BINARY_",
        "TRTR BINARIO (Train Real, Test Real) - BASELINE"
    )

    # Summary global
    generar_summary_global(
        results_tstr_multi, results_trtr_multi,
        results_tstr_binary, results_trtr_binary,
        label_encoder_multi, label_encoder_binary,
        output_dir, dataset_info['nombre']
    )

    # Save todos los results
    todos_results = []
    for tipo, clasificacion, results in [
        ('TSTR', 'Multiclass', results_tstr_multi),
        ('TRTR', 'Multiclass', results_trtr_multi),
        ('TSTR', 'Binaria', results_tstr_binary),
        ('TRTR', 'Binaria', results_trtr_binary),
    ]:
        for r in results:
            r_copy = {k: v for k, v in r.items() if k not in ['Report', 'Confusion_Matrix']}
            r_copy['Tipo'] = tipo
            r_copy['Clasificacion'] = clasificacion
            todos_results.append(r_copy)

    pd.DataFrame(todos_results).to_csv(os.path.join(output_dir, 'todos_results.csv'), index=False)

    print(f"\n\n{'='*100}")
    print(f"RESULTADOS GUARDADOS EN: {output_dir}")
    print('='*100)


if __name__ == "__main__":
    main()
