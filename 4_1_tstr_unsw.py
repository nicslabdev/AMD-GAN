"""
Script: Train on Synthetic, Test on Real - UNSW-NB15 (TSTR)

Compara CUATRO escenarios:
  1. TSTR Multiclass: Entrena en synthetics, testea en reales (7 classs)
  2. TRTR Multiclass: Entrena en reales, testea en reales (7 classs) - baseline
  3. TSTR Binario: Entrena en synthetics, testea en reales (Benign vs ATTACK)
  4. TRTR Binario: Entrena en reales, testea en reales (Benign vs ATTACK) - baseline

Usa los datasets synthetics generados por generate_synthetic_dataset_unsw.py

Usage:
    python train_synthetic_test_real_unsw.py                    # Interactivo
    python train_synthetic_test_real_unsw.py --dataset nombre   # Dataset específico
    python train_synthetic_test_real_unsw.py --list             # Listar datasets
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import polars as pl
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

# ----------------------------
# Configuration
# ----------------------------
REAL_DATA_PATH = '<PATH_TO_UNSWNB15_CSV>'
SYNTHETIC_DATASETS_DIR = '<PATH_TO_GENERATED_DATASETS_UNSW>'
OUTPUT_BASE_DIR = '<PATH_TO_RESULTS_TSTR_UNSW>'
RANDOM_STATE = 42
LABEL_COLUMN = 'Label'

# Features base (mismo orden que en GAN training)
FEATURES_BASE_RAW = [
    'Src IP', 'Dst IP',
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


# ----------------------------
# Utilities para manejo de datasets
# ----------------------------
def listar_datasets_disponibles():
    """Lista todos los datasets synthetics UNSW-NB15 disponibles."""
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
    print("DATASETS SINTÉTICOS UNSW-NB15 DISPONIBLES")
    print("=" * 80)

    if not datasets:
        print("  No hay datasets disponibles en:", SYNTHETIC_DATASETS_DIR)
        print("  Genera uno con: python generate_synthetic_dataset_unsw.py --interactive")
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
    print("CARGANDO DATASET SINTÉTICO UNSW-NB15")
    print("=" * 70)

    print(f"  Archivo: {dataset_info['archivo']}")
    df = pd.read_csv(dataset_info['path'])
    print(f"  Total samples: {len(df):,}")

    # Detectar columna de class
    if 'Label' in df.columns:
        label_col = 'Label'
    elif 'Label_Class' in df.columns:
        label_col = 'Label_Class'
    else:
        raise ValueError("No se encontró columna de class (Label o Label_Class)")

    df['Label_Class'] = df[label_col]

    print(f"\n  Class distribution:")
    for class, count in df['Label_Class'].value_counts().items():
        pct = (count / len(df)) * 100
        print(f"    {class}: {count:,} ({pct:.1f}%)")

    return df


def cargar_data_reales() -> pd.DataFrame:
    print("\n" + "=" * 70)
    print("CARGANDO DATOS REALES (UNSW-NB15)")
    print("=" * 70)

    file_size = os.path.getsize(REAL_DATA_PATH)
    print(f"  Size del archivo: {file_size / (1024 * 1024):.2f} MB")

    start_time = datetime.now()
    df_pl = pl.read_csv(REAL_DATA_PATH, low_memory=False)
    df = df_pl.to_pandas()
    df.columns = df.columns.str.strip()
    print(f"  Data cargados en {datetime.now() - start_time}")

    return df


def preparar_features_reales(df: pd.DataFrame) -> pd.DataFrame:
    """Prepara el dataset real con la misma estructura de features que el synthetic."""
    df = df.copy()
    df.columns = df.columns.str.strip()

    df['Label_Class'] = df[LABEL_COLUMN]

    features_disponibles = [f for f in FEATURES_BASE_RAW if f in df.columns]
    df = df[features_disponibles + ['Label_Class']].copy()

    # Expand IPs
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


def filtrar_classs(df: pd.DataFrame, classs_objetivo: list) -> pd.DataFrame:
    df_filtrado = df[df['Label_Class'].isin(classs_objetivo)].copy()
    print(f"\n  Filas después de filtrar classs objetivo: {len(df_filtrado):,} de {len(df):,}")
    return df_filtrado


def preparar_data_para_training(df_synth, df_real, label_encoder):
    cols_synth = set(df_synth.columns) - {'Label', 'Label_Class'}
    cols_real = set(df_real.columns) - {'Label', 'Label_Class'}
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
    summary.append(" " * 25 + "RESUMEN GLOBAL: TSTR vs TRTR - UNSW-NB15")
    summary.append(" " * 15 + "Clasificación Multiclass y Binaria")
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
    summary.append(f"\n  BINARIA:")
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
    summary.append("3. COMPARATIVA BINARIA (Benign vs ATTACK)")
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

    summary.append("\n" + "-" * 90)
    summary.append(f"{'Class':<20} | {'TSTR Prec':>10} {'TSTR Rec':>10} {'TSTR F1':>10} | {'TRTR F1':>10} | {'dF1':>8}")
    summary.append("-" * 90)

    for class in label_encoder_multi.classes_:
        if class in report_tstr and class in report_trtr:
            tstr_r = report_tstr[class]
            trtr_r = report_trtr[class]
            delta = tstr_r['f1-score'] - trtr_r['f1-score']
            summary.append(f"{class:<20} | {tstr_r['precision']:>10.4f} {tstr_r['recall']:>10.4f} "
                           f"{tstr_r['f1-score']:>10.4f} | {trtr_r['f1-score']:>10.4f} | {delta:>+8.4f}")

    # Insights
    summary.append("\n\n" + "=" * 120)
    summary.append("5. INSIGHTS")
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

    # Metrics finales
    summary.append("\n\n" + "=" * 120)
    summary.append("6. MÉTRICAS CLAVE")
    summary.append("=" * 120)
    summary.append(f"\n  BINARIA:")
    summary.append(f"    TSTR F1-Macro: {mejor_tstr_binary['F1_Macro']:.4f}    TRTR F1-Macro: {mejor_trtr_binary['F1_Macro']:.4f}    Gap: {gap_binary:+.4f}")
    summary.append(f"  MULTICLASE:")
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
        description='TSTR evaluation vs TRTR para UNSW-NB15',
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
    print("EXPERIMENTO TSTR vs TRTR - UNSW-NB15")
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
    label_encoder_binary.fit(['ATTACK', 'Benign'])
    print(f"  Classs binarias: {label_encoder_binary.classes_}")

    # Preparar data
    X_synth, y_synth_multi, X_real, y_real_multi, feature_cols = preparar_data_para_training(
        df_synth, df_real, label_encoder_multi
    )

    # Binario
    benign_idx = np.where(label_encoder_multi.classes_ == 'Benign')[0]
    if len(benign_idx) == 0:
        print("\n[WARNING] No se encontró class 'Benign'.")
        benign_idx = 0
    else:
        benign_idx = benign_idx[0]

    y_synth_binary = np.where(y_synth_multi == benign_idx, 1, 0)
    y_real_binary = np.where(y_real_multi == benign_idx, 1, 0)

    # Split data reales
    np.random.seed(RANDOM_STATE)

    X_real_train, X_real_test, y_real_train_multi, y_real_test_multi = train_test_split(
        X_real, y_real_multi, test_size=0.3, random_state=RANDOM_STATE, stratify=y_real_multi
    )

    y_real_train_binary = np.where(y_real_train_multi == benign_idx, 1, 0)
    y_real_test_binary = np.where(y_real_test_multi == benign_idx, 1, 0)

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
    print(f"    Benign: {np.sum(y_synth_binary == 1):,}")
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
