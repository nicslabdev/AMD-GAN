"""
Script: Comparación de Técnicas de Oversampling - GAN vs Métodos Tradicionales

Este experimento compara diferentes técnicas de oversampling para balancear
el dataset CIC-IDS2017:
  1. Original (sin oversampling) - baseline
  2. GAN v2 (WGAN-GP entrenada)
  3. SMOTE
  4. ADASYN
  5. BorderlineSMOTE
  6. SMOTE-ENN (oversampling + limpieza)
  7. Random Oversampling

Para cada técnica:
  - Se balancea el dataset de training
  - Se entrena el mismo model (LightGBM)
  - Se evalúa en el mismo conjunto de test (sin modificar)
"""

import os
import sys
import pickle
import argparse
import numpy as np
import pandas as pd
import polars as pl
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

# Reducir logs de TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ----------------------------
# Configuration
# ----------------------------
REAL_DATA_PATH = '<PATH_TO_CICIDS2017_CSV>'
MODELS_DIR_V2 = '<PATH_TO_WGAN_MODELS_DIR>'
MODELS_DIR_V1 = '<PATH_TO_WGAN_MODELS_DIR>'
OUTPUT_DIR = '<PATH_TO_RESULTS_OVERSAMPLING>'
RANDOM_STATE = 42
LATENT_DIM = 100

# Classs a usar en el experimento
CLASES_EXPERIMENTO = ['BENIGN', 'Brute Force', 'DDoS', 'DoS', 'Port Scan']

# Class name to folder mapping de models GAN
CLASS_TO_FOLDER = {
    'BENIGN': 'benign',
    'Bot': 'bot',
    'Brute Force': 'brute_force',
    'DDoS': 'ddos',
    'DoS': 'dos',
    'Port Scan': 'port_scan',
    'Web Attack': 'web_attack',
}

# Features base (mismo orden que en training GAN)
FEATURES_BASE = [
    'Source Port', 'Destination Port', 'Protocol',
    'Total Fwd Packets', 'Total Backward Packets',
    'Total Length of Fwd Packets', 'Total Length of Bwd Packets',
    'Flow Duration', 'Flow IAT Mean', 'Flow IAT Std',
    'Fwd IAT Mean', 'Bwd IAT Mean',
    'Fwd Packet Length Mean', 'Bwd Packet Length Mean',
    'Packet Length Std', 'Max Packet Length',
    'SYN Flag Count', 'ACK Flag Count', 'FIN Flag Count',
    'RST Flag Count', 'PSH Flag Count'
]

# Features con IPs expandidas
FEATURE_NAMES = FEATURES_BASE + [
    'Src_IP_1', 'Src_IP_2', 'Src_IP_3', 'Src_IP_4',
    'Dst_IP_1', 'Dst_IP_2', 'Dst_IP_3', 'Dst_IP_4'
]

# Columns requiring logarithmic transformation
LOG_COLUMNS = [
    'Total Fwd Packets', 'Total Backward Packets', 
    'Total Length of Fwd Packets', 'Total Length of Bwd Packets',
    'Flow Duration', 'Flow IAT Mean', 'Flow IAT Std', 
    'Fwd IAT Mean', 'Bwd IAT Mean',
    'Fwd Packet Length Mean', 'Bwd Packet Length Mean', 
    'Packet Length Std', 'Max Packet Length'
]


# ----------------------------
# Funciones de Carga de Data
# ----------------------------
def print_header():
    print("=" * 100)
    print("  EXPERIMENTO: COMPARACIÓN DE TÉCNICAS DE OVERSAMPLING")
    print("  GAN (WGAN-GP) vs SMOTE vs ADASYN vs BorderlineSMOTE vs SMOTE-ENN")
    print("=" * 100)


def cargar_data_reales():
    """Carga el dataset CIC-IDS2017"""
    print("\n[1] CARGANDO DATOS REALES")
    print("-" * 50)
    
    file_size = os.path.getsize(REAL_DATA_PATH)
    print(f"  Archivo: {REAL_DATA_PATH}")
    print(f"  Size: {file_size / (1024 * 1024):.2f} MB")
    
    start_time = datetime.now()
    df_pl = pl.read_csv(REAL_DATA_PATH, low_memory=False)
    df = df_pl.to_pandas()
    df.columns = df.columns.str.strip()
    print(f"  Cargado en {datetime.now() - start_time}")
    print(f"  Total registros: {len(df):,}")
    
    return df


def preparar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Prepara features para el model"""
    df = df.copy()
    
    # Obtener label
    label_col = 'Attack Type' if 'Attack Type' in df.columns else 'Label'
    df['Label_Class'] = df[label_col]
    
    # Filtrar solo classs del experimento
    df = df[df['Label_Class'].isin(CLASES_EXPERIMENTO)].copy()
    
    # Seleccionar features base
    features_df = df[FEATURES_BASE].copy()
    
    # Expand IPs to octets
    if 'Source IP' in df.columns:
        octetos = df['Source IP'].astype(str).str.split('.', expand=True)
        for i in range(4):
            features_df[f'Src_IP_{i+1}'] = pd.to_numeric(octetos[i], errors='coerce').fillna(0).astype(int)
    else:
        for i in range(4):
            features_df[f'Src_IP_{i+1}'] = 0
    
    if 'Destination IP' in df.columns:
        octetos = df['Destination IP'].astype(str).str.split('.', expand=True)
        for i in range(4):
            features_df[f'Dst_IP_{i+1}'] = pd.to_numeric(octetos[i], errors='coerce').fillna(0).astype(int)
    else:
        for i in range(4):
            features_df[f'Dst_IP_{i+1}'] = 0
    
    # Añadir label
    features_df['Label_Class'] = df['Label_Class'].values
    
    # Limpiar NaN e Inf
    features_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    features_df.fillna(0, inplace=True)
    
    return features_df


def mostrar_distribucion(y, label_encoder, titulo="Distribution"):
    """Muestra la distribution de classs"""
    print(f"\n  {titulo}:")
    counter = Counter(y)
    total = len(y)
    for label_idx in sorted(counter.keys()):
        class = label_encoder.inverse_transform([label_idx])[0]
        count = counter[label_idx]
        pct = (count / total) * 100
        print(f"    {class:<15}: {count:>10,} ({pct:>5.1f}%)")
    print(f"    {'TOTAL':<15}: {total:>10,}")


# ----------------------------
# Funciones de Oversampling
# ----------------------------
def oversample_gan(X_train, y_train, label_encoder, target_count):
    """
    Usa los generatores GAN para hacer oversampling de classs minoritarias.
    
    Args:
        X_train: features de training
        y_train: labels de training
        label_encoder: encoder de labels
        target_count: número objetivo de samples por class
    
    Returns:
        X_resampled, y_resampled
    """
    print("\n  [GAN] Aplicando oversampling con WGAN-GP v2...")
    
    X_resampled = X_train.copy()
    y_resampled = y_train.copy()
    
    for class_idx, class in enumerate(label_encoder.classes_):
        current_count = np.sum(y_train == class_idx)
        
        if current_count >= target_count:
            print(f"    {class}: {current_count:,} >= {target_count:,} (sin cambios)")
            continue
        
        n_generate = target_count - current_count
        print(f"    {class}: {current_count:,} -> {target_count:,} (generating {n_generate:,})")
        
        # Cargar generator y scaler
        folder = CLASS_TO_FOLDER.get(class)
        if not folder:
            print(f"      [WARN] No hay model GAN para {class}")
            continue
        
        # Buscar primero en v2, luego en v1
        generator_path = os.path.join(MODELS_DIR_V2, folder, f'generator_{folder}.h5')
        scaler_path = os.path.join(MODELS_DIR_V2, folder, 'scaler.pkl')
        
        if not os.path.exists(generator_path):
            generator_path = os.path.join(MODELS_DIR_V1, folder, f'generator_{folder}.h5')
            scaler_path = os.path.join(MODELS_DIR_V1, folder, 'scaler.pkl')
        
        if not os.path.exists(generator_path):
            print(f"      [WARN] No se encontró generator para {class}")
            continue
        
        # Cargar generator
        generator = load_model(generator_path, compile=False)
        
        # Cargar scaler
        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
        else:
            print(f"      [WARN] No se encontró scaler para {class}, usando data originales")
            continue
        
        # Generar samples
        noise = np.random.normal(0, 1, (n_generate, LATENT_DIM))
        X_synthetic_scaled = generator.predict(noise, verbose=0)
        
        # Invertir escalado
        X_synthetic = scaler.inverse_transform(X_synthetic_scaled)
        
        # Invertir transformación logarítmica para columnas específicas
        for i, col in enumerate(FEATURE_NAMES):
            if col in LOG_COLUMNS:
                X_synthetic[:, i] = np.expm1(X_synthetic[:, i])
        
        # Clip valores
        X_synthetic = np.clip(X_synthetic, 0, None)
        
        # Añadir al dataset
        X_resampled = np.vstack([X_resampled, X_synthetic])
        y_resampled = np.concatenate([y_resampled, np.full(n_generate, class_idx)])
        
        # Limpiar memoria
        del generator
        tf.keras.backend.clear_session()
    
    return X_resampled, y_resampled


def oversample_smote(X_train, y_train, label_encoder, target_count):
    """Aplica SMOTE para oversampling"""
    print("\n  [SMOTE] Aplicando SMOTE...")
    
    # Calcular sampling_strategy
    counter = Counter(y_train)
    sampling_strategy = {}
    for class_idx in counter.keys():
        if counter[class_idx] < target_count:
            sampling_strategy[class_idx] = target_count
    
    if not sampling_strategy:
        print("    No se requiere oversampling")
        return X_train, y_train
    
    smote = SMOTE(
        sampling_strategy=sampling_strategy,
        random_state=RANDOM_STATE,
        k_neighbors=5
    )
    
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    
    for class_idx, class in enumerate(label_encoder.classes_):
        old_count = counter[class_idx]
        new_count = np.sum(y_resampled == class_idx)
        if new_count > old_count:
            print(f"    {class}: {old_count:,} -> {new_count:,}")
    
    return X_resampled, y_resampled


def oversample_adasyn(X_train, y_train, label_encoder, target_count):
    """Aplica ADASYN para oversampling"""
    print("\n  [ADASYN] Aplicando ADASYN...")
    
    counter = Counter(y_train)
    sampling_strategy = {}
    for class_idx in counter.keys():
        if counter[class_idx] < target_count:
            sampling_strategy[class_idx] = target_count
    
    if not sampling_strategy:
        print("    No se requiere oversampling")
        return X_train, y_train
    
    try:
        adasyn = ADASYN(
            sampling_strategy=sampling_strategy,
            random_state=RANDOM_STATE,
            n_neighbors=5
        )
        X_resampled, y_resampled = adasyn.fit_resample(X_train, y_train)
        
        for class_idx, class in enumerate(label_encoder.classes_):
            old_count = counter[class_idx]
            new_count = np.sum(y_resampled == class_idx)
            if new_count > old_count:
                print(f"    {class}: {old_count:,} -> {new_count:,}")
        
        return X_resampled, y_resampled
    
    except Exception as e:
        print(f"    [ERROR] ADASYN falló: {e}")
        print(f"    Usando SMOTE como fallback...")
        return oversample_smote(X_train, y_train, label_encoder, target_count)


def oversample_borderline(X_train, y_train, label_encoder, target_count):
    """Aplica BorderlineSMOTE para oversampling"""
    print("\n  [BorderlineSMOTE] Aplicando BorderlineSMOTE...")
    
    counter = Counter(y_train)
    sampling_strategy = {}
    for class_idx in counter.keys():
        if counter[class_idx] < target_count:
            sampling_strategy[class_idx] = target_count
    
    if not sampling_strategy:
        print("    No se requiere oversampling")
        return X_train, y_train
    
    try:
        borderline = BorderlineSMOTE(
            sampling_strategy=sampling_strategy,
            random_state=RANDOM_STATE,
            k_neighbors=5
        )
        X_resampled, y_resampled = borderline.fit_resample(X_train, y_train)
        
        for class_idx, class in enumerate(label_encoder.classes_):
            old_count = counter[class_idx]
            new_count = np.sum(y_resampled == class_idx)
            if new_count > old_count:
                print(f"    {class}: {old_count:,} -> {new_count:,}")
        
        return X_resampled, y_resampled
    
    except Exception as e:
        print(f"    [ERROR] BorderlineSMOTE falló: {e}")
        print(f"    Usando SMOTE como fallback...")
        return oversample_smote(X_train, y_train, label_encoder, target_count)


def oversample_smoteenn(X_train, y_train, label_encoder, target_count):
    """Aplica SMOTE-ENN (oversampling + undersampling de ruido)"""
    print("\n  [SMOTE-ENN] Aplicando SMOTE + ENN cleaning...")
    
    counter = Counter(y_train)
    sampling_strategy = {}
    for class_idx in counter.keys():
        if counter[class_idx] < target_count:
            sampling_strategy[class_idx] = target_count
    
    if not sampling_strategy:
        print("    No se requiere oversampling")
        return X_train, y_train
    
    try:
        smoteenn = SMOTEENN(
            sampling_strategy=sampling_strategy,
            random_state=RANDOM_STATE
        )
        X_resampled, y_resampled = smoteenn.fit_resample(X_train, y_train)
        
        print(f"    Muestras antes: {len(y_train):,}")
        print(f"    Muestras después: {len(y_resampled):,}")
        
        for class_idx, class in enumerate(label_encoder.classes_):
            old_count = counter[class_idx]
            new_count = np.sum(y_resampled == class_idx)
            print(f"    {class}: {old_count:,} -> {new_count:,}")
        
        return X_resampled, y_resampled
    
    except Exception as e:
        print(f"    [ERROR] SMOTE-ENN falló: {e}")
        print(f"    Usando SMOTE como fallback...")
        return oversample_smote(X_train, y_train, label_encoder, target_count)


def oversample_random(X_train, y_train, label_encoder, target_count):
    """Aplica Random Oversampling (duplicación aleatoria)"""
    print("\n  [RandomOverSampler] Aplicando oversampling aleatorio...")
    
    counter = Counter(y_train)
    sampling_strategy = {}
    for class_idx in counter.keys():
        if counter[class_idx] < target_count:
            sampling_strategy[class_idx] = target_count
    
    if not sampling_strategy:
        print("    No se requiere oversampling")
        return X_train, y_train
    
    ros = RandomOverSampler(
        sampling_strategy=sampling_strategy,
        random_state=RANDOM_STATE
    )
    X_resampled, y_resampled = ros.fit_resample(X_train, y_train)
    
    for class_idx, class in enumerate(label_encoder.classes_):
        old_count = counter[class_idx]
        new_count = np.sum(y_resampled == class_idx)
        if new_count > old_count:
            print(f"    {class}: {old_count:,} -> {new_count:,}")
    
    return X_resampled, y_resampled


# ----------------------------
# Training y Evaluation
# ----------------------------
def crear_model(model_tipo='logistic'):
    """
    Crea el model a usar para todos los experimentos.
    
    Args:
        model_tipo: 'logistic', 'tree', o 'lgbm'
    """
    if model_tipo == 'logistic':
        # Model lineal simple - más sensible a la quality de los data
        return LogisticRegression(
            max_iter=500,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            class_weight='balanced',
            solver='lbfgs'
        )
    elif model_tipo == 'tree':
        # Árbol de decisión simple - muestra diferencias en data synthetics
        return DecisionTreeClassifier(
            max_depth=10,
            random_state=RANDOM_STATE,
            class_weight='balanced'
        )
    else:
        # LightGBM potente (results casi perfectos)
        return LGBMClassifier(
            n_estimators=200,
            max_depth=15,
            learning_rate=0.1,
            num_leaves=63,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1,
            class_weight='balanced'
        )

# Model a usar en el experimento (cambiar aquí)
MODELO_TIPO = 'logistic'  # Opciones: 'logistic', 'tree', 'lgbm'


def entrenar_y_evaluar(X_train, y_train, X_test, y_test, label_encoder, 
                       nombre_tecnica, output_dir):
    """Entrena y evalúa un model"""
    print(f"\n  Entrenando model con {nombre_tecnica}...")
    print(f"    Train: {len(X_train):,} samples")
    
    # Scale data
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Limpiar NaN/Inf
    X_train_scaled = np.nan_to_num(X_train_scaled, nan=0, posinf=0, neginf=0)
    X_test_scaled = np.nan_to_num(X_test_scaled, nan=0, posinf=0, neginf=0)
    
    # Train
    model = crear_model(MODELO_TIPO)
    start_time = datetime.now()
    model.fit(X_train_scaled, y_train)
    train_time = datetime.now() - start_time
    
    # Predecir
    y_pred = model.predict(X_test_scaled)
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
    f1_weighted = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    precision_macro = precision_score(y_test, y_pred, average='macro', zero_division=0)
    recall_macro = recall_score(y_test, y_pred, average='macro', zero_division=0)
    
    print(f"    Accuracy: {accuracy:.4f}")
    print(f"    F1-Macro: {f1_macro:.4f}")
    print(f"    F1-Weighted: {f1_weighted:.4f}")
    print(f"    Tiempo training: {train_time}")
    
    # Save classification report
    report = classification_report(
        y_test, y_pred,
        target_names=label_encoder.classes_,
        output_dict=True,
        zero_division=0
    )
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(os.path.join(output_dir, f'{nombre_tecnica}_classification_report.csv'))
    
    # Save confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm, 
                         index=label_encoder.classes_,
                         columns=label_encoder.classes_)
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
    """Genera summary comparativo de todas las técnicas"""
    
    summary = []
    summary.append("=" * 120)
    summary.append(" " * 25 + "COMPARACIÓN DE TÉCNICAS DE OVERSAMPLING")
    summary.append(" " * 30 + "GAN vs Métodos Tradicionales")
    summary.append(" " * 35 + f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    summary.append("=" * 120)
    
    # Ordenar por F1-Macro
    results_sorted = sorted(results, key=lambda x: x['F1_Macro'], reverse=True)
    
    # Table comparative
    summary.append("\n" + "=" * 120)
    summary.append("1. RANKING DE TÉCNICAS (ordenado por F1-Macro)")
    summary.append("=" * 120)
    
    summary.append("\n" + "-" * 120)
    header = f"{'Pos':<5} {'Técnica':<20} │ {'Train':>12} │ {'Accuracy':>10} {'F1-Macro':>10} {'F1-Weight':>10} │ {'Prec':>8} {'Recall':>8}"
    summary.append(header)
    summary.append("-" * 120)
    
    mejor = results_sorted[0]
    for i, r in enumerate(results_sorted, 1):
        delta = r['F1_Macro'] - mejor['F1_Macro']
        delta_str = f"({delta:+.4f})" if i > 1 else "(best)"
        linea = f"{i:<5} {r['Tecnica']:<20} │ {r['Train_Samples']:>12,} │ {r['Accuracy']:>10.4f} {r['F1_Macro']:>10.4f} {r['F1_Weighted']:>10.4f} │ {r['Precision_Macro']:>8.4f} {r['Recall_Macro']:>8.4f} {delta_str}"
        summary.append(linea)
    
    # Mejora de GAN sobre baseline
    summary.append("\n\n" + "=" * 120)
    summary.append("2. COMPARACIÓN GAN vs OTROS MÉTODOS")
    summary.append("=" * 120)
    
    gan_result = next((r for r in results if r['Tecnica'] == 'GAN_v2'), None)
    baseline_result = next((r for r in results if r['Tecnica'] == 'Original'), None)
    smote_result = next((r for r in results if r['Tecnica'] == 'SMOTE'), None)
    
    if gan_result and baseline_result:
        mejora_vs_baseline = gan_result['F1_Macro'] - baseline_result['F1_Macro']
        summary.append(f"\n  GAN vs Original (sin oversampling):")
        summary.append(f"    • Δ F1-Macro: {mejora_vs_baseline:+.4f}")
        summary.append(f"    • Δ Accuracy: {gan_result['Accuracy'] - baseline_result['Accuracy']:+.4f}")
    
    if gan_result and smote_result:
        mejora_vs_smote = gan_result['F1_Macro'] - smote_result['F1_Macro']
        summary.append(f"\n  GAN vs SMOTE:")
        summary.append(f"    • Δ F1-Macro: {mejora_vs_smote:+.4f}")
        summary.append(f"    • Δ Accuracy: {gan_result['Accuracy'] - smote_result['Accuracy']:+.4f}")
    
    # Análisis por class para la mejor técnica
    summary.append("\n\n" + "=" * 120)
    summary.append(f"3. ANÁLISIS POR CLASE - MEJOR TÉCNICA: {mejor['Tecnica']}")
    summary.append("=" * 120)
    
    summary.append("\n" + "-" * 90)
    summary.append(f"{'Class':<15} │ {'Precision':>10} {'Recall':>10} {'F1-Score':>10} │ {'Support':>10}")
    summary.append("-" * 90)
    
    report = mejor['Report']
    for class in label_encoder.classes_:
        if class in report:
            r = report[class]
            summary.append(f"{class:<15} │ {r['precision']:>10.4f} {r['recall']:>10.4f} {r['f1-score']:>10.4f} │ {r['support']:>10.0f}")
    
    # F1-Score por class para cada técnica
    summary.append("\n\n" + "=" * 120)
    summary.append("4. F1-SCORE POR CLASE PARA CADA TÉCNICA")
    summary.append("=" * 120)
    
    # Header
    tecnicas_nombres = [r['Tecnica'][:12] for r in results]
    header_line = f"{'Class':<15} │ " + " │ ".join([f"{t:>12}" for t in tecnicas_nombres])
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
        linea = f"{class:<15} │ " + " │ ".join(valores)
        summary.append(linea)
    
    # Conclusiones
    summary.append("\n\n" + "=" * 120)
    summary.append("5. CONCLUSIONES")
    summary.append("=" * 120)
    
    summary.append("\n┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐")
    summary.append(f"│  MEJOR TÉCNICA: {mejor['Tecnica']:<20} F1-Macro = {mejor['F1_Macro']:.4f}                                                  │")
    summary.append("├─────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤")
    
    if gan_result:
        gan_pos = next(i for i, r in enumerate(results_sorted, 1) if r['Tecnica'] == 'GAN_v2')
        if gan_pos == 1:
            summary.append(f"│  ✓ GAN supera a todas las técnicas tradicionales de oversampling                                              │")
        elif gan_pos <= 3:
            summary.append(f"│  ○ GAN en posición {gan_pos} - Competitivo con métodos tradicionales                                              │")
        else:
            summary.append(f"│  ✗ GAN en posición {gan_pos} - Los métodos tradicionales funcionan mejor en este caso                             │")
    
    summary.append("└─────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘")
    
    # Save
    summary_texto = "\n".join(summary)
    
    with open(os.path.join(output_dir, 'RESUMEN_COMPARATIVO.txt'), 'w') as f:
        f.write(summary_texto)
    
    print("\n" + summary_texto)
    
    # Save CSV comparativo
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
    
    pd.DataFrame(comparative).to_csv(
        os.path.join(output_dir, 'comparative_tecnicas.csv'), 
        index=False
    )
    
    return summary_texto


# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser(
        description='Oversampling technique comparison: GAN vs métodos tradicionales'
    )
    parser.add_argument('--max-samples', type=int, default=100000,
                       help='Máximo samples por class para balanceo (default: 100000)')
    parser.add_argument('--test-size', type=float, default=0.3,
                       help='Proporción para test set (default: 0.3)')
    parser.add_argument('--skip-gan', action='store_true',
                       help='Omitir oversampling con GAN')
    parser.add_argument('--output', type=str, default=None,
                       help='Directorio de salida personalizado')
    
    args = parser.parse_args()
    
    print_header()
    
    # Create output directory
    if args.output:
        output_dir = args.output
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(OUTPUT_DIR, f'comparison_{timestamp}')
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n  Results se guardarán en: {output_dir}")
    
    # ==========================================
    # CARGA Y PREPARACIÓN DE DATOS
    # ==========================================
    
    df = cargar_data_reales()
    df_features = preparar_features(df)
    
    print(f"\n[2] DISTRIBUCIÓN DE CLASES EN DATASET FILTRADO")
    print("-" * 50)
    print(f"  Classes: {CLASES_EXPERIMENTO}")
    print(f"  Total samples: {len(df_features):,}")
    print("\n  Distribution:")
    for class in CLASES_EXPERIMENTO:
        count = len(df_features[df_features['Label_Class'] == class])
        pct = (count / len(df_features)) * 100
        print(f"    {class:<15}: {count:>10,} ({pct:>5.1f}%)")
    
    # Separar features y labels
    feature_cols = [c for c in df_features.columns if c != 'Label_Class']
    X = df_features[feature_cols].values
    
    # Crear label encoder
    label_encoder = LabelEncoder()
    label_encoder.fit(CLASES_EXPERIMENTO)
    y = label_encoder.transform(df_features['Label_Class'])
    
    print(f"\n  Features: {len(feature_cols)}")
    print(f"  Classs codificadas: {list(label_encoder.classes_)}")
    
    # ==========================================
    # SPLIT TRAIN/TEST
    # ==========================================
    
    print(f"\n[3] DIVIDIENDO TRAIN/TEST")
    print("-" * 50)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=args.test_size, 
        random_state=RANDOM_STATE,
        stratify=y
    )
    
    print(f"  Train: {len(X_train):,} samples")
    print(f"  Test: {len(X_test):,} samples")
    
    mostrar_distribucion(y_train, label_encoder, "Distribution Train (original)")
    mostrar_distribucion(y_test, label_encoder, "Distribution Test")
    
    # Determinar target_count para balanceo
    counter = Counter(y_train)
    max_class_count = max(counter.values())
    target_count = min(max_class_count, args.max_samples)
    
    print(f"\n  Target para balanceo: {target_count:,} samples por class")
    
    # ==========================================
    # APLICAR TÉCNICAS DE OVERSAMPLING
    # ==========================================
    
    print(f"\n[4] APLICANDO TÉCNICAS DE OVERSAMPLING")
    print("=" * 70)
    
    tecnicas = {}
    
    # 1. Original (sin oversampling) - BASELINE
    print("\n" + "-" * 70)
    print("  [Original] No oversampling (baseline)")
    tecnicas['Original'] = (X_train.copy(), y_train.copy())
    mostrar_distribucion(y_train, label_encoder, "Distribution")
    
    # 2. GAN v2
    if not args.skip_gan:
        print("\n" + "-" * 70)
        X_gan, y_gan = oversample_gan(X_train, y_train, label_encoder, target_count)
        tecnicas['GAN_v2'] = (X_gan, y_gan)
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
    X_borderline, y_borderline = oversample_borderline(X_train, y_train, label_encoder, target_count)
    tecnicas['BorderlineSMOTE'] = (X_borderline, y_borderline)
    mostrar_distribucion(y_borderline, label_encoder, "Distribution BorderlineSMOTE")
    
    # 6. SMOTE-ENN
    print("\n" + "-" * 70)
    X_smoteenn, y_smoteenn = oversample_smoteenn(X_train, y_train, label_encoder, target_count)
    tecnicas['SMOTE_ENN'] = (X_smoteenn, y_smoteenn)
    mostrar_distribucion(y_smoteenn, label_encoder, "Distribution SMOTE-ENN")
    
    # 7. Random Oversampling
    print("\n" + "-" * 70)
    X_random, y_random = oversample_random(X_train, y_train, label_encoder, target_count)
    tecnicas['RandomOverSampler'] = (X_random, y_random)
    mostrar_distribucion(y_random, label_encoder, "Distribution Random")
    
    # ==========================================
    # ENTRENAR Y EVALUAR
    # ==========================================
    
    print(f"\n\n[5] ENTRENAMIENTO Y EVALUACIÓN")
    print("=" * 70)
    model_desc = {
        'logistic': 'Logistic Regression (multinomial, balanced)',
        'tree': 'Decision Tree (max_depth=10, balanced)',
        'lgbm': 'LightGBM (200 estimators, max_depth=15)'
    }
    print(f"  Model: {model_desc.get(MODELO_TIPO, MODELO_TIPO)}")
    print(f"  Test set: {len(X_test):,} samples (mismo para todas las técnicas)")
    
    results = []
    
    for nombre, (X_t, y_t) in tecnicas.items():
        print(f"\n" + "-" * 70)
        result = entrenar_y_evaluar(
            X_t, y_t, X_test, y_test,
            label_encoder, nombre, output_dir
        )
        results.append(result)
    
    # ==========================================
    # GENERAR RESUMEN
    # ==========================================
    
    print(f"\n\n[6] GENERANDO RESUMEN COMPARATIVO")
    print("=" * 70)
    
    generar_summary(results, label_encoder, output_dir)
    
    print(f"\n\n{'='*100}")
    print(f"  EXPERIMENTO COMPLETADO")
    print(f"  Results saved to: {output_dir}")
    print(f"{'='*100}")


if __name__ == "__main__":
    main()
