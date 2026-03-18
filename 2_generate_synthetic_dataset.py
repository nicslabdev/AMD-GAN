"""
Script: Generator de Dataset Sintético Personalizado

Genera un dataset synthetic especificando el número de samples por class,
utilizando los models WGAN-GP entrenados previamente.

Usage:
    python generate_synthetic_dataset.py --config config.json
    python generate_synthetic_dataset.py --benign 50000 --ddos 20000 --dos 15000
    python generate_synthetic_dataset.py --balanced 10000
    python generate_synthetic_dataset.py --interactive
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

# Configuration
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TensorFlow warnings

# ----------------------------
# Constantes y Configuration
# ----------------------------
# Directorio de models v1 (original)
MODELS_DIR_V1 = '<PATH_TO_WGAN_MODELS_DIR>'
# Directorio de models v2 (mejorado para classs minoritarias)
MODELS_DIR_V2 = '<PATH_TO_WGAN_MODELS_DIR>'

OUTPUT_DIR = '<PATH_TO_GENERATED_DATASETS>'
DATASET_PATH = '<PATH_TO_CICIDS2017_CSV>'

LATENT_DIM = 100

# Por defecto usar v1, pero se puede cambiar con --use-v2
MODELS_DIR = MODELS_DIR_V1

# Class name to folder mapping de models
CLASS_TO_FOLDER = {
    'BENIGN': 'benign',
    'Bot': 'bot',
    'Brute Force': 'brute_force',
    'DDoS': 'ddos',
    'DoS': 'dos',
    'Port Scan': 'port_scan',
    'Web Attack': 'web_attack',
}

# Mapeo inverso
FOLDER_TO_CLASS = {v: k for k, v in CLASS_TO_FOLDER.items()}

# Labels numéricos (consistentes con el training)
CLASS_TO_LABEL = {
    'BENIGN': 0,
    'Bot': 1,
    'Brute Force': 2,
    'DDoS': 3,
    'DoS': 4,
    'Port Scan': 5,
    'Web Attack': 6,
}

# Columns requiring logarithmic transformation inversa
LOG_COLUMNS = [
    'Total Fwd Packets', 'Total Backward Packets', 
    'Total Length of Fwd Packets', 'Total Length of Bwd Packets',
    'Flow Duration', 'Flow IAT Mean', 'Flow IAT Std', 
    'Fwd IAT Mean', 'Bwd IAT Mean',
    'Fwd Packet Length Mean', 'Bwd Packet Length Mean', 
    'Packet Length Std', 'Max Packet Length'
]

# Feature names (orden esperado por los models)
FEATURE_NAMES = [
    'Source Port', 'Destination Port', 'Protocol',
    'Total Fwd Packets', 'Total Backward Packets',
    'Total Length of Fwd Packets', 'Total Length of Bwd Packets',
    'Flow Duration', 'Flow IAT Mean', 'Flow IAT Std',
    'Fwd IAT Mean', 'Bwd IAT Mean',
    'Fwd Packet Length Mean', 'Bwd Packet Length Mean',
    'Packet Length Std', 'Max Packet Length',
    'SYN Flag Count', 'ACK Flag Count', 'FIN Flag Count',
    'RST Flag Count', 'PSH Flag Count',
    'Src_IP_1', 'Src_IP_2', 'Src_IP_3', 'Src_IP_4',
    'Dst_IP_1', 'Dst_IP_2', 'Dst_IP_3', 'Dst_IP_4'
]


# ----------------------------
# Funciones de Utilidad
# ----------------------------
def print_header():
    """Imprime cabecera del programa"""
    print("=" * 70)
    print("  GENERADOR DE DATASET SINTÉTICO - WGAN-GP")
    print("  AMD-GAN - Intrusion Detection System")
    print("=" * 70)


def get_available_classes(models_dir=None, prefer_v2=False):
    """
    Obtiene las classs disponibles verificando models existentes.
    
    Args:
        models_dir: directorio de models a usar (si None, usa MODELS_DIR)
        prefer_v2: si True, prefiere models v2 para classs que existan en ambos
    """
    if models_dir is None:
        models_dir = MODELS_DIR
    
    available = {}
    for class_name, folder in CLASS_TO_FOLDER.items():
        model_path = None
        source_dir = None
        
        # Si prefer_v2, buscar primero en v2
        if prefer_v2:
            v2_path = os.path.join(MODELS_DIR_V2, folder, f'generator_{folder}.h5')
            if os.path.exists(v2_path):
                model_path = v2_path
                source_dir = MODELS_DIR_V2
        
        # Si no encontrado en v2 o no prefer_v2, buscar en directorio principal
        if model_path is None:
            v1_path = os.path.join(models_dir, folder, f'generator_{folder}.h5')
            if os.path.exists(v1_path):
                model_path = v1_path
                source_dir = models_dir
        
        if model_path:
            available[class_name] = {
                'folder': folder,
                'model_path': model_path,
                'source_dir': source_dir,
                'version': 'v2' if source_dir == MODELS_DIR_V2 else 'v1'
            }
    
    return available


def print_available_classes(available_classes):
    """Muestra las classs disponibles"""
    print("\nClasss disponibles:")
    print("-" * 60)
    for i, (class_name, info) in enumerate(available_classes.items(), 1):
        version = info.get('version', 'v1')
        version_tag = f"[{version}]" if version == 'v2' else ""
        print(f"  {i}. {class_name:<15} -> {info['folder']:<15} {version_tag}")
    print("-" * 60)
    if any(info.get('version') == 'v2' for info in available_classes.values()):
        print("  [v2] = Model mejorado para classs minoritarias")


def load_generator(class_name, available_classes):
    """Carga el generator para una class específica"""
    if class_name not in available_classes:
        raise ValueError(f"Class '{class_name}' no disponible")
    
    model_path = available_classes[class_name]['model_path']
    print(f"  Loading model: {os.path.basename(model_path)}")
    generator = load_model(model_path, compile=False)
    return generator


def load_scaler_from_data(class_name, source_dir=None):
    """
    Carga/recrea el scaler para una class específica.
    Primero busca scaler saved, si no existe lo recrea.
    """
    import polars as pl
    
    folder = CLASS_TO_FOLDER[class_name]
    
    # Buscar scaler en el directorio fuente especificado o ambos
    search_dirs = [source_dir] if source_dir else [MODELS_DIR_V2, MODELS_DIR_V1]
    
    for search_dir in search_dirs:
        if search_dir is None:
            continue
        scaler_path = os.path.join(search_dir, folder, 'scaler.pkl')
        if os.path.exists(scaler_path):
            print(f"  Loading scaler desde: {os.path.dirname(scaler_path)}")
            with open(scaler_path, 'rb') as f:
                return pickle.load(f)
    
    # Si no existe, recrear desde data
    print(f"  Recreando scaler para {class_name} desde data originales...")
    
    df_pl = pl.read_csv(DATASET_PATH, low_memory=False)
    df = df_pl.to_pandas()
    df.columns = df.columns.str.strip()
    
    # Filtrar por class
    df_cls = df[df['Attack Type'] == class_name].copy()
    
    # Preparar features
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
    
    features = df_cls[FEATURES_BASE].copy()
    
    # IPs
    octetos = df_cls['Source IP'].astype(str).str.split('.', expand=True)
    for i in range(4):
        features[f'Src_IP_{i+1}'] = pd.to_numeric(octetos[i], errors='coerce').fillna(0).astype(int)
    
    octetos = df_cls['Destination IP'].astype(str).str.split('.', expand=True)
    for i in range(4):
        features[f'Dst_IP_{i+1}'] = pd.to_numeric(octetos[i], errors='coerce').fillna(0).astype(int)
    
    # Transformación logarítmica
    for col in LOG_COLUMNS:
        features[col] = np.log1p(features[col].clip(lower=0))
        features[col] = features[col].clip(lower=-20, upper=20)
    
    features.replace([np.inf, -np.inf], np.nan, inplace=True)
    features.fillna(0, inplace=True)
    
    # Ajustar scaler
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler.fit(features[FEATURE_NAMES])
    
    # Save para uso futuro
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    
    return scaler


def generate_samples(generator, n_samples, latent_dim=LATENT_DIM, batch_size=1024):
    """Generate synthetic samples usando el generator"""
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
    """Reconstructs features to original scale"""
    X_inv = scaler.inverse_transform(X_synthetic)
    df_rec = pd.DataFrame(X_inv, columns=FEATURE_NAMES)
    
    # Transformación inversa logarítmica
    for col in LOG_COLUMNS:
        df_rec[col] = np.expm1(df_rec[col])
    
    # Clipping y tipos de data
    for col in df_rec.columns:
        if col.startswith('Src_IP_') or col.startswith('Dst_IP_'):
            df_rec[col] = df_rec[col].round().clip(0, 255).astype(int)
    
    df_rec['Source Port'] = df_rec['Source Port'].round().clip(1, 65535).astype(int)
    df_rec['Destination Port'] = df_rec['Destination Port'].round().clip(1, 65535).astype(int)
    df_rec['Protocol'] = df_rec['Protocol'].round().clip(1, 255).astype(int)
    
    for col in LOG_COLUMNS + ['SYN Flag Count', 'ACK Flag Count', 
                               'FIN Flag Count', 'RST Flag Count', 'PSH Flag Count']:
        df_rec[col] = df_rec[col].clip(lower=0)
    
    integer_columns = ['Total Fwd Packets', 'Total Backward Packets', 
                        'SYN Flag Count', 'ACK Flag Count', 
                        'FIN Flag Count', 'RST Flag Count', 'PSH Flag Count']
    for col in integer_columns:
        df_rec[col] = df_rec[col].round().astype(int)
    
    return df_rec


def generate_dataset(samples_per_class, output_name=None, include_scaled=False, prefer_v2=False):
    """
    Genera el dataset synthetic completo.
    
    Args:
        samples_per_class: dict con {nombre_class: num_samples}
        output_name: nombre base para los archivos de salida
        include_scaled: si True, también guarda los data escalados
        prefer_v2: si True, usa models v2 cuando estén disponibles
    
    Returns:
        DataFrame con el dataset synthetic completo
    """
    print_header()
    
    available_classes = get_available_classes(prefer_v2=prefer_v2)
    print_available_classes(available_classes)
    
    # Validar classs solicitadas
    for class_name in samples_per_class:
        if class_name not in available_classes:
            print(f"\n[WARNING] Class '{class_name}' no disponible, se omitirá.")
    
    # Filter only valid classes
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
        print(f"  {class_name:<15}: {n:>10,} samples ({pct:>5.1f}%)")
    print("-" * 70)
    print(f"  {'TOTAL':<15}: {total_samples:>10,} samples")
    print("=" * 70)
    
    # Generar data
    synthetic_parts = []
    scaled_parts = []
    
    print("\nGenerating data synthetics...")
    start_time = datetime.now()
    
    for class_name, n_samples in valid_samples.items():
        print(f"\n[{class_name}] Generating {n_samples:,} samples...")
        
        # Cargar generator
        generator = load_generator(class_name, available_classes)
        
        # Cargar/recrear scaler (usando el directorio correcto)
        source_dir = available_classes[class_name].get('source_dir')
        scaler = load_scaler_from_data(class_name, source_dir=source_dir)
        
        # Generar samples
        print(f"  Generating con WGAN-GP...")
        X_synth = generate_samples(generator, n_samples)
        
        if include_scaled:
            scaled_parts.append({
                'class': class_name,
                'data': X_synth
            })
        
        # Reconstruir a escala original
        print(f"  Reconstruyendo features...")
        df_synth = reconstruct_features(X_synth, scaler)
        
        # Añadir etiquetas
        df_synth['Label'] = CLASS_TO_LABEL[class_name]
        df_synth['Attack Type'] = class_name
        
        synthetic_parts.append(df_synth)
        print(f"  [OK] {len(df_synth):,} samples generadas")
        
        # Free memory
        del generator
        tf.keras.backend.clear_session()
    
    # Combinar todo
    print("\nCombinando dataset...")
    df_all = pd.concat(synthetic_parts, ignore_index=True)
    
    # Shuffle
    df_all = df_all.sample(frac=1, random_state=42).reset_index(drop=True)
    
    elapsed = datetime.now() - start_time
    print(f"\nTiempo de generación: {elapsed}")
    
    # Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    if output_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_name = f"synthetic_dataset_{timestamp}"
    
    # CSV principal
    csv_path = os.path.join(OUTPUT_DIR, f"{output_name}.csv")
    df_all.to_csv(csv_path, index=False)
    print(f"\n[SAVED] Dataset: {csv_path}")
    
    # Save configuration
    config = {
        'generated_at': datetime.now().isoformat(),
        'samples_per_class': valid_samples,
        'total_samples': len(df_all),
        'models_dir': MODELS_DIR,
    }
    config_path = os.path.join(OUTPUT_DIR, f"{output_name}_config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"[SAVED] Config: {config_path}")
    
    # Data escalados (opcional)
    if include_scaled:
        scaled_path = os.path.join(OUTPUT_DIR, f"{output_name}_scaled.npz")
        np.savez(scaled_path, 
                 **{p['class']: p['data'] for p in scaled_parts})
        print(f"[SAVED] Scaled: {scaled_path}")
    
    # Summary final
    print("\n" + "=" * 70)
    print("RESUMEN DEL DATASET GENERADO")
    print("=" * 70)
    print(f"\nClass distribution:")
    for class_name in valid_samples:
        count = len(df_all[df_all['Attack Type'] == class_name])
        pct = (count / len(df_all)) * 100
        print(f"  {class_name:<15}: {count:>10,} ({pct:>5.1f}%)")
    print(f"\n  Total: {len(df_all):,} samples")
    print(f"  Archivo: {csv_path}")
    print("=" * 70)
    
    return df_all


def interactive_mode(prefer_v2=False):
    """Modo interactivo para especificar samples"""
    print_header()
    
    # Preguntar si usar v2 si no está especificado
    if not prefer_v2:
        use_v2_input = input("\n¿Usar models v2 mejorados para classs minoritarias? [s/N]: ").strip().lower()
        prefer_v2 = use_v2_input in ['s', 'si', 'y', 'yes']
    
    available_classes = get_available_classes(prefer_v2=prefer_v2)
    print_available_classes(available_classes)
    
    print("\nModo interactivo - Especifica el número de samples por class")
    print("(Ingresa 0 o deja vacío para omitir una class)\n")
    
    samples_per_class = {}
    
    for class_name in available_classes:
        while True:
            try:
                version = available_classes[class_name].get('version', 'v1')
                version_tag = f" [{version}]" if version == 'v2' else ""
                inp = input(f"  {class_name:<15}{version_tag}: ")
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
    
    # Nombre de salida
    output_name = input("\nNombre del dataset (Enter para auto): ").strip()
    if not output_name:
        output_name = None
    
    # Generar
    generate_dataset(samples_per_class, output_name, prefer_v2=prefer_v2)


def parse_arguments():
    """Parsea argumentos de línea de comandos"""
    parser = argparse.ArgumentParser(
        description='Genera datasets synthetics usando models WGAN-GP entrenados',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  # Modo interactivo
  python generate_synthetic_dataset.py --interactive
  
  # Especificar por class
  python generate_synthetic_dataset.py --benign 50000 --ddos 20000 --dos 15000
  
  # Dataset balanceado (mismo número por class)
  python generate_synthetic_dataset.py --balanced 10000
  
  # Desde archivo de configuration
  python generate_synthetic_dataset.py --config mi_config.json
  
  # Con nombre personalizado
  python generate_synthetic_dataset.py --balanced 5000 --output mi_dataset
        """
    )
    
    # Modos de ejecución
    parser.add_argument('--interactive', '-i', action='store_true',
                       help='Modo interactivo')
    parser.add_argument('--config', '-c', type=str,
                       help='Archivo JSON con configuration')
    parser.add_argument('--balanced', '-b', type=int,
                       help='Genera N samples de cada class')
    
    # Muestras por class
    parser.add_argument('--benign', type=int, default=0,
                       help='Número de samples BENIGN')
    parser.add_argument('--bot', type=int, default=0,
                       help='Número de samples Bot')
    parser.add_argument('--brute-force', '--bf', type=int, default=0,
                       help='Número de samples Brute Force')
    parser.add_argument('--ddos', type=int, default=0,
                       help='Número de samples DDoS')
    parser.add_argument('--dos', type=int, default=0,
                       help='Número de samples DoS')
    parser.add_argument('--port-scan', '--ps', type=int, default=0,
                       help='Número de samples Port Scan')
    parser.add_argument('--web-attack', '--wa', type=int, default=0,
                       help='Número de samples Web Attack')
    
    # Opciones adicionales
    parser.add_argument('--output', '-o', type=str,
                       help='Nombre base para archivos de salida')
    parser.add_argument('--include-scaled', action='store_true',
                       help='También guardar data escalados (.npz)')
    parser.add_argument('--list-classes', '-l', action='store_true',
                       help='Lista classs disponibles y sale')
    parser.add_argument('--use-v2', action='store_true',
                       help='Preferir models v2 (mejorados para classs minoritarias)')
    
    return parser.parse_args()


def main():
    args = parse_arguments()
    
    # Determinar si usar models v2
    prefer_v2 = getattr(args, 'use_v2', False)
    
    # Listar classs
    if args.list_classes:
        print_header()
        available = get_available_classes(prefer_v2=prefer_v2)
        print_available_classes(available)
        return
    
    # Modo interactivo
    if args.interactive:
        interactive_mode(prefer_v2=prefer_v2)
        return
    
    # Desde archivo de configuration
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
        samples_per_class = config.get('samples_per_class', {})
        output_name = config.get('output_name', args.output)
        generate_dataset(samples_per_class, output_name, args.include_scaled, prefer_v2=prefer_v2)
        return
    
    # Dataset balanceado
    if args.balanced:
        available = get_available_classes(prefer_v2=prefer_v2)
        samples_per_class = {cls: args.balanced for cls in available}
        generate_dataset(samples_per_class, args.output, args.include_scaled, prefer_v2=prefer_v2)
        return
    
    # Desde argumentos individuales
    samples_per_class = {}
    
    arg_mapping = {
        'BENIGN': args.benign,
        'Bot': args.bot,
        'Brute Force': args.brute_force,
        'DDoS': args.ddos,
        'DoS': args.dos,
        'Port Scan': args.port_scan,
        'Web Attack': args.web_attack,
    }
    
    for class_name, n in arg_mapping.items():
        if n > 0:
            samples_per_class[class_name] = n
    
    if samples_per_class:
        generate_dataset(samples_per_class, args.output, args.include_scaled, prefer_v2=prefer_v2)
    else:
        # Sin argumentos, mostrar ayuda
        print_header()
        print("\nNo se especificaron samples a generar.")
        print("Usa --help para ver las opciones disponibles.")
        print("\nEjemplos rápidos:")
        print("  python generate_synthetic_dataset.py --interactive")
        print("  python generate_synthetic_dataset.py --balanced 10000")
        print("  python generate_synthetic_dataset.py --balanced 10000 --use-v2")
        print("  python generate_synthetic_dataset.py --benign 50000 --ddos 20000")


if __name__ == "__main__":
    main()
