"""
Synthetic Network Traffic Generation from Pre-trained WGAN-GP Models

Generates synthetic datasets specifying number of samples per class
using trained WGAN-GP models.

Usage:
    python 02_generate_synthetic_data_cicids2017.py
"""

import os 
import sys 
import json 
import argparse 
import pickle 
import numpy as np 
import pandas as pd 
from datetime import datetime 
from sklearn .preprocessing import MinMaxScaler 
import tensorflow as tf 
from tensorflow .keras .models import load_model 

from dotenv import load_dotenv 


load_dotenv ()


os .environ ['TF_CPP_MIN_LOG_LEVEL']='2'

MODELS_DIR =os .getenv ('OUTPUT_MODELS_DIR','./outputs/models')
OUTPUT_DIR =os .getenv ('OUTPUT_SYNTHETIC_DIR','./outputs/synthetic_data')
DATASET_PATH =os .getenv ('DATA_CICIDS2017_PATH','./data/CIC-IDS2017.csv')
LATENT_DIM =int (os .getenv ('LATENT_DIM',100 ))


CLASS_TO_FOLDER ={
'BENIGN':'benign',
'Bot':'bot',
'Brute Force':'brute_force',
'DDoS':'ddos',
'DoS':'dos',
'Port Scan':'port_scan',
'Web Attack':'web_attack',
}


FOLDER_TO_CLASS ={v :k for k ,v in CLASS_TO_FOLDER .items ()}


CLASS_TO_LABEL ={
'BENIGN':0 ,
'Bot':1 ,
'Brute Force':2 ,
'DDoS':3 ,
'DoS':4 ,
'Port Scan':5 ,
'Web Attack':6 ,
}


COLUMNAS_LOG =[
'Total Fwd Packets','Total Backward Packets',
'Total Length of Fwd Packets','Total Length of Bwd Packets',
'Flow Duration','Flow IAT Mean','Flow IAT Std',
'Fwd IAT Mean','Bwd IAT Mean',
'Fwd Packet Length Mean','Bwd Packet Length Mean',
'Packet Length Std','Max Packet Length'
]


FEATURE_NAMES =[
'Source Port','Destination Port','Protocol',
'Total Fwd Packets','Total Backward Packets',
'Total Length of Fwd Packets','Total Length of Bwd Packets',
'Flow Duration','Flow IAT Mean','Flow IAT Std',
'Fwd IAT Mean','Bwd IAT Mean',
'Fwd Packet Length Mean','Bwd Packet Length Mean',
'Packet Length Std','Max Packet Length',
'SYN Flag Count','ACK Flag Count','FIN Flag Count',
'RST Flag Count','PSH Flag Count',
'Src_IP_1','Src_IP_2','Src_IP_3','Src_IP_4',
'Dst_IP_1','Dst_IP_2','Dst_IP_3','Dst_IP_4'
]





def print_header ():
    """Imprime cabecera del programa"""
    print ("="*70 )
    print ("  GENERADOR DE DATASET SINTÉTICO - WGAN-GP")
    print ("  CiberIA - Intrusion Detection System")
    print ("="*70 )


def get_available_classes (models_dir =None ,prefer_v2 =False ):
    """
    Obtiene las clases disponibles verificando modelos existentes.
    
    Args:
        models_dir: directorio de modelos a usar (si None, usa MODELS_DIR)
        prefer_v2: si True, prefiere modelos v2 para clases que existan en ambos
    """
    if models_dir is None :
        models_dir =MODELS_DIR 

    available ={}
    for class_name ,folder in CLASS_TO_FOLDER .items ():
        model_path =None 
        source_dir =None 


        if model_path is None :
            v1_path =os .path .join (models_dir ,folder ,f'generator_{folder }.h5')
            if os .path .exists (v1_path ):
                model_path =v1_path 
                source_dir =models_dir 

        if model_path :
            available [class_name ]={
            'folder':folder ,
            'model_path':model_path ,
            'source_dir':source_dir ,
            'version':'v2'
            }

    return available 


def print_available_classes (available_classes ):
    """Muestra las clases disponibles"""
    print ("\nClases disponibles:")
    print ("-"*60 )
    for i ,(class_name ,info )in enumerate (available_classes .items (),1 ):
        version =info .get ('version','v1')
        version_tag =f"[{version }]"if version =='v2'else ""
        print (f"  {i }. {class_name :<15} -> {info ['folder']:<15} {version_tag }")
    print ("-"*60 )
    if any (info .get ('version')=='v2'for info in available_classes .values ()):
        print ("  [v2] = Modelo mejorado para clases minoritarias")


def load_generator (class_name ,available_classes ):
    """Carga el generador para una clase específica"""
    if class_name not in available_classes :
        raise ValueError (f"Clase '{class_name }' no disponible")

    model_path =available_classes [class_name ]['model_path']
    print (f"  Cargando modelo: {os .path .basename (model_path )}")
    generator =load_model (model_path ,compile =False )
    return generator 


def load_scaler_from_data (class_name ,source_dir =None ):
    """
    Carga/recrea el scaler para una clase específica.
    Primero busca scaler guardado, si no existe lo recrea.
    """
    import polars as pl 

    folder =CLASS_TO_FOLDER [class_name ]


    search_dirs =[source_dir ]if source_dir else [MODELS_DIR ]

    for search_dir in search_dirs :
        if search_dir is None :
            continue 
        scaler_path =os .path .join (search_dir ,folder ,'scaler.pkl')
        if os .path .exists (scaler_path ):
            print (f"  Cargando scaler desde: {os .path .dirname (scaler_path )}")
            with open (scaler_path ,'rb')as f :
                return pickle .load (f )


    print (f"  Recreando scaler para {class_name } desde datos originales...")

    df_pl =pl .read_csv (DATASET_PATH ,low_memory =False )
    df =df_pl .to_pandas ()
    df .columns =df .columns .str .strip ()


    df_cls =df [df ['Attack Type']==class_name ].copy ()


    FEATURES_BASE =[
    'Source Port','Destination Port','Protocol',
    'Total Fwd Packets','Total Backward Packets',
    'Total Length of Fwd Packets','Total Length of Bwd Packets',
    'Flow Duration','Flow IAT Mean','Flow IAT Std',
    'Fwd IAT Mean','Bwd IAT Mean',
    'Fwd Packet Length Mean','Bwd Packet Length Mean',
    'Packet Length Std','Max Packet Length',
    'SYN Flag Count','ACK Flag Count','FIN Flag Count',
    'RST Flag Count','PSH Flag Count'
    ]

    features =df_cls [FEATURES_BASE ].copy ()


    octetos =df_cls ['Source IP'].astype (str ).str .split ('.',expand =True )
    for i in range (4 ):
        features [f'Src_IP_{i +1 }']=pd .to_numeric (octetos [i ],errors ='coerce').fillna (0 ).astype (int )

    octetos =df_cls ['Destination IP'].astype (str ).str .split ('.',expand =True )
    for i in range (4 ):
        features [f'Dst_IP_{i +1 }']=pd .to_numeric (octetos [i ],errors ='coerce').fillna (0 ).astype (int )


    for col in COLUMNAS_LOG :
        features [col ]=np .log1p (features [col ].clip (lower =0 ))
        features [col ]=features [col ].clip (lower =-20 ,upper =20 )

    features .replace ([np .inf ,-np .inf ],np .nan ,inplace =True )
    features .fillna (0 ,inplace =True )


    scaler =MinMaxScaler (feature_range =(-1 ,1 ))
    scaler .fit (features [FEATURE_NAMES ])


    with open (scaler_path ,'wb')as f :
        pickle .dump (scaler ,f )

    return scaler 


def generate_samples (generator ,n_samples ,latent_dim =LATENT_DIM ,batch_size =1024 ):
    """Genera muestras sintéticas usando el generador"""
    all_samples =[]
    remaining =n_samples 

    while remaining >0 :
        current_batch =min (batch_size ,remaining )
        noise =np .random .normal (0 ,1 ,(current_batch ,latent_dim ))
        samples =generator .predict (noise ,verbose =0 )
        all_samples .append (samples )
        remaining -=current_batch 

    return np .vstack (all_samples )


def reconstruct_features (X_synthetic ,scaler ):
    """Reconstruye features a escala original"""
    X_inv =scaler .inverse_transform (X_synthetic )
    df_rec =pd .DataFrame (X_inv ,columns =FEATURE_NAMES )


    for col in COLUMNAS_LOG :
        df_rec [col ]=np .expm1 (df_rec [col ])


    for col in df_rec .columns :
        if col .startswith ('Src_IP_')or col .startswith ('Dst_IP_'):
            df_rec [col ]=df_rec [col ].round ().clip (0 ,255 ).astype (int )

    df_rec ['Source Port']=df_rec ['Source Port'].round ().clip (1 ,65535 ).astype (int )
    df_rec ['Destination Port']=df_rec ['Destination Port'].round ().clip (1 ,65535 ).astype (int )
    df_rec ['Protocol']=df_rec ['Protocol'].round ().clip (1 ,255 ).astype (int )

    for col in COLUMNAS_LOG +['SYN Flag Count','ACK Flag Count',
    'FIN Flag Count','RST Flag Count','PSH Flag Count']:
        df_rec [col ]=df_rec [col ].clip (lower =0 )

    columnas_enteras =['Total Fwd Packets','Total Backward Packets',
    'SYN Flag Count','ACK Flag Count',
    'FIN Flag Count','RST Flag Count','PSH Flag Count']
    for col in columnas_enteras :
        df_rec [col ]=df_rec [col ].round ().astype (int )

    return df_rec 


def generate_dataset (samples_per_class ,output_name =None ,include_scaled =False ,prefer_v2 =False ):
    """
    Genera el dataset sintético completo.
    
    Args:
        samples_per_class: dict con {nombre_clase: num_muestras}
        output_name: nombre base para los archivos de salida
        include_scaled: si True, también guarda los datos escalados
        prefer_v2: si True, usa modelos v2 cuando estén disponibles
    
    Returns:
        DataFrame con el dataset sintético completo
    """
    print_header ()

    available_classes =get_available_classes (prefer_v2 =prefer_v2 )
    print_available_classes (available_classes )


    for class_name in samples_per_class :
        if class_name not in available_classes :
            print (f"\n[WARNING] Clase '{class_name }' no disponible, se omitirá.")


    valid_samples ={k :v for k ,v in samples_per_class .items ()
    if k in available_classes and v >0 }

    if not valid_samples :
        print ("\n[ERROR] No hay clases válidas para generar.")
        return None 

    print ("\n"+"="*70 )
    print ("CONFIGURACIÓN DE GENERACIÓN")
    print ("="*70 )
    total_samples =sum (valid_samples .values ())
    for class_name ,n in valid_samples .items ():
        pct =(n /total_samples )*100 
        print (f"  {class_name :<15}: {n :>10,} muestras ({pct :>5.1f}%)")
    print ("-"*70 )
    print (f"  {'TOTAL':<15}: {total_samples :>10,} muestras")
    print ("="*70 )


    synthetic_parts =[]
    scaled_parts =[]

    print ("\nGenerando datos sintéticos...")
    start_time =datetime .now ()

    for class_name ,n_samples in valid_samples .items ():
        print (f"\n[{class_name }] Generando {n_samples :,} muestras...")


        generator =load_generator (class_name ,available_classes )


        source_dir =available_classes [class_name ].get ('source_dir')
        scaler =load_scaler_from_data (class_name ,source_dir =source_dir )


        print (f"  Generando con WGAN-GP...")
        X_synth =generate_samples (generator ,n_samples )

        if include_scaled :
            scaled_parts .append ({
            'class':class_name ,
            'data':X_synth 
            })


        print (f"  Reconstruyendo features...")
        df_synth =reconstruct_features (X_synth ,scaler )


        df_synth ['Label']=CLASS_TO_LABEL [class_name ]
        df_synth ['Attack Type']=class_name 

        synthetic_parts .append (df_synth )
        print (f"  [OK] {len (df_synth ):,} muestras generadas")


        del generator 
        tf .keras .backend .clear_session ()


    print ("\nCombinando dataset...")
    df_all =pd .concat (synthetic_parts ,ignore_index =True )


    df_all =df_all .sample (frac =1 ,random_state =42 ).reset_index (drop =True )

    elapsed =datetime .now ()-start_time 
    print (f"\nTiempo de generación: {elapsed }")


    os .makedirs (OUTPUT_DIR ,exist_ok =True )

    if output_name is None :
        timestamp =datetime .now ().strftime ("%Y%m%d_%H%M%S")
        output_name =f"synthetic_dataset_{timestamp }"


    csv_path =os .path .join (OUTPUT_DIR ,f"{output_name }.csv")
    df_all .to_csv (csv_path ,index =False )
    print (f"\n[SAVED] Dataset: {csv_path }")


    config ={
    'generated_at':datetime .now ().isoformat (),
    'samples_per_class':valid_samples ,
    'total_samples':len (df_all ),
    'models_dir':MODELS_DIR ,
    }
    config_path =os .path .join (OUTPUT_DIR ,f"{output_name }_config.json")
    with open (config_path ,'w')as f :
        json .dump (config ,f ,indent =2 )
    print (f"[SAVED] Config: {config_path }")


    if include_scaled :
        scaled_path =os .path .join (OUTPUT_DIR ,f"{output_name }_scaled.npz")
        np .savez (scaled_path ,
        **{p ['class']:p ['data']for p in scaled_parts })
        print (f"[SAVED] Scaled: {scaled_path }")


    print ("\n"+"="*70 )
    print ("RESUMEN DEL DATASET GENERADO")
    print ("="*70 )
    print (f"\nDistribución de clases:")
    for class_name in valid_samples :
        count =len (df_all [df_all ['Attack Type']==class_name ])
        pct =(count /len (df_all ))*100 
        print (f"  {class_name :<15}: {count :>10,} ({pct :>5.1f}%)")
    print (f"\n  Total: {len (df_all ):,} muestras")
    print (f"  Archivo: {csv_path }")
    print ("="*70 )

    return df_all 


def interactive_mode (prefer_v2 =False ):
    """Modo interactivo para especificar muestras"""
    print_header ()


    if not prefer_v2 :
        use_v2_input =input ("\n¿Usar modelos v2 mejorados para clases minoritarias? [s/N]: ").strip ().lower ()
        prefer_v2 =use_v2_input in ['s','si','y','yes']

    available_classes =get_available_classes (prefer_v2 =prefer_v2 )
    print_available_classes (available_classes )

    print ("\nModo interactivo - Especifica el número de muestras por clase")
    print ("(Ingresa 0 o deja vacío para omitir una clase)\n")

    samples_per_class ={}

    for class_name in available_classes :
        while True :
            try :
                version =available_classes [class_name ].get ('version','v1')
                version_tag =f" [{version }]"if version =='v2'else ""
                inp =input (f"  {class_name :<15}{version_tag }: ")
                if inp .strip ()=='':
                    n =0 
                else :
                    n =int (inp )
                if n <0 :
                    print ("    [!] Debe ser >= 0")
                    continue 
                if n >0 :
                    samples_per_class [class_name ]=n 
                break 
            except ValueError :
                print ("    [!] Ingresa un número válido")

    if not samples_per_class :
        print ("\n[!] No se especificaron muestras. Saliendo.")
        return 


    output_name =input ("\nNombre del dataset (Enter para auto): ").strip ()
    if not output_name :
        output_name =None 


    generate_dataset (samples_per_class ,output_name ,prefer_v2 =prefer_v2 )


def parse_arguments ():
    """Parsea argumentos de línea de comandos"""
    parser =argparse .ArgumentParser (
    description ='Genera datasets sintéticos usando modelos WGAN-GP entrenados',
    formatter_class =argparse .RawDescriptionHelpFormatter ,
    epilog ="""
Ejemplos de uso:
  # Modo interactivo
  python generate_synthetic_dataset.py --interactive
  
  # Especificar por clase
  python generate_synthetic_dataset.py --benign 50000 --ddos 20000 --dos 15000
  
  # Dataset balanceado (mismo número por clase)
  python generate_synthetic_dataset.py --balanced 10000
  
  # Desde archivo de configuración
  python generate_synthetic_dataset.py --config mi_config.json
  
  # Con nombre personalizado
  python generate_synthetic_dataset.py --balanced 5000 --output mi_dataset
        """
    )


    parser .add_argument ('--interactive','-i',action ='store_true',
    help ='Modo interactivo')
    parser .add_argument ('--config','-c',type =str ,
    help ='Archivo JSON con configuración')
    parser .add_argument ('--balanced','-b',type =int ,
    help ='Genera N muestras de cada clase')


    parser .add_argument ('--benign',type =int ,default =0 ,
    help ='Número de muestras BENIGN')
    parser .add_argument ('--bot',type =int ,default =0 ,
    help ='Número de muestras Bot')
    parser .add_argument ('--brute-force','--bf',type =int ,default =0 ,
    help ='Número de muestras Brute Force')
    parser .add_argument ('--ddos',type =int ,default =0 ,
    help ='Número de muestras DDoS')
    parser .add_argument ('--dos',type =int ,default =0 ,
    help ='Número de muestras DoS')
    parser .add_argument ('--port-scan','--ps',type =int ,default =0 ,
    help ='Número de muestras Port Scan')
    parser .add_argument ('--web-attack','--wa',type =int ,default =0 ,
    help ='Número de muestras Web Attack')


    parser .add_argument ('--output','-o',type =str ,
    help ='Nombre base para archivos de salida')
    parser .add_argument ('--include-scaled',action ='store_true',
    help ='También guardar datos escalados (.npz)')
    parser .add_argument ('--list-classes','-l',action ='store_true',
    help ='Lista clases disponibles y sale')
    parser .add_argument ('--use-v2',action ='store_true',
    help ='Preferir modelos v2 (mejorados para clases minoritarias)')

    return parser .parse_args ()


def main ():
    args =parse_arguments ()


    prefer_v2 =getattr (args ,'use_v2',False )


    if args .list_classes :
        print_header ()
        available =get_available_classes (prefer_v2 =prefer_v2 )
        print_available_classes (available )
        return 


    if args .interactive :
        interactive_mode (prefer_v2 =prefer_v2 )
        return 


    if args .config :
        with open (args .config ,'r')as f :
            config =json .load (f )
        samples_per_class =config .get ('samples_per_class',{})
        output_name =config .get ('output_name',args .output )
        generate_dataset (samples_per_class ,output_name ,args .include_scaled ,prefer_v2 =prefer_v2 )
        return 


    if args .balanced :
        available =get_available_classes (prefer_v2 =prefer_v2 )
        samples_per_class ={cls :args .balanced for cls in available }
        generate_dataset (samples_per_class ,args .output ,args .include_scaled ,prefer_v2 =prefer_v2 )
        return 


    samples_per_class ={}

    arg_mapping ={
    'BENIGN':args .benign ,
    'Bot':args .bot ,
    'Brute Force':args .brute_force ,
    'DDoS':args .ddos ,
    'DoS':args .dos ,
    'Port Scan':args .port_scan ,
    'Web Attack':args .web_attack ,
    }

    for class_name ,n in arg_mapping .items ():
        if n >0 :
            samples_per_class [class_name ]=n 

    if samples_per_class :
        generate_dataset (samples_per_class ,args .output ,args .include_scaled ,prefer_v2 =prefer_v2 )
    else :

        print_header ()
        print ("\nNo se especificaron muestras a generar.")
        print ("Usa --help para ver las opciones disponibles.")
        print ("\nEjemplos rápidos:")
        print ("  python generate_synthetic_dataset.py --interactive")
        print ("  python generate_synthetic_dataset.py --balanced 10000")
        print ("  python generate_synthetic_dataset.py --balanced 10000 --use-v2")
        print ("  python generate_synthetic_dataset.py --benign 50000 --ddos 20000")


if __name__ =="__main__":
    main ()
