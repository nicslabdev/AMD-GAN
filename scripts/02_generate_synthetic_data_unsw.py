"""
Synthetic Dataset Generator — UNSW-NB15

Generates synthetic UNSW-NB15 dataset by specifying sample count per class,
using pre-trained WGAN-GP models.

Usage:
    python 02_generate_synthetic_data_unsw.py --interactive
    python 02_generate_synthetic_data_unsw.py --balanced 10000
    python 02_generate_synthetic_data_unsw.py --config config.json
    python 02_generate_synthetic_data_unsw.py --benign 50000 --exploits 10000
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
from sklearn .preprocessing import MinMaxScaler 
import tensorflow as tf 
from tensorflow .keras .models import load_model 

os .environ ['TF_CPP_MIN_LOG_LEVEL']='2'

from dotenv import load_dotenv 


load_dotenv ()


MODELS_DIR =os .getenv ('OUTPUT_MODELS_DIR','./outputs/models')
OUTPUT_DIR =os .getenv ('OUTPUT_SYNTHETIC_DIR','./outputs/synthetic_data')
DATASET_PATH =os .getenv ('DATA_UNSW_PATH','./data/UNSW-NB15.csv')
LATENT_DIM =int (os .getenv ('LATENT_DIM',100 ))
LABEL_COLUMN ='Label'


CLASS_TO_FOLDER ={
'Benign':'benign',
'Exploits':'exploits',
'Fuzzers':'fuzzers',
'Reconnaissance':'reconnaissance',
'Generic':'generic',
'DoS':'dos',
'Shellcode':'shellcode',
}

FOLDER_TO_CLASS ={v :k for k ,v in CLASS_TO_FOLDER .items ()}

CLASS_TO_LABEL ={
'Benign':0 ,
'DoS':1 ,
'Exploits':2 ,
'Fuzzers':3 ,
'Generic':4 ,
'Reconnaissance':5 ,
'Shellcode':6 ,
}


FEATURES_BASE =[
'Src Port','Dst Port','Protocol',
'Total Fwd Packet','Total Bwd packets',
'Total Length of Fwd Packet','Total Length of Bwd Packet',
'Flow Duration','Flow IAT Mean','Flow IAT Std',
'Fwd IAT Mean','Bwd IAT Mean',
'Fwd Packet Length Mean','Bwd Packet Length Mean',
'Packet Length Std','Packet Length Max',
'SYN Flag Count','ACK Flag Count','FIN Flag Count',
'RST Flag Count','PSH Flag Count'
]

FEATURE_NAMES =FEATURES_BASE +[
'Src_IP_1','Src_IP_2','Src_IP_3','Src_IP_4',
'Dst_IP_1','Dst_IP_2','Dst_IP_3','Dst_IP_4'
]

COLUMNAS_LOG =[
'Total Fwd Packet','Total Bwd packets',
'Total Length of Fwd Packet','Total Length of Bwd Packet',
'Flow Duration','Flow IAT Mean','Flow IAT Std',
'Fwd IAT Mean','Bwd IAT Mean',
'Fwd Packet Length Mean','Bwd Packet Length Mean',
'Packet Length Std','Packet Length Max'
]





def print_header ():
    print ("="*70 )
    print ("  GENERADOR DE DATASET SINTÉTICO UNSW-NB15 - WGAN-GP")
    print ("  CiberIA - Intrusion Detection System")
    print ("="*70 )


def get_available_classes ():
    """Obtiene las clases disponibles verificando modelos existentes."""
    available ={}
    for class_name ,folder in CLASS_TO_FOLDER .items ():
        model_path =os .path .join (MODELS_DIR ,folder ,f'generator_{folder }.h5')
        if os .path .exists (model_path ):
            available [class_name ]={
            'folder':folder ,
            'model_path':model_path ,
            'source_dir':MODELS_DIR ,
            }
    return available 


def print_available_classes (available_classes ):
    print ("\nClases disponibles:")
    print ("-"*60 )
    for i ,(class_name ,info )in enumerate (available_classes .items (),1 ):
        print (f"  {i }. {class_name :<20} -> {info ['folder']}")
    print ("-"*60 )


def load_generator (class_name ,available_classes ):
    if class_name not in available_classes :
        raise ValueError (f"Clase '{class_name }' no disponible")
    model_path =available_classes [class_name ]['model_path']
    print (f"  Cargando modelo: {os .path .basename (model_path )}")
    return load_model (model_path ,compile =False )


def load_scaler (class_name ,source_dir =None ):
    """Carga el scaler para una clase. Si no existe, lo recrea desde datos."""
    folder =CLASS_TO_FOLDER [class_name ]
    search_dir =source_dir or MODELS_DIR 

    scaler_path =os .path .join (search_dir ,folder ,'scaler.pkl')
    if os .path .exists (scaler_path ):
        print (f"  Cargando scaler desde: {os .path .dirname (scaler_path )}")
        with open (scaler_path ,'rb')as f :
            return pickle .load (f )


    print (f"  Recreando scaler para {class_name } desde datos originales...")

    df_pl =pl .read_csv (DATASET_PATH ,low_memory =False )
    df =df_pl .to_pandas ()
    df .columns =df .columns .str .strip ()

    df_cls =df [df [LABEL_COLUMN ]==class_name ].copy ()

    features =df_cls [FEATURES_BASE ].copy ()


    if 'Src IP'in df_cls .columns :
        octetos =df_cls ['Src IP'].astype (str ).str .split ('.',expand =True )
        for i in range (4 ):
            features [f'Src_IP_{i +1 }']=pd .to_numeric (octetos [i ],errors ='coerce').fillna (0 ).astype (int )

    if 'Dst IP'in df_cls .columns :
        octetos =df_cls ['Dst IP'].astype (str ).str .split ('.',expand =True )
        for i in range (4 ):
            features [f'Dst_IP_{i +1 }']=pd .to_numeric (octetos [i ],errors ='coerce').fillna (0 ).astype (int )


    for col in COLUMNAS_LOG :
        if col in features .columns :
            features [col ]=np .log1p (features [col ].clip (lower =0 ))
            features [col ]=features [col ].clip (lower =-20 ,upper =20 )

    features .replace ([np .inf ,-np .inf ],np .nan ,inplace =True )
    features .fillna (0 ,inplace =True )

    scaler =MinMaxScaler (feature_range =(-1 ,1 ))
    scaler .fit (features [FEATURE_NAMES ])

    os .makedirs (os .path .dirname (scaler_path ),exist_ok =True )
    with open (scaler_path ,'wb')as f :
        pickle .dump (scaler ,f )

    return scaler 


def generate_samples (generator ,n_samples ,latent_dim =LATENT_DIM ,batch_size =1024 ):
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
    """Reconstruye features a escala original."""
    X_inv =scaler .inverse_transform (X_synthetic )
    df_rec =pd .DataFrame (X_inv ,columns =FEATURE_NAMES )


    for col in COLUMNAS_LOG :
        if col in df_rec .columns :
            df_rec [col ]=np .expm1 (df_rec [col ])


    for col in df_rec .columns :
        if col .startswith ('Src_IP_')or col .startswith ('Dst_IP_'):
            df_rec [col ]=df_rec [col ].round ().clip (0 ,255 ).astype (int )

    df_rec ['Src Port']=df_rec ['Src Port'].round ().clip (1 ,65535 ).astype (int )
    df_rec ['Dst Port']=df_rec ['Dst Port'].round ().clip (1 ,65535 ).astype (int )
    df_rec ['Protocol']=df_rec ['Protocol'].round ().clip (0 ,255 ).astype (int )

    for col in COLUMNAS_LOG +['SYN Flag Count','ACK Flag Count',
    'FIN Flag Count','RST Flag Count','PSH Flag Count']:
        if col in df_rec .columns :
            df_rec [col ]=df_rec [col ].clip (lower =0 )

    columnas_enteras =['Total Fwd Packet','Total Bwd packets',
    'SYN Flag Count','ACK Flag Count',
    'FIN Flag Count','RST Flag Count','PSH Flag Count']
    for col in columnas_enteras :
        if col in df_rec .columns :
            df_rec [col ]=df_rec [col ].round ().astype (int )

    return df_rec 


def generate_dataset (samples_per_class ,output_name =None ,include_scaled =False ):
    """Genera el dataset sintético completo."""
    print_header ()

    available_classes =get_available_classes ()
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
        print (f"  {class_name :<20}: {n :>10,} muestras ({pct :>5.1f}%)")
    print ("-"*70 )
    print (f"  {'TOTAL':<20}: {total_samples :>10,} muestras")
    print ("="*70 )

    synthetic_parts =[]
    scaled_parts =[]

    print ("\nGenerando datos sintéticos...")
    start_time =datetime .now ()

    for class_name ,n_samples_cls in valid_samples .items ():
        print (f"\n[{class_name }] Generando {n_samples_cls :,} muestras...")

        generator =load_generator (class_name ,available_classes )
        source_dir =available_classes [class_name ].get ('source_dir')
        scaler =load_scaler (class_name ,source_dir =source_dir )

        print (f"  Generando con WGAN-GP...")
        X_synth =generate_samples (generator ,n_samples_cls )

        if include_scaled :
            scaled_parts .append ({'class':class_name ,'data':X_synth })

        print (f"  Reconstruyendo features...")
        df_synth =reconstruct_features (X_synth ,scaler )

        df_synth ['Label']=class_name 
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
        output_name =f"unsw_synthetic_{timestamp }"

    csv_path =os .path .join (OUTPUT_DIR ,f"{output_name }.csv")
    df_all .to_csv (csv_path ,index =False )
    print (f"\n[SAVED] Dataset: {csv_path }")

    config ={
    'generated_at':datetime .now ().isoformat (),
    'dataset':'UNSW-NB15',
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
        np .savez (scaled_path ,**{p ['class']:p ['data']for p in scaled_parts })
        print (f"[SAVED] Scaled: {scaled_path }")


    print ("\n"+"="*70 )
    print ("RESUMEN DEL DATASET GENERADO")
    print ("="*70 )
    print (f"\nDistribución de clases:")
    for class_name in valid_samples :
        count =len (df_all [df_all ['Label']==class_name ])
        pct =(count /len (df_all ))*100 
        print (f"  {class_name :<20}: {count :>10,} ({pct :>5.1f}%)")
    print (f"\n  Total: {len (df_all ):,} muestras")
    print (f"  Archivo: {csv_path }")
    print ("="*70 )

    return df_all 


def interactive_mode ():
    """Modo interactivo para especificar muestras."""
    print_header ()

    available_classes =get_available_classes ()
    print_available_classes (available_classes )

    if not available_classes :
        print ("\n[ERROR] No se encontraron modelos entrenados en:",MODELS_DIR )
        print ("  Entrena los modelos primero con: python gan_wgan_unsw.py --all")
        return 

    print ("\nModo interactivo - Especifica el número de muestras por clase")
    print ("(Ingresa 0 o deja vacío para omitir una clase)\n")

    samples_per_class ={}
    for class_name in available_classes :
        while True :
            try :
                inp =input (f"  {class_name :<20}: ")
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

    output_name =input ("\nNombre del dataset (Enter para auto): ").strip ()or None 
    generate_dataset (samples_per_class ,output_name )


def main ():
    parser =argparse .ArgumentParser (
    description ='Genera datasets sintéticos UNSW-NB15 usando modelos WGAN-GP',
    formatter_class =argparse .RawDescriptionHelpFormatter ,
    epilog ="""
Ejemplos:
  python generate_synthetic_dataset_unsw.py --interactive
  python generate_synthetic_dataset_unsw.py --balanced 10000
  python generate_synthetic_dataset_unsw.py --benign 50000 --exploits 10000
  python generate_synthetic_dataset_unsw.py --config config.json
        """
    )

    parser .add_argument ('--interactive','-i',action ='store_true')
    parser .add_argument ('--config','-c',type =str ,help ='Archivo JSON de configuración')
    parser .add_argument ('--balanced','-b',type =int ,help ='N muestras de cada clase')

    parser .add_argument ('--benign',type =int ,default =0 )
    parser .add_argument ('--exploits',type =int ,default =0 )
    parser .add_argument ('--fuzzers',type =int ,default =0 )
    parser .add_argument ('--reconnaissance',type =int ,default =0 )
    parser .add_argument ('--generic',type =int ,default =0 )
    parser .add_argument ('--dos',type =int ,default =0 )
    parser .add_argument ('--shellcode',type =int ,default =0 )

    parser .add_argument ('--output','-o',type =str ,help ='Nombre base para salida')
    parser .add_argument ('--include-scaled',action ='store_true')
    parser .add_argument ('--list-classes','-l',action ='store_true')

    args =parser .parse_args ()

    if args .list_classes :
        print_header ()
        available =get_available_classes ()
        print_available_classes (available )
        return 

    if args .interactive :
        interactive_mode ()
        return 

    if args .config :
        with open (args .config ,'r')as f :
            config =json .load (f )
        samples =config .get ('samples_per_class',{})
        generate_dataset (samples ,config .get ('output_name',args .output ),args .include_scaled )
        return 

    if args .balanced :
        available =get_available_classes ()
        samples ={cls :args .balanced for cls in available }
        generate_dataset (samples ,args .output ,args .include_scaled )
        return 


    arg_mapping ={
    'Benign':args .benign ,
    'Exploits':args .exploits ,
    'Fuzzers':args .fuzzers ,
    'Reconnaissance':args .reconnaissance ,
    'Generic':args .generic ,
    'DoS':args .dos ,
    'Shellcode':args .shellcode ,
    }

    samples_per_class ={k :v for k ,v in arg_mapping .items ()if v >0 }

    if samples_per_class :
        generate_dataset (samples_per_class ,args .output ,args .include_scaled )
    else :
        print_header ()
        print ("\nNo se especificaron muestras a generar.")
        print ("Usa --help para ver las opciones disponibles.")
        print ("\nEjemplos rápidos:")
        print ("  python generate_synthetic_dataset_unsw.py --interactive")
        print ("  python generate_synthetic_dataset_unsw.py --balanced 10000")
        print ("  python generate_synthetic_dataset_unsw.py --benign 50000 --exploits 10000")


if __name__ =="__main__":
    main ()
