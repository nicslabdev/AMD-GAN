"""
Oversampling Comparison: GAN vs Traditional Methods - Multi-Model Evaluation

Compares oversampling techniques for class imbalance in CIC-IDS2017 dataset:
- Original (no oversampling)
- GAN-generated synthetic data
- SMOTE, ADASYN, BorderlineSMOTE, SMOTE-ENN, Random Oversampling

Evaluates with multiple classifiers: RandomForest, KNN, MLP-DNN
"""

import os 
import sys 

os .environ ['TF_CPP_MIN_LOG_LEVEL']='2'

import pickle 
import numpy as np 
import pandas as pd 
import polars as pl 
from datetime import datetime 
from collections import Counter 

from sklearn .preprocessing import MinMaxScaler ,LabelEncoder 
from sklearn .model_selection import train_test_split 
from sklearn .metrics import (
classification_report ,confusion_matrix ,
accuracy_score ,f1_score ,precision_score ,recall_score 
)
from sklearn .neighbors import KNeighborsClassifier 
from sklearn .ensemble import RandomForestClassifier 

from imblearn .over_sampling import SMOTE ,ADASYN ,BorderlineSMOTE ,RandomOverSampler 
from imblearn .combine import SMOTEENN 

import tensorflow as tf 
from tensorflow .keras .models import load_model ,Sequential 
from tensorflow .keras .layers import Dense ,Dropout ,BatchNormalization 
from tensorflow .keras .callbacks import EarlyStopping 
from tensorflow .keras .utils import to_categorical 

import warnings 
warnings .filterwarnings ('ignore')

from dotenv import load_dotenv 


load_dotenv ()


REAL_DATA_PATH =os .getenv ('DATA_CICIDS2017_PATH','./data/CIC-IDS2017.csv')
MODELS_DIR =os .getenv ('OUTPUT_MODELS_DIR','./outputs/models')
OUTPUT_DIR =os .getenv ('OUTPUT_RESULTS_DIR','./outputs/results')
RANDOM_STATE =42 
LATENT_DIM =int (os .getenv ('LATENT_DIM',100 ))

EXPERIMENT_CLASSES =['BENIGN','Brute Force','DDoS','DoS','Port Scan']

CLASS_TO_FOLDER ={
'BENIGN':'benign',
'Bot':'bot',
'Brute Force':'brute_force',
'DDoS':'ddos',
'DoS':'dos',
'Port Scan':'port_scan',
'Web Attack':'web_attack',
}

BASE_FEATURES =[
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


FEATURE_NAMES =BASE_FEATURES +[
'Src_IP_1','Src_IP_2','Src_IP_3','Src_IP_4',
'Dst_IP_1','Dst_IP_2','Dst_IP_3','Dst_IP_4'
]


COLUMNAS_LOG =[
'Total Fwd Packets','Total Backward Packets',
'Total Length of Fwd Packets','Total Length of Bwd Packets',
'Flow Duration','Flow IAT Mean','Flow IAT Std',
'Fwd IAT Mean','Bwd IAT Mean',
'Fwd Packet Length Mean','Bwd Packet Length Mean',
'Packet Length Std','Max Packet Length'
]


MODELOS =['rf','knn','mlp']





def print_header ():
    print ("="*100 )
    print ("  EXPERIMENTO: COMPARACIÓN DE TÉCNICAS DE OVERSAMPLING (MULTI-MODEL)")
    print ("  GAN (WGAN-GP) vs SMOTE vs ADASYN vs BorderlineSMOTE vs SMOTE-ENN")
    print ("  Modelos: RandomForest | KNN | MLP-DNN")
    print ("="*100 )


def cargar_datos_reales ():
    """Carga el dataset CIC-IDS2017"""
    print ("\n[1] CARGANDO DATOS REALES")
    print ("-"*50 )

    file_size =os .path .getsize (REAL_DATA_PATH )
    print (f"  Archivo: {REAL_DATA_PATH }")
    print (f"  Tamaño: {file_size /(1024 *1024 ):.2f} MB")

    start_time =datetime .now ()
    df_pl =pl .read_csv (REAL_DATA_PATH ,low_memory =False )
    df =df_pl .to_pandas ()
    df .columns =df .columns .str .strip ()
    print (f"  Cargado en {datetime .now ()-start_time }")
    print (f"  Total registros: {len (df ):,}")

    return df 


def preparar_features (df :pd .DataFrame )->pd .DataFrame :
    """Prepara features para el modelo"""
    df =df .copy ()


    label_col ='Attack Type'if 'Attack Type'in df .columns else 'Label'
    df ['Label_Class']=df [label_col ]


    df =df [df ['Label_Class'].isin (EXPERIMENT_CLASSES )].copy ()


    features_df =df [BASE_FEATURES ].copy ()


    if 'Source IP'in df .columns :
        octetos =df ['Source IP'].astype (str ).str .split ('.',expand =True )
        for i in range (4 ):
            features_df [f'Src_IP_{i +1 }']=pd .to_numeric (octetos [i ],errors ='coerce').fillna (0 ).astype (int )
    else :
        for i in range (4 ):
            features_df [f'Src_IP_{i +1 }']=0 

    if 'Destination IP'in df .columns :
        octetos =df ['Destination IP'].astype (str ).str .split ('.',expand =True )
        for i in range (4 ):
            features_df [f'Dst_IP_{i +1 }']=pd .to_numeric (octetos [i ],errors ='coerce').fillna (0 ).astype (int )
    else :
        for i in range (4 ):
            features_df [f'Dst_IP_{i +1 }']=0 


    features_df ['Label_Class']=df ['Label_Class'].values 


    features_df .replace ([np .inf ,-np .inf ],np .nan ,inplace =True )
    features_df .fillna (0 ,inplace =True )

    return features_df 


def mostrar_distribucion (y ,label_encoder ,titulo ="Distribución"):
    """Muestra la distribución de clases"""
    print (f"\n  {titulo }:")
    counter =Counter (y )
    total =len (y )
    for label_idx in sorted (counter .keys ()):
        clase =label_encoder .inverse_transform ([label_idx ])[0 ]
        count =counter [label_idx ]
        pct =(count /total )*100 
        print (f"    {clase :<15}: {count :>10,} ({pct :>5.1f}%)")
    print (f"    {'TOTAL':<15}: {total :>10,}")





def oversample_gan (X_train ,y_train ,label_encoder ,target_count ):
    """
    Usa los generadores GAN para hacer oversampling de clases minoritarias.
    """
    print ("\n  [GAN] Aplicando oversampling con WGAN-GP v2...")

    X_resampled =X_train .copy ()
    y_resampled =y_train .copy ()

    for clase_idx ,clase in enumerate (label_encoder .classes_ ):
        current_count =np .sum (y_train ==clase_idx )

        if current_count >=target_count :
            print (f"    {clase }: {current_count :,} >= {target_count :,} (sin cambios)")
            continue 

        n_generate =target_count -current_count 
        print (f"    {clase }: {current_count :,} -> {target_count :,} (generando {n_generate :,})")


        folder =CLASS_TO_FOLDER .get (clase )
        if not folder :
            print (f"      [WARN] No hay modelo GAN para {clase }")
            continue 


        generator_path =os .path .join (MODELS_DIR ,folder ,f'generator_{folder }.h5')
        scaler_path =os .path .join (MODELS_DIR ,folder ,'scaler.pkl')

        if not os .path .exists (generator_path ):
            generator_path =os .path .join (MODELS_DIR ,folder ,f'generator_{folder }.h5')
            scaler_path =os .path .join (MODELS_DIR ,folder ,'scaler.pkl')

        if not os .path .exists (generator_path ):
            print (f"      [WARN] No se encontró generador para {clase }")
            continue 


        generator =load_model (generator_path ,compile =False )


        if os .path .exists (scaler_path ):
            with open (scaler_path ,'rb')as f :
                scaler =pickle .load (f )
        else :
            print (f"      [WARN] No se encontró scaler para {clase }, usando datos originales")
            continue 


        noise =np .random .normal (0 ,1 ,(n_generate ,LATENT_DIM ))
        X_synthetic_scaled =generator .predict (noise ,verbose =0 )


        X_synthetic =scaler .inverse_transform (X_synthetic_scaled )


        for i ,col in enumerate (FEATURE_NAMES ):
            if col in COLUMNAS_LOG :
                X_synthetic [:,i ]=np .expm1 (X_synthetic [:,i ])


        X_synthetic =np .clip (X_synthetic ,0 ,None )


        X_resampled =np .vstack ([X_resampled ,X_synthetic ])
        y_resampled =np .concatenate ([y_resampled ,np .full (n_generate ,clase_idx )])


        del generator 
        tf .keras .backend .clear_session ()

    return X_resampled ,y_resampled 


def oversample_smote (X_train ,y_train ,label_encoder ,target_count ):
    """Aplica SMOTE para oversampling"""
    print ("\n  [SMOTE] Aplicando SMOTE...")

    counter =Counter (y_train )
    sampling_strategy ={}
    for clase_idx in counter .keys ():
        if counter [clase_idx ]<target_count :
            sampling_strategy [clase_idx ]=target_count 

    if not sampling_strategy :
        print ("    No se requiere oversampling")
        return X_train ,y_train 

    smote =SMOTE (
    sampling_strategy =sampling_strategy ,
    random_state =RANDOM_STATE ,
    k_neighbors =5 
    )

    X_resampled ,y_resampled =smote .fit_resample (X_train ,y_train )

    for clase_idx ,clase in enumerate (label_encoder .classes_ ):
        old_count =counter [clase_idx ]
        new_count =np .sum (y_resampled ==clase_idx )
        if new_count >old_count :
            print (f"    {clase }: {old_count :,} -> {new_count :,}")

    return X_resampled ,y_resampled 


def oversample_adasyn (X_train ,y_train ,label_encoder ,target_count ):
    """Aplica ADASYN para oversampling"""
    print ("\n  [ADASYN] Aplicando ADASYN...")

    counter =Counter (y_train )
    sampling_strategy ={}
    for clase_idx in counter .keys ():
        if counter [clase_idx ]<target_count :
            sampling_strategy [clase_idx ]=target_count 

    if not sampling_strategy :
        print ("    No se requiere oversampling")
        return X_train ,y_train 

    try :
        adasyn =ADASYN (
        sampling_strategy =sampling_strategy ,
        random_state =RANDOM_STATE ,
        n_neighbors =5 
        )
        X_resampled ,y_resampled =adasyn .fit_resample (X_train ,y_train )

        for clase_idx ,clase in enumerate (label_encoder .classes_ ):
            old_count =counter [clase_idx ]
            new_count =np .sum (y_resampled ==clase_idx )
            if new_count >old_count :
                print (f"    {clase }: {old_count :,} -> {new_count :,}")

        return X_resampled ,y_resampled 

    except Exception as e :
        print (f"    [ERROR] ADASYN falló: {e }")
        print (f"    Usando SMOTE como fallback...")
        return oversample_smote (X_train ,y_train ,label_encoder ,target_count )


def oversample_borderline (X_train ,y_train ,label_encoder ,target_count ):
    """Aplica BorderlineSMOTE para oversampling"""
    print ("\n  [BorderlineSMOTE] Aplicando BorderlineSMOTE...")

    counter =Counter (y_train )
    sampling_strategy ={}
    for clase_idx in counter .keys ():
        if counter [clase_idx ]<target_count :
            sampling_strategy [clase_idx ]=target_count 

    if not sampling_strategy :
        print ("    No se requiere oversampling")
        return X_train ,y_train 

    try :
        borderline =BorderlineSMOTE (
        sampling_strategy =sampling_strategy ,
        random_state =RANDOM_STATE ,
        k_neighbors =5 
        )
        X_resampled ,y_resampled =borderline .fit_resample (X_train ,y_train )

        for clase_idx ,clase in enumerate (label_encoder .classes_ ):
            old_count =counter [clase_idx ]
            new_count =np .sum (y_resampled ==clase_idx )
            if new_count >old_count :
                print (f"    {clase }: {old_count :,} -> {new_count :,}")

        return X_resampled ,y_resampled 

    except Exception as e :
        print (f"    [ERROR] BorderlineSMOTE falló: {e }")
        print (f"    Usando SMOTE como fallback...")
        return oversample_smote (X_train ,y_train ,label_encoder ,target_count )


def oversample_smoteenn (X_train ,y_train ,label_encoder ,target_count ):
    """Aplica SMOTE-ENN (oversampling + undersampling de ruido)"""
    print ("\n  [SMOTE-ENN] Aplicando SMOTE + ENN cleaning...")

    counter =Counter (y_train )
    sampling_strategy ={}
    for clase_idx in counter .keys ():
        if counter [clase_idx ]<target_count :
            sampling_strategy [clase_idx ]=target_count 

    if not sampling_strategy :
        print ("    No se requiere oversampling")
        return X_train ,y_train 

    try :
        smoteenn =SMOTEENN (
        sampling_strategy =sampling_strategy ,
        random_state =RANDOM_STATE 
        )
        X_resampled ,y_resampled =smoteenn .fit_resample (X_train ,y_train )

        print (f"    Muestras antes: {len (y_train ):,}")
        print (f"    Muestras después: {len (y_resampled ):,}")

        for clase_idx ,clase in enumerate (label_encoder .classes_ ):
            old_count =counter [clase_idx ]
            new_count =np .sum (y_resampled ==clase_idx )
            print (f"    {clase }: {old_count :,} -> {new_count :,}")

        return X_resampled ,y_resampled 

    except Exception as e :
        print (f"    [ERROR] SMOTE-ENN falló: {e }")
        print (f"    Usando SMOTE como fallback...")
        return oversample_smote (X_train ,y_train ,label_encoder ,target_count )


def oversample_random (X_train ,y_train ,label_encoder ,target_count ):
    """Aplica Random Oversampling (duplicación aleatoria)"""
    print ("\n  [RandomOverSampler] Aplicando oversampling aleatorio...")

    counter =Counter (y_train )
    sampling_strategy ={}
    for clase_idx in counter .keys ():
        if counter [clase_idx ]<target_count :
            sampling_strategy [clase_idx ]=target_count 

    if not sampling_strategy :
        print ("    No se requiere oversampling")
        return X_train ,y_train 

    ros =RandomOverSampler (
    sampling_strategy =sampling_strategy ,
    random_state =RANDOM_STATE 
    )
    X_resampled ,y_resampled =ros .fit_resample (X_train ,y_train )

    for clase_idx ,clase in enumerate (label_encoder .classes_ ):
        old_count =counter [clase_idx ]
        new_count =np .sum (y_resampled ==clase_idx )
        if new_count >old_count :
            print (f"    {clase }: {old_count :,} -> {new_count :,}")

    return X_resampled ,y_resampled 





def balance_dataset (X ,y ,target_count ):
    """
    Balancea el dataset: recorta clases mayoritarias y mantiene las minoritarias.
    Esto asegura que TODAS las clases tengan ≤ target_count muestras,
    reduciendo drásticamente el tamaño del dataset y el tiempo de entrenamiento.
    """
    X_balanced =[]
    y_balanced =[]

    for clase_idx in np .unique (y ):
        mask =y ==clase_idx 
        X_class =X [mask ]
        y_class =y [mask ]

        if len (X_class )>target_count :

            np .random .seed (42 )
            indices =np .random .choice (len (X_class ),target_count ,replace =False )
            X_balanced .append (X_class [indices ])
            y_balanced .append (y_class [indices ])
        else :
            X_balanced .append (X_class )
            y_balanced .append (y_class )

    return np .vstack (X_balanced ),np .concatenate (y_balanced )





def crear_modelo (modelo_tipo ,n_classes =None ,n_features =None ):
    """
    Crea el modelo clasificador.
    
    Args:
        modelo_tipo: 'rf', 'knn', 'mlp'
        n_classes: número de clases (para MLP)
        n_features: número de features (para MLP)
    """
    if modelo_tipo =='rf':
        return RandomForestClassifier (
        n_estimators =100 ,
        max_depth =15 ,
        random_state =RANDOM_STATE ,
        n_jobs =-1 ,
        class_weight ='balanced'
        )
    elif modelo_tipo =='knn':
        return KNeighborsClassifier (
        n_neighbors =5 ,
        weights ='distance',
        n_jobs =-1 
        )
    elif modelo_tipo =='mlp':

        return None 
    else :
        raise ValueError (f"Modelo desconocido: {modelo_tipo }")


def build_mlp (n_features ,n_classes ):
    """Construye un modelo MLP (Deep Neural Network)"""
    model =Sequential ([
    Dense (256 ,activation ='relu',input_shape =(n_features ,)),
    BatchNormalization (),
    Dropout (0.3 ),
    Dense (128 ,activation ='relu'),
    BatchNormalization (),
    Dropout (0.3 ),
    Dense (64 ,activation ='relu'),
    BatchNormalization (),
    Dropout (0.2 ),
    Dense (n_classes ,activation ='softmax')
    ])
    model .compile (
    optimizer ='adam',
    loss ='categorical_crossentropy',
    metrics =['accuracy']
    )
    return model 





def entrenar_y_evaluar (X_train ,y_train ,X_test ,y_test ,label_encoder ,
nombre_tecnica ,modelo_tipo ,output_dir ):
    """Entrena y evalúa un modelo específico con una técnica de oversampling"""

    modelo_desc ={
    'rf':'RandomForest',
    'knn':'KNN',
    'mlp':'MLP-DNN'
    }

    nombre_completo =f"{nombre_tecnica }_{modelo_tipo }"
    print (f"\n  [{modelo_desc .get (modelo_tipo ,modelo_tipo )}] Entrenando con {nombre_tecnica }...")
    print (f"    Train: {len (X_train ):,} muestras")


    scaler =MinMaxScaler ()
    X_train_scaled =scaler .fit_transform (X_train )
    X_test_scaled =scaler .transform (X_test )


    X_train_scaled =np .nan_to_num (X_train_scaled ,nan =0 ,posinf =0 ,neginf =0 )
    X_test_scaled =np .nan_to_num (X_test_scaled ,nan =0 ,posinf =0 ,neginf =0 )

    n_classes =len (label_encoder .classes_ )
    n_features =X_train_scaled .shape [1 ]


    X_fit ,y_fit =X_train_scaled ,y_train 
    if modelo_tipo =='knn'and len (X_train_scaled )>100000 :
        print (f"    [KNN] Subsample: {len (X_train_scaled ):,} -> 100,000")
        np .random .seed (RANDOM_STATE )
        idx =np .random .choice (len (X_train_scaled ),100000 ,replace =False )
        X_fit ,y_fit =X_train_scaled [idx ],y_train [idx ]

    start_time =datetime .now ()

    if modelo_tipo =='mlp':

        model =build_mlp (n_features ,n_classes )
        y_train_cat =to_categorical (y_fit ,n_classes )

        early_stop =EarlyStopping (
        monitor ='val_loss',patience =5 ,restore_best_weights =True ,verbose =0 
        )

        model .fit (
        X_fit ,y_train_cat ,
        epochs =50 ,
        batch_size =2048 ,
        validation_split =0.15 ,
        callbacks =[early_stop ],
        verbose =0 
        )

        train_time =datetime .now ()-start_time 
        y_pred =np .argmax (model .predict (X_test_scaled ,verbose =0 ),axis =1 )


        del model 
        tf .keras .backend .clear_session ()
    else :

        modelo =crear_modelo (modelo_tipo ,n_classes ,n_features )
        modelo .fit (X_fit ,y_fit )
        train_time =datetime .now ()-start_time 
        y_pred =modelo .predict (X_test_scaled )


    accuracy =accuracy_score (y_test ,y_pred )
    f1_macro =f1_score (y_test ,y_pred ,average ='macro',zero_division =0 )
    f1_weighted =f1_score (y_test ,y_pred ,average ='weighted',zero_division =0 )
    precision_macro =precision_score (y_test ,y_pred ,average ='macro',zero_division =0 )
    recall_macro =recall_score (y_test ,y_pred ,average ='macro',zero_division =0 )

    print (f"    Accuracy: {accuracy :.4f}")
    print (f"    F1-Macro: {f1_macro :.4f}")
    print (f"    F1-Weighted: {f1_weighted :.4f}")
    print (f"    Tiempo entrenamiento: {train_time }")


    report =classification_report (
    y_test ,y_pred ,
    target_names =label_encoder .classes_ ,
    output_dict =True ,
    zero_division =0 
    )
    report_df =pd .DataFrame (report ).transpose ()
    report_df .to_csv (os .path .join (output_dir ,f'{nombre_completo }_classification_report.csv'))


    cm =confusion_matrix (y_test ,y_pred )
    cm_df =pd .DataFrame (cm ,
    index =label_encoder .classes_ ,
    columns =label_encoder .classes_ )
    cm_df .to_csv (os .path .join (output_dir ,f'{nombre_completo }_confusion_matrix.csv'))

    return {
    'Tecnica':nombre_tecnica ,
    'Modelo':modelo_desc .get (modelo_tipo ,modelo_tipo ),
    'Modelo_Key':modelo_tipo ,
    'Train_Samples':len (X_train ),
    'Test_Samples':len (X_test ),
    'Accuracy':accuracy ,
    'F1_Macro':f1_macro ,
    'F1_Weighted':f1_weighted ,
    'Precision_Macro':precision_macro ,
    'Recall_Macro':recall_macro ,
    'Train_Time':str (train_time ),
    'Report':report ,
    'Confusion_Matrix':cm 
    }


def generar_resumen (resultados ,label_encoder ,output_dir ):
    """Genera resumen comparativo de todas las técnicas y modelos"""

    resumen =[]
    resumen .append ("="*130 )
    resumen .append (" "*20 +"COMPARACIÓN DE TÉCNICAS DE OVERSAMPLING (MULTI-MODEL)")
    resumen .append (" "*25 +"GAN vs Métodos Tradicionales - CIC-IDS2017")
    resumen .append (" "*30 +"Modelos: RandomForest | KNN | MLP-DNN")
    resumen .append (" "*35 +f"Fecha: {datetime .now ().strftime ('%Y-%m-%d %H:%M:%S')}")
    resumen .append ("="*130 )


    modelos_unicos =list (dict .fromkeys (r ['Modelo']for r in resultados ))
    tecnicas_unicas =list (dict .fromkeys (r ['Tecnica']for r in resultados ))




    resultados_sorted =sorted (resultados ,key =lambda x :x ['F1_Macro'],reverse =True )

    resumen .append ("\n"+"="*130 )
    resumen .append ("1. RANKING GLOBAL (todas las combinaciones, ordenado por F1-Macro)")
    resumen .append ("="*130 )

    resumen .append ("\n"+"-"*130 )
    header =f"{'Pos':<5} {'Técnica':<20} {'Modelo':<15} │ {'Train':>12} │ {'Accuracy':>10} {'F1-Macro':>10} {'F1-Weight':>10} │ {'Prec':>8} {'Recall':>8}"
    resumen .append (header )
    resumen .append ("-"*130 )

    mejor =resultados_sorted [0 ]
    for i ,r in enumerate (resultados_sorted ,1 ):
        delta =r ['F1_Macro']-mejor ['F1_Macro']
        delta_str =f"({delta :+.4f})"if i >1 else "(best)"
        linea =f"{i :<5} {r ['Tecnica']:<20} {r ['Modelo']:<15} │ {r ['Train_Samples']:>12,} │ {r ['Accuracy']:>10.4f} {r ['F1_Macro']:>10.4f} {r ['F1_Weighted']:>10.4f} │ {r ['Precision_Macro']:>8.4f} {r ['Recall_Macro']:>8.4f} {delta_str }"
        resumen .append (linea )




    resumen .append ("\n\n"+"="*130 )
    resumen .append ("2. RESULTADOS POR MODELO")
    resumen .append ("="*130 )

    for modelo_nombre in modelos_unicos :
        resumen .append (f"\n  ── {modelo_nombre } ──")
        resultados_modelo =[r for r in resultados if r ['Modelo']==modelo_nombre ]
        resultados_modelo_sorted =sorted (resultados_modelo ,key =lambda x :x ['F1_Macro'],reverse =True )

        for i ,r in enumerate (resultados_modelo_sorted ,1 ):
            marca =" ★"if r ['Tecnica']in ('GAN_v2',)else ""
            linea =f"    {i }. {r ['Tecnica']:<20} Acc={r ['Accuracy']:.4f}  F1-M={r ['F1_Macro']:.4f}  F1-W={r ['F1_Weighted']:.4f}{marca }"
            resumen .append (linea )




    resumen .append ("\n\n"+"="*130 )
    resumen .append ("3. COMPARACIÓN GAN vs OTROS MÉTODOS (por clasificador)")
    resumen .append ("="*130 )

    for modelo_nombre in modelos_unicos :
        resultados_modelo =[r for r in resultados if r ['Modelo']==modelo_nombre ]
        gan_result =next ((r for r in resultados_modelo if r ['Tecnica']=='GAN_v2'),None )
        baseline_result =next ((r for r in resultados_modelo if r ['Tecnica']=='Original'),None )
        smote_result =next ((r for r in resultados_modelo if r ['Tecnica']=='SMOTE'),None )

        resumen .append (f"\n  ── {modelo_nombre } ──")

        if gan_result and baseline_result :
            mejora_f1 =gan_result ['F1_Macro']-baseline_result ['F1_Macro']
            mejora_acc =gan_result ['Accuracy']-baseline_result ['Accuracy']
            resumen .append (f"    GAN vs Original:  ΔF1-Macro={mejora_f1 :+.4f}  ΔAcc={mejora_acc :+.4f}")

        if gan_result and smote_result :
            mejora_f1 =gan_result ['F1_Macro']-smote_result ['F1_Macro']
            mejora_acc =gan_result ['Accuracy']-smote_result ['Accuracy']
            resumen .append (f"    GAN vs SMOTE:     ΔF1-Macro={mejora_f1 :+.4f}  ΔAcc={mejora_acc :+.4f}")




    resumen .append ("\n\n"+"="*130 )
    resumen .append (f"4. F1-SCORE POR CLASE - MEJOR COMBINACIÓN: {mejor ['Tecnica']} + {mejor ['Modelo']}")
    resumen .append ("="*130 )

    resumen .append ("\n"+"-"*90 )
    resumen .append (f"{'Clase':<15} │ {'Precision':>10} {'Recall':>10} {'F1-Score':>10} │ {'Support':>10}")
    resumen .append ("-"*90 )

    report =mejor ['Report']
    for clase in label_encoder .classes_ :
        if clase in report :
            r =report [clase ]
            resumen .append (f"{clase :<15} │ {r ['precision']:>10.4f} {r ['recall']:>10.4f} {r ['f1-score']:>10.4f} │ {r ['support']:>10.0f}")




    resumen .append ("\n\n"+"="*130 )
    resumen .append ("5. TABLA CRUZADA: F1-Macro por Técnica × Modelo")
    resumen .append ("="*130 )


    header_line =f"{'Técnica':<20} │ "+" │ ".join ([f"{m :>14}"for m in modelos_unicos ])
    resumen .append ("\n"+"-"*len (header_line ))
    resumen .append (header_line )
    resumen .append ("-"*len (header_line ))

    for tecnica in tecnicas_unicas :
        valores =[]
        for modelo in modelos_unicos :
            r =next ((x for x in resultados if x ['Tecnica']==tecnica and x ['Modelo']==modelo ),None )
            if r :
                valores .append (f"{r ['F1_Macro']:>14.4f}")
            else :
                valores .append (f"{'N/A':>14}")
        resumen .append (f"{tecnica :<20} │ "+" │ ".join (valores ))




    resumen .append ("\n\n"+"="*130 )
    resumen .append ("6. CONCLUSIONES")
    resumen .append ("="*130 )

    resumen .append ("\n┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐")
    resumen .append (f"│  MEJOR COMBINACIÓN: {mejor ['Tecnica']} + {mejor ['Modelo']:<15} F1-Macro = {mejor ['F1_Macro']:.4f}                                              │")
    resumen .append ("├─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤")


    for modelo_nombre in modelos_unicos :
        resultados_modelo =sorted ([r for r in resultados if r ['Modelo']==modelo_nombre ],
        key =lambda x :x ['F1_Macro'],reverse =True )
        gan_result =next ((r for r in resultados_modelo if r ['Tecnica']=='GAN_v2'),None )
        if gan_result :
            gan_pos =next (i for i ,r in enumerate (resultados_modelo ,1 )if r ['Tecnica']=='GAN_v2')
            total =len (resultados_modelo )
            if gan_pos ==1 :
                resumen .append (f"│  ✓ [{modelo_nombre }] GAN en 1º posición - SUPERA a todos los métodos tradicionales                                      │")
            elif gan_pos <=3 :
                resumen .append (f"│  ○ [{modelo_nombre }] GAN en posición {gan_pos }/{total } - Competitivo con métodos tradicionales                                       │")
            else :
                resumen .append (f"│  △ [{modelo_nombre }] GAN en posición {gan_pos }/{total } - Margen de mejora                                                             │")

    resumen .append ("└─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘")


    resumen_texto ="\n".join (resumen )

    with open (os .path .join (output_dir ,'RESUMEN_COMPARATIVO.txt'),'w')as f :
        f .write (resumen_texto )

    print ("\n"+resumen_texto )


    comparativa =[{
    'Tecnica':r ['Tecnica'],
    'Modelo':r ['Modelo'],
    'Train_Samples':r ['Train_Samples'],
    'Accuracy':r ['Accuracy'],
    'F1_Macro':r ['F1_Macro'],
    'F1_Weighted':r ['F1_Weighted'],
    'Precision_Macro':r ['Precision_Macro'],
    'Recall_Macro':r ['Recall_Macro'],
    'Train_Time':r ['Train_Time']
    }for r in resultados ]

    pd .DataFrame (comparativa ).to_csv (
    os .path .join (output_dir ,'comparativa_tecnicas.csv'),
    index =False 
    )

    return resumen_texto 





def main ():
    parser =argparse .ArgumentParser (
    description ='Comparación de técnicas de oversampling: GAN vs métodos tradicionales (multi-modelo)'
    )
    parser .add_argument ('--max-samples',type =int ,default =100000 ,
    help ='Máximo muestras por clase para balanceo (default: 100000)')
    parser .add_argument ('--test-size',type =float ,default =0.3 ,
    help ='Proporción para test set (default: 0.3)')
    parser .add_argument ('--skip-gan',action ='store_true',
    help ='Omitir oversampling con GAN')
    parser .add_argument ('--output',type =str ,default =None ,
    help ='Directorio de salida personalizado')
    parser .add_argument ('--modelos',type =str ,nargs ='+',default =MODELOS ,
    choices =['rf','knn','mlp'],
    help ='Modelos a evaluar (default: todos)')

    args =parser .parse_args ()

    print_header ()


    if args .output :
        output_dir =args .output 
    else :
        timestamp =datetime .now ().strftime ("%Y%m%d_%H%M%S")
        output_dir =os .path .join (OUTPUT_DIR ,f'comparison_multimodel_{timestamp }')

    os .makedirs (output_dir ,exist_ok =True )
    print (f"\n  Resultados se guardarán en: {output_dir }")
    print (f"  Modelos a evaluar: {args .modelos }")





    df =cargar_datos_reales ()
    df_features =preparar_features (df )

    print (f"\n[2] DISTRIBUCIÓN DE CLASES EN DATASET FILTRADO")
    print ("-"*50 )
    print (f"  Clases: {EXPERIMENT_CLASSES }")
    print (f"  Total muestras: {len (df_features ):,}")
    print ("\n  Distribución:")
    for clase in EXPERIMENT_CLASSES :
        count =len (df_features [df_features ['Label_Class']==clase ])
        pct =(count /len (df_features ))*100 
        print (f"    {clase :<15}: {count :>10,} ({pct :>5.1f}%)")


    feature_cols =[c for c in df_features .columns if c !='Label_Class']
    X =df_features [feature_cols ].values 


    label_encoder =LabelEncoder ()
    label_encoder .fit (EXPERIMENT_CLASSES )
    y =label_encoder .transform (df_features ['Label_Class'])

    print (f"\n  Features: {len (feature_cols )}")
    print (f"  Clases codificadas: {list (label_encoder .classes_ )}")





    print (f"\n[3] DIVIDIENDO TRAIN/TEST")
    print ("-"*50 )

    X_train ,X_test ,y_train ,y_test =train_test_split (
    X ,y ,
    test_size =args .test_size ,
    random_state =RANDOM_STATE ,
    stratify =y 
    )

    print (f"  Train: {len (X_train ):,} muestras")
    print (f"  Test: {len (X_test ):,} muestras")

    mostrar_distribucion (y_train ,label_encoder ,"Distribución Train (original)")
    mostrar_distribucion (y_test ,label_encoder ,"Distribución Test")


    counter =Counter (y_train )
    max_class_count =max (counter .values ())
    target_count =min (max_class_count ,args .max_samples )

    print (f"\n  Target para balanceo: {target_count :,} muestras por clase")





    print (f"\n[4] APLICANDO TÉCNICAS DE OVERSAMPLING")
    print ("="*70 )

    tecnicas ={}



    print ("\n"+"-"*70 )
    print ("  [Original] Sin oversampling (baseline, majority capped)")
    X_orig_bal ,y_orig_bal =balance_dataset (X_train ,y_train ,target_count )
    tecnicas ['Original']=(X_orig_bal ,y_orig_bal )
    mostrar_distribucion (y_orig_bal ,label_encoder ,"Distribución")


    if not args .skip_gan :
        print ("\n"+"-"*70 )
        X_gan ,y_gan =oversample_gan (X_train ,y_train ,label_encoder ,target_count )
        X_gan ,y_gan =balance_dataset (X_gan ,y_gan ,target_count )
        tecnicas ['GAN_v2']=(X_gan ,y_gan )
        mostrar_distribucion (y_gan ,label_encoder ,"Distribución GAN (balanced)")


    print ("\n"+"-"*70 )
    X_smote ,y_smote =oversample_smote (X_train ,y_train ,label_encoder ,target_count )
    X_smote ,y_smote =balance_dataset (X_smote ,y_smote ,target_count )
    tecnicas ['SMOTE']=(X_smote ,y_smote )
    mostrar_distribucion (y_smote ,label_encoder ,"Distribución SMOTE (balanced)")


    print ("\n"+"-"*70 )
    X_adasyn ,y_adasyn =oversample_adasyn (X_train ,y_train ,label_encoder ,target_count )
    X_adasyn ,y_adasyn =balance_dataset (X_adasyn ,y_adasyn ,target_count )
    tecnicas ['ADASYN']=(X_adasyn ,y_adasyn )
    mostrar_distribucion (y_adasyn ,label_encoder ,"Distribución ADASYN (balanced)")


    print ("\n"+"-"*70 )
    X_borderline ,y_borderline =oversample_borderline (X_train ,y_train ,label_encoder ,target_count )
    X_borderline ,y_borderline =balance_dataset (X_borderline ,y_borderline ,target_count )
    tecnicas ['BorderlineSMOTE']=(X_borderline ,y_borderline )
    mostrar_distribucion (y_borderline ,label_encoder ,"Distribución BorderlineSMOTE (balanced)")


    print ("\n"+"-"*70 )
    X_smoteenn ,y_smoteenn =oversample_smoteenn (X_train ,y_train ,label_encoder ,target_count )
    X_smoteenn ,y_smoteenn =balance_dataset (X_smoteenn ,y_smoteenn ,target_count )
    tecnicas ['SMOTE_ENN']=(X_smoteenn ,y_smoteenn )
    mostrar_distribucion (y_smoteenn ,label_encoder ,"Distribución SMOTE-ENN (balanced)")


    print ("\n"+"-"*70 )
    X_random ,y_random =oversample_random (X_train ,y_train ,label_encoder ,target_count )
    X_random ,y_random =balance_dataset (X_random ,y_random ,target_count )
    tecnicas ['RandomOverSampler']=(X_random ,y_random )
    mostrar_distribucion (y_random ,label_encoder ,"Distribución Random (balanced)")





    print (f"\n\n[5] ENTRENAMIENTO Y EVALUACIÓN (MULTI-MODEL)")
    print ("="*70 )
    print (f"  Modelos: {', '.join (args .modelos )}")
    print (f"  Técnicas: {list (tecnicas .keys ())}")
    print (f"  Total combinaciones: {len (tecnicas )*len (args .modelos )}")
    print (f"  Test set: {len (X_test ):,} muestras (mismo para todas las combinaciones)")

    resultados =[]

    for nombre_tecnica ,(X_t ,y_t )in tecnicas .items ():
        for modelo_tipo in args .modelos :
            print (f"\n"+"-"*70 )
            resultado =entrenar_y_evaluar (
            X_t ,y_t ,X_test ,y_test ,
            label_encoder ,nombre_tecnica ,modelo_tipo ,output_dir 
            )
            resultados .append (resultado )





    print (f"\n\n[6] GENERANDO RESUMEN COMPARATIVO")
    print ("="*70 )

    generar_resumen (resultados ,label_encoder ,output_dir )

    print (f"\n\n{'='*100 }")
    print (f"  EXPERIMENTO COMPLETADO")
    print (f"  Resultados guardados en: {output_dir }")
    print (f"{'='*100 }")


if __name__ =="__main__":
    main ()
