"""
Oversampling Techniques Comparison — Edge-IIoT 2022
                         MULTI-MODEL EVALUATION (LightGBM, KNN, RandomForest, MLP-DNN)

Compares different oversampling techniques for balancing Edge-IIoT dataset:
    1. Original (no oversampling) - baseline
    2. GAN (WGAN-GP trained on Edge-IIoT)
    3. SMOTE
    4. ADASYN
    5. BorderlineSMOTE
    6. SMOTE-ENN (oversampling + cleaning)
    7. Random Oversampling

For each technique, 3 different classifiers are evaluated:
    - RandomForest    (bagging - trees)
    - KNN             (k-nearest neighbors)
    - MLP-DNN         (deep neural network)

Usage:
        python 03_oversampling_edgeiiot.py
        python 03_oversampling_edgeiiot.py --skip-gan
        python 03_oversampling_edgeiiot.py --max-samples 50000
"""

import os 
import sys 




GPU_ID ='3'
os .environ ['CUDA_VISIBLE_DEVICES']=GPU_ID 
os .environ ['TF_CPP_MIN_LOG_LEVEL']='2'

import pickle 
import argparse 
import numpy as np 
import pandas as pd 
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


_tmpdir =os .path .join (os .path .expanduser ('~'),'.cache','tmp')
os .makedirs (_tmpdir ,exist_ok =True )
os .environ ['TMPDIR']=_tmpdir 
os .environ ['TEMP']=_tmpdir 
os .environ ['TMP']=_tmpdir 




from dotenv import load_dotenv 
load_dotenv ()
REAL_DATA_PATH =os .getenv ('DATA_EDGEIIOT_PATH','./data/Edge-IIoT.csv')
MODELS_DIR =os .getenv ('OUTPUT_MODELS_DIR','./outputs/models')
OUTPUT_DIR =os .getenv ('OUTPUT_RESULTS_DIR','./outputs/results')
RANDOM_STATE =42 
LATENT_DIM =100 
LABEL_COLUMN ='Attack_type'



D_FEATURES =58 
RHO_HIGH =2.5 
RHO_LOW =0.8 
THETA_HIGH =8410 
THETA_LOW =2692 


VALID_CLASSES =[
'Normal','DDoS_UDP','DDoS_ICMP','SQL_injection','Password',
'Vulnerability_scanner','DDoS_TCP','DDoS_HTTP','Uploading',
'Backdoor','Port_Scanning','XSS','Ransomware','MITM','Fingerprinting'
]

CLASS_TO_FOLDER ={c :c .lower ()for c in VALID_CLASSES }


FEATURES_BASE =[
'arp.opcode','arp.hw.size',
'icmp.checksum','icmp.seq_le','icmp.transmit_timestamp','icmp.unused',
'http.file_data','http.content_length','http.request.uri.query',
'http.request.method','http.referer','http.request.full_uri',
'http.request.version','http.response','http.tls_port',
'tcp.ack','tcp.ack_raw','tcp.checksum',
'tcp.connection.fin','tcp.connection.rst','tcp.connection.syn',
'tcp.connection.synack','tcp.dstport','tcp.flags','tcp.flags.ack',
'tcp.len','tcp.seq','tcp.srcport',
'udp.port','udp.stream','udp.time_delta',
'dns.qry.name','dns.qry.name.len','dns.qry.qu','dns.qry.type',
'dns.retransmission','dns.retransmit_request','dns.retransmit_request_in',
'mqtt.conflag.cleansess','mqtt.conflags','mqtt.hdrflags',
'mqtt.len','mqtt.msg_decoded_as','mqtt.msgtype',
'mqtt.proto_len','mqtt.topic_len','mqtt.ver',
'mbtcp.len','mbtcp.trans_id','mbtcp.unit_id',
]


FEATURE_NAMES =FEATURES_BASE +[
'Src_IP_1','Src_IP_2','Src_IP_3','Src_IP_4',
'Dst_IP_1','Dst_IP_2','Dst_IP_3','Dst_IP_4',
]

COLUMNAS_LOG =[
'tcp.ack_raw','tcp.checksum','tcp.seq','tcp.srcport','tcp.dstport',
'tcp.len','udp.port','http.content_length','http.file_data',
'icmp.checksum',
]


DROP_COLUMNS =[
'frame.time',
'arp.dst.proto_ipv4','arp.src.proto_ipv4',
'tcp.options','tcp.payload',
'mqtt.conack.flags','mqtt.msg','mqtt.protoname','mqtt.topic',
'Attack_label','Attack_type'
]



GAN_OVERGEN_FACTOR =1.5 
GAN_QUALITY_PERCENTILE =85 


MODELOS =['rf','knn','mlp']





def print_header ():
    print ("="*100 )
    print ("  EXPERIMENTO: COMPARACIÓN DE TÉCNICAS DE OVERSAMPLING - EDGE-IIoT 2022 (MULTI-MODEL)")
    print ("  GAN (WGAN-GP) vs SMOTE vs ADASYN vs BorderlineSMOTE vs SMOTE-ENN")
    print ("  AMD-GAN Adaptive Training Engine | θ_high={}, θ_low={}".format (THETA_HIGH ,THETA_LOW ))
    print ("  Modelos: RandomForest | KNN | MLP-DNN")
    print ("="*100 )


def cargar_datos_reales ():
    """Carga el dataset Edge-IIoT 2022."""
    print ("\n[1] CARGANDO DATOS REALES (Edge-IIoT 2022)")
    print ("-"*50 )

    file_size =os .path .getsize (REAL_DATA_PATH )
    print (f"  Archivo: {REAL_DATA_PATH }")
    print (f"  Tamaño: {file_size /(1024 *1024 ):.2f} MB")

    start_time =datetime .now ()
    df =pd .read_csv (REAL_DATA_PATH ,low_memory =False )
    df .columns =df .columns .str .strip ()
    print (f"  Cargado en {datetime .now ()-start_time }")
    print (f"  Total registros: {len (df ):,}")

    return df 


def preparar_features (df :pd .DataFrame )->pd .DataFrame :
    """Prepara features para el modelo, incluyendo expansión de IPs a octetos."""
    df =df .copy ()

    df ['Label_Class']=df [LABEL_COLUMN ]


    df =df [df ['Label_Class'].isin (VALID_CLASSES )].copy ()


    features_available =[f for f in FEATURES_BASE if f in df .columns ]
    features_df =df [features_available ].copy ()


    for col in features_df .columns :
        features_df [col ]=pd .to_numeric (features_df [col ],errors ='coerce').fillna (0 )


    if 'ip.src_host'in df .columns :
        octetos =df ['ip.src_host'].astype (str ).str .split ('.',expand =True )
        for i in range (4 ):
            features_df [f'Src_IP_{i +1 }']=pd .to_numeric (octetos [i ],errors ='coerce').fillna (0 ).astype (int )
    else :
        for i in range (4 ):
            features_df [f'Src_IP_{i +1 }']=0 

    if 'ip.dst_host'in df .columns :
        octetos =df ['ip.dst_host'].astype (str ).str .split ('.',expand =True )
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
    print (f"\n  {titulo }:")
    counter =Counter (y )
    total =len (y )
    for label_idx in sorted (counter .keys ()):
        clase =label_encoder .inverse_transform ([label_idx ])[0 ]
        count =counter [label_idx ]
        pct =(count /total )*100 
        print (f"    {clase :<25}: {count :>10,} ({pct :>5.1f}%)")
    print (f"    {'TOTAL':<25}: {total :>10,}")





def _quality_filter_gan_samples (X_synthetic ,X_real_class ):
    """
    Quality-aware filtering: keep generated samples closest to the
    real-class centroid (Mahalanobis-lite via L2 on normalized space).
    This discards outlier / low-quality synthetic rows.
    """
    if len (X_synthetic )==0 or len (X_real_class )==0 :
        return X_synthetic 

    mu =X_real_class .mean (axis =0 )
    std =X_real_class .std (axis =0 )+1e-8 
    X_norm =(X_synthetic -mu )/std 

    distances =np .linalg .norm (X_norm ,axis =1 )
    threshold =np .percentile (distances ,GAN_QUALITY_PERCENTILE )
    mask =distances <=threshold 
    return X_synthetic [mask ]


def oversample_gan (X_train ,y_train ,label_encoder ,target_count ):
    """Usa los generadores GAN para hacer oversampling de clases minoritarias
       con filtrado de calidad (Quality-Aware Generation)."""
    print ("\n  [GAN] Aplicando oversampling con WGAN-GP Edge-IIoT (quality-aware)...")

    X_resampled =X_train .copy ()
    y_resampled =y_train .copy ()

    for clase_idx ,clase in enumerate (label_encoder .classes_ ):
        current_count =np .sum (y_train ==clase_idx )

        if current_count >=target_count :
            print (f"    {clase }: {current_count :,} >= {target_count :,} (sin cambios)")
            continue 

        n_generate =target_count -current_count 
        n_generate_over =int (n_generate *GAN_OVERGEN_FACTOR )
        print (f"    {clase }: {current_count :,} -> {target_count :,} (generando {n_generate_over :,}, filtrando a ~{n_generate :,})")

        folder =CLASS_TO_FOLDER .get (clase )
        if not folder :
            print (f"      [WARN] No hay modelo GAN para {clase }")
            continue 

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
            print (f"      [WARN] No se encontró scaler para {clase }")
            continue 

        noise =np .random .normal (0 ,1 ,(n_generate_over ,LATENT_DIM ))
        X_synthetic_scaled =generator .predict (noise ,verbose =0 )

        X_synthetic =scaler .inverse_transform (X_synthetic_scaled )


        for i ,col in enumerate (FEATURE_NAMES ):
            if col in COLUMNAS_LOG and i <X_synthetic .shape [1 ]:
                X_synthetic [:,i ]=np .expm1 (X_synthetic [:,i ])

        X_synthetic =np .clip (X_synthetic ,0 ,None )


        X_real_class =X_train [y_train ==clase_idx ]
        X_synthetic =_quality_filter_gan_samples (X_synthetic ,X_real_class )


        if len (X_synthetic )>n_generate :
            X_synthetic =X_synthetic [:n_generate ]

        print (f"      Muestras finales tras filtrado: {len (X_synthetic ):,}")

        X_resampled =np .vstack ([X_resampled ,X_synthetic ])
        y_resampled =np .concatenate ([y_resampled ,np .full (len (X_synthetic ),clase_idx )])

        del generator 
        tf .keras .backend .clear_session ()

    return X_resampled ,y_resampled 


def oversample_smote (X_train ,y_train ,label_encoder ,target_count ):
    print ("\n  [SMOTE] Aplicando SMOTE...")
    counter =Counter (y_train )
    sampling_strategy ={}
    for k in counter :
        if counter [k ]<target_count :
            sampling_strategy [k ]=target_count 

    if not sampling_strategy :
        print ("    No se requiere oversampling")
        return X_train ,y_train 

    min_count =min (counter [k ]for k in sampling_strategy )
    k_neighbors =min (5 ,min_count -1 )if min_count >1 else 1 

    smote =SMOTE (sampling_strategy =sampling_strategy ,random_state =RANDOM_STATE ,
    k_neighbors =max (1 ,k_neighbors ))
    X_resampled ,y_resampled =smote .fit_resample (X_train ,y_train )

    for clase_idx ,clase in enumerate (label_encoder .classes_ ):
        old_count =counter .get (clase_idx ,0 )
        new_count =np .sum (y_resampled ==clase_idx )
        if new_count >old_count :
            print (f"    {clase }: {old_count :,} -> {new_count :,}")

    return X_resampled ,y_resampled 


def oversample_adasyn (X_train ,y_train ,label_encoder ,target_count ):
    print ("\n  [ADASYN] Aplicando ADASYN...")
    counter =Counter (y_train )
    sampling_strategy ={k :target_count for k in counter if counter [k ]<target_count }
    if not sampling_strategy :
        return X_train ,y_train 

    min_count =min (counter [k ]for k in sampling_strategy )
    n_neighbors =min (5 ,min_count -1 )if min_count >1 else 1 

    try :
        adasyn =ADASYN (sampling_strategy =sampling_strategy ,random_state =RANDOM_STATE ,
        n_neighbors =max (1 ,n_neighbors ))
        X_resampled ,y_resampled =adasyn .fit_resample (X_train ,y_train )
        for clase_idx ,clase in enumerate (label_encoder .classes_ ):
            old_count =counter .get (clase_idx ,0 )
            new_count =np .sum (y_resampled ==clase_idx )
            if new_count >old_count :
                print (f"    {clase }: {old_count :,} -> {new_count :,}")
        return X_resampled ,y_resampled 
    except Exception as e :
        print (f"    [ERROR] ADASYN falló: {e }")
        return oversample_smote (X_train ,y_train ,label_encoder ,target_count )


def oversample_borderline (X_train ,y_train ,label_encoder ,target_count ):
    print ("\n  [BorderlineSMOTE] Aplicando BorderlineSMOTE...")
    counter =Counter (y_train )
    sampling_strategy ={k :target_count for k in counter if counter [k ]<target_count }
    if not sampling_strategy :
        return X_train ,y_train 

    min_count =min (counter [k ]for k in sampling_strategy )
    k_neighbors =min (5 ,min_count -1 )if min_count >1 else 1 

    try :
        borderline =BorderlineSMOTE (sampling_strategy =sampling_strategy ,
        random_state =RANDOM_STATE ,
        k_neighbors =max (1 ,k_neighbors ))
        X_resampled ,y_resampled =borderline .fit_resample (X_train ,y_train )
        for clase_idx ,clase in enumerate (label_encoder .classes_ ):
            old_count =counter .get (clase_idx ,0 )
            new_count =np .sum (y_resampled ==clase_idx )
            if new_count >old_count :
                print (f"    {clase }: {old_count :,} -> {new_count :,}")
        return X_resampled ,y_resampled 
    except Exception as e :
        print (f"    [ERROR] BorderlineSMOTE falló: {e }")
        return oversample_smote (X_train ,y_train ,label_encoder ,target_count )


def oversample_smoteenn (X_train ,y_train ,label_encoder ,target_count ):
    print ("\n  [SMOTE-ENN] Aplicando SMOTE + ENN cleaning...")
    counter =Counter (y_train )
    sampling_strategy ={k :target_count for k in counter if counter [k ]<target_count }
    if not sampling_strategy :
        return X_train ,y_train 

    min_count =min (counter [k ]for k in sampling_strategy )
    k_neighbors =min (5 ,min_count -1 )if min_count >1 else 1 

    try :
        smote =SMOTE (sampling_strategy =sampling_strategy ,random_state =RANDOM_STATE ,
        k_neighbors =max (1 ,k_neighbors ))
        smoteenn =SMOTEENN (smote =smote ,random_state =RANDOM_STATE )
        X_resampled ,y_resampled =smoteenn .fit_resample (X_train ,y_train )
        print (f"    Muestras antes: {len (y_train ):,}")
        print (f"    Muestras después: {len (y_resampled ):,}")
        for clase_idx ,clase in enumerate (label_encoder .classes_ ):
            old_count =counter .get (clase_idx ,0 )
            new_count =np .sum (y_resampled ==clase_idx )
            print (f"    {clase }: {old_count :,} -> {new_count :,}")
        return X_resampled ,y_resampled 
    except Exception as e :
        print (f"    [ERROR] SMOTE-ENN falló: {e }")
        return oversample_smote (X_train ,y_train ,label_encoder ,target_count )


def oversample_random (X_train ,y_train ,label_encoder ,target_count ):
    print ("\n  [RandomOverSampler] Aplicando oversampling aleatorio...")
    counter =Counter (y_train )
    sampling_strategy ={k :target_count for k in counter if counter [k ]<target_count }
    if not sampling_strategy :
        return X_train ,y_train 

    ros =RandomOverSampler (sampling_strategy =sampling_strategy ,random_state =RANDOM_STATE )
    X_resampled ,y_resampled =ros .fit_resample (X_train ,y_train )
    for clase_idx ,clase in enumerate (label_encoder .classes_ ):
        old_count =counter .get (clase_idx ,0 )
        new_count =np .sum (y_resampled ==clase_idx )
        if new_count >old_count :
            print (f"    {clase }: {old_count :,} -> {new_count :,}")
    return X_resampled ,y_resampled 





def balance_dataset (X ,y ,target_count ):
    """Cap ALL classes at target_count for fair/fast comparison."""
    X_balanced ,y_balanced =[],[]
    for clase_idx in np .unique (y ):
        mask =y ==clase_idx 
        X_c ,y_c =X [mask ],y [mask ]
        if len (X_c )>target_count :
            np .random .seed (42 )
            idx =np .random .choice (len (X_c ),target_count ,replace =False )
            X_balanced .append (X_c [idx ]);y_balanced .append (y_c [idx ])
        else :
            X_balanced .append (X_c );y_balanced .append (y_c )
    return np .vstack (X_balanced ),np .concatenate (y_balanced )





def crear_modelo (modelo_tipo ,n_classes =None ,n_features =None ):
    """Crea el modelo clasificador."""
    if modelo_tipo =='rf':
        return RandomForestClassifier (
        n_estimators =100 ,max_depth =15 ,
        random_state =RANDOM_STATE ,n_jobs =-1 ,
        class_weight ='balanced'
        )
    elif modelo_tipo =='knn':
        return KNeighborsClassifier (
        n_neighbors =5 ,weights ='distance',n_jobs =-1 
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
    """Entrena y evalúa un modelo específico con una técnica de oversampling."""

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
        epochs =50 ,batch_size =2048 ,
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

    report =classification_report (y_test ,y_pred ,target_names =label_encoder .classes_ ,
    output_dict =True ,zero_division =0 )
    report_df =pd .DataFrame (report ).transpose ()
    report_df .to_csv (os .path .join (output_dir ,f'{nombre_completo }_classification_report.csv'))

    cm =confusion_matrix (y_test ,y_pred )
    cm_df =pd .DataFrame (cm ,index =label_encoder .classes_ ,columns =label_encoder .classes_ )
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
    """Genera resumen comparativo de todas las técnicas y modelos."""
    resumen =[]
    resumen .append ("="*130 )
    resumen .append (" "*10 +"COMPARACIÓN DE TÉCNICAS DE OVERSAMPLING - EDGE-IIoT 2022 (MULTI-MODEL)")
    resumen .append (" "*20 +"GAN vs Métodos Tradicionales")
    resumen .append (" "*15 +"AMD-GAN Adaptive Training Engine | θ_high={}, θ_low={}".format (THETA_HIGH ,THETA_LOW ))
    resumen .append (" "*20 +"Modelos: RandomForest | KNN | MLP-DNN")
    resumen .append (" "*30 +f"Fecha: {datetime .now ().strftime ('%Y-%m-%d %H:%M:%S')}")
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
        resultados_modelo =sorted (
        [r for r in resultados if r ['Modelo']==modelo_nombre ],
        key =lambda x :x ['F1_Macro'],reverse =True 
        )
        for i ,r in enumerate (resultados_modelo ,1 ):
            marca =" ★"if r ['Tecnica']=='GAN'else ""
            linea =f"    {i }. {r ['Tecnica']:<20} Acc={r ['Accuracy']:.4f}  F1-M={r ['F1_Macro']:.4f}  F1-W={r ['F1_Weighted']:.4f}{marca }"
            resumen .append (linea )


    resumen .append ("\n\n"+"="*130 )
    resumen .append ("3. COMPARACIÓN GAN vs OTROS MÉTODOS (por clasificador)")
    resumen .append ("="*130 )

    for modelo_nombre in modelos_unicos :
        resultados_modelo =[r for r in resultados if r ['Modelo']==modelo_nombre ]
        gan_result =next ((r for r in resultados_modelo if r ['Tecnica']=='GAN'),None )
        baseline_result =next ((r for r in resultados_modelo if r ['Tecnica']=='Original'),None )
        smote_result =next ((r for r in resultados_modelo if r ['Tecnica']=='SMOTE'),None )

        resumen .append (f"\n  ── {modelo_nombre } ──")
        if gan_result and baseline_result :
            resumen .append (f"    GAN vs Original:  ΔF1-Macro={gan_result ['F1_Macro']-baseline_result ['F1_Macro']:+.4f}  ΔAcc={gan_result ['Accuracy']-baseline_result ['Accuracy']:+.4f}")
        if gan_result and smote_result :
            resumen .append (f"    GAN vs SMOTE:     ΔF1-Macro={gan_result ['F1_Macro']-smote_result ['F1_Macro']:+.4f}  ΔAcc={gan_result ['Accuracy']-smote_result ['Accuracy']:+.4f}")


    resumen .append ("\n\n"+"="*130 )
    resumen .append ("4. ANÁLISIS POR CATEGORÍA AMD-GAN (θ_high={}, θ_low={})".format (THETA_HIGH ,THETA_LOW ))
    resumen .append ("="*130 )

    for categoria ,desc in [
    ('VERY_SMALL',f'< {THETA_LOW } muestras (MITM, Fingerprinting)'),
    ('LARGE',f'>= {THETA_HIGH } muestras (resto)')
    ]:
        resumen .append (f"\n  {categoria }: {desc }")

        best_model =modelos_unicos [0 ]if modelos_unicos else None 
        if best_model :
            gan_result =next ((r for r in resultados if r ['Tecnica']=='GAN'and r ['Modelo']==best_model ),None )
            if gan_result :
                for clase in label_encoder .classes_ :
                    if clase in gan_result ['Report']:
                        f1_val =gan_result ['Report'][clase ]['f1-score']
                        resumen .append (f"    {clase }: F1={f1_val :.4f}")


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

    resumen .append (f"\n  MEJOR COMBINACIÓN: {mejor ['Tecnica']} + {mejor ['Modelo']} (F1-Macro = {mejor ['F1_Macro']:.4f})")

    for modelo_nombre in modelos_unicos :
        resultados_modelo =sorted (
        [r for r in resultados if r ['Modelo']==modelo_nombre ],
        key =lambda x :x ['F1_Macro'],reverse =True 
        )
        gan_result =next ((r for r in resultados_modelo if r ['Tecnica']=='GAN'),None )
        if gan_result :
            gan_pos =next (i for i ,r in enumerate (resultados_modelo ,1 )if r ['Tecnica']=='GAN')
            total =len (resultados_modelo )
            if gan_pos ==1 :
                resumen .append (f"  ✓ [{modelo_nombre }] GAN en 1º posición - SUPERA a todos los métodos tradicionales")
            elif gan_pos <=3 :
                resumen .append (f"  ○ [{modelo_nombre }] GAN en posición {gan_pos }/{total } - Competitivo con métodos tradicionales")
            else :
                resumen .append (f"  △ [{modelo_nombre }] GAN en posición {gan_pos }/{total } - Margen de mejora")

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

    pd .DataFrame (comparativa ).to_csv (os .path .join (output_dir ,'comparativa_tecnicas.csv'),index =False )

    return resumen_texto 





def main ():
    parser =argparse .ArgumentParser (
    description ='Comparación de técnicas de oversampling para Edge-IIoT 2022 (multi-modelo)'
    )
    parser .add_argument ('--max-samples',type =int ,default =100000 ,
    help ='Máximo muestras por clase para balanceo (default: 100000)')
    parser .add_argument ('--test-size',type =float ,default =0.3 )
    parser .add_argument ('--skip-gan',action ='store_true',
    help ='Omitir técnica GAN')
    parser .add_argument ('--output',type =str ,default =None )
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


    class_counts =df_features ['Label_Class'].value_counts ()
    clases_experimento =[c for c in VALID_CLASSES if c in class_counts .index ]

    print (f"\n[2] DISTRIBUCIÓN DE CLASES")
    print ("-"*50 )
    print (f"  Clases: {clases_experimento }")
    print (f"  Total muestras: {len (df_features ):,}")
    for clase in clases_experimento :
        count =class_counts .get (clase ,0 )
        pct =(count /len (df_features ))*100 
        if count <THETA_LOW :
            cat ="VERY_SMALL"
        elif count <THETA_HIGH :
            cat ="SMALL"
        else :
            cat ="LARGE"
        print (f"    {clase :<25}: {count :>10,} ({pct :>5.1f}%) [{cat }]")

    feature_cols =[c for c in df_features .columns if c !='Label_Class']
    X =df_features [feature_cols ].values 

    label_encoder =LabelEncoder ()
    label_encoder .fit (clases_experimento )
    y =label_encoder .transform (df_features ['Label_Class'])

    print (f"\n  Features: {len (feature_cols )}")
    print (f"  Clases codificadas: {list (label_encoder .classes_ )}")


    print (f"\n[3] DIVIDIENDO TRAIN/TEST")
    print ("-"*50 )

    X_train ,X_test ,y_train ,y_test =train_test_split (
    X ,y ,test_size =args .test_size ,random_state =RANDOM_STATE ,stratify =y 
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
        tecnicas ['GAN']=(X_gan ,y_gan )
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
    X_bl ,y_bl =oversample_borderline (X_train ,y_train ,label_encoder ,target_count )
    X_bl ,y_bl =balance_dataset (X_bl ,y_bl ,target_count )
    tecnicas ['BorderlineSMOTE']=(X_bl ,y_bl )
    mostrar_distribucion (y_bl ,label_encoder ,"Distribución BorderlineSMOTE (balanced)")


    print ("\n"+"-"*70 )
    X_senn ,y_senn =oversample_smoteenn (X_train ,y_train ,label_encoder ,target_count )
    X_senn ,y_senn =balance_dataset (X_senn ,y_senn ,target_count )
    tecnicas ['SMOTE_ENN']=(X_senn ,y_senn )
    mostrar_distribucion (y_senn ,label_encoder ,"Distribución SMOTE-ENN (balanced)")


    print ("\n"+"-"*70 )
    X_rnd ,y_rnd =oversample_random (X_train ,y_train ,label_encoder ,target_count )
    X_rnd ,y_rnd =balance_dataset (X_rnd ,y_rnd ,target_count )
    tecnicas ['RandomOverSampler']=(X_rnd ,y_rnd )
    mostrar_distribucion (y_rnd ,label_encoder ,"Distribución Random (balanced)")


    print (f"\n\n[5] ENTRENAMIENTO Y EVALUACIÓN (MULTI-MODEL)")
    print ("="*70 )
    print (f"  Modelos: {', '.join (args .modelos )}")
    print (f"  Técnicas: {list (tecnicas .keys ())}")
    print (f"  Total combinaciones: {len (tecnicas )*len (args .modelos )}")
    print (f"  Test set: {len (X_test ):,} muestras")

    resultados =[]
    for nombre ,(X_t ,y_t )in tecnicas .items ():
        for modelo_tipo in args .modelos :
            print (f"\n"+"-"*70 )
            resultado =entrenar_y_evaluar (
            X_t ,y_t ,X_test ,y_test ,
            label_encoder ,nombre ,modelo_tipo ,output_dir 
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
