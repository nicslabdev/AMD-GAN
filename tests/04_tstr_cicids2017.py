"""
Train on Synthetic, Test on Real - Multi-class and Binary Evaluation

Compares four scenarios:
1. TSTR Multi-class: Train on synthetic, test on real data
2. TRTR Multi-class: Train on real, test on real (baseline)
3. TSTR Binary: Train on synthetic (BENIGN vs ATTACK), test on real
4. TRTR Binary: Train on real, test on real (baseline)
"""

import os 
import sys 
import argparse 
import numpy as np 
import pandas as pd 
import polars as pl 
from datetime import datetime 
from sklearn .preprocessing import MinMaxScaler ,LabelEncoder 
from sklearn .model_selection import train_test_split 
from sklearn .metrics import classification_report ,confusion_matrix ,accuracy_score ,f1_score 
from sklearn .ensemble import (
RandomForestClassifier ,ExtraTreesClassifier ,
AdaBoostClassifier ,BaggingClassifier ,
StackingClassifier ,VotingClassifier 
)
from sklearn .dummy import DummyClassifier 
from sklearn .linear_model import LogisticRegression 
from sklearn .neighbors import KNeighborsClassifier 
from sklearn .tree import DecisionTreeClassifier 
from sklearn .naive_bayes import GaussianNB 
from sklearn .neural_network import MLPClassifier 
from lightgbm import LGBMClassifier 
from xgboost import XGBClassifier 
import warnings 
warnings .filterwarnings ('ignore')

from dotenv import load_dotenv 


load_dotenv ()


REAL_DATA_PATH =os .getenv ('DATA_CICIDS2017_PATH','./data/CIC-IDS2017.csv')
SYNTHETIC_DATASETS_DIR =os .getenv ('OUTPUT_SYNTHETIC_DIR','./outputs/synthetic_data')
OUTPUT_BASE_DIR =os .getenv ('OUTPUT_RESULTS_DIR','./outputs/results')
RANDOM_STATE =42 




def listar_datasets_disponibles ():
    """Lista todos los datasets sintéticos disponibles"""
    if not os .path .exists (SYNTHETIC_DATASETS_DIR ):
        print (f"[ERROR] No existe el directorio: {SYNTHETIC_DATASETS_DIR }")
        return []

    datasets =[]
    for f in os .listdir (SYNTHETIC_DATASETS_DIR ):
        if f .endswith ('.csv')and not f .endswith ('_config.json'):
            csv_path =os .path .join (SYNTHETIC_DATASETS_DIR ,f )
            config_path =csv_path .replace ('.csv','_config.json')


            try :

                df_sample =pd .read_csv (csv_path ,nrows =5 )
                n_rows =sum (1 for _ in open (csv_path ))-1 


                config ={}
                if os .path .exists (config_path ):
                    import json 
                    with open (config_path ,'r')as cf :
                        config =json .load (cf )

                datasets .append ({
                'nombre':f .replace ('.csv',''),
                'archivo':f ,
                'path':csv_path ,
                'filas':n_rows ,
                'config':config ,
                'clases':config .get ('samples_per_class',{})
                })
            except Exception as e :
                print (f"  [WARN] Error leyendo {f }: {e }")

    return datasets 


def mostrar_datasets_disponibles (datasets ):
    """Muestra los datasets disponibles de forma formateada"""
    print ("\n"+"="*80 )
    print ("DATASETS SINTÉTICOS DISPONIBLES")
    print ("="*80 )

    if not datasets :
        print ("  No hay datasets disponibles en:",SYNTHETIC_DATASETS_DIR )
        print ("  Genera uno con: python generate_synthetic_dataset.py --interactive")
        return 

    for i ,ds in enumerate (datasets ,1 ):
        print (f"\n  [{i }] {ds ['nombre']}")
        print (f"      Archivo: {ds ['archivo']}")
        print (f"      Total muestras: {ds ['filas']:,}")
        if ds ['clases']:
            print (f"      Clases: {', '.join (ds ['clases'].keys ())}")
            for clase ,n in ds ['clases'].items ():
                print (f"        - {clase }: {n :,}")

    print ("\n"+"="*80 )


def seleccionar_dataset_interactivo (datasets ):
    """Permite al usuario seleccionar un dataset interactivamente"""
    mostrar_datasets_disponibles (datasets )

    if not datasets :
        return None 

    while True :
        try :
            inp =input ("\nSelecciona un dataset (número o nombre): ").strip ()


            if inp .isdigit ():
                idx =int (inp )-1 
                if 0 <=idx <len (datasets ):
                    return datasets [idx ]
                print (f"  [!] Número debe estar entre 1 y {len (datasets )}")
                continue 


            for ds in datasets :
                if ds ['nombre'].lower ()==inp .lower ()or ds ['archivo'].lower ()==inp .lower ():
                    return ds 

            print (f"  [!] Dataset '{inp }' no encontrado")

        except KeyboardInterrupt :
            print ("\n\nCancelado.")
            return None 


def cargar_dataset_sintetico (dataset_info )->pd .DataFrame :
    """Carga un dataset sintético y muestra su información"""
    print ("\n"+"="*70 )
    print ("CARGANDO DATASET SINTÉTICO")
    print ("="*70 )

    print (f"  Archivo: {dataset_info ['archivo']}")

    df =pd .read_csv (dataset_info ['path'])

    print (f"  Total muestras: {len (df ):,}")


    if 'Attack Type'in df .columns :
        label_col ='Attack Type'
    elif 'Label_Class'in df .columns :
        label_col ='Label_Class'
    else :
        raise ValueError ("No se encontró columna de clase (Attack Type o Label_Class)")


    df ['Label_Class']=df [label_col ]


    print (f"\n  Distribución de clases:")
    for clase ,count in df ['Label_Class'].value_counts ().items ():
        pct =(count /len (df ))*100 
        print (f"    {clase }: {count :,} ({pct :.1f}%)")

    return df 


def cargar_datos_reales ()->pd .DataFrame :
    """Carga el dataset real"""
    print ("\n"+"="*70 )
    print ("CARGANDO DATOS REALES")
    print ("="*70 )

    file_size =os .path .getsize (REAL_DATA_PATH )
    print (f"  Tamaño del archivo: {file_size /(1024 *1024 ):.2f} MB")

    start_time =datetime .now ()
    df_pl =pl .read_csv (REAL_DATA_PATH ,low_memory =False )
    df =df_pl .to_pandas ()
    print (f"  Datos cargados en {datetime .now ()-start_time }")

    return df 


def preparar_features_reales (df :pd .DataFrame )->pd .DataFrame :
    """
    Prepara el dataset real con la misma estructura de features que el sintético
    """
    FEATURES_BASE =[
    'Source IP','Destination IP',
    'Source Port','Destination Port','Protocol',
    'Total Fwd Packets','Total Backward Packets',
    'Total Length of Fwd Packets','Total Length of Bwd Packets',
    'Flow Duration','Flow IAT Mean','Flow IAT Std','Fwd IAT Mean','Bwd IAT Mean',
    'Fwd Packet Length Mean','Bwd Packet Length Mean','Packet Length Std','Max Packet Length',
    'SYN Flag Count','ACK Flag Count','FIN Flag Count','RST Flag Count','PSH Flag Count'
    ]

    df =df .copy ()
    df .columns =df .columns .str .strip ()


    label_col ='Attack Type'if 'Attack Type'in df .columns else 'Label'
    df ['Label_Class']=df [label_col ]


    features_disponibles =[f for f in FEATURES_BASE if f in df .columns ]
    df =df [features_disponibles +['Label_Class']].copy ()


    if 'Source IP'in df .columns :
        octetos =df ['Source IP'].astype (str ).str .split ('.',expand =True )
        for i in range (4 ):
            df [f'Src_IP_{i +1 }']=pd .to_numeric (octetos [i ],errors ='coerce').fillna (0 ).astype (int )
        df .drop (columns =['Source IP'],inplace =True )

    if 'Destination IP'in df .columns :
        octetos =df ['Destination IP'].astype (str ).str .split ('.',expand =True )
        for i in range (4 ):
            df [f'Dst_IP_{i +1 }']=pd .to_numeric (octetos [i ],errors ='coerce').fillna (0 ).astype (int )
        df .drop (columns =['Destination IP'],inplace =True )

    return df 


def filtrar_clases (df :pd .DataFrame ,clases_objetivo :list )->pd .DataFrame :
    """Filtra solo las clases objetivo para comparación justa"""
    df_filtrado =df [df ['Label_Class'].isin (clases_objetivo )].copy ()
    print (f"\n  Filas después de filtrar clases objetivo: {len (df_filtrado ):,} de {len (df ):,}")
    return df_filtrado 


def preparar_datos_para_entrenamiento (df_synth :pd .DataFrame ,df_real :pd .DataFrame ,label_encoder :LabelEncoder ):
    """
    Prepara los datos para entrenamiento alineando features y codificando labels
    """

    cols_synth =set (df_synth .columns )-{'Label','Label_Class','Attack Type'}
    cols_real =set (df_real .columns )-{'Label','Label_Class','Attack Type'}
    feature_cols =sorted (list (cols_synth .intersection (cols_real )))

    print (f"\n  Features comunes para entrenamiento: {len (feature_cols )}")


    X_synth =df_synth [feature_cols ].values 
    y_synth =label_encoder .transform (df_synth ['Label_Class'])

    X_real =df_real [feature_cols ].values 
    y_real =label_encoder .transform (df_real ['Label_Class'])


    X_synth =np .nan_to_num (X_synth ,nan =0 ,posinf =0 ,neginf =0 )
    X_real =np .nan_to_num (X_real ,nan =0 ,posinf =0 ,neginf =0 )

    return X_synth ,y_synth ,X_real ,y_real ,feature_cols 


def entrenar_y_evaluar_modelo (modelo ,nombre ,X_train ,y_train ,X_test ,y_test ,
label_encoder ,output_dir ,prefijo =""):
    """Entrena un modelo y retorna métricas"""
    print (f"\n  Entrenando {nombre }...",end =" ")

    start_time =datetime .now ()
    modelo .fit (X_train ,y_train )
    train_time =datetime .now ()-start_time 

    y_pred =modelo .predict (X_test )

    accuracy =accuracy_score (y_test ,y_pred )
    f1_macro =f1_score (y_test ,y_pred ,average ='macro',zero_division =0 )
    f1_weighted =f1_score (y_test ,y_pred ,average ='weighted',zero_division =0 )

    print (f"Acc: {accuracy :.4f}, F1-Macro: {f1_macro :.4f}, F1-Weighted: {f1_weighted :.4f}")


    report =classification_report (y_test ,y_pred ,
    target_names =label_encoder .classes_ ,
    output_dict =True ,
    zero_division =0 )
    report_df =pd .DataFrame (report ).transpose ()
    report_df .to_csv (os .path .join (output_dir ,f'{prefijo }{nombre }_classification_report.csv'))


    cm =confusion_matrix (y_test ,y_pred )
    cm_df =pd .DataFrame (cm ,
    index =label_encoder .classes_ ,
    columns =label_encoder .classes_ )
    cm_df .to_csv (os .path .join (output_dir ,f'{prefijo }{nombre }_confusion_matrix.csv'))

    return {
    'Modelo':nombre ,
    'Accuracy':accuracy ,
    'F1_Macro':f1_macro ,
    'F1_Weighted':f1_weighted ,
    'Train_Time':str (train_time ),
    'Train_Samples':len (y_train ),
    'Test_Samples':len (y_test ),
    'Report':report ,
    'Confusion_Matrix':cm 
    }


def ejecutar_experimento (X_train ,y_train ,X_test ,y_test ,label_encoder ,output_dir ,prefijo ,descripcion ):
    """Ejecuta todos los modelos para un experimento"""
    print (f"\n{'='*70 }")
    print (f"{descripcion }")
    print (f"  Train: {len (X_train ):,} muestras | Test: {len (X_test ):,} muestras")
    print ('='*70 )

    modelos ={
    'Dummy':DummyClassifier (strategy ='most_frequent',random_state =RANDOM_STATE ),
    'LogisticReg':LogisticRegression (max_iter =1000 ,
    n_jobs =-1 ,random_state =RANDOM_STATE ,solver ='saga'),
    'RandomForest':RandomForestClassifier (n_estimators =200 ,random_state =RANDOM_STATE ,
    n_jobs =-1 ,class_weight ='balanced_subsample'),
    'KNeighbors':KNeighborsClassifier (n_neighbors =5 ,n_jobs =-1 ),
    'LightGBM':LGBMClassifier (n_estimators =100 ,random_state =RANDOM_STATE ,
    n_jobs =-1 ,verbose =-1 ),
    'XGBoost':XGBClassifier (n_estimators =100 ,random_state =RANDOM_STATE ,
    n_jobs =-1 ,verbosity =0 ,use_label_encoder =False ),
    'MLP':MLPClassifier (hidden_layer_sizes =(128 ,64 ),max_iter =500 ,random_state =RANDOM_STATE ,early_stopping =True ,validation_fraction =0.1 ),
    'Voting':VotingClassifier (
    estimators =[
    ('rf',RandomForestClassifier (n_estimators =100 ,random_state =RANDOM_STATE ,n_jobs =-1 )),
    ('et',ExtraTreesClassifier (n_estimators =100 ,random_state =RANDOM_STATE ,n_jobs =-1 )),
    ('lgbm',LGBMClassifier (n_estimators =50 ,random_state =RANDOM_STATE ,n_jobs =-1 ,verbose =-1 ))
    ],
    voting ='soft',n_jobs =-1 
    ),
    }


    scaler =MinMaxScaler ()
    X_train_scaled =scaler .fit_transform (X_train )
    X_test_scaled =scaler .transform (X_test )

    resultados =[]
    for nombre ,modelo in modelos .items ():
        resultado =entrenar_y_evaluar_modelo (
        modelo ,nombre ,X_train_scaled ,y_train ,X_test_scaled ,y_test ,
        label_encoder ,output_dir ,prefijo 
        )
        resultado ['Experimento']=descripcion 
        resultados .append (resultado )

    return resultados 


def generar_resumen_global (resultados_tstr_multi ,resultados_trtr_multi ,
resultados_tstr_binary ,resultados_trtr_binary ,
label_encoder_multi ,label_encoder_binary ,
output_dir ,dataset_name ):
    """Genera un resumen global combinando los 4 experimentos"""

    resumen =[]
    resumen .append ("="*120 )
    resumen .append (" "*30 +"RESUMEN GLOBAL: TODOS LOS EXPERIMENTOS")
    resumen .append (" "*20 +"TSTR vs TRTR - Clasificación Multiclase y Binaria")
    resumen .append (f" "*30 +f"Dataset: {dataset_name }")
    resumen .append (f" "*45 +f"Fecha: {datetime .now ().strftime ('%Y-%m-%d %H:%M:%S')}")
    resumen .append ("="*120 )





    resumen .append ("\n"+"="*120 )
    resumen .append ("1. RESUMEN EJECUTIVO")
    resumen .append ("="*120 )


    mejor_tstr_multi =max (resultados_tstr_multi ,key =lambda x :x ['F1_Macro'])
    mejor_trtr_multi =max (resultados_trtr_multi ,key =lambda x :x ['F1_Macro'])
    mejor_tstr_binary =max (resultados_tstr_binary ,key =lambda x :x ['F1_Macro'])
    mejor_trtr_binary =max (resultados_trtr_binary ,key =lambda x :x ['F1_Macro'])

    gap_multi =mejor_tstr_multi ['F1_Macro']-mejor_trtr_multi ['F1_Macro']
    gap_binary =mejor_tstr_binary ['F1_Macro']-mejor_trtr_binary ['F1_Macro']

    resumen .append (f"\nClases utilizadas: {', '.join (label_encoder_multi .classes_ )}")

    resumen .append ("\n┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐")
    resumen .append ("│                                    MEJORES RESULTADOS POR EXPERIMENTO                                           │")
    resumen .append ("├─────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤")
    resumen .append (f"│  MULTICLASE ({len (label_encoder_multi .classes_ )} clases):                                                                                           │")
    resumen .append (f"│    • TSTR: {mejor_tstr_multi ['Modelo']:<15} F1-Macro = {mejor_tstr_multi ['F1_Macro']:.4f}  Accuracy = {mejor_tstr_multi ['Accuracy']:.4f}                                        │")
    resumen .append (f"│    • TRTR: {mejor_trtr_multi ['Modelo']:<15} F1-Macro = {mejor_trtr_multi ['F1_Macro']:.4f}  Accuracy = {mejor_trtr_multi ['Accuracy']:.4f}  (baseline)                            │")
    resumen .append (f"│    • Gap TSTR-TRTR: {gap_multi :+.4f}                                                                                │")
    resumen .append (f"│                                                                                                                 │")
    resumen .append (f"│  BINARIA:                                                                                                       │")
    resumen .append (f"│    • TSTR: {mejor_tstr_binary ['Modelo']:<15} F1-Macro = {mejor_tstr_binary ['F1_Macro']:.4f}  Accuracy = {mejor_tstr_binary ['Accuracy']:.4f}                                        │")
    resumen .append (f"│    • TRTR: {mejor_trtr_binary ['Modelo']:<15} F1-Macro = {mejor_trtr_binary ['F1_Macro']:.4f}  Accuracy = {mejor_trtr_binary ['Accuracy']:.4f}  (baseline)                            │")
    resumen .append (f"│    • Gap TSTR-TRTR: {gap_binary :+.4f}                                                                                │")
    resumen .append ("└─────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘")





    resumen .append ("\n\n"+"="*120 )
    resumen .append ("2. COMPARATIVA DETALLADA - CLASIFICACIÓN MULTICLASE")
    resumen .append ("="*120 )

    resumen .append ("\n"+"-"*120 )
    header =f"{'Modelo':<20} │ {'TSTR Acc':>10} {'TSTR F1-M':>10} {'TSTR F1-W':>10} │ {'TRTR Acc':>10} {'TRTR F1-M':>10} {'TRTR F1-W':>10} │ {'Δ F1-M':>8} {'Δ Acc':>8}"
    resumen .append (header )
    resumen .append ("-"*120 )

    for r_tstr in resultados_tstr_multi :
        modelo =r_tstr ['Modelo']
        r_trtr =next ((r for r in resultados_trtr_multi if r ['Modelo']==modelo ),None )
        if r_trtr :
            delta_f1 =r_tstr ['F1_Macro']-r_trtr ['F1_Macro']
            delta_acc =r_tstr ['Accuracy']-r_trtr ['Accuracy']
            linea =f"{modelo :<20} │ {r_tstr ['Accuracy']:>10.4f} {r_tstr ['F1_Macro']:>10.4f} {r_tstr ['F1_Weighted']:>10.4f} │ {r_trtr ['Accuracy']:>10.4f} {r_trtr ['F1_Macro']:>10.4f} {r_trtr ['F1_Weighted']:>10.4f} │ {delta_f1 :>+8.4f} {delta_acc :>+8.4f}"
            resumen .append (linea )





    resumen .append ("\n\n"+"="*120 )
    resumen .append ("3. COMPARATIVA DETALLADA - CLASIFICACIÓN BINARIA (BENIGN vs ATTACK)")
    resumen .append ("="*120 )

    resumen .append ("\n"+"-"*120 )
    resumen .append (header )
    resumen .append ("-"*120 )

    for r_tstr in resultados_tstr_binary :
        modelo =r_tstr ['Modelo']
        r_trtr =next ((r for r in resultados_trtr_binary if r ['Modelo']==modelo ),None )
        if r_trtr :
            delta_f1 =r_tstr ['F1_Macro']-r_trtr ['F1_Macro']
            delta_acc =r_tstr ['Accuracy']-r_trtr ['Accuracy']
            linea =f"{modelo :<20} │ {r_tstr ['Accuracy']:>10.4f} {r_tstr ['F1_Macro']:>10.4f} {r_tstr ['F1_Weighted']:>10.4f} │ {r_trtr ['Accuracy']:>10.4f} {r_trtr ['F1_Macro']:>10.4f} {r_trtr ['F1_Weighted']:>10.4f} │ {delta_f1 :>+8.4f} {delta_acc :>+8.4f}"
            resumen .append (linea )





    resumen .append ("\n\n"+"="*120 )
    resumen .append ("4. ANÁLISIS POR CLASE - MEJOR MODELO TSTR MULTICLASE")
    resumen .append ("="*120 )

    report_tstr =mejor_tstr_multi ['Report']
    report_trtr =mejor_trtr_multi ['Report']

    resumen .append (f"\nModelo: {mejor_tstr_multi ['Modelo']}")
    resumen .append ("\n"+"-"*90 )
    resumen .append (f"{'Clase':<15} │ {'TSTR Prec':>10} {'TSTR Rec':>10} {'TSTR F1':>10} │ {'TRTR F1':>10} │ {'Δ F1':>8}")
    resumen .append ("-"*90 )

    for clase in label_encoder_multi .classes_ :
        if clase in report_tstr and clase in report_trtr :
            tstr_r =report_tstr [clase ]
            trtr_r =report_trtr [clase ]
            delta =tstr_r ['f1-score']-trtr_r ['f1-score']
            resumen .append (f"{clase :<15} │ {tstr_r ['precision']:>10.4f} {tstr_r ['recall']:>10.4f} {tstr_r ['f1-score']:>10.4f} │ {trtr_r ['f1-score']:>10.4f} │ {delta :>+8.4f}")





    resumen .append ("\n\n"+"="*120 )
    resumen .append ("5. MATRIZ DE CONFUSIÓN - TSTR MULTICLASE (Mejor modelo: "+mejor_tstr_multi ['Modelo']+")")
    resumen .append ("="*120 )

    cm =mejor_tstr_multi ['Confusion_Matrix']


    clases_cortas =[c [:8 ]for c in label_encoder_multi .classes_ ]
    header_cm ="Pred→    "+"".join ([f"{c :>10}"for c in clases_cortas ])
    resumen .append ("\n"+header_cm )
    resumen .append ("Real↓    "+"-"*(10 *len (clases_cortas )))

    for i ,clase in enumerate (label_encoder_multi .classes_ ):
        fila =f"{clase [:8 ]:<9}"+"".join ([f"{cm [i ,j ]:>10}"for j in range (len (label_encoder_multi .classes_ ))])
        resumen .append (fila )





    resumen .append ("\n\n"+"="*120 )
    resumen .append ("6. INSIGHTS Y CONCLUSIONES")
    resumen .append ("="*120 )

    resumen .append ("\n┌─ CALIDAD DE LOS DATOS SINTÉTICOS ──────────────────────────────────────────────────────────────────────────────┐")

    if mejor_tstr_binary ['F1_Macro']>0.95 :
        resumen .append (f"│  ✓ EXCELENTE: TSTR Binario F1-Macro = {mejor_tstr_binary ['F1_Macro']:.4f} (>0.95)                                                    │")
    elif mejor_tstr_binary ['F1_Macro']>0.90 :
        resumen .append (f"│  ✓ MUY BUENO: TSTR Binario F1-Macro = {mejor_tstr_binary ['F1_Macro']:.4f} (>0.90)                                                    │")
    else :
        resumen .append (f"│  ○ ACEPTABLE: TSTR Binario F1-Macro = {mejor_tstr_binary ['F1_Macro']:.4f}                                                            │")

    if mejor_tstr_multi ['F1_Macro']>0.85 :
        resumen .append (f"│  ✓ ÉXITO: TSTR Multiclase F1-Macro = {mejor_tstr_multi ['F1_Macro']:.4f} (>0.85)                                                      │")
    else :
        resumen .append (f"│  ○ TSTR Multiclase F1-Macro = {mejor_tstr_multi ['F1_Macro']:.4f} (<0.85) - Hay margen de mejora                                      │")
    resumen .append ("└─────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘")


    resumen .append ("\n┌─ CLASES PROBLEMÁTICAS EN TSTR MULTICLASE ──────────────────────────────────────────────────────────────────────┐")

    clases_f1 =[(c ,report_tstr [c ]['f1-score'])for c in label_encoder_multi .classes_ if c in report_tstr ]
    clases_f1_sorted =sorted (clases_f1 ,key =lambda x :x [1 ])

    for clase ,f1 in clases_f1_sorted [:3 ]:
        if f1 <0.5 :
            resumen .append (f"│  ✗ {clase }: F1 = {f1 :.4f} - CRÍTICO                                                                           │")
        elif f1 <0.8 :
            resumen .append (f"│  ○ {clase }: F1 = {f1 :.4f} - MEJORABLE                                                                         │")
        else :
            resumen .append (f"│  ✓ {clase }: F1 = {f1 :.4f} - ACEPTABLE                                                                         │")
    resumen .append ("└─────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘")





    resumen .append ("\n\n"+"="*120 )
    resumen .append ("7. RESUMEN NUMÉRICO FINAL")
    resumen .append ("="*120 )

    resumen .append ("\n┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐")
    resumen .append ("│                                         MÉTRICAS CLAVE                                                          │")
    resumen .append ("├─────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤")
    resumen .append (f"│  BINARIA:                                                                                                       │")
    resumen .append (f"│    • TSTR F1-Macro: {mejor_tstr_binary ['F1_Macro']:.4f}    TRTR F1-Macro: {mejor_trtr_binary ['F1_Macro']:.4f}    Gap: {gap_binary :+.4f}                                    │")
    resumen .append (f"│    • TSTR Accuracy: {mejor_tstr_binary ['Accuracy']:.4f}    TRTR Accuracy: {mejor_trtr_binary ['Accuracy']:.4f}                                                  │")
    resumen .append (f"│                                                                                                                 │")
    resumen .append (f"│  MULTICLASE:                                                                                                    │")
    resumen .append (f"│    • TSTR F1-Macro: {mejor_tstr_multi ['F1_Macro']:.4f}    TRTR F1-Macro: {mejor_trtr_multi ['F1_Macro']:.4f}    Gap: {gap_multi :+.4f}                                    │")
    resumen .append (f"│    • TSTR Accuracy: {mejor_tstr_multi ['Accuracy']:.4f}    TRTR Accuracy: {mejor_trtr_multi ['Accuracy']:.4f}                                                  │")
    resumen .append ("└─────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘")


    resumen_texto ="\n".join (resumen )

    with open (os .path .join (output_dir ,'RESUMEN_GLOBAL.txt'),'w')as f :
        f .write (resumen_texto )

    print ("\n"+resumen_texto )


    comparativa_global =[]

    for exp_name ,resultados in [
    ('TSTR_Multi',resultados_tstr_multi ),
    ('TRTR_Multi',resultados_trtr_multi ),
    ('TSTR_Binary',resultados_tstr_binary ),
    ('TRTR_Binary',resultados_trtr_binary )
    ]:
        for r in resultados :
            comparativa_global .append ({
            'Experimento':exp_name ,
            'Modelo':r ['Modelo'],
            'Accuracy':r ['Accuracy'],
            'F1_Macro':r ['F1_Macro'],
            'F1_Weighted':r ['F1_Weighted']
            })

    pd .DataFrame (comparativa_global ).to_csv (os .path .join (output_dir ,'comparativa_global.csv'),index =False )

    return resumen_texto 





def main ():
    parser =argparse .ArgumentParser (
    description ='Evaluación TSTR vs TRTR con datasets sintéticos',
    formatter_class =argparse .RawDescriptionHelpFormatter 
    )
    parser .add_argument ('--dataset','-d',type =str ,help ='Nombre del dataset sintético a usar')
    parser .add_argument ('--list','-l',action ='store_true',help ='Listar datasets disponibles')
    parser .add_argument ('--max-train',type =int ,default =100000 ,help ='Máximo muestras de entrenamiento')
    parser .add_argument ('--max-test',type =int ,default =100000 ,help ='Máximo muestras de test')

    args =parser .parse_args ()


    datasets =listar_datasets_disponibles ()

    if args .list :
        mostrar_datasets_disponibles (datasets )
        return 

    print ("="*100 )
    print ("EXPERIMENTO COMPLETO: TSTR vs TRTR - MULTICLASE Y BINARIO")
    print ("="*100 )


    if args .dataset :

        dataset_info =None 
        for ds in datasets :
            if ds ['nombre'].lower ()==args .dataset .lower ()or ds ['archivo'].lower ()==args .dataset .lower ():
                dataset_info =ds 
                break 

        if not dataset_info :
            print (f"\n[ERROR] Dataset '{args .dataset }' no encontrado.")
            mostrar_datasets_disponibles (datasets )
            return 
    else :

        dataset_info =seleccionar_dataset_interactivo (datasets )

        if not dataset_info :
            return 


    output_dir =os .path .join (OUTPUT_BASE_DIR ,dataset_info ['nombre'])
    os .makedirs (output_dir ,exist_ok =True )






    df_synth =cargar_dataset_sintetico (dataset_info )


    clases_sinteticas =sorted (df_synth ['Label_Class'].unique ().tolist ())
    print (f"\n  Clases detectadas en dataset sintético: {clases_sinteticas }")


    df_real_raw =cargar_datos_reales ()


    df_real =preparar_features_reales (df_real_raw )


    df_real =filtrar_clases (df_real ,clases_sinteticas )

    print (f"\n  Distribución de clases reales (filtradas):")
    print (df_real ['Label_Class'].value_counts ())


    label_encoder_multi =LabelEncoder ()
    label_encoder_multi .fit (clases_sinteticas )
    print (f"\n  Clases del encoder multiclase: {label_encoder_multi .classes_ }")


    label_encoder_binary =LabelEncoder ()
    label_encoder_binary .fit (['ATTACK','BENIGN'])
    print (f"  Clases del encoder binario: {label_encoder_binary .classes_ }")


    X_synth ,y_synth_multi ,X_real ,y_real_multi ,feature_cols =preparar_datos_para_entrenamiento (
    df_synth ,df_real ,label_encoder_multi 
    )


    benign_idx =np .where (label_encoder_multi .classes_ =='BENIGN')[0 ]
    if len (benign_idx )==0 :
        print ("\n[WARNING] No se encontró clase 'BENIGN'. La clasificación binaria usará la primera clase como 'no-ataque'.")
        benign_idx =0 
    else :
        benign_idx =benign_idx [0 ]


    y_synth_binary =np .where (y_synth_multi ==benign_idx ,1 ,0 )
    y_real_binary =np .where (y_real_multi ==benign_idx ,1 ,0 )





    np .random .seed (RANDOM_STATE )


    X_real_train ,X_real_test ,y_real_train_multi ,y_real_test_multi =train_test_split (
    X_real ,y_real_multi ,test_size =0.3 ,random_state =RANDOM_STATE ,stratify =y_real_multi 
    )


    y_real_train_binary =np .where (y_real_train_multi ==benign_idx ,1 ,0 )
    y_real_test_binary =np .where (y_real_test_multi ==benign_idx ,1 ,0 )


    max_train =args .max_train 
    max_test =args .max_test 

    if len (X_real_train )>max_train :
        idx_train =[]
        for clase in range (len (label_encoder_multi .classes_ )):
            idx_clase =np .where (y_real_train_multi ==clase )[0 ]
            n_sample =min (len (idx_clase ),max_train //len (label_encoder_multi .classes_ ))
            if n_sample >0 :
                idx_train .extend (np .random .choice (idx_clase ,n_sample ,replace =False ))
        idx_train =np .array (idx_train )
        np .random .shuffle (idx_train )
        X_real_train =X_real_train [idx_train ]
        y_real_train_multi =y_real_train_multi [idx_train ]
        y_real_train_binary =y_real_train_binary [idx_train ]

    if len (X_real_test )>max_test :
        idx_test =np .random .choice (len (X_real_test ),max_test ,replace =False )
        X_real_test =X_real_test [idx_test ]
        y_real_test_multi =y_real_test_multi [idx_test ]
        y_real_test_binary =y_real_test_binary [idx_test ]





    print (f"\n{'='*100 }")
    print ("RESUMEN DE DATOS PARA TODOS LOS EXPERIMENTOS")
    print ('='*100 )
    print (f"  Dataset sintético: {dataset_info ['nombre']}")
    print (f"  Datos Sintéticos (Train TSTR): {len (X_synth ):,} muestras")
    print (f"  Datos Reales Train (TRTR): {len (X_real_train ):,} muestras")
    print (f"  Datos Reales Test (ambos): {len (X_real_test ):,} muestras")
    print (f"  Features: {len (feature_cols )}")

    print (f"\n  Distribución MULTICLASE en Train Sintético:")
    for i ,clase in enumerate (label_encoder_multi .classes_ ):
        count =np .sum (y_synth_multi ==i )
        print (f"    {clase }: {count :,}")

    print (f"\n  Distribución BINARIA en Train Sintético:")
    print (f"    BENIGN: {np .sum (y_synth_binary ==1 ):,}")
    print (f"    ATTACK: {np .sum (y_synth_binary ==0 ):,}")





    print ("\n\n"+"#"*100 )
    print ("#"+" "*35 +"CLASIFICACIÓN MULTICLASE"+" "*37 +"#")
    print ("#"*100 )


    resultados_tstr_multi =ejecutar_experimento (
    X_synth ,y_synth_multi ,X_real_test ,y_real_test_multi ,
    label_encoder_multi ,output_dir ,"TSTR_MULTI_",
    "EXPERIMENTO 1: TSTR MULTICLASE (Train Synthetic, Test Real)"
    )


    resultados_trtr_multi =ejecutar_experimento (
    X_real_train ,y_real_train_multi ,X_real_test ,y_real_test_multi ,
    label_encoder_multi ,output_dir ,"TRTR_MULTI_",
    "EXPERIMENTO 2: TRTR MULTICLASE (Train Real, Test Real) - BASELINE"
    )





    print ("\n\n"+"#"*100 )
    print ("#"+" "*37 +"CLASIFICACIÓN BINARIA"+" "*38 +"#")
    print ("#"*100 )


    resultados_tstr_binary =ejecutar_experimento (
    X_synth ,y_synth_binary ,X_real_test ,y_real_test_binary ,
    label_encoder_binary ,output_dir ,"TSTR_BINARY_",
    "EXPERIMENTO 3: TSTR BINARIO (Train Synthetic, Test Real)"
    )


    resultados_trtr_binary =ejecutar_experimento (
    X_real_train ,y_real_train_binary ,X_real_test ,y_real_test_binary ,
    label_encoder_binary ,output_dir ,"TRTR_BINARY_",
    "EXPERIMENTO 4: TRTR BINARIO (Train Real, Test Real) - BASELINE"
    )





    generar_resumen_global (
    resultados_tstr_multi ,resultados_trtr_multi ,
    resultados_tstr_binary ,resultados_trtr_binary ,
    label_encoder_multi ,label_encoder_binary ,
    output_dir ,dataset_info ['nombre']
    )





    todos_resultados =[]

    for r in resultados_tstr_multi :
        r_copy ={k :v for k ,v in r .items ()if k not in ['Report','Confusion_Matrix']}
        r_copy ['Tipo']='TSTR'
        r_copy ['Clasificacion']='Multiclase'
        todos_resultados .append (r_copy )

    for r in resultados_trtr_multi :
        r_copy ={k :v for k ,v in r .items ()if k not in ['Report','Confusion_Matrix']}
        r_copy ['Tipo']='TRTR'
        r_copy ['Clasificacion']='Multiclase'
        todos_resultados .append (r_copy )

    for r in resultados_tstr_binary :
        r_copy ={k :v for k ,v in r .items ()if k not in ['Report','Confusion_Matrix']}
        r_copy ['Tipo']='TSTR'
        r_copy ['Clasificacion']='Binaria'
        todos_resultados .append (r_copy )

    for r in resultados_trtr_binary :
        r_copy ={k :v for k ,v in r .items ()if k not in ['Report','Confusion_Matrix']}
        r_copy ['Tipo']='TRTR'
        r_copy ['Clasificacion']='Binaria'
        todos_resultados .append (r_copy )

    pd .DataFrame (todos_resultados ).to_csv (os .path .join (output_dir ,'todos_resultados.csv'),index =False )

    print (f"\n\n{'='*100 }")
    print (f"TODOS LOS RESULTADOS GUARDADOS EN: {output_dir }")
    print ('='*100 )


if __name__ =="__main__":
    main ()
