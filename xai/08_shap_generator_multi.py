"""
XAI-1 Multi-Dataset — SHAP Generator Analysis: 3 Datasets × 2 Variants

Executes XAI-1 on CIC-IDS2017, UNSW-NB15 and Edge-IIoTset using
both uniform and balanced synthetic datasets (6 total executions).

Outputs (per dataset_variant):
  - results_xai/XAI1_{dataset}_{variant}_beeswarm.pdf
  - results_xai/XAI1_{dataset}_{variant}_top_features.json
  - results_xai/XAI1_multi_summary.tex   (combined summary table)
"""

import os 
import json 
import numpy as np 
import pandas as pd 
import matplotlib 
matplotlib .use ('Agg')
import matplotlib .pyplot as plt 
import shap 
from sklearn .model_selection import train_test_split 
from sklearn .preprocessing import MinMaxScaler ,LabelEncoder 
from lightgbm import LGBMClassifier 
from dotenv import load_dotenv 
import warnings 
warnings .filterwarnings ('ignore')

load_dotenv ()


OUTPUT_DIR =os .getenv ('OUTPUT_RESULTS_DIR','./outputs/results')
RANDOM_STATE =42 
N_SHAP_SAMPLES =500 
MAX_TRAIN =100000 
N_TOP_FEATURES =10 


DATASETS ={
'CICIDS':{
'name':'CIC-IDS2017',
'syn_uniforme':os .getenv ('OUTPUT_SYNTHETIC_CICIDS_UNIFORM','./outputs/synthetic_data/cicids_uniform.csv'),
'syn_balanceado':os .getenv ('OUTPUT_SYNTHETIC_CICIDS_BALANCED','./outputs/synthetic_data/cicids_balanced.csv'),
'label_col':'Attack Type',
},
'UNSW':{
'name':'UNSW-NB15',
'syn_uniforme':os .getenv ('OUTPUT_SYNTHETIC_UNSW_UNIFORM','./outputs/synthetic_data/unsw_uniform.csv'),
'syn_balanceado':os .getenv ('OUTPUT_SYNTHETIC_UNSW_BALANCED','./outputs/synthetic_data/unsw_balanced.csv'),
'label_col':'Label',
},
'EdgeIIoT':{
'name':'Edge-IIoTset',
'syn_uniforme':os .getenv ('OUTPUT_SYNTHETIC_EDGEIIOT_UNIFORM','./outputs/synthetic_data/edgeiiot_uniform.csv'),
'syn_balanceado':os .getenv ('OUTPUT_SYNTHETIC_EDGEIIOT_BALANCED','./outputs/synthetic_data/edgeiiot_balanced.csv'),
'label_col':'Attack_type',
},
}


def detect_feature_cols (df ,label_col ):
    """Detects feature columns (everything except label)."""
    exclude ={'Label','Label_Class','Attack Type','Attack_type',label_col }
    return sorted ([c for c in df .columns if c not in exclude ])


def run_xai1 (dataset_key ,variant ,syn_path ,label_col ,dataset_name ):
    """Runs XAI-1 for a dataset + variant."""

    tag =f"{dataset_key }_{variant }"
    print (f"\n{'='*70 }")
    print (f"  XAI-1: {dataset_name } — {variant }")
    print (f"{'='*70 }")


    print (f"  [1] Cargando {syn_path }...")
    df =pd .read_csv (syn_path )
    print (f"      Muestras: {len (df ):,}")


    if label_col in df .columns :
        df ['Label_Class']=df [label_col ]
    elif 'Label_Class'in df .columns :
        pass 
    elif 'Label'in df .columns :
        df ['Label_Class']=df ['Label']
    else :
        raise ValueError (f"No se encontró columna de label en {syn_path }")

    feat_cols =detect_feature_cols (df ,label_col )
    classes =sorted (df ['Label_Class'].unique ().tolist ())
    n_classes =len (classes )

    print (f"      Clases: {n_classes }")
    print (f"      Features: {len (feat_cols )}")


    le =LabelEncoder ()
    le .fit (classes )

    X =df [feat_cols ].replace ([np .inf ,-np .inf ],np .nan ).fillna (0 )
    y =le .transform (df ['Label_Class'])


    for col in X .columns :
        X [col ]=pd .to_numeric (X [col ],errors ='coerce').fillna (0 )



    scaler =MinMaxScaler ()
    X_sc =pd .DataFrame (scaler .fit_transform (X ),columns =feat_cols ,index =X .index )

    X_train_sc ,X_test_sc ,y_train ,y_test =train_test_split (
    X_sc ,y ,test_size =0.2 ,random_state =RANDOM_STATE ,stratify =y 
    )


    if len (X_train_sc )>MAX_TRAIN :
        idx =np .random .RandomState (RANDOM_STATE ).choice (len (X_train_sc ),MAX_TRAIN ,replace =False )
        X_train_sc =X_train_sc .iloc [idx ]
        y_train =y_train [idx ]


    print (f"  [2] Entrenando LightGBM...")
    clf =LGBMClassifier (
    n_estimators =200 ,random_state =RANDOM_STATE ,
    n_jobs =-1 ,verbose =-1 ,num_leaves =63 ,learning_rate =0.05 
    )
    clf .fit (X_train_sc ,y_train )
    acc =clf .score (X_test_sc ,y_test )
    print (f"      Accuracy: {acc :.4f}")


    print (f"  [3] Computando SHAP por generador...")

    explainer =shap .TreeExplainer (clf )
    all_top_features ={}


    n_cols =min (4 ,n_classes )
    n_rows =(n_classes +n_cols -1 )//n_cols 

    fig ,axes =plt .subplots (n_rows ,n_cols ,figsize =(5 *n_cols ,4.5 *n_rows ))
    if n_rows ==1 and n_cols ==1 :
        axes =np .array ([[axes ]])
    elif n_rows ==1 :
        axes =axes .reshape (1 ,-1 )
    elif n_cols ==1 :
        axes =axes .reshape (-1 ,1 )

    for idx ,class_name in enumerate (classes ):
        row_idx =idx //n_cols 
        col_idx =idx %n_cols 

        print (f"      [{idx +1 }/{n_classes }] {class_name }...",end =" ")


        class_mask =df ['Label_Class']==class_name 
        subset_sc =X_sc [class_mask ]


        n_sample =min (N_SHAP_SAMPLES ,len (subset_sc ))
        subset_sc =subset_sc .sample (n_sample ,random_state =RANDOM_STATE )


        shap_values =explainer .shap_values (subset_sc )

        class_idx =int (le .transform ([class_name ])[0 ])

        if isinstance (shap_values ,list ):
            shap_for_class =shap_values [class_idx ]
        elif len (shap_values .shape )==3 :
            shap_for_class =shap_values [:,:,class_idx ]
        else :
            shap_for_class =shap_values 


        mean_abs =np .mean (np .abs (shap_for_class ),axis =0 )
        if len (mean_abs .shape )>1 :
            mean_abs =mean_abs .flatten ()
        top_n =min (N_TOP_FEATURES ,len (feat_cols ))
        top_idx =np .argsort (mean_abs )[-top_n :][::-1 ]
        top_feats =[(feat_cols [i ],float (mean_abs [i ]))for i in top_idx ]
        all_top_features [class_name ]=top_feats 

        top3 =[feat_cols [i ]for i in top_idx [:3 ]]
        print (f"top-3: {top3 }")


        ax =axes [row_idx ,col_idx ]
        plt .sca (ax )
        shap .summary_plot (
        shap_for_class [:,top_idx ],
        subset_sc .iloc [:,top_idx ],
        feature_names =[feat_cols [i ]for i in top_idx ],
        show =False ,
        plot_size =None ,
        )
        safe_name =class_name .replace (" ",r"\ ")
        ax .set_title (r'$G_{\mathrm{'+safe_name +r'}}$',
        fontsize =11 ,fontweight ='bold',pad =8 )


    for idx in range (n_classes ,n_rows *n_cols ):
        row_idx =idx //n_cols 
        col_idx =idx %n_cols 
        axes [row_idx ,col_idx ].set_visible (False )

    plt .tight_layout ()

    fig_path =os .path .join (OUTPUT_DIR ,f'XAI1_{tag }_beeswarm.pdf')
    plt .savefig (fig_path ,bbox_inches ='tight',dpi =500 )
    plt .savefig (fig_path .replace ('.pdf','.png'),bbox_inches ='tight',dpi =500 )
    plt .close ()
    print (f"  [4] Figura: {fig_path }")


    json_path =os .path .join (OUTPUT_DIR ,f'XAI1_{tag }_top_features.json')
    json_data ={cls :{f :v for f ,v in feats }
    for cls ,feats in all_top_features .items ()}
    with open (json_path ,'w')as f :
        json .dump (json_data ,f ,indent =2 )

    return {
    'dataset':dataset_name ,
    'variant':variant ,
    'tag':tag ,
    'n_classes':n_classes ,
    'n_features':len (feat_cols ),
    'accuracy':acc ,
    'top_features':all_top_features ,
    }


def main ():
    os .makedirs (OUTPUT_DIR ,exist_ok =True )

    print ("="*70 )
    print ("XAI-1 Multi-Dataset: SHAP por Generador")
    print ("  3 datasets × 2 variantes = 6 ejecuciones")
    print ("="*70 )

    all_results =[]

    for ds_key ,ds_config in DATASETS .items ():
        for variant ,path_key in [('uniforme','syn_uniforme'),('balanceado','syn_balanceado')]:
            syn_path =ds_config [path_key ]
            if not os .path .exists (syn_path ):
                print (f"\n  [SKIP] {ds_config ['name']} {variant }: {syn_path } no encontrado")
                continue 

            result =run_xai1 (
            ds_key ,variant ,syn_path ,
            ds_config ['label_col'],ds_config ['name']
            )
            all_results .append (result )


    print ("\n\n"+"="*70 )
    print ("TABLA RESUMEN XAI-1 MULTI-DATASET")
    print ("="*70 )

    summary_rows =[]
    for r in all_results :

        first_features =[feats [0 ][0 ]for feats in r ['top_features'].values ()]
        from collections import Counter 
        most_common =Counter (first_features ).most_common (3 )
        top_global =', '.join ([f"{f }({c })"for f ,c in most_common ])

        summary_rows .append ({
        'Dataset':r ['dataset'],
        'Variant':r ['variant'],
        'Classes':r ['n_classes'],
        'Features':r ['n_features'],
        'Clf Accuracy':f"{r ['accuracy']:.4f}",
        'Most Discriminative Features':top_global ,
        })

    summary_df =pd .DataFrame (summary_rows )
    print ("\n"+summary_df .to_string (index =False ))


    latex_str =summary_df .to_latex (
    index =False ,
    caption ='XAI-1 Summary: SHAP analysis per generator across datasets and sampling variants.',
    label ='tab:xai1_multi_dataset',
    column_format ='l l c c c l'
    )
    latex_path =os .path .join (OUTPUT_DIR ,'XAI1_multi_summary.tex')
    with open (latex_path ,'w')as f :
        f .write (latex_str )


    summary_df .to_csv (os .path .join (OUTPUT_DIR ,'XAI1_multi_summary.csv'),index =False )



    detail_rows =[]
    for r in all_results :
        for cls ,feats in r ['top_features'].items ():
            top3 =feats [:3 ]
            detail_rows .append ({
            'Dataset':r ['dataset'],
            'Variant':r ['variant'],
            'Class':cls ,
            'Feature_1':top3 [0 ][0 ]if len (top3 )>0 else '',
            'SHAP_1':f"{top3 [0 ][1 ]:.4f}"if len (top3 )>0 else '',
            'Feature_2':top3 [1 ][0 ]if len (top3 )>1 else '',
            'SHAP_2':f"{top3 [1 ][1 ]:.4f}"if len (top3 )>1 else '',
            'Feature_3':top3 [2 ][0 ]if len (top3 )>2 else '',
            'SHAP_3':f"{top3 [2 ][1 ]:.4f}"if len (top3 )>2 else '',
            })

    detail_df =pd .DataFrame (detail_rows )
    detail_df .to_csv (os .path .join (OUTPUT_DIR ,'XAI1_multi_detail.csv'),index =False )

    print (f"\n  Tabla detallada guardada: XAI1_multi_detail.csv")
    print (f"  Tabla LaTeX: {latex_path }")

    print ("\n"+"="*70 )
    print ("XAI-1 MULTI-DATASET COMPLETADO")
    print ("="*70 )


if __name__ =="__main__":
    main ()
