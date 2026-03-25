"""
XAI-2 Multi-Dataset — Top-K Feature Overlap SHAP: 3 Datasets

Instead of Spearman correlation over 29+ features, evaluates how many
of the top-K features from the REAL classifier also appear in the
top-K of the SYNTHETIC classifier. More robust and easier to explain:
  "7 of the 10 most important features from the real classifier also
   appear in the top-10 of the synthetic classifier."

Executes on CIC-IDS2017, UNSW-NB15 and Edge-IIoTset.

Outputs:
  - results_xai/XAI2_topk_overlap_comparison.pdf    (3-panel bar chart)
  - results_xai/XAI2_topk_overlap_table.tex          (LaTeX table)
  - results_xai/XAI2_topk_overlap_metrics.txt        (detailed metrics)
  - results_xai/XAI2_topk_overlap_table.csv          (CSV)
"""

import os 
import sys 
import numpy as np 
import pandas as pd 
import polars as pl 
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
N_SHAP_SAMPLES =1000 
MAX_TRAIN =100000 
MAX_TEST =50000 
K_VALUES =[5 ,10 ,15 ]



DATASETS ={
'CIC-IDS2017':{
'real_path':os .getenv ('DATA_CICIDS2017_PATH','./data/CIC-IDS2017.csv'),
'syn_bal_path':os .getenv ('OUTPUT_SYNTHETIC_CICIDS_UNIFORM','./outputs/synthetic_data/cicids_uniform.csv'),
'syn_real_path':os .getenv ('OUTPUT_SYNTHETIC_CICIDS_BALANCED','./outputs/synthetic_data/cicids_balanced.csv'),
'label_col_real':'Attack Type',
'label_col_syn':'Attack Type',
'reader':'polars',
'prepare_fn':'prepare_cicids',
},
'UNSW-NB15':{
'real_path':os .getenv ('DATA_UNSW_PATH','./data/UNSW-NB15.csv'),
'syn_bal_path':os .getenv ('OUTPUT_SYNTHETIC_UNSW_UNIFORM','./outputs/synthetic_data/unsw_uniform.csv'),
'syn_real_path':os .getenv ('OUTPUT_SYNTHETIC_UNSW_BALANCED','./outputs/synthetic_data/unsw_balanced.csv'),
'label_col_real':'Label',
'label_col_syn':'Label',
'reader':'polars',
'prepare_fn':'prepare_unsw',
},
'Edge-IIoTset':{
'real_path':os .getenv ('DATA_EDGEIIOT_PATH','./data/edgeiiot.csv'),
'syn_bal_path':os .getenv ('OUTPUT_SYNTHETIC_EDGEIIOT_UNIFORM','./outputs/synthetic_data/edgeiiot_uniform.csv'),
'syn_real_path':os .getenv ('OUTPUT_SYNTHETIC_EDGEIIOT_BALANCED','./outputs/synthetic_data/edgeiiot_balanced.csv'),
'label_col_real':'Attack_type',
'label_col_syn':'Attack_type',
'reader':'pandas',
'prepare_fn':'prepare_edgeiiot',
},
}




def prepare_cicids (df_raw ):
    """Prepara CIC-IDS2017 real data."""
    df =df_raw .copy ()
    df .columns =df .columns .str .strip ()

    FEATURES_BASE =[
    'Source IP','Destination IP',
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


def prepare_unsw (df_raw ):
    """Prepara UNSW-NB15 real data."""
    df =df_raw .copy ()
    df .columns =df .columns .str .strip ()

    FEATURES_BASE =[
    'Src IP','Dst IP',
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

    df ['Label_Class']=df ['Label']

    features_disponibles =[f for f in FEATURES_BASE if f in df .columns ]
    df =df [features_disponibles +['Label_Class']].copy ()

    if 'Src IP'in df .columns :
        octetos =df ['Src IP'].astype (str ).str .split ('.',expand =True )
        for i in range (4 ):
            df [f'Src_IP_{i +1 }']=pd .to_numeric (octetos [i ],errors ='coerce').fillna (0 ).astype (int )
        df .drop (columns =['Src IP'],inplace =True )

    if 'Dst IP'in df .columns :
        octetos =df ['Dst IP'].astype (str ).str .split ('.',expand =True )
        for i in range (4 ):
            df [f'Dst_IP_{i +1 }']=pd .to_numeric (octetos [i ],errors ='coerce').fillna (0 ).astype (int )
        df .drop (columns =['Dst IP'],inplace =True )

    return df 


def prepare_edgeiiot (df_raw ):
    """Prepara Edge-IIoTset real data."""
    df =df_raw .copy ()
    df .columns =df .columns .str .strip ()

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

    df ['Label_Class']=df ['Attack_type']

    features_disponibles =[f for f in FEATURES_BASE if f in df .columns ]
    result =df [features_disponibles ].copy ()


    for col in result .columns :
        result [col ]=pd .to_numeric (result [col ],errors ='coerce').fillna (0 )


    if 'ip.src_host'in df .columns :
        octetos =df ['ip.src_host'].astype (str ).str .split ('.',expand =True )
        for i in range (4 ):
            result [f'Src_IP_{i +1 }']=pd .to_numeric (octetos [i ],errors ='coerce').fillna (0 ).astype (int )
    else :
        for i in range (4 ):
            result [f'Src_IP_{i +1 }']=0 

    if 'ip.dst_host'in df .columns :
        octetos =df ['ip.dst_host'].astype (str ).str .split ('.',expand =True )
        for i in range (4 ):
            result [f'Dst_IP_{i +1 }']=pd .to_numeric (octetos [i ],errors ='coerce').fillna (0 ).astype (int )
    else :
        for i in range (4 ):
            result [f'Dst_IP_{i +1 }']=0 

    result ['Label_Class']=df ['Label_Class'].values 
    result .replace ([np .inf ,-np .inf ],np .nan ,inplace =True )
    result .fillna (0 ,inplace =True )

    return result 


PREPARE_FNS ={
'prepare_cicids':prepare_cicids ,
'prepare_unsw':prepare_unsw ,
'prepare_edgeiiot':prepare_edgeiiot ,
}






def compute_global_shap_ranking (clf ,X_eval ,feature_names ,
n_samples =1000 ,random_state =42 ):
    """
    Devuelve Series con importancia SHAP global por feature,
    ordenada descendente. Funciona para cualquier n_classes.
    """
    X_sample =X_eval .sample (
    min (n_samples ,len (X_eval )),
    random_state =random_state 
    )
    explainer =shap .TreeExplainer (clf )
    shap_values =explainer .shap_values (X_sample )


    if isinstance (shap_values ,list ):
        stacked =np .abs (np .array (shap_values ))
        global_importance =stacked .mean (axis =(0 ,1 ))
    else :
        global_importance =np .abs (shap_values ).mean (axis =0 )
        if global_importance .ndim >1 :
            global_importance =global_importance .mean (axis =-1 )

    return pd .Series (
    global_importance ,
    index =feature_names ,
    name ='shap_importance'
    ).sort_values (ascending =False )


def top_k_overlap (ranking_a ,ranking_b ,k =10 ):
    """
    Proporción de las top-k features de ranking_a
    que aparecen en las top-k de ranking_b.
    Devuelve (overlap_ratio, set_of_shared_features).
    """
    top_a =set (ranking_a .nlargest (k ).index )
    top_b =set (ranking_b .nlargest (k ).index )
    intersection =top_a &top_b 
    return len (intersection )/k ,intersection 






def run_xai2_for_dataset (dataset_name ,config ):
    """Ejecuta XAI-2 completo para un dataset. Devuelve dict con métricas."""
    print (f"\n{'='*70 }")
    print (f"  Procesando: {dataset_name }")
    print (f"{'='*70 }")


    print (f"  [1] Cargando datos reales...")
    if config ['reader']=='polars':
        df_pl =pl .read_csv (config ['real_path'],low_memory =False )
        df_raw =df_pl .to_pandas ()
    else :
        df_raw =pd .read_csv (config ['real_path'],low_memory =False )
    df_raw .columns =df_raw .columns .str .strip ()

    prepare_fn =PREPARE_FNS [config ['prepare_fn']]
    df_real =prepare_fn (df_raw )
    print (f"      Muestras reales: {len (df_real ):,}")


    print (f"  [2] Cargando datasets sintéticos...")
    df_syn_bal =pd .read_csv (config ['syn_bal_path'])
    df_syn_real =pd .read_csv (config ['syn_real_path'])
    print (f"      Syn-Balanced: {len (df_syn_bal ):,}")
    print (f"      Syn-Real:     {len (df_syn_real ):,}")


    label_col_syn =config ['label_col_syn']
    for df_syn in [df_syn_bal ,df_syn_real ]:
        if label_col_syn in df_syn .columns :
            df_syn ['Label_Class']=df_syn [label_col_syn ]
        elif 'Label_Class'not in df_syn .columns :
            for col in ['Label','Attack Type','Attack_type']:
                if col in df_syn .columns :
                    df_syn ['Label_Class']=df_syn [col ]
                    break 


    all_classes =sorted (set (df_real ['Label_Class'].unique ())&
    set (df_syn_bal ['Label_Class'].unique ())&
    set (df_syn_real ['Label_Class'].unique ()))
    n_classes =len (all_classes )
    print (f"      Clases comunes: {n_classes }")

    df_real =df_real [df_real ['Label_Class'].isin (all_classes )]
    df_syn_bal =df_syn_bal [df_syn_bal ['Label_Class'].isin (all_classes )]
    df_syn_real =df_syn_real [df_syn_real ['Label_Class'].isin (all_classes )]

    le =LabelEncoder ()
    le .fit (all_classes )


    non_feature_cols ={'Label','Label_Class','Attack Type','Attack_type'}
    feat_cols_real =set (df_real .columns )-non_feature_cols 
    feat_cols_syn_bal =set (df_syn_bal .columns )-non_feature_cols 
    feat_cols_syn_real =set (df_syn_real .columns )-non_feature_cols 
    feat_cols =sorted (feat_cols_real &feat_cols_syn_bal &feat_cols_syn_real )
    n_feats =len (feat_cols )
    print (f"      Features comunes: {n_feats }")


    X_real =df_real [feat_cols ].replace ([np .inf ,-np .inf ],np .nan ).fillna (0 )
    y_real =le .transform (df_real ['Label_Class'])

    X_real_train ,X_real_test ,y_real_train ,y_real_test =train_test_split (
    X_real ,y_real ,test_size =0.3 ,random_state =RANDOM_STATE ,stratify =y_real 
    )


    rng =np .random .RandomState (RANDOM_STATE )
    if len (X_real_train )>MAX_TRAIN :
        idx =rng .choice (len (X_real_train ),MAX_TRAIN ,replace =False )
        X_real_train =X_real_train .iloc [idx ]
        y_real_train =y_real_train [idx ]

    if len (X_real_test )>MAX_TEST :
        idx =rng .choice (len (X_real_test ),MAX_TEST ,replace =False )
        X_real_test_shap =X_real_test .iloc [idx ]
    else :
        X_real_test_shap =X_real_test 

    X_syn_bal =df_syn_bal [feat_cols ].replace ([np .inf ,-np .inf ],np .nan ).fillna (0 )
    y_syn_bal =le .transform (df_syn_bal ['Label_Class'])

    X_syn_real =df_syn_real [feat_cols ].replace ([np .inf ,-np .inf ],np .nan ).fillna (0 )
    y_syn_real =le .transform (df_syn_real ['Label_Class'])

    if len (X_syn_bal )>MAX_TRAIN :
        idx =rng .choice (len (X_syn_bal ),MAX_TRAIN ,replace =False )
        X_syn_bal_train =X_syn_bal .iloc [idx ]
        y_syn_bal_train =y_syn_bal [idx ]
    else :
        X_syn_bal_train =X_syn_bal 
        y_syn_bal_train =y_syn_bal 

    if len (X_syn_real )>MAX_TRAIN :
        idx =rng .choice (len (X_syn_real ),MAX_TRAIN ,replace =False )
        X_syn_real_train =X_syn_real .iloc [idx ]
        y_syn_real_train =y_syn_real [idx ]
    else :
        X_syn_real_train =X_syn_real 
        y_syn_real_train =y_syn_real 


    scaler =MinMaxScaler ()
    X_real_train_sc =pd .DataFrame (scaler .fit_transform (X_real_train ),columns =feat_cols )
    X_real_test_sc =pd .DataFrame (scaler .transform (X_real_test_shap ),columns =feat_cols )
    X_syn_bal_train_sc =pd .DataFrame (scaler .transform (X_syn_bal_train ),columns =feat_cols )
    X_syn_real_train_sc =pd .DataFrame (scaler .transform (X_syn_real_train ),columns =feat_cols )


    print (f"  [3] Entrenando clasificadores...")
    clf_params =dict (n_estimators =200 ,random_state =RANDOM_STATE ,
    n_jobs =-1 ,verbose =-1 ,num_leaves =63 ,learning_rate =0.05 )

    clf_real =LGBMClassifier (**clf_params )
    clf_real .fit (X_real_train_sc ,y_real_train )

    clf_syn_bal =LGBMClassifier (**clf_params )
    clf_syn_bal .fit (X_syn_bal_train_sc ,y_syn_bal_train )

    clf_syn_real =LGBMClassifier (**clf_params )
    clf_syn_real .fit (X_syn_real_train_sc ,y_syn_real_train )


    print (f"  [4] Computando rankings SHAP...")
    ranking_real =compute_global_shap_ranking (clf_real ,X_real_test_sc ,
    feat_cols ,N_SHAP_SAMPLES )
    ranking_syn_bal =compute_global_shap_ranking (clf_syn_bal ,X_real_test_sc ,
    feat_cols ,N_SHAP_SAMPLES )
    ranking_syn_real =compute_global_shap_ranking (clf_syn_real ,X_real_test_sc ,
    feat_cols ,N_SHAP_SAMPLES )


    print (f"  [5] Calculando Top-K Feature Overlap...")
    topk_results =[]
    for k in K_VALUES :
        overlap_bal ,shared_bal =top_k_overlap (ranking_real ,ranking_syn_bal ,k )
        overlap_real ,shared_real =top_k_overlap (ranking_real ,ranking_syn_real ,k )

        topk_results .append ({
        'K':k ,
        'overlap_bal':overlap_bal ,
        'overlap_real':overlap_real ,
        'shared_bal':shared_bal ,
        'shared_real':shared_real ,
        })

        print (f"      K={k :>2}: Syn-Balanced={overlap_bal :.0%} ({len (shared_bal )}/{k })  "
        f"Syn-Real={overlap_real :.0%} ({len (shared_real )}/{k })")


    top10_real =list (ranking_real .nlargest (10 ).index )
    top10_syn_bal =list (ranking_syn_bal .nlargest (10 ).index )
    top10_syn_real =list (ranking_syn_real .nlargest (10 ).index )

    return {
    'dataset':dataset_name ,
    'n_classes':n_classes ,
    'n_features':n_feats ,
    'topk_results':topk_results ,
    'top10_real':top10_real ,
    'top10_syn_bal':top10_syn_bal ,
    'top10_syn_real':top10_syn_real ,
    'ranking_real':ranking_real ,
    'ranking_syn_bal':ranking_syn_bal ,
    'ranking_syn_real':ranking_syn_real ,
    }






def main ():
    os .makedirs (OUTPUT_DIR ,exist_ok =True )

    print ("="*70 )
    print ("XAI-2 Multi-Dataset: Top-K Feature Overlap (SHAP)")
    print ("="*70 )


    all_results =[]

    for ds_name ,ds_config in DATASETS .items ():
        for key in ['real_path','syn_bal_path','syn_real_path']:
            if not os .path .exists (ds_config [key ]):
                print (f"\n  [SKIP] {ds_name }: Archivo no encontrado: {ds_config [key ]}")
                break 
        else :
            result =run_xai2_for_dataset (ds_name ,ds_config )
            all_results .append (result )

    if not all_results :
        print ("\n[ERROR] No se pudo procesar ningún dataset.")
        return 




    print ("\n\n"+"="*70 )
    print ("TABLA TOP-K OVERLAP MULTI-DATASET")
    print ("="*70 )


    table_rows =[]
    for r in all_results :
        for tk in r ['topk_results']:
            table_rows .append ({
            'Dataset':r ['dataset'],
            'K':tk ['K'],
            'Syn-Balanced Overlap':tk ['overlap_bal'],
            'Syn-Real Overlap':tk ['overlap_real'],
            'Shared (Balanced)':', '.join (sorted (tk ['shared_bal'])),
            'Shared (Real)':', '.join (sorted (tk ['shared_real'])),
            })

    table_df =pd .DataFrame (table_rows )


    print (f"\n{'Dataset':<16} {'K':>3}  {'Syn-Balanced':>14}  {'Syn-Real':>14}")
    print ("-"*55 )
    for _ ,row in table_df .iterrows ():
        print (f"{row ['Dataset']:<16} {row ['K']:>3}  "
        f"{row ['Syn-Balanced Overlap']:>13.0%}  "
        f"{row ['Syn-Real Overlap']:>13.0%}")


    latex_df =table_df [['Dataset','K','Syn-Balanced Overlap','Syn-Real Overlap']].copy ()
    latex_df ['Syn-Balanced Overlap']=latex_df ['Syn-Balanced Overlap'].apply (
    lambda x :f'{x :.0%}')
    latex_df ['Syn-Real Overlap']=latex_df ['Syn-Real Overlap'].apply (
    lambda x :f'{x :.0%}')

    latex_str =latex_df .to_latex (
    index =False ,
    caption ='Top-K feature overlap between SHAP rankings of classifiers '
    'trained on synthetic vs.\\ real data across three NIDS benchmarks. '
    'Values indicate the proportion of the top-K most important features '
    'of the real-data classifier also found in the top-K of the synthetic-data classifier.',
    label ='tab:xai2_topk_overlap',
    column_format ='l c c c'
    )
    latex_path =os .path .join (OUTPUT_DIR ,'XAI2_topk_overlap_table.tex')
    with open (latex_path ,'w')as f :
        f .write (latex_str )
    print (f"\n  Tabla LaTeX: {latex_path }")


    table_df .to_csv (os .path .join (OUTPUT_DIR ,'XAI2_topk_overlap_table.csv'),index =False )




    metrics_path =os .path .join (OUTPUT_DIR ,'XAI2_topk_overlap_metrics.txt')
    with open (metrics_path ,'w')as f :
        f .write ("XAI-2 Multi-Dataset: Top-K Feature Overlap (SHAP)\n")
        f .write ("="*70 +"\n\n")

        for r in all_results :
            f .write (f"{r ['dataset']}:\n")
            f .write (f"  Classes: {r ['n_classes']}, Features: {r ['n_features']}\n\n")


            f .write (f"  Top-10 SHAP Features (Real Classifier):\n")
            for i ,feat in enumerate (r ['top10_real'],1 ):
                imp =r ['ranking_real'][feat ]
                f .write (f"    {i :>2}. {feat :<35} (SHAP = {imp :.6f})\n")

            f .write (f"\n  Top-10 SHAP Features (Syn-Balanced Classifier):\n")
            for i ,feat in enumerate (r ['top10_syn_bal'],1 ):
                imp =r ['ranking_syn_bal'][feat ]
                f .write (f"    {i :>2}. {feat :<35} (SHAP = {imp :.6f})\n")

            f .write (f"\n  Top-10 SHAP Features (Syn-Real Classifier):\n")
            for i ,feat in enumerate (r ['top10_syn_real'],1 ):
                imp =r ['ranking_syn_real'][feat ]
                f .write (f"    {i :>2}. {feat :<35} (SHAP = {imp :.6f})\n")

            f .write (f"\n  Top-K Overlap Results:\n")
            f .write (f"  {'K':>3}  {'Syn-Balanced':>14}  {'Syn-Real':>14}  Shared features (Balanced)\n")
            f .write (f"  {'-'*75 }\n")
            for tk in r ['topk_results']:
                shared_str =', '.join (sorted (tk ['shared_bal']))
                f .write (f"  {tk ['K']:>3}  "
                f"{tk ['overlap_bal']:>13.0%}  "
                f"{tk ['overlap_real']:>13.0%}  "
                f"{shared_str }\n")

            f .write ("\n"+"-"*70 +"\n\n")


        f .write ("="*70 +"\n")
        f .write ("Cross-Domain Analysis (K=10):\n\n")

        k10_bal_all =[]
        k10_real_all =[]
        for r in all_results :
            for tk in r ['topk_results']:
                if tk ['K']==10 :
                    k10_bal_all .append (tk ['overlap_bal'])
                    k10_real_all .append (tk ['overlap_real'])

        mean_bal =np .mean (k10_bal_all )if k10_bal_all else 0 
        mean_real =np .mean (k10_real_all )if k10_real_all else 0 

        f .write (f"  Mean Top-10 Overlap (Syn-Balanced): {mean_bal :.0%}\n")
        f .write (f"  Mean Top-10 Overlap (Syn-Real):     {mean_real :.0%}\n\n")

        threshold_met =all (x >=0.60 for x in k10_bal_all )
        f .write (f"  Syn-Balanced overlap ≥ 60% in all datasets: {threshold_met }\n\n")

        if threshold_met :
            f .write ("  FINDING: At least 6 of the 10 most important features learned by\n")
            f .write ("  the real-data classifier are also captured by the AMD-GAN balanced\n")
            f .write ("  classifier across all three NIDS benchmarks. This demonstrates\n")
            f .write ("  that AMD-GAN preserves semantically meaningful feature structure.\n")

    print (f"  Métricas: {metrics_path }")




    print ("\n  Generando figura comparativa Top-K Overlap...")

    n_datasets =len (all_results )
    fig ,axes =plt .subplots (1 ,n_datasets ,figsize =(6 *n_datasets ,5 ))
    if n_datasets ==1 :
        axes =[axes ]

    bar_colors ={'Syn-Balanced':'#3498db','Syn-Real':'#e67e22'}

    for ax_idx ,r in enumerate (all_results ):
        ax =axes [ax_idx ]
        k_vals =[tk ['K']for tk in r ['topk_results']]
        overlap_bal =[tk ['overlap_bal']for tk in r ['topk_results']]
        overlap_real =[tk ['overlap_real']for tk in r ['topk_results']]

        x =np .arange (len (k_vals ))
        width =0.32 

        bars1 =ax .bar (x -width /2 ,overlap_bal ,width ,
        label ='Syn-Balanced',color =bar_colors ['Syn-Balanced'],
        edgecolor ='white',linewidth =1.2 ,alpha =0.9 )
        bars2 =ax .bar (x +width /2 ,overlap_real ,width ,
        label ='Syn-Real',color =bar_colors ['Syn-Real'],
        edgecolor ='white',linewidth =1.2 ,alpha =0.9 )


        for bar in bars1 :
            h =bar .get_height ()
            ax .text (bar .get_x ()+bar .get_width ()/2 ,h +0.02 ,
            f'{h :.0%}',ha ='center',va ='bottom',fontsize =10 ,
            fontweight ='bold',color =bar_colors ['Syn-Balanced'])
        for bar in bars2 :
            h =bar .get_height ()
            ax .text (bar .get_x ()+bar .get_width ()/2 ,h +0.02 ,
            f'{h :.0%}',ha ='center',va ='bottom',fontsize =10 ,
            fontweight ='bold',color =bar_colors ['Syn-Real'])

        ax .set_xlabel ('K (Top-K Features)',fontsize =11 )
        ax .set_ylabel ('Feature Overlap',fontsize =11 )
        ax .set_title (f'{r ["dataset"]}\n({r ["n_features"]} features, {r ["n_classes"]} classes)',
        fontsize =12 ,fontweight ='bold')
        ax .set_xticks (x )
        ax .set_xticklabels ([f'K={k }'for k in k_vals ],fontsize =10 )
        ax .set_ylim (0 ,1.15 )
        ax .axhline (y =0.7 ,color ='#2ecc71',linestyle ='--',alpha =0.5 ,
        linewidth =1.5 ,label ='70% threshold')
        ax .legend (fontsize =9 ,loc ='upper right')
        ax .grid (axis ='y',alpha =0.3 )
        ax .spines ['top'].set_visible (False )
        ax .spines ['right'].set_visible (False )


        ax .set_yticklabels ([f'{int (t *100 )}%'for t in ax .get_yticks ()])

    plt .suptitle ('XAI-2: Top-K SHAP Feature Overlap\nReal Classifier vs Synthetic Classifiers',
    fontsize =14 ,fontweight ='bold',y =1.02 )
    plt .tight_layout ()

    fig_path =os .path .join (OUTPUT_DIR ,'XAI2_topk_overlap_comparison.pdf')
    plt .savefig (fig_path ,bbox_inches ='tight',dpi =300 )
    plt .savefig (fig_path .replace ('.pdf','.png'),bbox_inches ='tight',dpi =150 )
    plt .close ()
    print (f"  Figura: {fig_path }")


    print ("  Generando figura de rankings Top-10...")

    fig2 ,axes2 =plt .subplots (n_datasets ,1 ,figsize =(12 ,5 *n_datasets ))
    if n_datasets ==1 :
        axes2 =[axes2 ]

    for ax_idx ,r in enumerate (all_results ):
        ax =axes2 [ax_idx ]


        top10 =r ['top10_real'][::-1 ]
        importances_real =[r ['ranking_real'][f ]for f in top10 ]


        top10_bal_set =set (r ['top10_syn_bal'])
        top10_real_set =set (r ['top10_syn_real'])

        colors_bars =[]
        for f in top10 :
            in_bal =f in top10_bal_set 
            in_real =f in top10_real_set 
            if in_bal and in_real :
                colors_bars .append ('#2ecc71')
            elif in_bal :
                colors_bars .append ('#3498db')
            elif in_real :
                colors_bars .append ('#e67e22')
            else :
                colors_bars .append ('#e74c3c')

        y_pos =np .arange (len (top10 ))
        ax .barh (y_pos ,importances_real ,color =colors_bars ,edgecolor ='white',
        linewidth =0.8 ,alpha =0.85 ,height =0.7 )
        ax .set_yticks (y_pos )


        labels =[f [:30 ]+'...'if len (f )>30 else f for f in top10 ]
        ax .set_yticklabels (labels ,fontsize =9 )
        ax .set_xlabel ('SHAP Importance (Real Classifier)',fontsize =10 )
        ax .set_title (f'{r ["dataset"]} — Top-10 Features (Real Classifier)\n'
        f'Color = presence in synthetic Top-10',
        fontsize =11 ,fontweight ='bold')


        from matplotlib .patches import Patch 
        legend_elements =[
        Patch (facecolor ='#2ecc71',label ='In both Syn Top-10'),
        Patch (facecolor ='#3498db',label ='In Syn-Balanced Top-10 only'),
        Patch (facecolor ='#e67e22',label ='In Syn-Real Top-10 only'),
        Patch (facecolor ='#e74c3c',label ='Not in any Syn Top-10'),
        ]
        ax .legend (handles =legend_elements ,fontsize =8 ,loc ='lower right',
        framealpha =0.8 )
        ax .grid (axis ='x',alpha =0.3 )
        ax .spines ['top'].set_visible (False )
        ax .spines ['right'].set_visible (False )

    plt .tight_layout ()
    fig2_path =os .path .join (OUTPUT_DIR ,'XAI2_topk_ranking_comparison.pdf')
    plt .savefig (fig2_path ,bbox_inches ='tight',dpi =300 )
    plt .savefig (fig2_path .replace ('.pdf','.png'),bbox_inches ='tight',dpi =150 )
    plt .close ()
    print (f"  Figura rankings: {fig2_path }")




    print ("\n"+"="*70 )
    print ("RESUMEN XAI-2 MULTI-DATASET (Top-K Feature Overlap)")
    print ("="*70 )

    for r in all_results :
        print (f"\n  {r ['dataset']}:")
        for tk in r ['topk_results']:
            print (f"    K={tk ['K']:>2}: "
            f"Syn-Balanced={tk ['overlap_bal']:.0%}  "
            f"Syn-Real={tk ['overlap_real']:.0%}")


    print ("\n  Resumen K=10:")
    for r in all_results :
        for tk in r ['topk_results']:
            if tk ['K']==10 :
                n_shared =int (tk ['overlap_bal']*10 )
                print (f"    {r ['dataset']}: {n_shared }/10 features del clasificador real "
                f"aparecen en el top-10 del clasificador Syn-Balanced")

    print ("\n"+"="*70 )
    print ("XAI-2 MULTI-DATASET COMPLETADO")
    print ("="*70 )


if __name__ =='__main__':
    main ()
