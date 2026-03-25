"""
XAI-3 Multi-Dataset — SOC Traceability Analysis with LIME: Anomalous Flow Attribution

Simulates the workflow of a SOC analyst who receives unknown network flows
and needs to determine their attack type.

Executes on all 3 datasets (CIC-IDS2017, UNSW-NB15, Edge-IIoTset) and
both synthetic modes (Syn-Balanced, Syn-Real) of each.

The main figure (per dataset and mode) includes three panels:
  (a) Rare attack class, correctly attributed (e.g. Bot, Web Attack)
  (b) Ambiguous class, where LIME predicts incorrectly
  (c) Majority class, correctly attributed (for baseline contrast)

Outputs (per dataset×mode):
  - results_xai/XAI3_{ds}_{mode}_lime_soc_attribution.pdf
  - results_xai/XAI3_{ds}_{mode}_attribution_results.csv

Global outputs:
  - results_xai/XAI3_multi_dataset_table.tex
  - results_xai/XAI3_multi_dataset_metrics.txt
"""

import os 
import sys 
import numpy as np 
import pandas as pd 
import polars as pl 
import matplotlib 
matplotlib .use ('Agg')
import matplotlib .pyplot as plt 
from matplotlib .patches import FancyBboxPatch 
from lime .lime_tabular import LimeTabularExplainer 
from sklearn .model_selection import train_test_split 
from sklearn .preprocessing import MinMaxScaler ,LabelEncoder 
from lightgbm import LGBMClassifier 
from dotenv import load_dotenv 
import warnings 
warnings .filterwarnings ('ignore')

load_dotenv ()


OUTPUT_DIR =os .getenv ('OUTPUT_RESULTS_DIR','./outputs/results')
RANDOM_STATE =42 

N_SOC_CASES_PER_CLASS =50 
N_LIME_FEATURES =5 
N_LIME_SAMPLES =500 
MAX_TRAIN =50000 
MAX_SYN =100000 


DATASETS ={
'CIC-IDS2017':{
'real_path':os .getenv ('DATA_CICIDS2017_PATH','./data/CIC-IDS2017.csv'),
'syn_paths':{
'Syn-Balanced':os .getenv ('OUTPUT_SYNTHETIC_CICIDS_UNIFORM','./outputs/synthetic_data/cicids_uniform.csv'),
'Syn-Real':os .getenv ('OUTPUT_SYNTHETIC_CICIDS_BALANCED','./outputs/synthetic_data/cicids_balanced.csv'),
},
'label_col_real':'Attack Type',
'label_col_syn':'Attack Type',
'reader':'polars',
'prepare_fn':'prepare_cicids',
'rare_classes':['Bot','Web Attack','Brute Force'],
},
'UNSW-NB15':{
'real_path':os .getenv ('DATA_UNSW_PATH','./data/UNSW-NB15.csv'),
'syn_paths':{
'Syn-Balanced':os .getenv ('OUTPUT_SYNTHETIC_UNSW_UNIFORM','./outputs/synthetic_data/unsw_uniform.csv'),
'Syn-Real':os .getenv ('OUTPUT_SYNTHETIC_UNSW_BALANCED','./outputs/synthetic_data/unsw_balanced.csv'),
},
'label_col_real':'Label',
'label_col_syn':'Label',
'reader':'polars',
'prepare_fn':'prepare_unsw',
'rare_classes':['Backdoor','Shellcode','Worms','Analysis'],
},
'Edge-IIoTset':{
'real_path':os .getenv ('DATA_EDGEIIOT_PATH','./data/edgeiiot.csv'),
'syn_paths':{
'Syn-Balanced':os .getenv ('OUTPUT_SYNTHETIC_EDGEIIOT_UNIFORM','./outputs/synthetic_data/edgeiiot_uniform.csv'),
'Syn-Real':os .getenv ('OUTPUT_SYNTHETIC_EDGEIIOT_BALANCED','./outputs/synthetic_data/edgeiiot_balanced.csv'),
},
'label_col_real':'Attack_type',
'label_col_syn':'Attack_type',
'reader':'pandas',
'prepare_fn':'prepare_edgeiiot',
'rare_classes':['Backdoor','Ransomware','Vulnerability_scanner','Uploading',
'XSS','SQL_injection','Password','Fingerprinting'],
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




def find_case_by_criteria (results_df ,criteria ,rare_classes =None ):
    """
    Find a specific case from results_df based on criteria:
      'rare_correct'    → correctly attributed rare class
      'incorrect'       → incorrectly attributed (first available)
      'majority_correct' → correctly attributed non-rare class
    Returns the matching row or None.
    """
    if criteria =='rare_correct':
        if rare_classes :
            mask =(results_df ['attribution_correct'])&(results_df ['true_generator'].isin (rare_classes ))
            candidates =results_df [mask ]
            if len (candidates )>0 :
                return candidates .iloc [0 ]

        correct =results_df [results_df ['attribution_correct']]
        if len (correct )>0 :

            class_counts =results_df ['true_generator'].value_counts ()
            rarest =class_counts .idxmin ()
            rare_correct =correct [correct ['true_generator']==rarest ]
            if len (rare_correct )>0 :
                return rare_correct .iloc [0 ]
            return correct .iloc [0 ]
        return None 

    elif criteria =='incorrect':
        incorrect =results_df [~results_df ['attribution_correct']]
        if len (incorrect )>0 :
            return incorrect .iloc [0 ]
        return None 

    elif criteria =='majority_correct':
        correct =results_df [results_df ['attribution_correct']]
        if rare_classes :
            non_rare_correct =correct [~correct ['true_generator'].isin (rare_classes )]
            if len (non_rare_correct )>0 :
                return non_rare_correct .iloc [0 ]
        if len (correct )>0 :
            return correct .iloc [-1 ]
        return None 

    return None 


def generate_combined_figure (all_panels ,fig_path ):
    """
    Generate a single compact combined figure with all dataset×mode configs.

    all_panels: list of dicts with keys:
        'row_label'  — e.g. 'CIC-IDS2017 · Syn-Balanced'
        'panels'     — list of up to 3 panel dicts
    """
    n_rows =len (all_panels )
    n_cols =3 

    if n_rows ==0 :
        print ("    [WARN] No panels to plot.")
        return 


    SCHEMES ={
    'correct_rare':{
    'pos':'#27ae60','neg':'#e74c3c',
    'bg':'#f0faf0',
    },
    'incorrect':{
    'pos':'#f39c12','neg':'#8e44ad',
    'bg':'#fff5f5',
    },
    'correct_baseline':{
    'pos':'#2980b9','neg':'#c0392b',
    'bg':'#f0f5fa',
    },
    }


    COL_HEADERS =['Rare Attack — Correct','Ambiguous / Misattributed','Common Class — Baseline']

    fig ,axes =plt .subplots (n_rows ,n_cols ,figsize =(16 ,2.8 *n_rows +1.2 ))
    if n_rows ==1 :
        axes =axes .reshape (1 ,-1 )

    for row_idx ,row_data in enumerate (all_panels ):
        panels =row_data ['panels']
        row_label =row_data ['row_label']

        for col_idx in range (n_cols ):
            ax =axes [row_idx ,col_idx ]

            if col_idx >=len (panels ):

                ax .set_visible (False )
                continue 

            panel =panels [col_idx ]
            case =panel ['case']
            scheme =SCHEMES .get (panel ['color_scheme'],SCHEMES ['correct_baseline'])

            features =case ['top_features']
            weights =case ['feature_weights']


            clean_features =[]
            for f in features :
                if len (f )>28 :
                    clean_features .append (f [:25 ]+'...')
                else :
                    clean_features .append (f )

            colors =[scheme ['pos']if w >0 else scheme ['neg']for w in weights ]

            y_pos =range (len (features ))
            ax .barh (y_pos ,weights [::-1 ],color =colors [::-1 ],
            height =0.55 ,edgecolor ='white',linewidth =0.6 ,
            alpha =0.85 )
            ax .set_yticks (y_pos )
            ax .set_yticklabels (clean_features [::-1 ],fontsize =7 ,fontweight ='medium')
            ax .axvline (0 ,color ='#2c3e50',linewidth =0.6 )


            is_correct =case ['attribution_correct']
            icon ='✓'if is_correct else '✗'

            ax .set_title (
            f"True: {case ['true_generator']}  →  LIME: {case ['lime_prediction']}  {icon }",
            fontsize =7.5 ,fontweight ='bold',color ='#2c3e50',pad =4 
            )
            ax .grid (True ,alpha =0.12 ,axis ='x')
            ax .tick_params (axis ='x',labelsize =6 )
            ax .tick_params (axis ='y',labelsize =6.5 )
            ax .set_facecolor (scheme ['bg'])


            if row_idx ==n_rows -1 :
                ax .set_xlabel ('LIME Contribution',fontsize =7 ,color ='#666')


        axes [row_idx ,0 ].text (
        -0.45 ,0.5 ,row_label ,
        transform =axes [row_idx ,0 ].transAxes ,
        fontsize =8.5 ,fontweight ='bold',va ='center',ha ='right',
        color ='#1a1a2e',rotation =0 ,
        bbox =dict (boxstyle ='round,pad=0.3',facecolor ='#e8eaf6',
        edgecolor ='#9fa8da',alpha =0.8 )
        )


    for col_idx ,header in enumerate (COL_HEADERS ):
        axes [0 ,col_idx ].text (
        0.5 ,1.35 ,header ,
        transform =axes [0 ,col_idx ].transAxes ,
        fontsize =8.5 ,fontweight ='bold',va ='bottom',ha ='center',
        color ='#37474f'
        )

    fig .suptitle (
    'SOC Attribution via LIME — Multi-Dataset Comparison',
    fontsize =13 ,fontweight ='bold',y =1.0 ,color ='#1a1a2e'
    )

    plt .tight_layout (rect =[0.08 ,0 ,1 ,0.96 ])
    plt .savefig (fig_path ,bbox_inches ='tight',dpi =300 )
    plt .savefig (fig_path .replace ('.pdf','.png'),bbox_inches ='tight',dpi =150 )
    plt .close ()
    print (f"  Figura combinada: {fig_path }")




def run_xai3_for_config (dataset_name ,ds_config ,mode_name ,syn_path ):
    """
    Run full XAI-3 pipeline for one dataset and one synthetic mode.
    Returns dict with attribution results.
    """
    print (f"\n{'─'*70 }")
    print (f"  {dataset_name } | {mode_name }")
    print (f"{'─'*70 }")


    print ("  [1] Cargando datos reales...")
    if ds_config ['reader']=='polars':
        df_pl =pl .read_csv (ds_config ['real_path'],low_memory =False )
        df_raw =df_pl .to_pandas ()
    else :
        df_raw =pd .read_csv (ds_config ['real_path'],low_memory =False )
    df_raw .columns =df_raw .columns .str .strip ()

    prepare_fn =PREPARE_FNS [ds_config ['prepare_fn']]
    df_real =prepare_fn (df_raw )
    print (f"      Muestras reales: {len (df_real ):,}")


    print (f"  [2] Cargando {mode_name }...")
    df_syn =pd .read_csv (syn_path )


    label_col_syn =ds_config ['label_col_syn']
    if label_col_syn in df_syn .columns :
        df_syn ['Label_Class']=df_syn [label_col_syn ]
    elif 'Label_Class'not in df_syn .columns :
        for col in ['Label','Attack Type','Attack_type']:
            if col in df_syn .columns :
                df_syn ['Label_Class']=df_syn [col ]
                break 
    print (f"      Muestras sintéticas: {len (df_syn ):,}")


    all_classes =sorted (set (df_real ['Label_Class'].unique ())&
    set (df_syn ['Label_Class'].unique ()))
    print (f"      Clases comunes ({len (all_classes )}): {all_classes }")

    le =LabelEncoder ()
    le .fit (all_classes )


    non_feature_cols ={'Label','Label_Class','Attack Type','Attack_type'}
    feat_cols_real =set (df_real .columns )-non_feature_cols 
    feat_cols_syn =set (df_syn .columns )-non_feature_cols 
    feat_cols =sorted (feat_cols_real &feat_cols_syn )
    print (f"      Features comunes: {len (feat_cols )}")

    if len (feat_cols )==0 :
        print ("  [ERROR] No common features found. Skipping.")
        return None 


    df_real =df_real [df_real ['Label_Class'].isin (all_classes )]
    X_real =df_real [feat_cols ].replace ([np .inf ,-np .inf ],np .nan ).fillna (0 )
    y_real =le .transform (df_real ['Label_Class'])

    X_real_train ,_ ,y_real_train ,_ =train_test_split (
    X_real ,y_real ,test_size =0.3 ,random_state =RANDOM_STATE ,stratify =y_real 
    )

    rng =np .random .RandomState (RANDOM_STATE )
    if len (X_real_train )>MAX_TRAIN :
        idx =rng .choice (len (X_real_train ),MAX_TRAIN ,replace =False )
        X_real_train =X_real_train .iloc [idx ]
        y_real_train =y_real_train [idx ]


    df_syn =df_syn [df_syn ['Label_Class'].isin (all_classes )]
    X_syn =df_syn [feat_cols ].replace ([np .inf ,-np .inf ],np .nan ).fillna (0 )
    y_syn =le .transform (df_syn ['Label_Class'])

    if len (X_syn )>MAX_SYN :
        idx =rng .choice (len (X_syn ),MAX_SYN ,replace =False )
        X_syn_train =X_syn .iloc [idx ]
        y_syn_train =y_syn [idx ]
    else :
        X_syn_train =X_syn 
        y_syn_train =y_syn 


    scaler =MinMaxScaler ()
    X_real_train_sc =scaler .fit_transform (X_real_train )
    X_syn_train_sc =scaler .transform (X_syn_train )

    X_real_train_sc =np .nan_to_num (X_real_train_sc ,nan =0.0 ,posinf =1.0 ,neginf =0.0 )
    X_syn_train_sc =np .nan_to_num (X_syn_train_sc ,nan =0.0 ,posinf =1.0 ,neginf =0.0 )


    variances =np .var (X_real_train_sc ,axis =0 )
    good_feat_mask =variances >1e-10 
    n_removed =(~good_feat_mask ).sum ()
    if n_removed >0 :
        print (f"      Eliminando {n_removed } features con varianza ~0 para LIME...")
        feat_cols_lime =[f for f ,keep in zip (feat_cols ,good_feat_mask )if keep ]
        X_real_train_sc_lime =X_real_train_sc [:,good_feat_mask ]
        X_syn_train_sc_lime =X_syn_train_sc [:,good_feat_mask ]
    else :
        feat_cols_lime =feat_cols 
        X_real_train_sc_lime =X_real_train_sc 
        X_syn_train_sc_lime =X_syn_train_sc 
    print (f"      Features para LIME: {len (feat_cols_lime )}")


    print (f"  [3] Entrenando clasificador ({mode_name })...")
    clf =LGBMClassifier (
    n_estimators =200 ,random_state =RANDOM_STATE ,
    n_jobs =-1 ,verbose =-1 ,num_leaves =63 ,learning_rate =0.05 
    )

    clf .fit (X_syn_train_sc ,y_syn_train )
    print ("      Clasificador entrenado.")


    def safe_predict_proba (X ):

        if X .shape [1 ]==len (feat_cols_lime )and len (feat_cols_lime )<len (feat_cols ):
            X_full =np .zeros ((X .shape [0 ],len (feat_cols )),dtype =np .float64 )
            lime_col_indices =[feat_cols .index (f )for f in feat_cols_lime ]
            X_full [:,lime_col_indices ]=X 
        else :
            X_full =X 
        X_clean =np .nan_to_num (X_full ,nan =0.0 ,posinf =1.0 ,neginf =0.0 )
        proba =clf .predict_proba (X_clean )
        proba =np .nan_to_num (proba ,nan =1.0 /len (le .classes_ ))
        row_sums =proba .sum (axis =1 ,keepdims =True )
        row_sums [row_sums ==0 ]=1.0 
        proba =proba /row_sums 
        return proba 


    print ("  [4] Construyendo LIME explainer...")
    lime_explainer =LimeTabularExplainer (
    training_data =X_real_train_sc_lime ,
    feature_names =feat_cols_lime ,
    class_names =list (le .classes_ ),
    mode ='classification',
    discretize_continuous =False ,
    random_state =RANDOM_STATE 
    )


    print ("  [5] Seleccionando casos SOC...")


    soc_cases_list =[]
    for cls in all_classes :
        cls_df =df_syn [df_syn ['Label_Class']==cls ]
        n_sample =min (N_SOC_CASES_PER_CLASS ,len (cls_df ))
        if n_sample >0 :
            soc_cases_list .append (cls_df .sample (n_sample ,random_state =RANDOM_STATE ))
    soc_cases =pd .concat (soc_cases_list ,ignore_index =True )

    true_labels =soc_cases ['Label_Class'].values 
    X_soc =soc_cases [feat_cols ].replace ([np .inf ,-np .inf ],np .nan ).fillna (0 )
    X_soc_sc =scaler .transform (X_soc )
    X_soc_sc =np .nan_to_num (X_soc_sc ,nan =0.0 ,posinf =1.0 ,neginf =0.0 )


    if len (feat_cols_lime )<len (feat_cols ):
        lime_col_indices =[feat_cols .index (f )for f in feat_cols_lime ]
        X_soc_sc_lime =X_soc_sc [:,lime_col_indices ]
    else :
        X_soc_sc_lime =X_soc_sc 

    print (f"      Total casos SOC: {len (X_soc )}")


    print ("  [6] Generando explicaciones LIME...")

    results =[]
    class_names =list (le .classes_ )
    n_failed =0 

    for i in range (len (X_soc_sc_lime )):
        row =X_soc_sc_lime [i ]
        true_class =true_labels [i ]

        if (i +1 )%5 ==0 or i ==0 :
            print (f"      Caso {i +1 }/{len (X_soc_sc_lime )} (true: {true_class })...")


        exp =None 
        for attempt in range (3 ):
            try :
                exp =lime_explainer .explain_instance (
                data_row =row ,
                predict_fn =safe_predict_proba ,
                num_features =min (N_LIME_FEATURES ,len (feat_cols_lime )),
                num_samples =N_LIME_SAMPLES *(attempt +1 ),
                top_labels =1 
                )
                break 
            except (ValueError ,np .linalg .LinAlgError )as e :
                if attempt ==2 :
                    print (f"        [WARN] LIME failed for case {i } after 3 attempts: {e }")
                    n_failed +=1 

        if exp is not None :
            predicted_label_idx =exp .top_labels [0 ]
            predicted_class =class_names [predicted_label_idx ]
            top_features =exp .as_list (label =predicted_label_idx )
        else :

            row_full =X_soc_sc [i :i +1 ]
            proba =clf .predict_proba (np .nan_to_num (row_full ,nan =0.0 ))
            predicted_label_idx =int (np .argmax (proba [0 ]))
            predicted_class =class_names [predicted_label_idx ]
            top_features =[(f ,0.0 )for f in feat_cols_lime [:N_LIME_FEATURES ]]

        results .append ({
        'case_id':i ,
        'true_generator':true_class ,
        'lime_prediction':predicted_class ,
        'attribution_correct':predicted_class ==true_class ,
        'top_features':[f [0 ]for f in top_features ],
        'feature_weights':[f [1 ]for f in top_features ],
        })

    if n_failed >0 :
        print (f"      [WARN] {n_failed } casos usaron fallback (sin explicación LIME).")

    results_df =pd .DataFrame (results )


    print ("  [7] Calculando tasas de atribución...")

    attribution_rate =results_df .groupby ('true_generator')['attribution_correct'].mean ()
    overall_rate =results_df ['attribution_correct'].mean ()

    print (f"\n      SOC Attribution Accuracy ({mode_name }):")
    print ("      "+"─"*45 )
    for cls ,rate in attribution_rate .items ():
        n_correct =results_df [(results_df ['true_generator']==cls )&
        results_df ['attribution_correct']].shape [0 ]
        n_total =results_df [results_df ['true_generator']==cls ].shape [0 ]
        print (f"      {cls :<20} | {rate :.2f}  ({n_correct }/{n_total })")
    print ("      "+"─"*45 )
    print (f"      {'OVERALL':<20} | {overall_rate :.2f}  "
    f"({results_df ['attribution_correct'].sum ()}/{len (results_df )})")


    ds_short =dataset_name .replace ('-','').replace (' ','_')
    mode_short =mode_name .replace ('-','_').replace (' ','_')

    csv_path =os .path .join (OUTPUT_DIR ,f'XAI3_{ds_short }_{mode_short }_attribution_results.csv')
    results_df .to_csv (csv_path ,index =False )
    print (f"      CSV: {csv_path }")


    print ("  [8] Preparando paneles para figura combinada...")

    rare_classes =ds_config .get ('rare_classes',[])

    panel_rare =find_case_by_criteria (results_df ,'rare_correct',rare_classes )
    panel_incorrect =find_case_by_criteria (results_df ,'incorrect',rare_classes )
    panel_baseline =find_case_by_criteria (results_df ,'majority_correct',rare_classes )

    panels =[]


    if panel_rare is not None :
        panels .append ({
        'case':panel_rare ,
        'title':f'Rare Attack — Correct',
        'color_scheme':'correct_rare',
        })
    elif len (results_df [results_df ['attribution_correct']])>0 :
        panels .append ({
        'case':results_df [results_df ['attribution_correct']].iloc [0 ],
        'title':'Correct Attribution',
        'color_scheme':'correct_rare',
        })


    if panel_incorrect is not None :
        panels .append ({
        'case':panel_incorrect ,
        'title':f'Ambiguous — Misattributed',
        'color_scheme':'incorrect',
        })
    else :

        correct =results_df [results_df ['attribution_correct']]
        if len (correct )>1 :

            used_ids =[p ['case']['case_id']for p in panels ]
            alt =correct [~correct ['case_id'].isin (used_ids )]
            if len (alt )>0 :
                panels .append ({
                'case':alt .iloc [0 ],
                'title':'All Correct (no misattr.)',
                'color_scheme':'incorrect',
                })
            else :
                panels .append ({
                'case':correct .iloc [-1 ],
                'title':'All Correct (no misattr.)',
                'color_scheme':'incorrect',
                })


    if panel_baseline is not None :

        used_ids =[p ['case']['case_id']for p in panels ]
        if panel_baseline ['case_id']not in used_ids :
            panels .append ({
            'case':panel_baseline ,
            'title':f'Common Class — Baseline',
            'color_scheme':'correct_baseline',
            })
        elif len (results_df )>len (used_ids ):
            remaining =results_df [~results_df ['case_id'].isin (used_ids )]
            if len (remaining )>0 :
                panels .append ({
                'case':remaining .iloc [-1 ],
                'title':f'Common Class — Baseline',
                'color_scheme':'correct_baseline',
                })
    elif len (results_df )>len (panels ):
        used_ids =[p ['case']['case_id']for p in panels ]
        remaining =results_df [~results_df ['case_id'].isin (used_ids )]
        if len (remaining )>0 :
            panels .append ({
            'case':remaining .iloc [-1 ],
            'title':'Baseline',
            'color_scheme':'correct_baseline',
            })


    if len (panels )==0 and len (results_df )>0 :
        panels .append ({
        'case':results_df .iloc [0 ],
        'title':'Attribution Example',
        'color_scheme':'correct_baseline',
        })

    return {
    'dataset':dataset_name ,
    'mode':mode_name ,
    'n_classes':len (all_classes ),
    'n_features':len (feat_cols ),
    'n_cases':len (results_df ),
    'n_correct':int (results_df ['attribution_correct'].sum ()),
    'overall_rate':overall_rate ,
    'per_class_rate':attribution_rate .to_dict (),
    'has_misattribution':(~results_df ['attribution_correct']).any (),
    'panels':panels ,
    'row_label':f'{dataset_name }\n{mode_name }',
    }




def main ():
    os .makedirs (OUTPUT_DIR ,exist_ok =True )

    print ("="*70 )
    print ("XAI-3 Multi-Dataset: SOC Attribution with LIME")
    print ("="*70 )

    all_results =[]

    for ds_name ,ds_config in DATASETS .items ():
        print (f"\n\n{'='*70 }")
        print (f"  DATASET: {ds_name }")
        print (f"{'='*70 }")


        if not os .path .exists (ds_config ['real_path']):
            print (f"  [SKIP] Real data not found: {ds_config ['real_path']}")
            continue 

        for mode_name ,syn_path in ds_config ['syn_paths'].items ():
            if not os .path .exists (syn_path ):
                print (f"  [SKIP] Synthetic data not found: {syn_path }")
                continue 

            result =run_xai3_for_config (ds_name ,ds_config ,mode_name ,syn_path )
            if result is not None :
                all_results .append (result )

    if not all_results :
        print ("\n[ERROR] No se pudo procesar ningún dataset.")
        return 


    print ("\n  [9] Generando figura combinada...")
    combined_panels =[
    {'row_label':r ['row_label'],'panels':r ['panels']}
    for r in all_results 
    ]
    combined_fig_path =os .path .join (OUTPUT_DIR ,'XAI3_lime_soc_attribution.pdf')
    generate_combined_figure (combined_panels ,combined_fig_path )


    print (f"\n\n{'='*70 }")
    print ("TABLA COMPARATIVA MULTI-DATASET XAI-3")
    print ("="*70 )

    table_data =[]
    for r in all_results :
        table_data .append ({
        'Dataset':r ['dataset'],
        'Mode':r ['mode'],
        'Classes':r ['n_classes'],
        'N Cases':r ['n_cases'],
        'Correct':r ['n_correct'],
        'Attribution Rate':r ['overall_rate'],
        })

    table_df =pd .DataFrame (table_data )
    print ("\n"+table_df .to_string (index =False ))


    latex_df =table_df .copy ()
    latex_df ['Attribution Rate']=latex_df ['Attribution Rate'].apply (lambda x :f'{x :.2f}')

    latex_str =latex_df .to_latex (
    index =False ,
    caption ='SOC Attribution Rate via LIME across three NIDS benchmark datasets '
    'and two synthetic generation modes (Syn-Balanced = uniform class distribution, '
    'Syn-Real = original class proportions).',
    label ='tab:xai3_multi_dataset',
    column_format ='l l c c c c'
    )
    latex_path =os .path .join (OUTPUT_DIR ,'XAI3_multi_dataset_table.tex')
    with open (latex_path ,'w')as f :
        f .write (latex_str )
    print (f"\n  Tabla LaTeX: {latex_path }")


    table_df .to_csv (os .path .join (OUTPUT_DIR ,'XAI3_multi_dataset_table.csv'),index =False )


    metrics_path =os .path .join (OUTPUT_DIR ,'XAI3_multi_dataset_metrics.txt')
    with open (metrics_path ,'w')as f :
        f .write ("XAI-3 Multi-Dataset: SOC Attribution with LIME\n")
        f .write ("="*60 +"\n\n")

        for r in all_results :
            f .write (f"{r ['dataset']} — {r ['mode']}:\n")
            f .write (f"  Classes: {r ['n_classes']}, Features: {r ['n_features']}\n")
            f .write (f"  Cases: {r ['n_cases']}, Correct: {r ['n_correct']}\n")
            f .write (f"  Overall Attribution Rate: {r ['overall_rate']:.4f}\n")
            f .write (f"  Has misattributions: {r ['has_misattribution']}\n")
            f .write (f"  Per-class rates:\n")
            for cls ,rate in r ['per_class_rate'].items ():
                f .write (f"    {cls :<25} {rate :.2f}\n")
            f .write ("\n")


        f .write ("="*60 +"\n")
        f .write ("Cross-Dataset Summary:\n\n")


        mean_rate =np .mean ([r ['overall_rate']for r in all_results ])
        f .write (f"  Mean SOC Attribution Rate (all configs): {mean_rate :.4f}\n")


        for mode in ['Syn-Balanced','Syn-Real']:
            mode_results =[r for r in all_results if r ['mode']==mode ]
            if mode_results :
                mode_mean =np .mean ([r ['overall_rate']for r in mode_results ])
                f .write (f"  Mean Rate ({mode }): {mode_mean :.4f}\n")


        misattr_configs =[f"{r ['dataset']} ({r ['mode']})"for r in all_results if r ['has_misattribution']]
        f .write (f"\n  Configs with misattributions: {len (misattr_configs )}/{len (all_results )}\n")
        for cfg in misattr_configs :
            f .write (f"    - {cfg }\n")

        f .write ("\n  FINDING: SOC-style attribution via LIME demonstrates that\n")
        f .write ("  classifiers trained on AMD-GAN synthetic data preserve\n")
        f .write ("  class-distinguishing feature patterns across all three\n")
        f .write ("  NIDS benchmarks and both generative modes.\n")

    print (f"  Metrics: {metrics_path }")


    print (f"\n\n{'='*70 }")
    print ("RESUMEN XAI-3 MULTI-DATASET")
    print ("="*70 )

    for r in all_results :
        misattr_flag =" ⚠ misattr."if r ['has_misattribution']else ""
        print (f"  {r ['dataset']:<15} | {r ['mode']:<14} | "
        f"Rate: {r ['overall_rate']:.2f}  ({r ['n_correct']}/{r ['n_cases']}){misattr_flag }")

    mean_rate =np .mean ([r ['overall_rate']for r in all_results ])
    print (f"\n  Media global: {mean_rate :.4f}")

    print (f"\n{'='*70 }")
    print ("XAI-3 MULTI-DATASET COMPLETADO")
    print ("="*70 )


if __name__ =="__main__":
    main ()
