"""
WGAN-GP Training for Class-Specific Synthetic Network Traffic Generation (Edge-IIoT)

Trains adaptive WGAN-GP models using AMD-GAN framework for Edge-IIoT 2022 dataset
with dynamic configuration based on class cardinality.

Classes: Normal, DDoS (4 types), SQL Injection, Password, Vulnerability Scanner,
Uploading, Backdoor, Port Scanning, XSS, Ransomware, MITM, Fingerprinting
"""

import os 
import sys 
import math 
import json 
import argparse 
import pickle 
import numpy as np 
import pandas as pd 
import matplotlib .pyplot as plt 
from scipy .stats import gaussian_kde 
from datetime import datetime 
from sklearn .preprocessing import MinMaxScaler ,LabelEncoder 
import tensorflow as tf 
from tensorflow .keras import layers ,models ,optimizers 
from dotenv import load_dotenv 


load_dotenv ()


os .environ ['TF_CPP_MIN_LOG_LEVEL']='2'


DATASET_PATH =os .getenv ('DATA_EDGEIIOT_PATH','./data/Edge-IIoT.csv')
OUTPUT_DIR =os .getenv ('OUTPUT_MODELS_DIR','./outputs/models')
LATENT_DIM =int (os .getenv ('LATENT_DIM',100 ))
LABEL_COLUMN ='Attack_type'


VALID_CLASSES =[
'Normal','DDoS_UDP','DDoS_ICMP','SQL_injection','Password',
'Vulnerability_scanner','DDoS_TCP','DDoS_HTTP','Uploading',
'Backdoor','Port_Scanning','XSS','Ransomware','MITM','Fingerprinting'
]


MINORITY_CLASSES =['MITM','Fingerprinting','Ransomware','XSS']





D_FEATURES =58 


RHO_HIGH =2.5 
RHO_LOW =0.8 


THETA_HIGH =math .ceil (RHO_HIGH *D_FEATURES **2 )
THETA_LOW =math .ceil (RHO_LOW *D_FEATURES **2 )


CLASS_TO_FOLDER ={c :c .lower ()for c in VALID_CLASSES }



DROP_COLUMNS =[
'frame.time',
'arp.dst.proto_ipv4','arp.src.proto_ipv4',
'tcp.options','tcp.payload',
'mqtt.conack.flags','mqtt.msg','mqtt.protoname','mqtt.topic',
'Attack_label','Attack_type'
]


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


CONFIG_LARGE ={
'batch_size':128 ,
'epochs':15000 ,
'n_critic':5 ,
'lambda_gp':10.0 ,
'generator_layers':[256 ,512 ,256 ],
'critic_layers':[512 ,256 ,128 ],
'learning_rate':1e-4 ,
'oversample_factor':1 ,
'noise_std':0.0 ,
}


CONFIG_SMALL ={
'batch_size':32 ,
'epochs':25000 ,
'n_critic':3 ,
'lambda_gp':15.0 ,
'generator_layers':[128 ,256 ,128 ],
'critic_layers':[256 ,128 ,64 ],
'learning_rate':5e-5 ,
'oversample_factor':10 ,
'noise_std':0.02 ,
}


CONFIG_VERY_SMALL ={
'batch_size':16 ,
'epochs':30000 ,
'n_critic':2 ,
'lambda_gp':20.0 ,
'generator_layers':[64 ,128 ,64 ],
'critic_layers':[128 ,64 ,32 ],
'learning_rate':2e-5 ,
'oversample_factor':20 ,
'noise_std':0.03 ,
}





class PreprocesadorEdgeIIoT :
    """Preprocesador específico para Edge-IIoT 2022."""

    def __init__ (self ,dataset_path :str ):
        self .path =dataset_path 
        self .scaler =MinMaxScaler (feature_range =(-1 ,1 ))
        self .columnas_features =None 

    def cargar (self )->pd .DataFrame :
        file_size =os .path .getsize (self .path )
        print (f"  Dataset size: {file_size /(1024 *1024 ):.2f} MB")

        start_time =datetime .now ()
        print ("  Reading data with pandas...")
        df =pd .read_csv (self .path ,low_memory =False )
        print (f"  Data read in {datetime .now ()-start_time }")

        df .columns =df .columns .str .strip ()
        return df 

    @staticmethod 
    def preparar_df_base (df :pd .DataFrame )->pd .DataFrame :
        """Prepara el DataFrame base: selecciona features numéricas, expande IPs y filtra clases."""
        df =df .copy ()
        df .columns =df .columns .str .strip ()


        df =df [df [LABEL_COLUMN ].isin (VALID_CLASSES )].copy ()


        cols_needed =['ip.src_host','ip.dst_host']+FEATURES_BASE +[LABEL_COLUMN ]
        cols_available =[c for c in cols_needed if c in df .columns ]
        df_result =df [cols_available ].copy ()


        for col in [c for c in FEATURES_BASE if c in df_result .columns ]:
            df_result [col ]=pd .to_numeric (df_result [col ],errors ='coerce').fillna (0 )


        if 'ip.src_host'in df_result .columns :
            octetos =df_result ['ip.src_host'].astype (str ).str .split ('.',expand =True )
            for i in range (4 ):
                df_result [f'Src_IP_{i +1 }']=pd .to_numeric (octetos [i ],errors ='coerce').fillna (0 ).astype (int )
            df_result .drop (columns =['ip.src_host'],inplace =True )
        else :
            for i in range (4 ):
                df_result [f'Src_IP_{i +1 }']=0 

        if 'ip.dst_host'in df_result .columns :
            octetos =df_result ['ip.dst_host'].astype (str ).str .split ('.',expand =True )
            for i in range (4 ):
                df_result [f'Dst_IP_{i +1 }']=pd .to_numeric (octetos [i ],errors ='coerce').fillna (0 ).astype (int )
            df_result .drop (columns =['ip.dst_host'],inplace =True )
        else :
            for i in range (4 ):
                df_result [f'Dst_IP_{i +1 }']=0 

        df_result .replace ([np .inf ,-np .inf ],np .nan ,inplace =True )
        df_result .fillna (0 ,inplace =True )

        return df_result 

    def preparacion_gan_subset (self ,df_subset :pd .DataFrame ):
        """Prepara un subconjunto para entrenamiento GAN: log-transform + scale."""
        features =df_subset .drop (columns =[LABEL_COLUMN ])
        labels =df_subset [LABEL_COLUMN ]
        self .columnas_features =features .columns .tolist ()

        features_proc =features .copy ()

        for col in COLUMNAS_LOG :
            if col in features_proc .columns :
                features_proc [col ]=np .log1p (features_proc [col ].clip (lower =0 ))
                features_proc [col ]=features_proc [col ].clip (lower =-20 ,upper =20 )

        features_proc .replace ([np .inf ,-np .inf ],np .nan ,inplace =True )
        features_proc .fillna (0 ,inplace =True )

        X =self .scaler .fit_transform (features_proc )
        return X ,labels 





def oversample_with_noise (X ,factor =10 ,noise_std =0.02 ):
    """Oversampling con ruido gaussiano para densificar la distribución."""
    if factor <=1 and noise_std ==0 :
        return X 

    print (f"  Aplicando oversampling x{factor } con noise_std={noise_std }")

    X_augmented =[X ]
    for _ in range (factor -1 ):
        noise =np .random .normal (0 ,noise_std ,X .shape )
        X_noisy =np .clip (X +noise ,-1 ,1 )
        X_augmented .append (X_noisy )

    X_final =np .vstack (X_augmented )
    np .random .shuffle (X_final )
    print (f"  Datos aumentados: {len (X )} -> {len (X_final )}")
    return X_final 





def build_generator (latent_dim ,output_dim ,layer_sizes =[256 ,512 ,256 ]):
    noise =layers .Input (shape =(latent_dim ,))
    x =noise 
    for i ,units in enumerate (layer_sizes ):
        x =layers .Dense (units )(x )
        x =layers .LeakyReLU (0.2 )(x )
        if i <len (layer_sizes )-1 :
            x =layers .BatchNormalization (momentum =0.8 )(x )
    output =layers .Dense (output_dim ,activation ='tanh')(x )
    return models .Model (noise ,output ,name ="Generator")


def build_critic (input_dim ,layer_sizes =[512 ,256 ,128 ],dropout_rate =0.0 ):
    inp =layers .Input (shape =(input_dim ,))
    x =inp 
    for units in layer_sizes :
        x =layers .Dense (units )(x )
        x =layers .LeakyReLU (0.2 )(x )
        if dropout_rate >0 :
            x =layers .Dropout (dropout_rate )(x )
    output =layers .Dense (1 )(x )
    return models .Model (inp ,output ,name ="Critic")


def gradient_penalty (critic ,real_samples ,fake_samples ):
    batch_size =tf .shape (real_samples )[0 ]
    alpha =tf .random .uniform ([batch_size ,1 ],0.0 ,1.0 )
    alpha =tf .broadcast_to (alpha ,tf .shape (real_samples ))
    interpolated =alpha *real_samples +(1 -alpha )*fake_samples 

    with tf .GradientTape ()as tape :
        tape .watch (interpolated )
        validity =critic (interpolated )

    gradients =tape .gradient (validity ,interpolated )
    gradients =tf .reshape (gradients ,[batch_size ,-1 ])
    return tf .reduce_mean ((tf .norm (gradients ,axis =1 )-1.0 )**2 )


def train_wgan_gp (generator ,critic ,X_train ,config ,print_interval =1000 ):
    """Entrenamiento WGAN-GP con configuración personalizada."""
    latent_dim =LATENT_DIM 
    batch_size =config ['batch_size']
    epochs =config ['epochs']
    n_critic =config ['n_critic']
    lambda_gp =config ['lambda_gp']
    lr =config ['learning_rate']

    gen_optimizer =optimizers .Adam (learning_rate =lr ,beta_1 =0.0 ,beta_2 =0.9 )
    critic_optimizer =optimizers .Adam (learning_rate =lr ,beta_1 =0.0 ,beta_2 =0.9 )

    X_train =tf .convert_to_tensor (X_train ,dtype =tf .float32 )
    n_samples =X_train .shape [0 ]

    g_losses ,c_losses =[],[]

    print (f"\n  Iniciando entrenamiento WGAN-GP:")
    print (f"    - Batch size: {batch_size }")
    print (f"    - Epochs: {epochs }")
    print (f"    - N_critic: {n_critic }")
    print (f"    - Lambda GP: {lambda_gp }")
    print (f"    - Learning rate: {lr }")
    print (f"    - Datos de entrenamiento: {n_samples }")

    start_time =datetime .now ()

    for epoch in range (1 ,epochs +1 ):

        for _ in range (n_critic ):
            idx =np .random .randint (0 ,n_samples ,batch_size )
            real_samples =tf .gather (X_train ,idx )
            noise =tf .random .normal ((batch_size ,latent_dim ))
            fake_samples =generator (noise ,training =True )

            with tf .GradientTape ()as tape :
                real_validity =critic (real_samples ,training =True )
                fake_validity =critic (fake_samples ,training =True )
                gp =gradient_penalty (critic ,real_samples ,fake_samples )
                critic_loss =(tf .reduce_mean (fake_validity )
                -tf .reduce_mean (real_validity )
                +lambda_gp *gp )

            grads =tape .gradient (critic_loss ,critic .trainable_variables )
            critic_optimizer .apply_gradients (zip (grads ,critic .trainable_variables ))


        noise =tf .random .normal ((batch_size ,latent_dim ))
        with tf .GradientTape ()as tape :
            fake_samples =generator (noise ,training =True )
            fake_validity =critic (fake_samples ,training =True )
            generator_loss =-tf .reduce_mean (fake_validity )

        grads =tape .gradient (generator_loss ,generator .trainable_variables )
        gen_optimizer .apply_gradients (zip (grads ,generator .trainable_variables ))

        g_losses .append (generator_loss .numpy ())
        c_losses .append (critic_loss .numpy ())

        if epoch %print_interval ==0 or epoch ==1 :
            elapsed =datetime .now ()-start_time 
            eta =elapsed /epoch *(epochs -epoch )
            print (f"    [Epoch {epoch :>6}/{epochs }] C_loss: {critic_loss .numpy ():>8.4f} | "
            f"G_loss: {generator_loss .numpy ():>8.4f} | ETA: {str (eta ).split ('.')[0 ]}")

    total_time =datetime .now ()-start_time 
    print (f"\n  Entrenamiento completado en {total_time }")
    return g_losses ,c_losses 





def generate_samples (generator ,n_samples ,latent_dim ):
    noise =np .random .normal (0 ,1 ,(n_samples ,latent_dim ))
    return generator .predict (noise ,verbose =0 )


def reconstruir_features (X_synthetic ,scaler ,feature_names ):
    """Reconstruye features a escala original."""
    X_inv =scaler .inverse_transform (X_synthetic )
    df_rec =pd .DataFrame (X_inv ,columns =feature_names )


    for col in COLUMNAS_LOG :
        if col in df_rec .columns :
            df_rec [col ]=np .expm1 (df_rec [col ])


    port_cols =['tcp.srcport','tcp.dstport','udp.port']
    for col in port_cols :
        if col in df_rec .columns :
            df_rec [col ]=df_rec [col ].round ().clip (0 ,65535 ).astype (int )


    for col in df_rec .columns :
        if col .startswith ('Src_IP_')or col .startswith ('Dst_IP_'):
            df_rec [col ]=df_rec [col ].round ().clip (0 ,255 ).astype (int )


    flag_cols =[c for c in df_rec .columns if 'connection.'in c or 'flags.ack'in c 
    or 'retransmission'in c or 'retransmit'in c 
    or 'conflag'in c or 'conflags'in c ]
    for col in flag_cols :
        if col in df_rec .columns :
            df_rec [col ]=df_rec [col ].round ().clip (0 ,1 ).astype (int )


    for col in df_rec .columns :
        df_rec [col ]=df_rec [col ].clip (lower =0 )


    int_cols =['arp.opcode','arp.hw.size','tcp.flags','tcp.len',
    'dns.qry.name.len','dns.qry.type','mqtt.msgtype',
    'mqtt.proto_len','mqtt.topic_len','mqtt.ver',
    'mbtcp.len','mbtcp.trans_id','mbtcp.unit_id']
    for col in int_cols :
        if col in df_rec .columns :
            df_rec [col ]=df_rec [col ].round ().astype (int )

    return df_rec 


def plot_training_curves (g_losses ,c_losses ,output_path ):
    fig ,axes =plt .subplots (1 ,2 ,figsize =(12 ,4 ))

    window =min (100 ,len (g_losses )//10 )
    if window >1 :
        g_smooth =pd .Series (g_losses ).rolling (window ).mean ()
        c_smooth =pd .Series (c_losses ).rolling (window ).mean ()
    else :
        g_smooth ,c_smooth =g_losses ,c_losses 

    axes [0 ].plot (g_losses ,alpha =0.3 ,label ='Raw')
    axes [0 ].plot (g_smooth ,label ='Smoothed')
    axes [0 ].set_title ('Generator Loss')
    axes [0 ].set_xlabel ('Epoch')
    axes [0 ].legend ()
    axes [0 ].grid (True ,alpha =0.3 )

    axes [1 ].plot (c_losses ,alpha =0.3 ,label ='Raw')
    axes [1 ].plot (c_smooth ,label ='Smoothed')
    axes [1 ].set_title ('Critic Loss')
    axes [1 ].set_xlabel ('Epoch')
    axes [1 ].legend ()
    axes [1 ].grid (True ,alpha =0.3 )

    plt .tight_layout ()
    plt .savefig (output_path ,dpi =150 )
    plt .close ()


def plot_kde_comparison (X_real ,X_synth ,feature_names ,scaler ,output_path ,sample_size =5000 ):
    n_features =X_real .shape [1 ]
    cols_per_row =5 
    rows =math .ceil (n_features /cols_per_row )

    idx_real =np .random .choice (len (X_real ),min (sample_size ,len (X_real )),replace =False )
    idx_synth =np .random .choice (len (X_synth ),min (sample_size ,len (X_synth )),replace =False )
    Xr ,Xs =X_real [idx_real ],X_synth [idx_synth ]

    plt .figure (figsize =(cols_per_row *4 ,rows *3 ))

    for i in range (n_features ):
        r =Xr [:,i ].reshape (-1 ,1 )
        s =Xs [:,i ].reshape (-1 ,1 )
        r =scaler .inverse_transform (
        np .pad (r ,((0 ,0 ),(i ,scaler .n_features_in_ -i -1 )),mode ='constant'))[:,i ]
        s =scaler .inverse_transform (
        np .pad (s ,((0 ,0 ),(i ,scaler .n_features_in_ -i -1 )),mode ='constant'))[:,i ]
        r ,s =r [np .isfinite (r )],s [np .isfinite (s )]

        if np .std (r )<1e-6 :
            continue 
        try :
            kde_r =gaussian_kde (r )
            kde_s =gaussian_kde (s )
            xmin =min (r .min (),s .min ())
            xmax =max (r .max (),s .max ())
            x =np .linspace (xmin ,xmax ,300 )

            plt .subplot (rows ,cols_per_row ,i +1 )
            plt .plot (x ,kde_r (x ),label ="Real",linewidth =1.5 )
            plt .plot (x ,kde_s (x ),'--',label ="Synthetic",linewidth =1.5 )
            plt .title (feature_names [i ],fontsize =7 )
            plt .grid (alpha =0.3 )
            if i ==0 :
                plt .legend (fontsize =7 )
        except Exception :
            continue 

    plt .tight_layout ()
    plt .savefig (output_path ,dpi =150 )
    plt .close ()


def get_config_for_class (n_samples ):
    """Adaptive Training Engine: selecciona configuración basada en cardinalidad."""
    if n_samples <THETA_LOW :
        print (f"  Usando CONFIG_VERY_SMALL (N_k={n_samples :,} < θ_low={THETA_LOW :,})")
        return CONFIG_VERY_SMALL .copy ()
    elif n_samples <THETA_HIGH :
        print (f"  Usando CONFIG_SMALL (θ_low={THETA_LOW :,} ≤ N_k={n_samples :,} < θ_high={THETA_HIGH :,})")
        return CONFIG_SMALL .copy ()
    else :
        print (f"  Usando CONFIG_LARGE (N_k={n_samples :,} ≥ θ_high={THETA_HIGH :,})")
        return CONFIG_LARGE .copy ()





def main ():
    parser =argparse .ArgumentParser (
    description ='Entrenamiento WGAN-GP para Edge-IIoT 2022 (AMD-GAN)',
    formatter_class =argparse .RawDescriptionHelpFormatter ,
    epilog =f"""
Adaptive Thresholds (d={D_FEATURES }):
  θ_high = ⌈ρ_high·d²⌉ = {THETA_HIGH :,}  →  CONFIG_LARGE
  θ_low  = ⌈ρ_low·d²⌉  = {THETA_LOW :,}   →  CONFIG_SMALL
  N_k < θ_low            →  CONFIG_VERY_SMALL
        """
    )
    parser .add_argument ('--classes',nargs ='+',default =None ,
    help ='Clases específicas a entrenar')
    parser .add_argument ('--all',action ='store_true',
    help ='Entrenar TODAS las clases con configuración adaptativa')
    parser .add_argument ('--gpu',type =str ,default ='0',
    help ='GPU a utilizar (default: 0)')
    parser .add_argument ('--samples',type =int ,default =10000 ,
    help ='Número de muestras sintéticas a generar por clase')
    parser .add_argument ('--dry-run',action ='store_true',
    help ='Solo mostrar configuración sin entrenar')

    args =parser .parse_args ()

    os .environ ["CUDA_VISIBLE_DEVICES"]=args .gpu 

    print ("="*80 )
    print ("ENTRENAMIENTO WGAN-GP - Edge-IIoT 2022 (AMD-GAN)")
    print ("="*80 )
    print (f"\n  Adaptive Thresholds (d={D_FEATURES }):")
    print (f"    θ_high = ⌈{RHO_HIGH }·{D_FEATURES }²⌉ = {THETA_HIGH :,}")
    print (f"    θ_low  = ⌈{RHO_LOW }·{D_FEATURES }²⌉  = {THETA_LOW :,}")


    print ("\n[1] Cargando dataset Edge-IIoT...")
    prep_global =PreprocesadorEdgeIIoT (DATASET_PATH )
    df_raw =prep_global .cargar ()
    df_base =PreprocesadorEdgeIIoT .preparar_df_base (df_raw )


    print ("\n[2] Distribución de clases en el dataset:")
    print ("-"*60 )
    class_counts =df_base [LABEL_COLUMN ].value_counts ()
    for clase ,count in class_counts .items ():
        if count <THETA_LOW :
            marker ="!! "
            config_name ="VERY_SMALL"
        elif count <THETA_HIGH :
            marker ="!  "
            config_name ="SMALL"
        else :
            marker ="OK "
            config_name ="LARGE"
        print (f"  {marker }{clase :<25}: {count :>10,} muestras  [{config_name }]")
    print ("-"*60 )


    if args .all :
        classes_to_train =[c for c in VALID_CLASSES if c in class_counts .index ]
    elif args .classes :
        classes_to_train =args .classes 
    else :
        classes_to_train =MINORITY_CLASSES 

    print (f"\n[3] Clases a entrenar: {classes_to_train }")

    os .makedirs (OUTPUT_DIR ,exist_ok =True )

    label_encoder =LabelEncoder ().fit (df_base [LABEL_COLUMN ])

    for class_name in classes_to_train :
        print ("\n"+"="*80 )
        print (f"ENTRENANDO CLASE: {class_name }")
        print ("="*80 )

        df_cls =df_base [df_base [LABEL_COLUMN ]==class_name ].copy ()

        if len (df_cls )<100 :
            print (f"  [SKIP] Muy pocas muestras ({len (df_cls )})")
            continue 

        n_original =len (df_cls )
        print (f"\n  Muestras originales: {n_original :,}")


        MAX_TRAIN_SAMPLES =100000 
        if n_original >MAX_TRAIN_SAMPLES :
            print (f"  Submuestreando de {n_original :,} a {MAX_TRAIN_SAMPLES :,} para entrenamiento GAN")
            df_cls =df_cls .sample (n =MAX_TRAIN_SAMPLES ,random_state =42 )

        config =get_config_for_class (len (df_cls ))

        if args .dry_run :
            print (f"\n  [DRY-RUN] Configuración que se usaría:")
            for k ,v in config .items ():
                print (f"    {k }: {v }")
            continue 


        prep_cls =PreprocesadorEdgeIIoT (DATASET_PATH )
        X_cls ,_ =prep_cls .preparacion_gan_subset (df_cls )


        X_train =oversample_with_noise (
        X_cls ,
        factor =config ['oversample_factor'],
        noise_std =config ['noise_std']
        )


        generator =build_generator (
        LATENT_DIM ,X_cls .shape [1 ],
        layer_sizes =config ['generator_layers']
        )
        critic =build_critic (
        X_cls .shape [1 ],
        layer_sizes =config ['critic_layers']
        )

        print (f"\n  Arquitectura del Generador: {config ['generator_layers']}")
        print (f"  Arquitectura del Crítico: {config ['critic_layers']}")


        g_losses ,c_losses =train_wgan_gp (
        generator ,critic ,X_train ,config ,
        print_interval =max (1000 ,config ['epochs']//20 )
        )


        safe_name =CLASS_TO_FOLDER [class_name ]
        cls_dir =os .path .join (OUTPUT_DIR ,safe_name )
        os .makedirs (cls_dir ,exist_ok =True )

        generator .save (os .path .join (cls_dir ,f'generator_{safe_name }.h5'))
        critic .save (os .path .join (cls_dir ,f'critic_{safe_name }.h5'))

        with open (os .path .join (cls_dir ,'scaler.pkl'),'wb')as f :
            pickle .dump (prep_cls .scaler ,f )

        with open (os .path .join (cls_dir ,'training_config.json'),'w')as f :
            json .dump ({
            'class_name':class_name ,
            'dataset':'Edge-IIoT-2022',
            'original_samples':n_original ,
            'train_samples':len (df_cls ),
            'augmented_samples':len (X_train ),
            'n_features':X_cls .shape [1 ],
            'feature_names':prep_cls .columnas_features ,
            'd':D_FEATURES ,
            'theta_high':THETA_HIGH ,
            'theta_low':THETA_LOW ,
            'rho_high':RHO_HIGH ,
            'rho_low':RHO_LOW ,
            'config_regime':('VERY_SMALL'if n_original <THETA_LOW 
            else 'SMALL'if n_original <THETA_HIGH 
            else 'LARGE'),
            'config':config ,
            'timestamp':datetime .now ().isoformat ()
            },f ,indent =2 )


        print (f"\n  Generando {args .samples :,} muestras sintéticas...")
        X_synth =generate_samples (generator ,args .samples ,LATENT_DIM )
        np .save (os .path .join (cls_dir ,f'synthetic_scaled_{safe_name }.npy'),X_synth )

        df_synth =reconstruir_features (X_synth ,prep_cls .scaler ,prep_cls .columnas_features )
        df_synth [LABEL_COLUMN ]=class_name 
        df_synth .to_csv (os .path .join (cls_dir ,f'synthetic_reconstructed_{safe_name }.csv'),index =False )


        print ("  Generando gráficas...")
        plot_training_curves (g_losses ,c_losses ,
        os .path .join (cls_dir ,f'training_curves_{safe_name }.png'))
        plot_kde_comparison (X_cls ,X_synth ,prep_cls .columnas_features ,prep_cls .scaler ,
        os .path .join (cls_dir ,f'kde_comparison_{safe_name }.png'))

        print (f"\n  [OK] Resultados guardados en: {cls_dir }")

        del generator ,critic 
        tf .keras .backend .clear_session ()

    print ("\n"+"="*80 )
    print ("ENTRENAMIENTO COMPLETADO")
    print ("="*80 )
    print (f"\nResultados guardados en: {OUTPUT_DIR }")
    print ("\nPróximos pasos:")
    print ("  1. Generar dataset sintético:")
    print ("     python 2_2_generate_synthetic_dataset_edgeiiot.py --balanced 10000")
    print ("  2. Evaluar con TSTR:")
    print ("     python 4_2_tstr_edgeiiot.py")


if __name__ =="__main__":
    main ()
