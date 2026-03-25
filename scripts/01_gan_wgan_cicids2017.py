"""
WGAN-GP Training for Class-Specific Synthetic Network Traffic Generation (CIC-IDS2017)

Trains adaptive WGAN-GP models for minority attack classes with configuration
tiers based on class cardinality:
- Large classes: Standard configuration
- Small classes: Smaller batch size, more epochs, stronger regularization
- Very small classes: Enhanced adaptive strategies with noise injection

Usage:
    python 01_gan_wgan_cicids2017.py
"""

import os 
import sys 
import math 
import argparse 
import numpy as np 
import pandas as pd 
import polars as pl 
import pickle 
import matplotlib .pyplot as plt 
from scipy .stats import gaussian_kde 
from datetime import datetime 
from sklearn .preprocessing import MinMaxScaler ,LabelEncoder 
import tensorflow as tf 
from tensorflow .keras import layers ,models ,optimizers 
from dotenv import load_dotenv 


load_dotenv ()


os .environ ['TF_CPP_MIN_LOG_LEVEL']='2'


DATASET_PATH =os .getenv ('DATA_CICIDS2017_PATH','./data/CIC-IDS2017.csv')
OUTPUT_DIR =os .getenv ('OUTPUT_MODELS_DIR','./outputs/models')
LATENT_DIM =int (os .getenv ('LATENT_DIM',100 ))


MINORITY_CLASSES =['Bot','Web Attack']
SMALL_CLASS_THRESHOLD =15000 


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



class PreprocessorCIC :
    def __init__ (self ,dataset_path :str ):
        self .path =dataset_path 
        self .scaler =MinMaxScaler (feature_range =(-1 ,1 ))
        self .feature_columns =None 

    def load (self )->pd .DataFrame :
        """Load dataset with Polars and convert to Pandas"""
        df_pl =pl .read_csv (self .path ,low_memory =False )
        return df_pl .to_pandas ()

    @staticmethod 
    def prepare_base_df (df :pd .DataFrame ):
        """Prepare base features and expand IP addresses"""
        BASE_FEATURES =[
        'Source IP','Destination IP',
        'Source Port','Destination Port','Protocol',
        'Total Fwd Packets','Total Backward Packets',
        'Total Length of Fwd Packets','Total Length of Bwd Packets',
        'Flow Duration','Flow IAT Mean','Flow IAT Std','Fwd IAT Mean','Bwd IAT Mean',
        'Fwd Packet Length Mean','Bwd Packet Length Mean','Packet Length Std','Max Packet Length',
        'SYN Flag Count','ACK Flag Count','FIN Flag Count','RST Flag Count','PSH Flag Count'
        ]
        LABEL_COLUMN ='Label'

        df =df .copy ()
        df .columns =df .columns .str .strip ()
        df =df [BASE_FEATURES +[LABEL_COLUMN ]].copy ()


        ip_parts =df ['Source IP'].astype (str ).str .split ('.',expand =True )
        for i in range (4 ):
            df [f'Src_IP_{i +1 }']=pd .to_numeric (ip_parts [i ],errors ='coerce').fillna (0 ).astype (int )

        ip_parts =df ['Destination IP'].astype (str ).str .split ('.',expand =True )
        for i in range (4 ):
            df [f'Dst_IP_{i +1 }']=pd .to_numeric (ip_parts [i ],errors ='coerce').fillna (0 ).astype (int )

        df .drop (columns =['Source IP','Destination IP'],inplace =True )
        return df 

    def prepare_gan_subset (self ,df_subset :pd .DataFrame ):
        """Preprocess data for GAN training (log transform, normalization)"""
        LABEL_COLUMN ='Label'
        features =df_subset .drop (columns =[LABEL_COLUMN ])
        labels =df_subset [LABEL_COLUMN ]
        self .feature_columns =features .columns .tolist ()

        log_columns =[
        'Total Fwd Packets','Total Backward Packets','Total Length of Fwd Packets',
        'Total Length of Bwd Packets','Flow Duration','Flow IAT Mean','Flow IAT Std',
        'Fwd IAT Mean','Bwd IAT Mean','Fwd Packet Length Mean','Bwd Packet Length Mean',
        'Packet Length Std','Max Packet Length'
        ]

        features_processed =features .copy ()

        for col in log_columns :
            features_processed [col ]=np .log1p (features_processed [col ].clip (lower =0 ))
            features_processed [col ]=features_processed [col ].clip (lower =-20 ,upper =20 )

        features_processed .replace ([np .inf ,-np .inf ],np .nan ,inplace =True )
        features_processed .fillna (0 ,inplace =True )

        X =self .scaler .fit_transform (features_processed )
        return X ,labels 





def oversample_with_noise (X ,factor =10 ,noise_std =0.02 ):
    """
    Oversampling con ruido gaussiano para densificar la distribución.
    
    Args:
        X: datos originales (ya escalados)
        factor: cuántas veces multiplicar los datos
        noise_std: desviación estándar del ruido gaussiano
    
    Returns:
        X aumentado
    """
    if factor <=1 and noise_std ==0 :
        return X 

    print (f"  Aplicando oversampling x{factor } con noise_std={noise_std }")

    X_augmented =[X ]

    for i in range (factor -1 ):

        noise =np .random .normal (0 ,noise_std ,X .shape )
        X_noisy =X +noise 

        X_noisy =np .clip (X_noisy ,-1 ,1 )
        X_augmented .append (X_noisy )

    X_final =np .vstack (X_augmented )
    np .random .shuffle (X_final )

    print (f"  Datos aumentados: {len (X )} -> {len (X_final )}")

    return X_final 





def build_generator_configurable (latent_dim ,output_dim ,layer_sizes =[256 ,512 ,256 ]):
    """Generador con arquitectura configurable"""
    noise =layers .Input (shape =(latent_dim ,))
    x =noise 

    for i ,units in enumerate (layer_sizes ):
        x =layers .Dense (units )(x )
        x =layers .LeakyReLU (0.2 )(x )
        if i <len (layer_sizes )-1 :
            x =layers .BatchNormalization (momentum =0.8 )(x )

    output =layers .Dense (output_dim ,activation ='tanh')(x )
    return models .Model (noise ,output ,name ="Generator")


def build_critic_configurable (input_dim ,layer_sizes =[512 ,256 ,128 ],dropout_rate =0.0 ):
    """Crítico con arquitectura configurable y dropout opcional"""
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
    """Gradient penalty para WGAN-GP"""
    batch_size =tf .shape (real_samples )[0 ]
    alpha =tf .random .uniform ([batch_size ,1 ],0.0 ,1.0 )
    alpha =tf .broadcast_to (alpha ,tf .shape (real_samples ))
    interpolated =alpha *real_samples +(1 -alpha )*fake_samples 

    with tf .GradientTape ()as tape :
        tape .watch (interpolated )
        validity =critic (interpolated )

    gradients =tape .gradient (validity ,interpolated )
    gradients =tf .reshape (gradients ,[batch_size ,-1 ])
    gp =tf .reduce_mean ((tf .norm (gradients ,axis =1 )-1.0 )**2 )
    return gp 


def train_wgan_gp_v2 (generator ,critic ,X_train ,config ,print_interval =1000 ):
    """
    Entrenamiento WGAN-GP con configuración personalizada.
    """
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


    g_losses =[]
    c_losses =[]

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
                critic_loss =tf .reduce_mean (fake_validity )-tf .reduce_mean (real_validity )+lambda_gp *gp 

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
            print (f"    [Epoch {epoch :>6}/{epochs }] C_loss: {critic_loss .numpy ():>8.4f} | G_loss: {generator_loss .numpy ():>8.4f} | ETA: {str (eta ).split ('.')[0 ]}")

    total_time =datetime .now ()-start_time 
    print (f"\n  Entrenamiento completado en {total_time }")

    return g_losses ,c_losses 





COLUMNAS_LOG =[
'Total Fwd Packets','Total Backward Packets','Total Length of Fwd Packets','Total Length of Bwd Packets',
'Flow Duration','Flow IAT Mean','Flow IAT Std','Fwd IAT Mean','Bwd IAT Mean',
'Fwd Packet Length Mean','Bwd Packet Length Mean','Packet Length Std','Max Packet Length'
]

def generate_samples (generator ,n_samples ,latent_dim ):
    """Genera muestras sintéticas"""
    noise =np .random .normal (0 ,1 ,(n_samples ,latent_dim ))
    return generator .predict (noise ,verbose =0 )


def reconstruir_features_originales (X_synthetic ,scaler ,feature_names ,columnas_log ):
    """Reconstruye features a escala original"""
    X_inv =scaler .inverse_transform (X_synthetic )
    df_rec =pd .DataFrame (X_inv ,columns =feature_names )

    for col in columnas_log :
        df_rec [col ]=np .expm1 (df_rec [col ])

    for col in df_rec .columns :
        if col .startswith ('Src_IP_')or col .startswith ('Dst_IP_'):
            df_rec [col ]=df_rec [col ].round ().clip (0 ,255 ).astype (int )

    df_rec ['Source Port']=df_rec ['Source Port'].round ().clip (1 ,65535 ).astype (int )
    df_rec ['Destination Port']=df_rec ['Destination Port'].round ().clip (1 ,65535 ).astype (int )
    df_rec ['Protocol']=df_rec ['Protocol'].round ().clip (1 ,255 ).astype (int )

    for col in columnas_log +['SYN Flag Count','ACK Flag Count','FIN Flag Count','RST Flag Count','PSH Flag Count']:
        df_rec [col ]=df_rec [col ].clip (lower =0 )

    columnas_enteras =['Total Fwd Packets','Total Backward Packets','SYN Flag Count',
    'ACK Flag Count','FIN Flag Count','RST Flag Count','PSH Flag Count']
    for col in columnas_enteras :
        df_rec [col ]=df_rec [col ].round ().astype (int )

    return df_rec 


def plot_training_curves (g_losses ,c_losses ,output_path ):
    """Grafica curvas de entrenamiento"""
    fig ,axes =plt .subplots (1 ,2 ,figsize =(12 ,4 ))


    window =min (100 ,len (g_losses )//10 )
    if window >1 :
        g_smooth =pd .Series (g_losses ).rolling (window ).mean ()
        c_smooth =pd .Series (c_losses ).rolling (window ).mean ()
    else :
        g_smooth =g_losses 
        c_smooth =c_losses 

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
    """Compara distribuciones real vs sintético"""
    n_features =X_real .shape [1 ]
    cols_per_row =4 
    rows =math .ceil (n_features /cols_per_row )

    idx_real =np .random .choice (len (X_real ),min (sample_size ,len (X_real )),replace =False )
    idx_synth =np .random .choice (len (X_synth ),min (sample_size ,len (X_synth )),replace =False )
    Xr =X_real [idx_real ]
    Xs =X_synth [idx_synth ]

    plt .figure (figsize =(cols_per_row *4 ,rows *3 ))

    for i in range (n_features ):
        r =Xr [:,i ].reshape (-1 ,1 )
        s =Xs [:,i ].reshape (-1 ,1 )
        r =scaler .inverse_transform (np .pad (r ,((0 ,0 ),(i ,scaler .n_features_in_ -i -1 )),mode ='constant'))[:,i ]
        s =scaler .inverse_transform (np .pad (s ,((0 ,0 ),(i ,scaler .n_features_in_ -i -1 )),mode ='constant'))[:,i ]
        r =r [np .isfinite (r )]
        s =s [np .isfinite (s )]

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
            plt .title (feature_names [i ],fontsize =9 )
            plt .grid (alpha =0.3 )
            if i ==0 :
                plt .legend ()
        except :
            continue 

    plt .tight_layout ()
    plt .savefig (output_path ,dpi =150 )
    plt .close ()


def get_config_for_class (n_samples ):
    """Selecciona configuración según tamaño de la clase"""
    if n_samples <5000 :
        print (f"  Usando CONFIG_VERY_SMALL (n={n_samples } < 5000)")
        return CONFIG_VERY_SMALL .copy ()
    elif n_samples <SMALL_CLASS_THRESHOLD :
        print (f"  Usando CONFIG_SMALL (n={n_samples } < {SMALL_CLASS_THRESHOLD })")
        return CONFIG_SMALL .copy ()
    else :
        print (f"  Usando CONFIG_LARGE (n={n_samples } >= {SMALL_CLASS_THRESHOLD })")
        return CONFIG_LARGE .copy ()





def main ():
    parser =argparse .ArgumentParser (
    description ='Reentrenamiento WGAN-GP optimizado para clases minoritarias',
    formatter_class =argparse .RawDescriptionHelpFormatter 
    )
    parser .add_argument ('--classes',nargs ='+',default =None ,
    help ='Clases específicas a entrenar (por defecto: Bot, Web Attack)')
    parser .add_argument ('--all',action ='store_true',
    help ='Entrenar TODAS las clases con configuración adaptativa')
    parser .add_argument ('--gpu',type =str ,default ='0',
    help ='GPU a utilizar (default: 0)')
    parser .add_argument ('--samples',type =int ,default =10000 ,
    help ='Número de muestras sintéticas a generar')
    parser .add_argument ('--dry-run',action ='store_true',
    help ='Solo mostrar configuración sin entrenar')

    args =parser .parse_args ()


    os .environ ["CUDA_VISIBLE_DEVICES"]=args .gpu 

    print ("="*80 )
    print ("REENTRENAMIENTO WGAN-GP - CLASES MINORITARIAS (v2)")
    print ("="*80 )


    print ("\n[1] Cargando dataset...")
    prep_global =PreprocessorCIC (DATASET_PATH )
    df_raw =prep_global .cargar ()
    df_base =PreprocessorCIC .preparar_df_base (df_raw )
    label_col ='Attack Type'


    print ("\n[2] Distribución de clases en el dataset:")
    print ("-"*50 )
    class_counts =df_base [label_col ].value_counts ()
    for clase ,count in class_counts .items ():
        marker ="⚠️ "if count <SMALL_CLASS_THRESHOLD else "✓ "
        print (f"  {marker }{clase :<15}: {count :>10,} muestras")
    print ("-"*50 )


    if args .all :
        classes_to_train =class_counts .index .tolist ()
    elif args .classes :
        classes_to_train =args .classes 
    else :
        classes_to_train =MINORITY_CLASSES 

    print (f"\n[3] Clases a entrenar: {classes_to_train }")


    os .makedirs (OUTPUT_DIR ,exist_ok =True )


    label_encoder =LabelEncoder ().fit (df_base [label_col ])


    for class_name in classes_to_train :
        print ("\n"+"="*80 )
        print (f"ENTRENANDO CLASE: {class_name }")
        print ("="*80 )


        df_cls =df_base [df_base [label_col ]==class_name ].copy ()

        if len (df_cls )<100 :
            print (f"  [SKIP] Muy pocas muestras ({len (df_cls )})")
            continue 

        n_original =len (df_cls )
        print (f"\n  Muestras originales: {n_original :,}")


        config =get_config_for_class (n_original )

        if args .dry_run :
            print (f"\n  [DRY-RUN] Configuración que se usaría:")
            for k ,v in config .items ():
                print (f"    {k }: {v }")
            continue 


        prep_cls =PreprocessorCIC (DATASET_PATH )
        X_cls ,_ =prep_cls .preparacion_gan_subset (df_cls )


        X_train =oversample_with_noise (
        X_cls ,
        factor =config ['oversample_factor'],
        noise_std =config ['noise_std']
        )


        generator =build_generator_configurable (
        LATENT_DIM ,
        X_cls .shape [1 ],
        layer_sizes =config ['generator_layers']
        )
        critic =build_critic_configurable (
        X_cls .shape [1 ],
        layer_sizes =config ['critic_layers']
        )

        print (f"\n  Arquitectura del Generador: {config ['generator_layers']}")
        print (f"  Arquitectura del Crítico: {config ['critic_layers']}")


        g_losses ,c_losses =train_wgan_gp_v2 (
        generator ,critic ,X_train ,config ,
        print_interval =max (1000 ,config ['epochs']//20 )
        )


        safe_name =class_name .replace (' ','_').lower ()
        cls_dir =os .path .join (OUTPUT_DIR ,safe_name )
        os .makedirs (cls_dir ,exist_ok =True )


        generator .save (os .path .join (cls_dir ,f'generator_{safe_name }.h5'))
        critic .save (os .path .join (cls_dir ,f'critic_{safe_name }.h5'))


        with open (os .path .join (cls_dir ,'scaler.pkl'),'wb')as f :
            pickle .dump (prep_cls .scaler ,f )


        import json 
        with open (os .path .join (cls_dir ,'training_config.json'),'w')as f :
            json .dump ({
            'class_name':class_name ,
            'original_samples':n_original ,
            'augmented_samples':len (X_train ),
            'config':config ,
            'timestamp':datetime .now ().isoformat ()
            },f ,indent =2 )


        print (f"\n  Generando {args .samples :,} muestras sintéticas...")
        X_synth =generate_samples (generator ,args .samples ,LATENT_DIM )


        np .save (os .path .join (cls_dir ,f'synthetic_scaled_{safe_name }.npy'),X_synth )


        df_synth =reconstruir_features_originales (
        X_synth ,prep_cls .scaler ,prep_cls .columnas_features ,COLUMNAS_LOG 
        )
        df_synth ['Label']=label_encoder .transform ([class_name ])[0 ]
        df_synth ['Attack Type']=class_name 
        df_synth .to_csv (os .path .join (cls_dir ,f'synthetic_reconstructed_{safe_name }.csv'),index =False )


        print ("  Generando gráficas...")
        plot_training_curves (g_losses ,c_losses ,os .path .join (cls_dir ,f'training_curves_{safe_name }.png'))
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
    print ("  1. Generar dataset sintético con los nuevos modelos:")
    print ("     python generate_synthetic_dataset.py --balanced 10000")
    print ("  2. Evaluar con TSTR:")
    print ("     python train_synthetic_test_real_v2.py")


if __name__ =="__main__":
    main ()
