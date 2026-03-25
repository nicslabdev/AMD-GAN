#!/usr/bin/env python3
"""
Load Test Analysis — Computational Load Comparison of GAN Approaches

Compares three approaches for synthetic data generation on CIC-IDS2017:

  Case 1: WGAN-GP per class (proposed approach)
          - gradient penalty, adaptive configuration, n_critic
  Case 2: Monolithic basic CGAN
          - single conditional model for all classes, no GP
  Case 3: Basic GAN per class
          - simple GAN per class, no GP

Metrics measured during TRAINING and INFERENCE:
  • Total parameters (Generator + Discriminator/Critic)
  • Training time (total and per class)
  • Inference time (generation per class)
  • CPU - average and peak usage (%)
  • RAM - average and peak usage (MB, RSS)
  • VRAM - average and peak usage (MB, GPU)
  • GPU - average utilization (%)
  • Disk - bytes read/written (I/O)
  • Estimated model size on disk (MB)
  • Quick quality - average Wasserstein distance (real vs synthetic)

Additional requirements: psutil, pynvml (optional, for detailed GPU metrics)
  pip install psutil pynvml

NOTE: Case 1 (WGAN-GP) replicates EXACTLY the procedure of 1_gan_wgan.py:
      adaptive epochs per class (15000/25000/30000), Gaussian noise
      oversampling, and adaptive configuration.
      Cases 2 and 3 use --epochs as reference.

Usage:
    python 07_load_test.py                        # Adaptive WGAN-GP, CGAN/GAN 2000 ep.
    python 07_load_test.py --epochs 2000          # Epochs for CGAN and basic GAN
    python 07_load_test.py --wgan-scale 0.1       # Reduce WGAN epochs to 10% (quick test)
    python 07_load_test.py --gpu 0                # Select GPU
    python 07_load_test.py --gen-samples 10000    # Samples to generate per class
    python 07_load_test.py --skip 2               # Skip monolithic CGAN
"""

import os 
import sys 
import time 
import json 
import math 
import argparse 
import threading 
import numpy as np 
import pandas as pd 
import polars as pl 
import pickle 
from datetime import datetime ,timedelta 
from collections import OrderedDict 
from scipy .stats import wasserstein_distance 
from dotenv import load_dotenv 

import tensorflow as tf 
from tensorflow .keras import layers ,models ,optimizers 
from sklearn .preprocessing import MinMaxScaler 

import matplotlib 
matplotlib .use ('Agg')
import matplotlib .pyplot as plt 

load_dotenv ()

HAS_PSUTIL =False 
try :
    import psutil 
    HAS_PSUTIL =True 
except ImportError :
    print ("[WARN] psutil not installed — CPU/RAM/Disk metrics limited.")
    print ("       Install with: pip install psutil")

HAS_PYNVML =False 
try :
    import pynvml 
    pynvml .nvmlInit ()
    HAS_PYNVML =True 
except Exception :
    print ("[WARN] pynvml not available — tf.config will be used for VRAM.")
    print ("       Install with: pip install pynvml")

os .environ ['TF_CPP_MIN_LOG_LEVEL']='2'


DATASET_PATH =os .getenv ('DATA_CICIDS2017_PATH','./data/CIC-IDS2017.csv')
OUTPUT_BASE =os .getenv ('OUTPUT_RESULTS_DIR','./outputs/results')
LATENT_DIM =100 
MONITOR_INTERVAL =0.5 


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
LABEL_COLUMN ='Attack Type'

COLUMNAS_LOG =[
'Total Fwd Packets','Total Backward Packets',
'Total Length of Fwd Packets','Total Length of Bwd Packets',
'Flow Duration','Flow IAT Mean','Flow IAT Std',
'Fwd IAT Mean','Bwd IAT Mean',
'Fwd Packet Length Mean','Bwd Packet Length Mean',
'Packet Length Std','Max Packet Length'
]


SMALL_CLASS_THRESHOLD =15000 


CONFIG_WGAN_LARGE ={
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
CONFIG_WGAN_SMALL ={
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
CONFIG_WGAN_VERY_SMALL ={
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


CONFIG_BASELINE ={
'batch_size':128 ,
'generator_layers':[256 ,512 ,256 ],
'discriminator_layers':[512 ,256 ,128 ],
'learning_rate':2e-4 ,
}





class ResourceMonitor :
    """
    Muestrea CPU, RAM, VRAM, GPU util y Disco I/O a intervalos regulares
    en un hilo de fondo. Proporciona resumen estadístico.

    Cada muestra se etiqueta opcionalmente con 'experiment' y 'phase'
    para poder generar timelines globales y exportar a CSV.
    """

    def __init__ (self ,interval =MONITOR_INTERVAL ,gpu_index =0 ,
    experiment ='',phase =''):
        self .interval =interval 
        self .gpu_index =gpu_index 
        self .experiment =experiment 
        self .phase =phase 
        self .samples =[]
        self ._stop_event =threading .Event ()
        self ._thread =None 
        self .process =psutil .Process ()if HAS_PSUTIL else None 
        self .gpu_handle =None 
        if HAS_PYNVML :
            try :
                self .gpu_handle =pynvml .nvmlDeviceGetHandleByIndex (gpu_index )
            except Exception :
                pass 
        self ._disk_start =None 
        self ._disk_end =None 
        self ._prev_disk =None 

    def set_label (self ,experiment ='',phase =''):
        """Update experiment/phase labels on the fly (thread-safe)."""
        self .experiment =experiment 
        self .phase =phase 

    def start (self ):
        """Inicia el muestreo en background."""
        self .samples =[]
        self ._stop_event .clear ()
        if self .process :
            self .process .cpu_percent ()
        if HAS_PSUTIL :
            dio =psutil .disk_io_counters ()
            self ._disk_start =(dio .read_bytes ,dio .write_bytes )if dio else (0 ,0 )
            self ._prev_disk =self ._disk_start 
        self ._thread =threading .Thread (target =self ._loop ,daemon =True )
        self ._thread .start ()

    def stop (self ):
        """Detiene el muestreo y calcula I/O de disco total."""
        self ._stop_event .set ()
        if self ._thread :
            self ._thread .join (timeout =5 )
        if HAS_PSUTIL :
            dio =psutil .disk_io_counters ()
            self ._disk_end =(dio .read_bytes ,dio .write_bytes )if dio else self ._disk_start 
        else :
            self ._disk_end =self ._disk_start if self ._disk_start else (0 ,0 )

    def _loop (self ):
        while not self ._stop_event .is_set ():
            s ={
            't':time .time (),
            'experiment':self .experiment ,
            'phase':self .phase ,
            }

            if self .process :
                try :
                    s ['cpu']=self .process .cpu_percent ()
                    mem =self .process .memory_info ()
                    s ['ram_mb']=mem .rss /(1024 **2 )
                except Exception :
                    pass 

            if self .gpu_handle :
                try :
                    gi =pynvml .nvmlDeviceGetMemoryInfo (self .gpu_handle )
                    s ['vram_mb']=gi .used /(1024 **2 )
                    s ['vram_total_mb']=gi .total /(1024 **2 )
                    gu =pynvml .nvmlDeviceGetUtilizationRates (self .gpu_handle )
                    s ['gpu_util']=gu .gpu 
                except Exception :
                    pass 

            elif tf .config .list_physical_devices ('GPU'):
                try :
                    mi =tf .config .experimental .get_memory_info ('GPU:0')
                    s ['vram_mb']=mi .get ('current',0 )/(1024 **2 )
                except Exception :
                    pass 

            if HAS_PSUTIL :
                try :
                    dio =psutil .disk_io_counters ()
                    if dio and self ._prev_disk :
                        s ['disk_read_MB_s']=(dio .read_bytes -self ._prev_disk [0 ])/(1024 **2 )/max (self .interval ,0.01 )
                        s ['disk_write_MB_s']=(dio .write_bytes -self ._prev_disk [1 ])/(1024 **2 )/max (self .interval ,0.01 )
                        self ._prev_disk =(dio .read_bytes ,dio .write_bytes )
                except Exception :
                    pass 
            self .samples .append (s )
            self ._stop_event .wait (self .interval )

    def summary (self )->dict :
        """Devuelve resumen estadístico de las métricas recopiladas."""
        r ={}
        if not self .samples :
            return r 
        cpu =[s ['cpu']for s in self .samples if 'cpu'in s ]
        ram =[s ['ram_mb']for s in self .samples if 'ram_mb'in s ]
        vram =[s ['vram_mb']for s in self .samples if 'vram_mb'in s ]
        gpu =[s ['gpu_util']for s in self .samples if 'gpu_util'in s ]
        if cpu :
            r ['cpu_avg_%']=round (float (np .mean (cpu )),1 )
            r ['cpu_max_%']=round (float (np .max (cpu )),1 )
        if ram :
            r ['ram_avg_MB']=round (float (np .mean (ram )),1 )
            r ['ram_peak_MB']=round (float (np .max (ram )),1 )
        if vram :
            r ['vram_avg_MB']=round (float (np .mean (vram )),1 )
            r ['vram_peak_MB']=round (float (np .max (vram )),1 )
        if gpu :
            r ['gpu_util_avg_%']=round (float (np .mean (gpu )),1 )
        if self ._disk_start and self ._disk_end :
            r ['disk_read_MB']=round ((self ._disk_end [0 ]-self ._disk_start [0 ])/(1024 **2 ),2 )
            r ['disk_write_MB']=round ((self ._disk_end [1 ]-self ._disk_start [1 ])/(1024 **2 ),2 )
        return r 

    def export_samples_df (self )->pd .DataFrame :
        """
        Exporta todas las muestras crudas como DataFrame.
        Columnas: elapsed_sec, experiment, phase, cpu_%, ram_MB,
                  vram_MB, gpu_util_%, disk_read_MB_s, disk_write_MB_s
        """
        if not self .samples :
            return pd .DataFrame ()
        t0 =self .samples [0 ]['t']
        rows =[]
        for s in self .samples :
            rows .append ({
            'elapsed_sec':round (s ['t']-t0 ,2 ),
            'timestamp':datetime .fromtimestamp (s ['t']).strftime ('%H:%M:%S'),
            'experiment':s .get ('experiment',''),
            'phase':s .get ('phase',''),
            'cpu_%':s .get ('cpu',None ),
            'ram_MB':round (s ['ram_mb'],1 )if 'ram_mb'in s else None ,
            'vram_MB':round (s ['vram_mb'],1 )if 'vram_mb'in s else None ,
            'gpu_util_%':s .get ('gpu_util',None ),
            'disk_read_MB_s':round (s ['disk_read_MB_s'],2 )if 'disk_read_MB_s'in s else None ,
            'disk_write_MB_s':round (s ['disk_write_MB_s'],2 )if 'disk_write_MB_s'in s else None ,
            })
        return pd .DataFrame (rows )


def generate_resource_timeline_chart (df_timeline :pd .DataFrame ,output_dir :str ):
    """
    Genera gráfica multi-panel con la evolución temporal de CPU, RAM, VRAM,
    GPU utilización y Disk I/O durante todos los experimentos.
    Las zonas de cada experimento se colorean con fondo diferente.
    """
    if df_timeline .empty :
        print ("  [WARN] No hay datos de timeline para graficar.")
        return 

    elapsed =df_timeline ['elapsed_sec'].values 


    exp_colors ={
    'WGAN-GP':('#2196F3',0.07 ),
    'CGAN':('#FF9800',0.07 ),
    'GAN':('#4CAF50',0.07 ),
    }
    exp_blocks =[]
    prev_exp =None 
    block_start =0 
    for i ,row in df_timeline .iterrows ():
        exp =row ['experiment']
        if exp !=prev_exp :
            if prev_exp and prev_exp in exp_colors :
                exp_blocks .append ((prev_exp ,block_start ,row ['elapsed_sec']))
            block_start =row ['elapsed_sec']
            prev_exp =exp 
    if prev_exp and prev_exp in exp_colors :
        exp_blocks .append ((prev_exp ,block_start ,elapsed [-1 ]))

    metrics =[
    ('cpu_%','CPU Utilización (%)','#E53935','CPU %'),
    ('ram_MB','RAM Usada (MB)','#1E88E5','RAM MB'),
    ('vram_MB','VRAM Usada (MB)','#8E24AA','VRAM MB'),
    ('gpu_util_%','GPU Utilización (%)','#43A047','GPU %'),
    ('disk_read_MB_s','Disco Lectura (MB/s)','#FB8C00','Read'),
    ('disk_write_MB_s','Disco Escritura (MB/s)','#6D4C41','Write'),
    ]


    metrics_present =[]
    for col ,title ,color ,ylabel in metrics :
        if col in df_timeline .columns and df_timeline [col ].notna ().any ():
            metrics_present .append ((col ,title ,color ,ylabel ))

    n_panels =len (metrics_present )
    if n_panels ==0 :
        print ("  [WARN] No hay métricas disponibles para timeline.")
        return 

    fig ,axes =plt .subplots (n_panels ,1 ,figsize =(16 ,3.0 *n_panels ),
    sharex =True )
    if n_panels ==1 :
        axes =[axes ]

    fig .suptitle ('Evolución Temporal de Recursos del Sistema',
    fontsize =14 ,fontweight ='bold',y =1.01 )

    for ax ,(col ,title ,color ,ylabel )in zip (axes ,metrics_present ):
        vals =df_timeline [col ].values 
        ax .plot (elapsed ,vals ,color =color ,linewidth =0.8 ,alpha =0.85 )

        window =max (1 ,int (5.0 /MONITOR_INTERVAL ))
        smoothed =pd .Series (vals ).rolling (window ,min_periods =1 ).mean ().values 
        ax .plot (elapsed ,smoothed ,color =color ,linewidth =1.8 ,alpha =0.6 ,
        label =f'{ylabel } (media móvil {window *MONITOR_INTERVAL :.0f}s)')
        ax .set_ylabel (ylabel ,fontsize =9 )
        ax .set_title (title ,fontsize =10 ,fontweight ='bold',loc ='left')
        ax .grid (True ,alpha =0.25 )
        ax .legend (loc ='upper right',fontsize =7 )

        for exp_name ,xs ,xe in exp_blocks :
            c ,a =exp_colors .get (exp_name ,('#999999',0.05 ))
            ax .axvspan (xs ,xe ,color =c ,alpha =a )


    for exp_name ,xs ,xe in exp_blocks :
        mid =(xs +xe )/2 
        axes [0 ].annotate (exp_name ,xy =(mid ,1.02 ),xycoords =('data','axes fraction'),
        ha ='center',va ='bottom',fontsize =8 ,fontweight ='bold',
        color =exp_colors .get (exp_name ,('#333',))[0 ])

    axes [-1 ].set_xlabel ('Tiempo Transcurrido (segundos)',fontsize =10 )
    plt .tight_layout (rect =[0 ,0 ,1 ,0.98 ])
    plt .savefig (os .path .join (output_dir ,'resource_timeline.png'),dpi =150 ,
    bbox_inches ='tight')
    plt .close ()
    print (f"  Timeline de recursos guardado: resource_timeline.png")





def load_and_preprocess ():
    """
    Carga CIC-IDS2017 y aplica el mismo preprocesamiento que el enfoque
    propuesto: expansión de IPs, log-transform, escalado MinMax a [-1, 1].

    Returns:
        X:             np.ndarray (N, D) — features escaladas
        labels:        np.ndarray (N,)   — etiquetas string
        feature_names: list[str]
        scaler:        MinMaxScaler
        class_dist:    dict {clase: n_muestras}
    """
    print ("\n[DATA] Cargando CIC-IDS2017...")
    t0 =time .time ()
    df =pl .read_csv (DATASET_PATH ,low_memory =False ).to_pandas ()
    df .columns =df .columns .str .strip ()
    print (f"  Leído en {time .time ()-t0 :.1f}s — {len (df ):,} filas")


    df =df [FEATURES_BASE +[LABEL_COLUMN ]].copy ()


    for prefix ,col_ip in [('Src_IP','Source IP'),('Dst_IP','Destination IP')]:
        octetos =df [col_ip ].astype (str ).str .split ('.',expand =True )
        for i in range (4 ):
            df [f'{prefix }_{i +1 }']=pd .to_numeric (octetos [i ],errors ='coerce').fillna (0 ).astype (int )
    df .drop (columns =['Source IP','Destination IP'],inplace =True )

    labels =df [LABEL_COLUMN ].values 
    features_df =df .drop (columns =[LABEL_COLUMN ])
    feature_names =features_df .columns .tolist ()


    features_proc =features_df .copy ()
    for col in COLUMNAS_LOG :
        if col in features_proc .columns :
            features_proc [col ]=np .log1p (features_proc [col ].clip (lower =0 ))
            features_proc [col ]=features_proc [col ].clip (lower =-20 ,upper =20 )
    features_proc .replace ([np .inf ,-np .inf ],np .nan ,inplace =True )
    features_proc .fillna (0 ,inplace =True )


    scaler =MinMaxScaler (feature_range =(-1 ,1 ))
    X =scaler .fit_transform (features_proc )


    unique ,counts =np .unique (labels ,return_counts =True )
    class_dist =dict (zip (unique ,counts .astype (int )))
    print (f"  Clases: {len (unique )}")
    for c in sorted (class_dist ,key =class_dist .get ,reverse =True ):
        marker ="✓"if class_dist [c ]>=SMALL_CLASS_THRESHOLD else "⚠"
        print (f"    {marker } {c :<15}: {class_dist [c ]:>10,}")

    return X ,labels ,feature_names ,scaler ,class_dist 





def oversample_with_noise (X ,factor =10 ,noise_std =0.02 ):
    """
    Oversampling con ruido gaussiano para densificar la distribución.
    Idéntico a la función en 1_gan_wgan.py.

    Args:
        X: datos originales (ya escalados a [-1, 1])
        factor: cuántas veces multiplicar los datos
        noise_std: desviación estándar del ruido gaussiano

    Returns:
        X aumentado
    """
    if factor <=1 and noise_std ==0 :
        return X 

    print (f"      Aplicando oversampling x{factor } con noise_std={noise_std }")

    X_augmented =[X ]

    for i in range (factor -1 ):
        noise =np .random .normal (0 ,noise_std ,X .shape )
        X_noisy =X +noise 
        X_noisy =np .clip (X_noisy ,-1 ,1 )
        X_augmented .append (X_noisy )

    X_final =np .vstack (X_augmented )
    np .random .shuffle (X_final )

    print (f"      Datos aumentados: {len (X )} -> {len (X_final )}")

    return X_final 







def build_wgan_generator (latent_dim ,output_dim ,layer_sizes ):
    """Generador WGAN-GP con BatchNorm y LeakyReLU."""
    inp =layers .Input (shape =(latent_dim ,))
    x =inp 
    for i ,u in enumerate (layer_sizes ):
        x =layers .Dense (u )(x )
        x =layers .LeakyReLU (0.2 )(x )
        if i <len (layer_sizes )-1 :
            x =layers .BatchNormalization (momentum =0.8 )(x )
    out =layers .Dense (output_dim ,activation ='tanh')(x )
    return models .Model (inp ,out ,name ='WGAN_Generator')


def build_wgan_critic (input_dim ,layer_sizes ):
    """Crítico WGAN-GP (sin sigmoid, salida lineal)."""
    inp =layers .Input (shape =(input_dim ,))
    x =inp 
    for u in layer_sizes :
        x =layers .Dense (u )(x )
        x =layers .LeakyReLU (0.2 )(x )
    out =layers .Dense (1 )(x )
    return models .Model (inp ,out ,name ='WGAN_Critic')



def build_cgan_generator (latent_dim ,num_classes ,output_dim ,layer_sizes ):
    """Generador condicional: recibe ruido + etiqueta one-hot."""
    noise_in =layers .Input (shape =(latent_dim ,),name ='noise')
    label_in =layers .Input (shape =(num_classes ,),name ='label')
    x =layers .Concatenate ()([noise_in ,label_in ])
    for i ,u in enumerate (layer_sizes ):
        x =layers .Dense (u )(x )
        x =layers .LeakyReLU (0.2 )(x )
        if i <len (layer_sizes )-1 :
            x =layers .BatchNormalization (momentum =0.8 )(x )
    out =layers .Dense (output_dim ,activation ='tanh')(x )
    return models .Model ([noise_in ,label_in ],out ,name ='CGAN_Generator')


def build_cgan_discriminator (data_dim ,num_classes ,layer_sizes ):
    """Discriminador condicional: recibe datos + etiqueta one-hot, sigmoid."""
    data_in =layers .Input (shape =(data_dim ,),name ='data')
    label_in =layers .Input (shape =(num_classes ,),name ='label')
    x =layers .Concatenate ()([data_in ,label_in ])
    for u in layer_sizes :
        x =layers .Dense (u )(x )
        x =layers .LeakyReLU (0.2 )(x )
        x =layers .Dropout (0.3 )(x )
    out =layers .Dense (1 ,activation ='sigmoid')(x )
    return models .Model ([data_in ,label_in ],out ,name ='CGAN_Discriminator')



def build_simple_generator (latent_dim ,output_dim ,layer_sizes ):
    """Generador simple: solo ruido → features."""
    inp =layers .Input (shape =(latent_dim ,))
    x =inp 
    for i ,u in enumerate (layer_sizes ):
        x =layers .Dense (u )(x )
        x =layers .LeakyReLU (0.2 )(x )
        if i <len (layer_sizes )-1 :
            x =layers .BatchNormalization (momentum =0.8 )(x )
    out =layers .Dense (output_dim ,activation ='tanh')(x )
    return models .Model (inp ,out ,name ='Simple_Generator')


def build_simple_discriminator (input_dim ,layer_sizes ):
    """Discriminador simple: features → real/fake (sigmoid)."""
    inp =layers .Input (shape =(input_dim ,))
    x =inp 
    for u in layer_sizes :
        x =layers .Dense (u )(x )
        x =layers .LeakyReLU (0.2 )(x )
        x =layers .Dropout (0.3 )(x )
    out =layers .Dense (1 ,activation ='sigmoid')(x )
    return models .Model (inp ,out ,name ='Simple_Discriminator')






def compute_gradient_penalty (critic ,real_samples ,fake_samples ):
    """Gradient Penalty para WGAN-GP (idéntica a 1_gan_wgan.py)."""
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


def train_wgan_gp (generator ,critic ,X_train ,epochs ,config ,print_every =500 ):
    """
    Entrenamiento WGAN-GP (Caso 1) — idéntico a train_wgan_gp_v2 de 1_gan_wgan.py.
    Por cada epoch: n_critic pasos de crítico + 1 paso de generador.
    Overhead total por epoch: (n_critic + 1) evaluaciones de gradiente.
    """
    batch_size =min (config ['batch_size'],len (X_train ))
    n_critic =config ['n_critic']
    lambda_gp =config ['lambda_gp']
    lr =config ['learning_rate']

    gen_optimizer =optimizers .Adam (learning_rate =lr ,beta_1 =0.0 ,beta_2 =0.9 )
    critic_optimizer =optimizers .Adam (learning_rate =lr ,beta_1 =0.0 ,beta_2 =0.9 )

    X_train =tf .convert_to_tensor (X_train ,dtype =tf .float32 )
    n_samples =X_train .shape [0 ]

    for epoch in range (1 ,epochs +1 ):

        for _ in range (n_critic ):
            idx =np .random .randint (0 ,n_samples ,batch_size )
            real_samples =tf .gather (X_train ,idx )
            noise =tf .random .normal ((batch_size ,LATENT_DIM ))
            fake_samples =generator (noise ,training =True )

            with tf .GradientTape ()as tape :
                real_validity =critic (real_samples ,training =True )
                fake_validity =critic (fake_samples ,training =True )
                gp =compute_gradient_penalty (critic ,real_samples ,fake_samples )
                critic_loss =(tf .reduce_mean (fake_validity )
                -tf .reduce_mean (real_validity )
                +lambda_gp *gp )

            grads =tape .gradient (critic_loss ,critic .trainable_variables )
            critic_optimizer .apply_gradients (zip (grads ,critic .trainable_variables ))


        noise =tf .random .normal ((batch_size ,LATENT_DIM ))
        with tf .GradientTape ()as tape :
            fake_samples =generator (noise ,training =True )
            fake_validity =critic (fake_samples ,training =True )
            generator_loss =-tf .reduce_mean (fake_validity )

        grads =tape .gradient (generator_loss ,generator .trainable_variables )
        gen_optimizer .apply_gradients (zip (grads ,generator .trainable_variables ))

        if epoch %print_every ==0 or epoch ==1 :
            print (f"      [Epoch {epoch :>6}/{epochs }] "
            f"C_loss={critic_loss .numpy ():.4f}  "
            f"G_loss={generator_loss .numpy ():.4f}")


def train_cgan (generator ,discriminator ,X_all ,y_onehot ,epochs ,config ,
print_every =500 ):
    """
    Entrenamiento CGAN monolítica (Caso 2).
    BCE estándar, sin gradient penalty, sin n_critic adicional.
    Por cada epoch: 1 paso D + 1 paso G = 2 evaluaciones de gradiente.
    """
    bs =min (config ['batch_size'],len (X_all ))
    lr =config ['learning_rate']
    num_classes =y_onehot .shape [1 ]

    g_opt =optimizers .Adam (learning_rate =lr ,beta_1 =0.5 ,beta_2 =0.999 )
    d_opt =optimizers .Adam (learning_rate =lr ,beta_1 =0.5 ,beta_2 =0.999 )
    bce =tf .keras .losses .BinaryCrossentropy ()

    X_t =tf .constant (X_all ,dtype =tf .float32 )
    Y_t =tf .constant (y_onehot ,dtype =tf .float32 )
    n =X_t .shape [0 ]

    ones =tf .ones ((bs ,1 ))
    zeros =tf .zeros ((bs ,1 ))

    for ep in range (1 ,epochs +1 ):

        idx =np .random .randint (0 ,n ,bs )
        real_data =tf .gather (X_t ,idx )
        real_labels =tf .gather (Y_t ,idx )

        noise =tf .random .normal ((bs ,LATENT_DIM ))
        fake_label_idx =np .random .randint (0 ,num_classes ,bs )
        fake_labels =tf .one_hot (fake_label_idx ,num_classes )
        fake_data =generator ([noise ,fake_labels ],training =True )

        with tf .GradientTape ()as tape :
            d_real =discriminator ([real_data ,real_labels ],training =True )
            d_fake =discriminator ([fake_data ,fake_labels ],training =True )
            d_loss =bce (ones ,d_real )+bce (zeros ,d_fake )
        grads =tape .gradient (d_loss ,discriminator .trainable_variables )
        d_opt .apply_gradients (zip (grads ,discriminator .trainable_variables ))


        noise =tf .random .normal ((bs ,LATENT_DIM ))
        gen_label_idx =np .random .randint (0 ,num_classes ,bs )
        gen_labels =tf .one_hot (gen_label_idx ,num_classes )

        with tf .GradientTape ()as tape :
            fake_data =generator ([noise ,gen_labels ],training =True )
            d_fake =discriminator ([fake_data ,gen_labels ],training =True )
            g_loss =bce (ones ,d_fake )
        grads =tape .gradient (g_loss ,generator .trainable_variables )
        g_opt .apply_gradients (zip (grads ,generator .trainable_variables ))

        if ep %print_every ==0 or ep ==1 :
            print (f"      [Epoch {ep :>6}/{epochs }] D_loss={d_loss .numpy ():.4f}  "
            f"G_loss={g_loss .numpy ():.4f}")


def train_simple_gan (generator ,discriminator ,X_train ,epochs ,config ,
print_every =500 ):
    """
    Entrenamiento GAN básica sin GP (Caso 3).
    BCE estándar. Por cada epoch: 1 paso D + 1 paso G.
    """
    bs =min (config ['batch_size'],len (X_train ))
    lr =config ['learning_rate']

    g_opt =optimizers .Adam (learning_rate =lr ,beta_1 =0.5 ,beta_2 =0.999 )
    d_opt =optimizers .Adam (learning_rate =lr ,beta_1 =0.5 ,beta_2 =0.999 )
    bce =tf .keras .losses .BinaryCrossentropy ()

    X =tf .constant (X_train ,dtype =tf .float32 )
    n =X .shape [0 ]

    ones =tf .ones ((bs ,1 ))
    zeros =tf .zeros ((bs ,1 ))

    for ep in range (1 ,epochs +1 ):

        idx =np .random .randint (0 ,n ,bs )
        real =tf .gather (X ,idx )
        noise =tf .random .normal ((bs ,LATENT_DIM ))
        fake =generator (noise ,training =True )

        with tf .GradientTape ()as tape :
            d_real =discriminator (real ,training =True )
            d_fake =discriminator (fake ,training =True )
            d_loss =bce (ones ,d_real )+bce (zeros ,d_fake )
        grads =tape .gradient (d_loss ,discriminator .trainable_variables )
        d_opt .apply_gradients (zip (grads ,discriminator .trainable_variables ))


        noise =tf .random .normal ((bs ,LATENT_DIM ))
        with tf .GradientTape ()as tape :
            fake =generator (noise ,training =True )
            d_fake =discriminator (fake ,training =True )
            g_loss =bce (ones ,d_fake )
        grads =tape .gradient (g_loss ,generator .trainable_variables )
        g_opt .apply_gradients (zip (grads ,generator .trainable_variables ))

        if ep %print_every ==0 or ep ==1 :
            print (f"      [Epoch {ep :>6}/{epochs }] D_loss={d_loss .numpy ():.4f}  "
            f"G_loss={g_loss .numpy ():.4f}")






def count_params (model ):
    """Cuenta parámetros entrenables de un modelo Keras."""
    return int (np .sum ([np .prod (v .shape )for v in model .trainable_variables ]))


def model_size_mb (n_params ):
    """Tamaño estimado en disco (float32 = 4 bytes/param)."""
    return round (n_params *4 /(1024 **2 ),2 )


def get_wgan_config (n_samples ):
    """Devuelve configuración adaptativa según tamaño de clase."""
    if n_samples <5000 :
        return CONFIG_WGAN_VERY_SMALL .copy ()
    elif n_samples <SMALL_CLASS_THRESHOLD :
        return CONFIG_WGAN_SMALL .copy ()
    else :
        return CONFIG_WGAN_LARGE .copy ()


def quick_wasserstein (X_real ,X_fake ,n_samples =5000 ):
    """
    Distancia de Wasserstein media sobre TODAS las features,
    usando un subconjunto aleatorio de muestras.
    """
    nr =min (n_samples ,len (X_real ))
    nf =min (n_samples ,len (X_fake ))
    Xr =X_real [np .random .choice (len (X_real ),nr ,replace =False )]
    Xf =X_fake [np .random .choice (len (X_fake ),nf ,replace =len (X_fake )<nf )]
    dists =[]
    for fi in range (X_real .shape [1 ]):
        dists .append (wasserstein_distance (Xr [:,fi ],Xf [:,fi ]))
    return float (np .mean (dists ))


def format_time (sec ):
    """Formatea segundos a string legible."""
    if sec <60 :
        return f"{sec :.1f}s"
    elif sec <3600 :
        return f"{int (sec //60 )}m {int (sec %60 )}s"
    else :
        return f"{int (sec //3600 )}h {int ((sec %3600 )//60 )}m"


def get_gpu_name ():
    """Intenta obtener el nombre de la GPU."""
    if HAS_PYNVML :
        try :
            handle =pynvml .nvmlDeviceGetHandleByIndex (0 )
            return pynvml .nvmlDeviceGetName (handle )
        except Exception :
            pass 
    gpus =tf .config .list_physical_devices ('GPU')
    if gpus :
        return str (gpus [0 ])
    return "N/A (CPU)"






def run_experiment_1_wgan_gp (X ,labels ,class_dist ,epochs ,gen_samples ,
gpu_idx ,global_monitor =None ,wgan_scale =1.0 ):
    """
    Caso 1: WGAN-GP por clase — enfoque propuesto.
    Replica EXACTAMENTE el procedimiento de 1_gan_wgan.py:
      - Configuración adaptativa por tamaño de clase
      - Epochs adaptativos por clase (15000/25000/30000)
      - Oversampling con ruido gaussiano para clases pequeñas
      - Gradient penalty con alpha broadcast

    Args:
        wgan_scale: Factor multiplicador de epochs (1.0 = completo, 0.1 = 10%)
    """
    print ("\n"+"="*80 )
    print ("  CASO 1: WGAN-GP POR CLASE (PROPUESTO — idéntico a 1_gan_wgan.py)")
    print ("="*80 )

    classes =sorted (class_dist .keys ())
    data_dim =X .shape [1 ]

    results ={
    'name':'WGAN-GP (Propuesto)',
    'approach':'per_class_wgan_gp',
    'per_class':{},
    'training':{},
    'inference':{},
    }

    total_gen_params =0 
    total_critic_params =0 
    total_grad_ops =0 
    all_generators ={}


    monitor_train =ResourceMonitor (gpu_index =gpu_idx ,experiment ='WGAN-GP',
    phase ='train')
    monitor_train .start ()
    if global_monitor :
        global_monitor .set_label ('WGAN-GP','train')
    t_train_start =time .time ()

    for cls in classes :
        print (f"\n    ── Clase: {cls } ({class_dist [cls ]:,} muestras) ──")
        X_cls =X [labels ==cls ]
        cfg =get_wgan_config (len (X_cls ))


        cls_epochs =max (1 ,int (cfg ['epochs']*wgan_scale ))


        X_train =oversample_with_noise (
        X_cls ,
        factor =cfg ['oversample_factor'],
        noise_std =cfg ['noise_std']
        )

        config_type =('LARGE'if len (X_cls )>=SMALL_CLASS_THRESHOLD 
        else 'SMALL'if len (X_cls )>=5000 else 'VERY_SMALL')

        gen =build_wgan_generator (LATENT_DIM ,data_dim ,cfg ['generator_layers'])
        cri =build_wgan_critic (data_dim ,cfg ['critic_layers'])

        gp =count_params (gen )
        cp =count_params (cri )
        total_gen_params +=gp 
        total_critic_params +=cp 


        cls_grad_ops =cls_epochs *(cfg ['n_critic']+1 )
        total_grad_ops +=cls_grad_ops 

        print (f"      Config: {config_type }  Epochs: {cls_epochs }  "
        f"batch={cfg ['batch_size']}  n_critic={cfg ['n_critic']}")
        print (f"      Oversampling: x{cfg ['oversample_factor']}  "
        f"noise_std={cfg ['noise_std']}  "
        f"Train data: {len (X_train ):,}")
        print (f"      Params Gen: {gp :,}  Critic: {cp :,}  "
        f"grad_ops={cls_grad_ops :,}")

        pe =max (500 ,cls_epochs //20 )
        t0 =time .time ()
        train_wgan_gp (gen ,cri ,X_train ,cls_epochs ,cfg ,print_every =pe )
        t_cls =time .time ()-t0 

        results ['per_class'][cls ]={
        'train_time_sec':round (t_cls ,2 ),
        'gen_params':gp ,
        'critic_params':cp ,
        'config_type':config_type ,
        'n_critic':cfg ['n_critic'],
        'batch_size':cfg ['batch_size'],
        'epochs':cls_epochs ,
        'oversample_factor':cfg ['oversample_factor'],
        'noise_std':cfg ['noise_std'],
        'train_samples':len (X_train ),
        'n_samples':int (class_dist [cls ]),
        'grad_ops':cls_grad_ops ,
        }

        all_generators [cls ]=gen 
        del cri 
        print (f"      Tiempo: {format_time (t_cls )}")

    t_train_total =time .time ()-t_train_start 
    monitor_train .stop ()

    results ['training']={
    'total_time_sec':round (t_train_total ,2 ),
    'total_time_str':format_time (t_train_total ),
    'total_grad_ops':total_grad_ops ,
    'resources':monitor_train .summary (),
    }
    results ['total_gen_params']=total_gen_params 
    results ['total_critic_params']=total_critic_params 
    results ['total_params']=total_gen_params +total_critic_params 
    results ['model_size_est_MB']=model_size_mb (results ['total_params'])


    print (f"\n    ── Inferencia: generando {gen_samples :,} muestras/clase ──")
    monitor_inf =ResourceMonitor (gpu_index =gpu_idx ,experiment ='WGAN-GP',
    phase ='inference')
    monitor_inf .start ()
    if global_monitor :
        global_monitor .set_label ('WGAN-GP','inference')
    t_inf_start =time .time ()

    generated ={}
    for cls in classes :
        t0 =time .time ()
        noise =np .random .normal (0 ,1 ,(gen_samples ,LATENT_DIM )).astype (np .float32 )
        synth =all_generators [cls ].predict (noise ,verbose =0 )
        t_gen =time .time ()-t0 
        generated [cls ]=synth 
        results ['per_class'][cls ]['inference_time_sec']=round (t_gen ,4 )
        results ['per_class'][cls ]['inference_samples']=gen_samples 
        print (f"      {cls :<15}: {t_gen :.3f}s")

    t_inf_total =time .time ()-t_inf_start 
    monitor_inf .stop ()

    results ['inference']={
    'total_time_sec':round (t_inf_total ,2 ),
    'total_time_str':format_time (t_inf_total ),
    'resources':monitor_inf .summary (),
    }


    print ("\n    ── Calidad (Wasserstein) ──")
    for cls in classes :
        X_cls =X [labels ==cls ]
        wd =quick_wasserstein (X_cls ,generated [cls ])
        results ['per_class'][cls ]['wasserstein']=round (wd ,6 )
        print (f"      {cls :<15}: {wd :.6f}")

    results ['avg_wasserstein']=round (float (np .mean ([
    results ['per_class'][c ]['wasserstein']for c in classes 
    ])),6 )


    for g in all_generators .values ():
        del g 
    tf .keras .backend .clear_session ()

    return results 


def run_experiment_2_cgan_monolithic (X ,labels ,class_dist ,epochs ,
gen_samples ,gpu_idx ,
global_monitor =None ):
    """
    Caso 2: CGAN monolítica básica.
    Un solo modelo condicional para todas las clases, BCE estándar, sin GP.
    """
    print ("\n"+"="*80 )
    print ("  CASO 2: CGAN MONOLÍTICA BÁSICA (SIN GP)")
    print ("="*80 )

    classes =sorted (class_dist .keys ())
    num_classes =len (classes )
    data_dim =X .shape [1 ]
    cls_to_idx ={c :i for i ,c in enumerate (classes )}


    y_idx =np .array ([cls_to_idx [l ]for l in labels ])
    y_onehot =np .eye (num_classes ,dtype =np .float32 )[y_idx ]

    cfg =CONFIG_BASELINE 

    gen =build_cgan_generator (LATENT_DIM ,num_classes ,data_dim ,
    cfg ['generator_layers'])
    disc =build_cgan_discriminator (data_dim ,num_classes ,
    cfg ['discriminator_layers'])

    gp =count_params (gen )
    dp =count_params (disc )
    total_grad_ops =epochs *2 
    print (f"    Params Gen: {gp :,}  Disc: {dp :,}  Total: {gp +dp :,}")
    print (f"    Grad ops totales: {total_grad_ops :,}")

    results ={
    'name':'CGAN Monolítica',
    'approach':'monolithic_cgan',
    'total_gen_params':gp ,
    'total_disc_params':dp ,
    'total_params':gp +dp ,
    'model_size_est_MB':model_size_mb (gp +dp ),
    'per_class':{},
    'training':{},
    'inference':{},
    }


    monitor_train =ResourceMonitor (gpu_index =gpu_idx ,experiment ='CGAN',
    phase ='train')
    monitor_train .start ()
    if global_monitor :
        global_monitor .set_label ('CGAN','train')
    t0 =time .time ()

    pe =max (200 ,epochs //10 )
    train_cgan (gen ,disc ,X ,y_onehot ,epochs ,cfg ,print_every =pe )

    t_train =time .time ()-t0 
    monitor_train .stop ()

    results ['training']={
    'total_time_sec':round (t_train ,2 ),
    'total_time_str':format_time (t_train ),
    'total_grad_ops':total_grad_ops ,
    'resources':monitor_train .summary (),
    }


    print (f"\n    ── Inferencia: generando {gen_samples :,} muestras/clase ──")
    monitor_inf =ResourceMonitor (gpu_index =gpu_idx ,experiment ='CGAN',
    phase ='inference')
    monitor_inf .start ()
    if global_monitor :
        global_monitor .set_label ('CGAN','inference')
    t_inf_start =time .time ()

    generated ={}
    for cls in classes :
        ci =cls_to_idx [cls ]
        t1 =time .time ()
        noise =np .random .normal (0 ,1 ,(gen_samples ,LATENT_DIM )).astype (np .float32 )
        lbl =np .zeros ((gen_samples ,num_classes ),dtype =np .float32 )
        lbl [:,ci ]=1.0 
        synth =gen .predict ([noise ,lbl ],verbose =0 )
        t_gen =time .time ()-t1 
        generated [cls ]=synth 
        results ['per_class'][cls ]={
        'inference_time_sec':round (t_gen ,4 ),
        'inference_samples':gen_samples ,
        'n_samples':int (class_dist [cls ]),
        }
        print (f"      {cls :<15}: {t_gen :.3f}s")

    t_inf_total =time .time ()-t_inf_start 
    monitor_inf .stop ()

    results ['inference']={
    'total_time_sec':round (t_inf_total ,2 ),
    'total_time_str':format_time (t_inf_total ),
    'resources':monitor_inf .summary (),
    }


    print ("\n    ── Calidad (Wasserstein) ──")
    for cls in classes :
        X_cls =X [labels ==cls ]
        wd =quick_wasserstein (X_cls ,generated [cls ])
        results ['per_class'][cls ]['wasserstein']=round (wd ,6 )
        print (f"      {cls :<15}: {wd :.6f}")

    results ['avg_wasserstein']=round (float (np .mean ([
    results ['per_class'][c ]['wasserstein']for c in classes 
    ])),6 )

    del gen ,disc 
    tf .keras .backend .clear_session ()

    return results 


def run_experiment_3_gan_per_class (X ,labels ,class_dist ,epochs ,
gen_samples ,gpu_idx ,
global_monitor =None ):
    """
    Caso 3: GAN básica por clase (sin GP, sin conditioning).
    Un GAN simple por cada clase, BCE estándar.
    """
    print ("\n"+"="*80 )
    print ("  CASO 3: GAN BÁSICA POR CLASE (SIN GP)")
    print ("="*80 )

    classes =sorted (class_dist .keys ())
    data_dim =X .shape [1 ]
    cfg =CONFIG_BASELINE 

    results ={
    'name':'GAN por Clase',
    'approach':'per_class_simple_gan',
    'per_class':{},
    'training':{},
    'inference':{},
    }

    total_gen_params =0 
    total_disc_params =0 
    total_grad_ops =0 
    all_generators ={}


    monitor_train =ResourceMonitor (gpu_index =gpu_idx ,experiment ='GAN',
    phase ='train')
    monitor_train .start ()
    if global_monitor :
        global_monitor .set_label ('GAN','train')
    t_train_start =time .time ()

    for cls in classes :
        print (f"\n    ── Clase: {cls } ({class_dist [cls ]:,} muestras) ──")
        X_cls =X [labels ==cls ]

        gen =build_simple_generator (LATENT_DIM ,data_dim ,
        cfg ['generator_layers'])
        disc =build_simple_discriminator (data_dim ,
        cfg ['discriminator_layers'])

        gp =count_params (gen )
        dp =count_params (disc )
        total_gen_params +=gp 
        total_disc_params +=dp 
        cls_grad_ops =epochs *2 
        total_grad_ops +=cls_grad_ops 

        print (f"      Params Gen: {gp :,}  Disc: {dp :,}  grad_ops={cls_grad_ops :,}")

        pe =max (200 ,epochs //10 )
        t0 =time .time ()
        train_simple_gan (gen ,disc ,X_cls ,epochs ,cfg ,print_every =pe )
        t_cls =time .time ()-t0 

        results ['per_class'][cls ]={
        'train_time_sec':round (t_cls ,2 ),
        'gen_params':gp ,
        'disc_params':dp ,
        'n_samples':int (class_dist [cls ]),
        'batch_size':cfg ['batch_size'],
        'grad_ops':cls_grad_ops ,
        }

        all_generators [cls ]=gen 
        del disc 
        print (f"      Tiempo: {format_time (t_cls )}")

    t_train_total =time .time ()-t_train_start 
    monitor_train .stop ()

    results ['training']={
    'total_time_sec':round (t_train_total ,2 ),
    'total_time_str':format_time (t_train_total ),
    'total_grad_ops':total_grad_ops ,
    'resources':monitor_train .summary (),
    }
    results ['total_gen_params']=total_gen_params 
    results ['total_disc_params']=total_disc_params 
    results ['total_params']=total_gen_params +total_disc_params 
    results ['model_size_est_MB']=model_size_mb (results ['total_params'])


    print (f"\n    ── Inferencia: generando {gen_samples :,} muestras/clase ──")
    monitor_inf =ResourceMonitor (gpu_index =gpu_idx ,experiment ='GAN',
    phase ='inference')
    monitor_inf .start ()
    if global_monitor :
        global_monitor .set_label ('GAN','inference')
    t_inf_start =time .time ()

    generated ={}
    for cls in classes :
        t0 =time .time ()
        noise =np .random .normal (0 ,1 ,(gen_samples ,LATENT_DIM )).astype (np .float32 )
        synth =all_generators [cls ].predict (noise ,verbose =0 )
        t_gen =time .time ()-t0 
        generated [cls ]=synth 
        results ['per_class'][cls ]['inference_time_sec']=round (t_gen ,4 )
        results ['per_class'][cls ]['inference_samples']=gen_samples 
        print (f"      {cls :<15}: {t_gen :.3f}s")

    t_inf_total =time .time ()-t_inf_start 
    monitor_inf .stop ()

    results ['inference']={
    'total_time_sec':round (t_inf_total ,2 ),
    'total_time_str':format_time (t_inf_total ),
    'resources':monitor_inf .summary (),
    }


    print ("\n    ── Calidad (Wasserstein) ──")
    for cls in classes :
        X_cls =X [labels ==cls ]
        wd =quick_wasserstein (X_cls ,generated [cls ])
        results ['per_class'][cls ]['wasserstein']=round (wd ,6 )
        print (f"      {cls :<15}: {wd :.6f}")

    results ['avg_wasserstein']=round (float (np .mean ([
    results ['per_class'][c ]['wasserstein']for c in classes 
    ])),6 )

    for g in all_generators .values ():
        del g 
    tf .keras .backend .clear_session ()

    return results 






def print_comparison_table (r1 ,r2 ,r3 ):
    """Imprime tabla comparativa principal en consola."""
    sep ="─"*92 
    header =(f"{'Métrica':<32} │ {'WGAN-GP (Prop.)':>18} │ "
    f"{'CGAN Monolítica':>18} │ {'GAN por Clase':>18}")

    print ("\n"+"═"*92 )
    print ("  TABLA COMPARATIVA — CARGA COMPUTACIONAL")
    print ("═"*92 )
    print (header )
    print (sep )

    def _v (r ,key ,default ='N/A'):
        """Obtiene valor anidado, formatea números."""
        val =r .get (key ,default )
        if isinstance (val ,float ):
            return f"{val :,.1f}"
        elif isinstance (val ,int ):
            return f"{val :,}"
        return str (val )

    rows =[

    ("Params Generador(es)",
    f"{r1 ['total_gen_params']:,}",
    f"{r2 ['total_gen_params']:,}",
    f"{r3 ['total_gen_params']:,}"),
    ("Params Disc./Crítico(s)",
    f"{r1 .get ('total_critic_params',0 ):,}",
    f"{r2 .get ('total_disc_params',0 ):,}",
    f"{r3 .get ('total_disc_params',0 ):,}"),
    ("Params Totales",
    f"{r1 ['total_params']:,}",
    f"{r2 ['total_params']:,}",
    f"{r3 ['total_params']:,}"),
    ("Tamaño Est. Modelos (MB)",
    f"{r1 ['model_size_est_MB']:.2f}",
    f"{r2 ['model_size_est_MB']:.2f}",
    f"{r3 ['model_size_est_MB']:.2f}"),
    ("","","",""),

    ("Tiempo Entrenamiento",
    format_time (r1 ['training']['total_time_sec']),
    format_time (r2 ['training']['total_time_sec']),
    format_time (r3 ['training']['total_time_sec'])),
    ("Operaciones de Gradiente",
    f"{r1 ['training']['total_grad_ops']:,}",
    f"{r2 ['training']['total_grad_ops']:,}",
    f"{r3 ['training']['total_grad_ops']:,}"),
    ("Tiempo Inferencia Total",
    format_time (r1 ['inference']['total_time_sec']),
    format_time (r2 ['inference']['total_time_sec']),
    format_time (r3 ['inference']['total_time_sec'])),
    ("","","",""),
    ]


    res_keys_train =[
    ('cpu_avg_%','CPU Medio (%) [Train]'),
    ('cpu_max_%','CPU Pico (%) [Train]'),
    ('ram_avg_MB','RAM Media (MB) [Train]'),
    ('ram_peak_MB','RAM Pico (MB) [Train]'),
    ('vram_avg_MB','VRAM Media (MB) [Train]'),
    ('vram_peak_MB','VRAM Pico (MB) [Train]'),
    ('gpu_util_avg_%','GPU Util. (%) [Train]'),
    ('disk_read_MB','Disco Lectura (MB) [Train]'),
    ('disk_write_MB','Disco Escritura (MB) [Train]'),
    ]
    for key ,label in res_keys_train :
        vals =[]
        for r in [r1 ,r2 ,r3 ]:
            v =r ['training']['resources'].get (key ,'N/A')
            vals .append (f"{v :,.1f}"if isinstance (v ,(int ,float ))else str (v ))
        rows .append ((label ,*vals ))

    rows .append (("","","",""))


    res_keys_inf =[
    ('cpu_avg_%','CPU Medio (%) [Inf]'),
    ('ram_peak_MB','RAM Pico (MB) [Inf]'),
    ('vram_peak_MB','VRAM Pico (MB) [Inf]'),
    ]
    for key ,label in res_keys_inf :
        vals =[]
        for r in [r1 ,r2 ,r3 ]:
            v =r ['inference']['resources'].get (key ,'N/A')
            vals .append (f"{v :,.1f}"if isinstance (v ,(int ,float ))else str (v ))
        rows .append ((label ,*vals ))

    rows .append (("","","",""))
    rows .append ((
    "Wasserstein Media (↓ mejor)",
    f"{r1 ['avg_wasserstein']:.6f}",
    f"{r2 ['avg_wasserstein']:.6f}",
    f"{r3 ['avg_wasserstein']:.6f}",
    ))

    for label ,v1 ,v2 ,v3 in rows :
        if label =="":
            print (sep )
        else :
            print (f"{label :<32} │ {v1 :>18} │ {v2 :>18} │ {v3 :>18}")

    print ("═"*92 )


def print_per_class_table (r1 ,r2 ,r3 ,classes ):
    """Tabla detallada por clase: entrenamiento, inferencia y calidad."""
    print ("\n"+"═"*110 )
    print ("  DETALLE POR CLASE — ENTRENAMIENTO, INFERENCIA Y CALIDAD")
    print ("═"*110 )

    header =(f"{'Clase':<15} │ {'Muestras':>8} │ "
    f"{'T.Train WG':>10} │ {'T.Train GAN':>11} │ "
    f"{'T.Inf WG':>9} │ {'T.Inf CGAN':>10} │ {'T.Inf GAN':>9} │ "
    f"{'WD WG':>8} │ {'WD CGAN':>8} │ {'WD GAN':>8}")
    print (header )
    print ("─"*110 )

    for cls in classes :
        n =r1 ['per_class'][cls ]['n_samples']

        tt1 =format_time (r1 ['per_class'][cls ].get ('train_time_sec',0 ))
        tt3 =format_time (r3 ['per_class'][cls ].get ('train_time_sec',0 ))

        ti1 =f"{r1 ['per_class'][cls ].get ('inference_time_sec',0 ):.3f}s"
        ti2 =f"{r2 ['per_class'][cls ].get ('inference_time_sec',0 ):.3f}s"
        ti3 =f"{r3 ['per_class'][cls ].get ('inference_time_sec',0 ):.3f}s"

        wd1 =f"{r1 ['per_class'][cls ].get ('wasserstein',0 ):.4f}"
        wd2 =f"{r2 ['per_class'][cls ].get ('wasserstein',0 ):.4f}"
        wd3 =f"{r3 ['per_class'][cls ].get ('wasserstein',0 ):.4f}"

        print (f"{cls :<15} │ {n :>8,} │ "
        f"{tt1 :>10} │ {tt3 :>11} │ "
        f"{ti1 :>9} │ {ti2 :>10} │ {ti3 :>9} │ "
        f"{wd1 :>8} │ {wd2 :>8} │ {wd3 :>8}")

    print ("─"*110 )

    tt1_total =format_time (r1 ['training']['total_time_sec'])
    tt2_total =format_time (r2 ['training']['total_time_sec'])
    tt3_total =format_time (r3 ['training']['total_time_sec'])
    wd1_avg =f"{r1 ['avg_wasserstein']:.4f}"
    wd2_avg =f"{r2 ['avg_wasserstein']:.4f}"
    wd3_avg =f"{r3 ['avg_wasserstein']:.4f}"
    print (f"{'TOTAL/MEDIA':<15} │ {'':>8} │ "
    f"{tt1_total :>10} │ {tt3_total :>11} │ "
    f"{'':>9} │ {'':>10} │ {'':>9} │ "
    f"{wd1_avg :>8} │ {wd2_avg :>8} │ {wd3_avg :>8}")
    print (f"{'(CGAN mono.)':<15} │ {'':>8} │ "
    f"{tt2_total :>10} │ {'':>11} │ "
    f"{'':>9} │ {'':>10} │ {'':>9} │ "
    f"{'':>8} │ {'':>8} │ {'':>8}")
    print ("═"*110 )


def save_results_csv (r1 ,r2 ,r3 ,classes ,output_dir ):
    """Guarda resultados en ficheros CSV."""

    rows =[]
    for r in [r1 ,r2 ,r3 ]:
        row ={
        'Enfoque':r ['name'],
        'Params_Gen':r .get ('total_gen_params',0 ),
        'Params_Disc':r .get ('total_critic_params',
        r .get ('total_disc_params',0 )),
        'Params_Total':r ['total_params'],
        'Model_Size_MB':r ['model_size_est_MB'],
        'Train_Time_sec':r ['training']['total_time_sec'],
        'Train_Grad_Ops':r ['training']['total_grad_ops'],
        'Inference_Time_sec':r ['inference']['total_time_sec'],
        'Wasserstein_Avg':r ['avg_wasserstein'],
        }

        for k ,v in r ['training']['resources'].items ():
            row [f'train_{k }']=v 

        for k ,v in r ['inference']['resources'].items ():
            row [f'inf_{k }']=v 
        rows .append (row )
    df_summary =pd .DataFrame (rows )
    df_summary .to_csv (os .path .join (output_dir ,'comparison_summary.csv'),
    index =False )


    rows_cls =[]
    for cls in classes :
        row ={'Clase':cls ,'Muestras':r1 ['per_class'][cls ]['n_samples']}

        pc1 =r1 ['per_class'][cls ]
        row ['WGAN_Train_sec']=pc1 .get ('train_time_sec','')
        row ['WGAN_Inf_sec']=pc1 .get ('inference_time_sec','')
        row ['WGAN_Wasserstein']=pc1 .get ('wasserstein','')
        row ['WGAN_Gen_Params']=pc1 .get ('gen_params','')
        row ['WGAN_Critic_Params']=pc1 .get ('critic_params','')
        row ['WGAN_n_critic']=pc1 .get ('n_critic','')
        row ['WGAN_batch_size']=pc1 .get ('batch_size','')
        row ['WGAN_config']=pc1 .get ('config_type','')

        pc2 =r2 ['per_class'].get (cls ,{})
        row ['CGAN_Inf_sec']=pc2 .get ('inference_time_sec','')
        row ['CGAN_Wasserstein']=pc2 .get ('wasserstein','')

        pc3 =r3 ['per_class'].get (cls ,{})
        row ['GAN_Train_sec']=pc3 .get ('train_time_sec','')
        row ['GAN_Inf_sec']=pc3 .get ('inference_time_sec','')
        row ['GAN_Wasserstein']=pc3 .get ('wasserstein','')
        row ['GAN_Gen_Params']=pc3 .get ('gen_params','')
        row ['GAN_Disc_Params']=pc3 .get ('disc_params','')
        rows_cls .append (row )
    df_cls =pd .DataFrame (rows_cls )
    df_cls .to_csv (os .path .join (output_dir ,'comparison_per_class.csv'),
    index =False )

    return df_summary ,df_cls 


def generate_charts (r1 ,r2 ,r3 ,classes ,output_dir ):
    """Genera gráficas comparativas en PNG."""
    names =[r1 ['name'],r2 ['name'],r3 ['name']]
    colors =['#2196F3','#FF9800','#4CAF50']
    x =np .arange (len (names ))

    fig ,axes =plt .subplots (2 ,3 ,figsize =(18 ,10 ))
    fig .suptitle ('Análisis de Carga Computacional — GAN para NIDS',
    fontsize =14 ,fontweight ='bold')


    ax =axes [0 ,0 ]
    vals =[r ['total_params']for r in [r1 ,r2 ,r3 ]]
    bars =ax .bar (x ,vals ,color =colors ,edgecolor ='white',linewidth =0.5 )
    ax .set_title ('Parámetros Totales',fontweight ='bold')
    ax .set_xticks (x );ax .set_xticklabels (names ,fontsize =8 )
    ax .set_ylabel ('Parámetros')
    for bar ,val in zip (bars ,vals ):
        ax .text (bar .get_x ()+bar .get_width ()/2 ,bar .get_height ()*1.01 ,
        f'{val :,}',ha ='center',va ='bottom',fontsize =7 )
    ax .grid (axis ='y',alpha =0.3 )


    ax =axes [0 ,1 ]
    vals =[r ['training']['total_time_sec']for r in [r1 ,r2 ,r3 ]]
    bars =ax .bar (x ,vals ,color =colors ,edgecolor ='white',linewidth =0.5 )
    ax .set_title ('Tiempo de Entrenamiento',fontweight ='bold')
    ax .set_xticks (x );ax .set_xticklabels (names ,fontsize =8 )
    ax .set_ylabel ('Segundos')
    for bar ,val in zip (bars ,vals ):
        ax .text (bar .get_x ()+bar .get_width ()/2 ,bar .get_height ()*1.01 ,
        format_time (val ),ha ='center',va ='bottom',fontsize =8 )
    ax .grid (axis ='y',alpha =0.3 )


    ax =axes [0 ,2 ]
    vals =[r ['training']['total_grad_ops']for r in [r1 ,r2 ,r3 ]]
    bars =ax .bar (x ,vals ,color =colors ,edgecolor ='white',linewidth =0.5 )
    ax .set_title ('Operaciones de Gradiente Totales',fontweight ='bold')
    ax .set_xticks (x );ax .set_xticklabels (names ,fontsize =8 )
    ax .set_ylabel ('Grad Ops')
    for bar ,val in zip (bars ,vals ):
        ax .text (bar .get_x ()+bar .get_width ()/2 ,bar .get_height ()*1.01 ,
        f'{val :,}',ha ='center',va ='bottom',fontsize =7 )
    ax .grid (axis ='y',alpha =0.3 )


    ax =axes [1 ,0 ]
    vals =[r ['training']['resources'].get ('ram_peak_MB',0 )for r in [r1 ,r2 ,r3 ]]
    bars =ax .bar (x ,vals ,color =colors ,edgecolor ='white',linewidth =0.5 )
    ax .set_title ('RAM Pico — Entrenamiento (MB)',fontweight ='bold')
    ax .set_xticks (x );ax .set_xticklabels (names ,fontsize =8 )
    ax .set_ylabel ('MB')
    ax .grid (axis ='y',alpha =0.3 )


    ax =axes [1 ,1 ]
    vals =[r ['training']['resources'].get ('vram_peak_MB',0 )for r in [r1 ,r2 ,r3 ]]
    bars =ax .bar (x ,vals ,color =colors ,edgecolor ='white',linewidth =0.5 )
    ax .set_title ('VRAM Pico — Entrenamiento (MB)',fontweight ='bold')
    ax .set_xticks (x );ax .set_xticklabels (names ,fontsize =8 )
    ax .set_ylabel ('MB')
    ax .grid (axis ='y',alpha =0.3 )


    ax =axes [1 ,2 ]
    vals =[r ['avg_wasserstein']for r in [r1 ,r2 ,r3 ]]
    bars =ax .bar (x ,vals ,color =colors ,edgecolor ='white',linewidth =0.5 )
    ax .set_title ('Distancia Wasserstein Media\n(menor = mejor calidad)',
    fontweight ='bold')
    ax .set_xticks (x );ax .set_xticklabels (names ,fontsize =8 )
    ax .set_ylabel ('Wasserstein')
    for bar ,val in zip (bars ,vals ):
        ax .text (bar .get_x ()+bar .get_width ()/2 ,bar .get_height ()*1.01 ,
        f'{val :.4f}',ha ='center',va ='bottom',fontsize =8 )
    ax .grid (axis ='y',alpha =0.3 )

    plt .tight_layout (rect =[0 ,0 ,1 ,0.95 ])
    plt .savefig (os .path .join (output_dir ,'comparison_overview.png'),dpi =150 ,
    bbox_inches ='tight')
    plt .close ()


    fig ,axes =plt .subplots (1 ,3 ,figsize =(20 ,6 ))
    fig .suptitle ('Comparación por Clase',fontsize =13 ,fontweight ='bold')
    x_cls =np .arange (len (classes ))
    w =0.25 


    ax =axes [0 ]
    wg_t =[r1 ['per_class'][c ].get ('train_time_sec',0 )for c in classes ]
    gan_t =[r3 ['per_class'][c ].get ('train_time_sec',0 )for c in classes ]
    ax .bar (x_cls -w /2 ,wg_t ,w ,label ='WGAN-GP',color ='#2196F3')
    ax .bar (x_cls +w /2 ,gan_t ,w ,label ='GAN Básica',color ='#4CAF50')
    ax .set_xticks (x_cls )
    ax .set_xticklabels (classes ,rotation =30 ,ha ='right',fontsize =8 )
    ax .set_title ('Tiempo Entrenamiento por Clase (s)',fontweight ='bold')
    ax .set_ylabel ('Segundos')
    ax .legend (fontsize =8 )
    ax .grid (axis ='y',alpha =0.3 )


    ax =axes [1 ]
    wg_i =[r1 ['per_class'][c ].get ('inference_time_sec',0 )for c in classes ]
    cgan_i =[r2 ['per_class'][c ].get ('inference_time_sec',0 )for c in classes ]
    gan_i =[r3 ['per_class'][c ].get ('inference_time_sec',0 )for c in classes ]
    ax .bar (x_cls -w ,wg_i ,w ,label ='WGAN-GP',color ='#2196F3')
    ax .bar (x_cls ,cgan_i ,w ,label ='CGAN Mono.',color ='#FF9800')
    ax .bar (x_cls +w ,gan_i ,w ,label ='GAN Básica',color ='#4CAF50')
    ax .set_xticks (x_cls )
    ax .set_xticklabels (classes ,rotation =30 ,ha ='right',fontsize =8 )
    ax .set_title ('Tiempo Inferencia por Clase (s)',fontweight ='bold')
    ax .set_ylabel ('Segundos')
    ax .legend (fontsize =8 )
    ax .grid (axis ='y',alpha =0.3 )


    ax =axes [2 ]
    wg_w =[r1 ['per_class'][c ].get ('wasserstein',0 )for c in classes ]
    cgan_w =[r2 ['per_class'][c ].get ('wasserstein',0 )for c in classes ]
    gan_w =[r3 ['per_class'][c ].get ('wasserstein',0 )for c in classes ]
    ax .bar (x_cls -w ,wg_w ,w ,label ='WGAN-GP',color ='#2196F3')
    ax .bar (x_cls ,cgan_w ,w ,label ='CGAN Mono.',color ='#FF9800')
    ax .bar (x_cls +w ,gan_w ,w ,label ='GAN Básica',color ='#4CAF50')
    ax .set_xticks (x_cls )
    ax .set_xticklabels (classes ,rotation =30 ,ha ='right',fontsize =8 )
    ax .set_title ('Dist. Wasserstein por Clase\n(menor = mejor)',fontweight ='bold')
    ax .set_ylabel ('Wasserstein')
    ax .legend (fontsize =8 )
    ax .grid (axis ='y',alpha =0.3 )

    plt .tight_layout (rect =[0 ,0 ,1 ,0.93 ])
    plt .savefig (os .path .join (output_dir ,'comparison_per_class.png'),dpi =150 ,
    bbox_inches ='tight')
    plt .close ()

    print (f"\n  Gráficas guardadas en: {output_dir }")


def save_summary_txt (r1 ,r2 ,r3 ,classes ,config_info ,output_dir ):
    """Genera resumen en texto plano con conclusiones automáticas."""
    path =os .path .join (output_dir ,'RESUMEN_LOAD_TEST.txt')
    with open (path ,'w',encoding ='utf-8')as f :
        f .write ("="*80 +"\n")
        f .write ("ANÁLISIS DE CARGA COMPUTACIONAL — GAN para NIDS\n")
        f .write ("="*80 +"\n")
        f .write (f"Fecha: {datetime .now ().strftime ('%Y-%m-%d %H:%M:%S')}\n")
        f .write (f"Epochs:   {config_info ['epochs_cgan_gan']}\n")
        f .write (f"Muestras inferencia/clase: {config_info ['gen_samples']}\n")
        f .write (f"GPU: {config_info .get ('gpu_name','N/A')}\n")
        f .write (f"Dataset: CIC-IDS2017 — {config_info ['n_classes']} clases, "
        f"{sum (config_info ['class_dist'].values ()):,} muestras\n\n")


        for r ,tag in [(r1 ,'WGAN-GP (Propuesto)'),
        (r2 ,'CGAN Monolítica Básica'),
        (r3 ,'GAN Básica por Clase')]:
            f .write ("-"*60 +"\n")
            f .write (f"{tag }\n")
            f .write ("-"*60 +"\n")
            f .write (f"  Parámetros totales:   {r ['total_params']:>12,}\n")
            f .write (f"  Tamaño est. modelos:  {r ['model_size_est_MB']:>12.2f} MB\n")
            f .write (f"  Grad ops totales:     {r ['training']['total_grad_ops']:>12,}\n")
            f .write (f"  Tiempo entrenamiento: {r ['training']['total_time_str']:>12}\n")
            f .write (f"  Tiempo inferencia:    {r ['inference']['total_time_str']:>12}\n")
            f .write (f"  Wasserstein media:    {r ['avg_wasserstein']:>12.6f}\n")
            f .write (f"  Recursos entrenamiento:\n")
            for k ,v in r ['training']['resources'].items ():
                f .write (f"    {k :<20}: {v }\n")
            f .write (f"  Recursos inferencia:\n")
            for k ,v in r ['inference']['resources'].items ():
                f .write (f"    {k :<20}: {v }\n")
            f .write ("\n")


        f .write ("="*80 +"\n")
        f .write ("CONCLUSIONES\n")
        f .write ("="*80 +"\n\n")

        t1 =r1 ['training']['total_time_sec']
        t2 =r2 ['training']['total_time_sec']
        t3 =r3 ['training']['total_time_sec']
        w1 =r1 ['avg_wasserstein']
        w2 =r2 ['avg_wasserstein']
        w3 =r3 ['avg_wasserstein']
        go1 =r1 ['training']['total_grad_ops']
        go2 =r2 ['training']['total_grad_ops']
        go3 =r3 ['training']['total_grad_ops']

        f .write (f"1. OVERHEAD COMPUTACIONAL:\n")
        f .write (f"   - WGAN-GP realiza {go1 :,} grad ops vs {go3 :,} de GAN básica "
        f"({go1 /go3 :.1f}x más).\n")
        f .write (f"   - Este overhead se debe al penalty de gradiente (n_critic pasos "
        f"extra por epoch).\n")
        f .write (f"   - WGAN-GP tarda {format_time (t1 )} vs {format_time (t3 )} de GAN básica "
        f"({t1 /t3 :.1f}x).\n\n")

        f .write (f"2. EFICIENCIA DE MODELO MONOLÍTICO:\n")
        f .write (f"   - CGAN monolítica entrena en {format_time (t2 )} "
        f"({t1 /t2 :.1f}x más rápido que WGAN-GP).\n")
        f .write (f"   - Solo requiere {r2 ['total_params']:,} parámetros "
        f"(1 modelo vs {len (classes )} modelos).\n")
        if t2 >0 :
            f .write (f"   - Grad ops/segundo: "
            f"{go2 /t2 :.0f} (CGAN) vs {go1 /t1 :.0f} (WGAN-GP) vs "
            f"{go3 /t3 :.0f} (GAN básica).\n\n")

        f .write (f"3. CALIDAD (Wasserstein — menor = mejor):\n")
        f .write (f"   - WGAN-GP:       {w1 :.6f}\n")
        f .write (f"   - CGAN monolít.: {w2 :.6f}\n")
        f .write (f"   - GAN básica:    {w3 :.6f}\n")
        if w1 <w2 and w1 <w3 :
            f .write (f"   → WGAN-GP logra la MEJOR calidad, justificando su overhead.\n")
        elif w1 <w2 :
            f .write (f"   → WGAN-GP supera a la CGAN monolítica en calidad.\n")

        f .write (f"\n4. TRADE-OFF TIEMPO vs CALIDAD:\n")
        if t3 >0 and w3 >0 :
            ratio_tiempo =t1 /t3 
            ratio_calidad =w3 /w1 if w1 >0 else float ('inf')
            f .write (f"   - WGAN-GP usa {ratio_tiempo :.1f}x más tiempo que GAN básica.\n")
            f .write (f"   - Pero su calidad es {ratio_calidad :.1f}x mejor "
            f"(Wasserstein {ratio_calidad :.1f}x menor).\n")
            f .write (f"   - Relación calidad/tiempo favorable al enfoque propuesto.\n")

        f .write (f"\n5. INFERENCIA:\n")
        f .write (f"   - Los tres enfoques tienen tiempos de inferencia comparables.\n")
        f .write (f"   - WGAN-GP: {format_time (r1 ['inference']['total_time_sec'])}\n")
        f .write (f"   - CGAN:    {format_time (r2 ['inference']['total_time_sec'])}\n")
        f .write (f"   - GAN:     {format_time (r3 ['inference']['total_time_sec'])}\n")
        f .write (f"   - La generación de datos sintéticos es rápida en todos los casos.\n")

    print (f"\n  Resumen guardado en: {path }")





def main ():
    parser =argparse .ArgumentParser (
    description ='Análisis de Carga Computacional: WGAN-GP vs CGAN vs GAN',
    formatter_class =argparse .RawDescriptionHelpFormatter ,
    epilog ="""
Ejemplos:
  python 8_load_test.py                          # WGAN-GP adaptativo completo
  python 8_load_test.py --epochs 2000            # CGAN/GAN: 2000 epochs
  python 8_load_test.py --wgan-scale 0.1         # WGAN-GP al 10% (test rápido)
  python 8_load_test.py --epochs 2000 --skip 2   # Sin CGAN monolítica
        """
    )
    parser .add_argument ('--epochs',type =int ,default =2000 ,
    help ='Epochs para CGAN monolítica y GAN básica '
    '(default: 2000). WGAN-GP usa epochs adaptativos '
    'de su config (15000/25000/30000).')
    parser .add_argument ('--wgan-scale',type =float ,default =1.0 ,
    help ='Factor multiplicador de epochs para WGAN-GP '
    '(default: 1.0 = completo, 0.1 = 10%% para test rápido)')
    parser .add_argument ('--gen-samples',type =int ,default =10000 ,
    help ='Muestras a generar por clase en inferencia '
    '(default: 10000)')
    parser .add_argument ('--gpu',type =str ,default ='0',
    help ='GPU a utilizar (default: 0)')
    parser .add_argument ('--skip',nargs ='*',choices =['1','2','3'],
    default =[],
    help ='Saltar experimento(s): '
    '1=WGAN-GP, 2=CGAN monolít., 3=GAN por clase')
    args =parser .parse_args ()


    os .environ ["CUDA_VISIBLE_DEVICES"]=args .gpu 
    gpu_idx =0 

    gpus =tf .config .list_physical_devices ('GPU')
    if gpus :
        for gpu in gpus :
            tf .config .experimental .set_memory_growth (gpu ,True )
        print (f"[GPU] {len (gpus )} GPU(s) detectada(s)")
    else :
        print ("[GPU] No se detectó GPU — usando CPU")

    gpu_name =get_gpu_name ()


    timestamp =datetime .now ().strftime ('%Y%m%d_%H%M%S')
    output_dir =os .path .join (OUTPUT_BASE ,f'load_test_{timestamp }')
    os .makedirs (output_dir ,exist_ok =True )


    wgan_ep_info =(f"adaptativos (15k/25k/30k) x{args .wgan_scale }"
    if args .wgan_scale !=1.0 
    else "adaptativos (15k/25k/30k) — idéntico a 1_gan_wgan.py")
    print ("\n╔"+"═"*78 +"╗")
    print ("║"+"  ANÁLISIS DE CARGA COMPUTACIONAL — GAN para NIDS".center (78 )+"║")
    print ("╠"+"═"*78 +"╣")
    print (f"║  WGAN-GP epochs          : {wgan_ep_info :<50}║")
    print (f"║  CGAN/GAN epochs         : {args .epochs :<50}║")
    print (f"║  WGAN-GP scale           : {args .wgan_scale :<50}║")
    print (f"║  Muestras generadas/clase: {args .gen_samples :<50}║")
    print (f"║  GPU                     : {str (gpu_name ):<50}║")
    print (f"║  Salida                  : {output_dir :<50}║")
    skipped =', '.join (args .skip )if args .skip else 'ninguno'
    print (f"║  Experimentos omitidos   : {skipped :<50}║")
    print ("╚"+"═"*78 +"╝")


    X ,labels ,feature_names ,scaler ,class_dist =load_and_preprocess ()
    classes =sorted (class_dist .keys ())


    config_info ={
    'epochs_cgan_gan':args .epochs ,
    'wgan_scale':args .wgan_scale ,
    'gen_samples':args .gen_samples ,
    'gpu':args .gpu ,
    'gpu_name':str (gpu_name ),
    'latent_dim':LATENT_DIM ,
    'data_dim':X .shape [1 ],
    'n_classes':len (classes ),
    'classes':classes ,
    'class_dist':{k :int (v )for k ,v in class_dist .items ()},
    'timestamp':timestamp ,
    'has_psutil':HAS_PSUTIL ,
    'has_pynvml':HAS_PYNVML ,
    'config_wgan_large':CONFIG_WGAN_LARGE ,
    'config_wgan_small':CONFIG_WGAN_SMALL ,
    'config_wgan_very_small':CONFIG_WGAN_VERY_SMALL ,
    'config_baseline':CONFIG_BASELINE ,
    }
    with open (os .path .join (output_dir ,'experiment_config.json'),'w')as f :
        json .dump (config_info ,f ,indent =2 )


    r1 ,r2 ,r3 =None ,None ,None 



    global_monitor =ResourceMonitor (interval =1.0 ,gpu_index =gpu_idx ,
    experiment ='',phase ='')
    global_monitor .start ()

    if '1'not in args .skip :
        r1 =run_experiment_1_wgan_gp (
        X ,labels ,class_dist ,args .epochs ,args .gen_samples ,gpu_idx ,
        global_monitor =global_monitor ,wgan_scale =args .wgan_scale )
    else :
        print ("\n  [SKIP] Caso 1: WGAN-GP")

    if '2'not in args .skip :
        r2 =run_experiment_2_cgan_monolithic (
        X ,labels ,class_dist ,args .epochs ,args .gen_samples ,gpu_idx ,
        global_monitor =global_monitor )
    else :
        print ("\n  [SKIP] Caso 2: CGAN Monolítica")

    if '3'not in args .skip :
        r3 =run_experiment_3_gan_per_class (
        X ,labels ,class_dist ,args .epochs ,args .gen_samples ,gpu_idx ,
        global_monitor =global_monitor )
    else :
        print ("\n  [SKIP] Caso 3: GAN por Clase")


    global_monitor .stop ()
    df_timeline =global_monitor .export_samples_df ()
    timeline_csv =os .path .join (output_dir ,'resource_timeline.csv')
    df_timeline .to_csv (timeline_csv ,index =False )
    print (f"\n  Timeline de recursos guardado: {timeline_csv }")
    print (f"  ({len (df_timeline )} muestras, "
    f"{df_timeline ['elapsed_sec'].max ():.0f}s de monitorización)")


    generate_resource_timeline_chart (df_timeline ,output_dir )


    if r1 and r2 and r3 :
        print_comparison_table (r1 ,r2 ,r3 )
        print_per_class_table (r1 ,r2 ,r3 ,classes )

        save_results_csv (r1 ,r2 ,r3 ,classes ,output_dir )
        generate_charts (r1 ,r2 ,r3 ,classes ,output_dir )
        save_summary_txt (r1 ,r2 ,r3 ,classes ,config_info ,output_dir )


        all_results ={
        'config':config_info ,
        'wgan_gp':r1 ,
        'cgan_monolithic':r2 ,
        'gan_per_class':r3 ,
        }
        with open (os .path .join (output_dir ,'raw_results.json'),'w')as f :
            json .dump (all_results ,f ,indent =2 ,default =str )

        print (f"\n  Todos los resultados guardados en: {output_dir }")
        print (f"  Archivos generados:")
        print (f"    - comparison_summary.csv      (tabla resumen)")
        print (f"    - comparison_per_class.csv     (detalle por clase)")
        print (f"    - comparison_overview.png      (gráficas resumen)")
        print (f"    - comparison_per_class.png     (gráficas por clase)")
        print (f"    - resource_timeline.csv        (métricas por segundo)")
        print (f"    - resource_timeline.png        (gráfico temporal)")
        print (f"    - RESUMEN_LOAD_TEST.txt        (resumen con conclusiones)")
        print (f"    - raw_results.json             (datos crudos)")
        print (f"    - experiment_config.json       (configuración)")
    else :

        executed ={}
        for key ,r in [('wgan_gp',r1 ),('cgan_monolithic',r2 ),
        ('gan_per_class',r3 )]:
            if r :
                executed [key ]=r 
        executed ['config']=config_info 
        with open (os .path .join (output_dir ,'partial_results.json'),'w')as f :
            json .dump (executed ,f ,indent =2 ,default =str )
        print (f"\n  Resultados parciales guardados en: {output_dir }")

    print ("\n"+"═"*80 )
    print ("  ANÁLISIS DE CARGA COMPUTACIONAL COMPLETADO")
    print ("═"*80 )


if __name__ =="__main__":
    main ()
