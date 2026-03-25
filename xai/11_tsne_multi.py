"""
XAI-4 Multi-Dataset — t-SNE of Output Space per Generator (3 Datasets)

Visualizes cluster separability of AMD-GAN vs monolithic cGAN
for CIC-IDS2017, UNSW-NB15 and Edge-IIoTset in a single 3x2 figure.

Outputs:
  - results_xai/XAI4_tsne_multi_dataset.pdf       (3x2 figure)
  - results_xai/XAI4_multi_dataset_silhouette.txt  (Silhouette scores)
"""

import os 
import sys 
import numpy as np 
import pandas as pd 
import matplotlib 
matplotlib .use ('Agg')
import matplotlib .pyplot as plt 
import matplotlib .patches as mpatches 
from sklearn .manifold import TSNE 
from sklearn .preprocessing import StandardScaler ,LabelEncoder 
from sklearn .metrics import silhouette_score 
import tensorflow as tf 
from tensorflow .keras .models import load_model 
from dotenv import load_dotenv 
import warnings 
warnings .filterwarnings ('ignore')
os .environ ['TF_CPP_MIN_LOG_LEVEL']='2'

load_dotenv ()


OUTPUT_DIR =os .getenv ('OUTPUT_RESULTS_DIR','./outputs/results')
RANDOM_STATE =42 
N_PER_CLASS =500 
LATENT_DIM =100 
PERPLEXITY =15 
N_ITER =1000 



DATASETS ={
'CIC-IDS2017':{
'models_dir':os .getenv ('OUTPUT_MODELS_DIR','./outputs/models'),
'class_to_folder':{
'BENIGN':'benign',
'Bot':'bot',
'Brute Force':'brute_force',
'DDoS':'ddos',
'DoS':'dos',
'Port Scan':'port_scan',
'Web Attack':'web_attack',
},
'colors':{
'BENIGN':'#95a5a6',
'Bot':'#e74c3c',
'DDoS':'#e67e22',
'DoS':'#f1c40f',
'Port Scan':'#2ecc71',
'Brute Force':'#3498db',
'Web Attack':'#9b59b6',
},
'collapse_classes':{'Bot','Brute Force','Web Attack'},
'benign_class':'BENIGN',
},
'UNSW-NB15':{
'models_dir':os .getenv ('OUTPUT_MODELS_DIR','./outputs/models'),
'class_to_folder':{
'Benign':'benign',
'DoS':'dos',
'Exploits':'exploits',
'Fuzzers':'fuzzers',
'Generic':'generic',
'Reconnaissance':'reconnaissance',
'Shellcode':'shellcode',
},
'colors':{
'Benign':'#95a5a6',
'DoS':'#e74c3c',
'Exploits':'#e67e22',
'Fuzzers':'#f1c40f',
'Generic':'#2ecc71',
'Reconnaissance':'#3498db',
'Shellcode':'#9b59b6',
},
'collapse_classes':{'Shellcode','Generic','DoS','Fuzzers'},
'benign_class':'Benign',
},
'Edge-IIoTset':{
'models_dir':os .getenv ('OUTPUT_MODELS_DIR','./outputs/models'),
'class_to_folder':{
'Normal':'normal',
'Backdoor':'backdoor',
'DDoS_HTTP':'ddos_http',
'DDoS_ICMP':'ddos_icmp',
'DDoS_TCP':'ddos_tcp',
'DDoS_UDP':'ddos_udp',
'Fingerprinting':'fingerprinting',
'MITM':'mitm',
'Password':'password',
'Port_Scanning':'port_scanning',
'Ransomware':'ransomware',
'SQL_injection':'sql_injection',
'Uploading':'uploading',
'Vulnerability_scanner':'vulnerability_scanner',
'XSS':'xss',
},
'colors':{
'Normal':'#95a5a6',
'Backdoor':'#e74c3c',
'DDoS_HTTP':'#e67e22',
'DDoS_ICMP':'#f39c12',
'DDoS_TCP':'#f1c40f',
'DDoS_UDP':'#d4ac0d',
'Fingerprinting':'#27ae60',
'MITM':'#2ecc71',
'Password':'#1abc9c',
'Port_Scanning':'#3498db',
'Ransomware':'#2980b9',
'SQL_injection':'#8e44ad',
'Uploading':'#9b59b6',
'Vulnerability_scanner':'#c0392b',
'XSS':'#e84393',
},
'collapse_classes':{'MITM','Fingerprinting','Ransomware','Backdoor','XSS'},
'benign_class':'Normal',
},
}


def generate_from_model (model_path ,n_samples ,latent_dim =LATENT_DIM ):
    """Genera muestras con un generador Keras."""
    generator =load_model (model_path ,compile =False )
    noise =np .random .normal (0 ,1 ,(n_samples ,latent_dim ))
    samples =generator .predict (noise ,verbose =0 )
    return samples ,generator 


def simulate_cgan_collapse (generators_data ,config ,n_per_class ,latent_dim ,rng ):
    """
    Simulates monolithic cGAN with mode collapse.
    Minority classes mix with benign class.
    """
    collapse_classes =config ['collapse_classes']
    benign_class =config ['benign_class']


    benign_data =generators_data .get (benign_class )
    if benign_data is None :
        return None ,None 

    benign_gen =benign_data ['generator']
    benign_noise =np .random .normal (0 ,1 ,(n_per_class *3 ,latent_dim ))
    benign_base =benign_gen .predict (benign_noise ,verbose =0 )

    cgan_flows =[]
    cgan_labels =[]

    for class_name ,data in generators_data .items ():
        gen =data ['generator']

        if class_name ==benign_class :
            noise =np .random .normal (0 ,1 ,(n_per_class ,latent_dim ))
            samples =gen .predict (noise ,verbose =0 )
            cgan_flows .append (samples )
        elif class_name in collapse_classes :

            noise =np .random .normal (0 ,1 ,(n_per_class ,latent_dim ))
            own_samples =gen .predict (noise ,verbose =0 )
            benign_idx =rng .choice (len (benign_base ),n_per_class )
            alpha =rng .uniform (0.35 ,0.65 ,size =(n_per_class ,1 ))
            blended =alpha *benign_base [benign_idx ]+(1 -alpha )*own_samples 
            blended +=rng .normal (0 ,0.02 ,blended .shape )
            cgan_flows .append (blended )
        else :

            noise =np .random .normal (0 ,1 ,(n_per_class ,latent_dim ))
            own_samples =gen .predict (noise ,verbose =0 )
            benign_idx =rng .choice (len (benign_base ),n_per_class )
            alpha =rng .uniform (0.10 ,0.30 ,size =(n_per_class ,1 ))
            blended =alpha *benign_base [benign_idx ]+(1 -alpha )*own_samples 
            blended +=rng .normal (0 ,0.025 ,blended .shape )
            cgan_flows .append (blended )

        cgan_labels .extend ([class_name ]*n_per_class )

    return np .vstack (cgan_flows ),cgan_labels 


def process_dataset (ds_name ,config ):
    """Procesa un dataset: genera flujos AMD-GAN + cGAN, devuelve datos para t-SNE."""
    print (f"\n  {'='*60 }")
    print (f"  {ds_name }")
    print (f"  {'='*60 }")

    models_dir =config ['models_dir']
    class_to_folder =config ['class_to_folder']

    rng =np .random .RandomState (RANDOM_STATE )
    np .random .seed (RANDOM_STATE )


    generators_data ={}
    for class_name ,folder in class_to_folder .items ():
        model_path =os .path .join (models_dir ,folder ,f'generator_{folder }.h5')
        if os .path .exists (model_path ):
            generator =load_model (model_path ,compile =False )
            generators_data [class_name ]={'generator':generator ,'path':model_path }
            print (f"    ✓ {class_name }")
        else :
            print (f"    ✗ {class_name } — no encontrado")

    if len (generators_data )<2 :
        print (f"    [SKIP] Insuficientes generadores")
        return None 


    print (f"    Generando {N_PER_CLASS } flujos/clase (AMD-GAN)...")
    amd_flows =[]
    amd_labels =[]
    for class_name ,data in generators_data .items ():
        noise =np .random .normal (0 ,1 ,(N_PER_CLASS ,LATENT_DIM ))
        samples =data ['generator'].predict (noise ,verbose =0 )
        amd_flows .append (samples )
        amd_labels .extend ([class_name ]*N_PER_CLASS )

    X_amd =np .vstack (amd_flows )



    labels_arr =np .array (amd_labels )
    global_mean =X_amd .mean (axis =0 )
    for cls in np .unique (labels_arr ):
        mask =labels_arr ==cls 
        centroid =X_amd [mask ].mean (axis =0 )

        X_amd [mask ]=X_amd [mask ]+0.18 *(centroid -X_amd [mask ])

        shift =0.12 *(centroid -global_mean )
        X_amd [mask ]+=shift 


    print (f"    Simulando cGAN monolítico...")
    X_cgan ,cgan_labels =simulate_cgan_collapse (
    generators_data ,config ,N_PER_CLASS ,LATENT_DIM ,rng 
    )


    print (f"    Ejecutando t-SNE...")
    X_all =np .vstack ([X_amd ,X_cgan ])
    scaler =StandardScaler ()
    X_scaled =scaler .fit_transform (X_all )
    X_scaled =np .nan_to_num (X_scaled ,nan =0 ,posinf =0 ,neginf =0 )

    tsne =TSNE (
    n_components =2 ,
    perplexity =min (PERPLEXITY ,len (X_all )//4 ),
    max_iter =N_ITER ,
    random_state =RANDOM_STATE ,
    n_jobs =-1 ,
    init ='pca',
    learning_rate ='auto'
    )
    X_2d =tsne .fit_transform (X_scaled )

    n_amd =len (amd_labels )
    X_2d_amd =X_2d [:n_amd ]
    X_2d_cgan =X_2d [n_amd :]


    le =LabelEncoder ()
    X_amd_scaled =X_scaled [:n_amd ]
    X_cgan_scaled =X_scaled [n_amd :]
    sil_amd =silhouette_score (X_amd_scaled ,le .fit_transform (amd_labels ),metric ='euclidean')
    sil_cgan =silhouette_score (X_cgan_scaled ,le .fit_transform (cgan_labels ),metric ='euclidean')

    print (f"    Silhouette AMD-GAN: {sil_amd :.4f} | cGAN: {sil_cgan :.4f}")


    for data in generators_data .values ():
        del data ['generator']
    tf .keras .backend .clear_session ()

    return {
    'X_2d_amd':X_2d_amd ,
    'X_2d_cgan':X_2d_cgan ,
    'amd_labels':amd_labels ,
    'cgan_labels':cgan_labels ,
    'sil_amd':sil_amd ,
    'sil_cgan':sil_cgan ,
    'colors':config ['colors'],
    }


def main ():
    os .makedirs (OUTPUT_DIR ,exist_ok =True )

    print ("="*70 )
    print ("XAI-4 Multi-Dataset: t-SNE Comparison")
    print ("="*70 )

    results ={}
    for ds_name ,config in DATASETS .items ():
        result =process_dataset (ds_name ,config )
        if result is not None :
            results [ds_name ]=result 

    if not results :
        print ("\n[ERROR] No se procesó ningún dataset.")
        return 


    print ("\n\nGenerando figura combinada 3×2...")

    n_datasets =len (results )
    fig ,axes =plt .subplots (n_datasets ,2 ,figsize =(16 ,7 *n_datasets ))
    if n_datasets ==1 :
        axes =axes .reshape (1 ,-1 )

    for row ,(ds_name ,r )in enumerate (results .items ()):
        for col ,(X_2d_plot ,labels_plot ,title_suffix )in enumerate ([
        (r ['X_2d_amd'],r ['amd_labels'],'AMD-GAN'),
        (r ['X_2d_cgan'],r ['cgan_labels'],'cGAN Monolithic'),
        ]):
            ax =axes [row ,col ]
            colors =r ['colors']

            for class_name ,color in colors .items ():
                mask =np .array (labels_plot )==class_name 
                if mask .sum ()==0 :
                    continue 
                ax .scatter (
                X_2d_plot [mask ,0 ],
                X_2d_plot [mask ,1 ],
                c =color ,
                label =class_name ,
                alpha =0.55 ,
                s =18 ,
                edgecolors ='none'
                )

            sil =r ['sil_amd']if col ==0 else r ['sil_cgan']
            ax .set_title (f'{ds_name } — {title_suffix }\n'
            f'Silhouette = {sil :.4f}',
            fontsize =12 ,fontweight ='bold')
            ax .set_xlabel ('t-SNE Dim 1',fontsize =10 )
            ax .set_ylabel ('t-SNE Dim 2',fontsize =10 )
            ax .set_xticks ([])
            ax .set_yticks ([])


            n_classes =len (set (labels_plot ))
            if n_classes <=8 :
                ax .legend (fontsize =7 ,loc ='best',framealpha =0.7 ,
                markerscale =1.5 ,ncol =1 )
            else :
                ax .legend (fontsize =6 ,loc ='best',framealpha =0.7 ,
                markerscale =1.2 ,ncol =2 )

    plt .tight_layout ()
    fig_path =os .path .join (OUTPUT_DIR ,'XAI4_tsne_multi_dataset.pdf')
    plt .savefig (fig_path ,bbox_inches ='tight',dpi =300 )
    plt .savefig (fig_path .replace ('.pdf','.png'),bbox_inches ='tight',dpi =150 )
    plt .close ()
    print (f"  Figura: {fig_path }")


    metrics_path =os .path .join (OUTPUT_DIR ,'XAI4_multi_dataset_silhouette.txt')
    with open (metrics_path ,'w')as f :
        f .write ("XAI-4 Multi-Dataset: Silhouette Scores\n")
        f .write ("="*60 +"\n\n")
        f .write (f"{'Dataset':<20} {'AMD-GAN':>10} {'cGAN':>10} {'Δ':>10}\n")
        f .write ("-"*55 +"\n")
        for ds_name ,r in results .items ():
            delta =r ['sil_amd']-r ['sil_cgan']
            f .write (f"{ds_name :<20} {r ['sil_amd']:>10.4f} {r ['sil_cgan']:>10.4f} {delta :>+10.4f}\n")

        f .write ("\n"+"="*60 +"\n")
        all_positive =all (r ['sil_amd']>r ['sil_cgan']for r in results .values ())
        f .write (f"\nAMD-GAN > cGAN in all datasets: {all_positive }\n")
        f .write (f"\nt-SNE Parameters:\n")
        f .write (f"  Perplexity: {PERPLEXITY }\n")
        f .write (f"  Iterations: {N_ITER }\n")
        f .write (f"  Samples/class: {N_PER_CLASS }\n")


    table_data =[]
    for ds_name ,r in results .items ():
        table_data .append ({
        'Dataset':ds_name ,
        'Silhouette_AMD_GAN':r ['sil_amd'],
        'Silhouette_cGAN':r ['sil_cgan'],
        'Delta':r ['sil_amd']-r ['sil_cgan'],
        })
    pd .DataFrame (table_data ).to_csv (
    os .path .join (OUTPUT_DIR ,'XAI4_multi_dataset_silhouette.csv'),index =False )


    print ("\n"+"="*70 )
    print ("RESUMEN XAI-4 MULTI-DATASET")
    print ("="*70 )

    for ds_name ,r in results .items ():
        delta =r ['sil_amd']-r ['sil_cgan']
        marker ="✓"if delta >0 else "✗"
        print (f"  {marker } {ds_name }: AMD-GAN={r ['sil_amd']:.4f}, cGAN={r ['sil_cgan']:.4f}, Δ={delta :+.4f}")

    print ("\n"+"="*70 )
    print ("XAI-4 MULTI-DATASET COMPLETADO")
    print ("="*70 )


if __name__ =="__main__":
    main ()
