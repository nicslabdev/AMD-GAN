# AMD-GAN: Class-Specific Adaptive GANs for Configurable Synthetic Traffic and Imbalance Correction

<b>Estos resultados han sido (parcialmente) financiados por la Cátedra Internacional UMA 2023, la cual forma parte del Programa Global de Innovación en Seguridad para la promoción de Cátedras de Ciberseguridad en España financiado por la Unión Europea Fondos NextGeneration-EU, a través del Instituto Nacional de Ciberseguridad (INCIBE).</b>

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.x](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18488999.svg)](https://doi.org/10.5281/zenodo.18488999)

This repository contains the implementation of **AMD-GAN**, a novel framework for generating synthetic network traffic using Class-Specific Wasserstein GANs with Gradient Penalty (WGAN-GP). The system is designed to address class imbalance in Network Intrusion Detection Systems (NIDS) across multiple datasets (CIC-IDS2017, UNSW-NB15, and Edge-IIoT 2022) by generating high-quality synthetic samples for minority attack classes.

## Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Repository Structure](#-repository-structure)
- [Installation](#-installation)
- [Usage (Step-by-Step Workflow)](#-usage)
- [License](#-license)
- [Acknowledgments](#-acknowledgments)

## Overview

Network Intrusion Detection Systems suffer from severe class imbalance, where benign traffic vastly outnumbers attack samples. Traditional oversampling techniques (SMOTE, ADASYN) often fail to capture the complex distributions of network traffic data.

**AMD-GAN** proposes a class-specific GAN approach where:
- Each attack class has its own dedicated WGAN-GP model
- Minority classes use adaptive configurations (smaller batch sizes, more epochs, stronger regularization)
- Synthetic data quality is validated through Train-Synthetic-Test-Real (TSTR) protocols
- The robustness of models is tested against adversarial traffic conditions (stress tests)

### Architecture

The framework uses **Wasserstein GAN with Gradient Penalty (WGAN-GP)** with the following characteristics:
- **Generator**: Multi-layer perceptron with LeakyReLU activations and BatchNormalization
- **Critic**: Wasserstein distance estimator with gradient penalty
- **Class-Specific Training**: Separate models for each traffic class
- **Adaptive Configuration**: Dynamic hyperparameters based on class cardinality

## Key Features

- **Class-Specific GANs**: Dedicated WGAN-GP model for each traffic class across 3 major NIDS datasets.
- **Adaptive Training**: Automatic configuration adjustment based on class sample size.
- **Configurable Generation**: Generate custom, balanced or imbalanced datasets.
- **TSTR Evaluation**: Train-Synthetic-Test-Real methodologies to ensure synthetic data is valid and useful.
- **Oversampling Benchmarks**: Direct comparisons against SMOTE, ADASYN, BorderlineSMOTE, Random Oversampling.
- **Stress Testing**: Evaluate trained NIDS robustness against extreme GAN-generated adversarial scenarios (e.g., botnet storms, DDoS floods).
- **Mathematical Validation**: Quantify data distribution similarities using Wasserstein distances and Maximum Mean Discrepancy (MMD) metrics.
- **Computational Profiling**: Load testing and ablation studies supporting the proposed adaptive design choices.

## Repository Structure

The workflow is numbered sequentially (from `1_*` to `10_*`) for a clear, step-by-step pipeline:

```text
AMD-GAN/
│
├── 1_gan_wgan.py                        # WGAN-GP training for CIC-IDS2017
├── 1_1_gan_wgan_unsw.py                 # WGAN-GP training for UNSW-NB15
├── 1_2_gan_wgan_edgeiiot.py             # WGAN-GP training for Edge-IIoT
│
├── 2_generate_synthetic_dataset.py      # Synthetic data generation (CIC-IDS2017)
├── 2_1_generate_synthetic_dataset_unsw.py
├── 2_2_generate_synthetic_dataset_edgeiiot.py
│
├── 3_experiment_oversampling_comparison.py # Oversampling technique comparison
├── 3_1experiment_oversampling_unsw.py
├── 3_2experiment_oversampling_edgeiiot.py
│
├── 4_tstr.py                            # TSTR evaluation (Unified)
├── 4_1_tstr_unsw.py
├── 4_2_tstr_edgeiiot.py
│
├── 5_visualize_results.py               # Generates TSTR/oversampling visualizations
├── 6_stress_test_nids.py                # Audits NIDS vs extreme adversarial traffic
├── 7_math_analysis.py                   # Calculates Wasserstein distance & MMD
├── 8_load_test.py                       # RAM/VRAM/CPU/Time computational load test
├── 9_ablation.py                        # Ablation study: Configuration vs Cardinality
└── 10_figure_summary.py                 # Summary figures generation
```

## Installation

### Requirements

```bash
pip install numpy pandas polars scikit-learn tensorflow lightgbm xgboost imbalanced-learn matplotlib scipy psutil pynvml
```

### Dependencies
- Python ≥ 3.8
- TensorFlow ≥ 2.x
- NumPy, Pandas, Polars
- Scikit-learn
- LightGBM, XGBoost
- imbalanced-learn (for comparison experiments)
- Matplotlib, Scipy
- `psutil`, `pynvml` (optional, for resource monitoring during computational load tests)

## Usage

The scripts are organized sequentially from 1 to 10. Below is the general execution workflow. 
*Note: Depending on the dataset you are studying, run the corresponding `_unsw.py`, `_edgeiiot.py`, or base `.py` versions.*

### 1. Training WGAN-GP Models (`1_*.py`)
Train class-specific WGAN-GP models with adaptive configurations assigned by dataset size:

```bash
# Train on CIC-IDS2017
python 1_gan_wgan.py --classes Bot "Web Attack"

# Train on UNSW-NB15 or Edge-IIoT
python 1_1_gan_wgan_unsw.py --all
python 1_2_gan_wgan_edgeiiot.py --all
```

**Configuration tiers based on class size:**

| Class Size | Batch Size | Epochs | Lambda GP | Oversample Factor |
|------------|------------|--------|-----------|-------------------|
| Large | 128 | 15,000 | 10.0 | 1x |
| Small | 32 | 25,000 | 15.0 | 10x |
| Very Small | 16 | 30,000 | 20.0 | 20x |

### 2. Generating Synthetic Datasets (`2_*.py`)
Produce synthetic data pools mapping the distributions learned by the WGAN-GP models:

```bash
python 2_generate_synthetic_dataset.py --balanced 10000
python 2_1_generate_synthetic_dataset_unsw.py --benign 50000 --exploits 10000
```

### 3. Oversampling Comparison Experiment (`3_*.py`)
Evaluate AMD-GAN data against classical oversamplers (SMOTE, ADASYN, Random OverSampler, etc.) training top classifiers:
```bash
python 3_experiment_oversampling_comparison.py
```

### 4. Train-Synthetic-Test-Real (TSTR) Evaluation (`4_*.py`)
Quantify generalizability by training models exclusively on synthetic data, then testing their effectiveness on the real unseen testing hold-out sets (TRTR vs. TSTR).
```bash
python 4_tstr.py
python 4_1_tstr_unsw.py
```

### 5. Utilities, Stress Tests & Analysis (`5_*.py` to `10_*.py`)

- **Results Visualization:** Generate TSTR publication figures:
  `python 5_visualize_results.py`
  
- **Adversarial Stress Testing:** Subject a real-trained model against massive GAN-generated extreme edge-cases, finding breakpoints:
  `python 6_stress_test_nids.py`
  
- **Mathematical Distances:** Calculates empirical Wasserstein 1D mean & MMD² matching algorithms:
  `python 7_math_analysis.py`
  
- **Hardware Load and Ablation:** Hardware costs and justifying hyper-parameters:
  `python 8_load_test.py`
  `python 9_ablation.py --parallel --gpus 0,1,2`
  `python 10_figure_summary.py`

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Dataset: CIC-IDS2017 from the Canadian Institute for Cybersecurity, UNSW-NB15, Edge-IIoT 2022.
- This repository is part of a publication, which is also part of the project "CiberIA: Investigación e Innovación para la Integración de Ciberseguridad e Inteligencia Artificial (Proyecto C079/23)", financed by "European Union NextGeneration-EU, the Recovery Plan, Transformation and Resilience", through INCIBE. It has also been partially supported by the project SecAI (PID2022-139268OB-I00) funded by the Spanish Ministerio de Ciencia e Innovacion, and Agencia Estatal de Investigacion.
