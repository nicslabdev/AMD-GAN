# AMD-GAN: Class-Specific Adaptive GANs for Configurable Synthetic Traffic and Imbalance Correction

<b>Estos resultados han sido (parcialmente) financiados por la Cátedra Internacional UMA 2023, la cual forma parte del Programa Global de Innovación en Seguridad para la promoción de Cátedras de Ciberseguridad en España financiado por la Unión Europea Fondos NextGeneration-EU, a través del Instituto Nacional de Ciberseguridad (INCIBE).</b>

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.x](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19212815.svg)](https://doi.org/10.5281/zenodo.19212815)

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

The workflow is organized into three main directories for clarity and maintainability:

```text
AMD-GAN/
├── scripts/
│   ├── 01_gan_wgan_cicids2017.py
│   ├── 01_gan_wgan_edgeiiot.py
│   ├── 01_gan_wgan_unsw.py
│   ├── 02_generate_synthetic_data_cicids2017.py
│   ├── 02_generate_synthetic_data_edgeiiot.py
│   └── 02_generate_synthetic_data_unsw.py
│
├── tests/
│   ├── 03_oversampling_cicids2017.py
│   ├── 03_oversampling_edgeiiot.py
│   ├── 03_oversampling_unsw.py
│   ├── 04_tstr_cicids2017.py
│   ├── 04_tstr_edgeiiot.py
│   ├── 04_tstr_unsw.py
│   ├── 05_stress_test_nids.py
│   ├── 06_math_analysis.py
│   └── 07_load_test.py
│
├── xai/
│   ├── 08_shap_generator_multi.py
│   ├── 09_shap_rank_multi.py
│   ├── 10_lime_soc_attribution.py
│   └── 11_tsne_multi.py
│
├── data/                                # Datasets (create this directory)
│   ├── CIC-IDS2017.csv                  # Download from official sources
│   ├── UNSW-NB15.csv
│   └── Edge-IIoT.csv
│
├── outputs/                             # Generated during execution
│   ├── models/                          # Trained GAN models
│   ├── synthetic_data/                  # Generated synthetic samples
│   ├── results/                         # Evaluation metrics
│   └── plots/                           # Visualizations
│
├── .env.example                         # Configuration template
├── .env                                 # Local configuration (not committed)
├── LICENSE
├── README.md
└── requirements.txt
```

## Installation

### Prerequisites
- Python ≥ 3.8
- pip or conda

### Quick Setup

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/AMD-GAN.git
cd AMD-GAN
```

2. **Create virtual environment (recommended)**
```bash
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Configure environment**
```bash
cp .env.example .env
# Edit .env with your dataset paths and settings
```

5. **Prepare datasets**
Create a `data/` directory and download datasets:
- CIC-IDS2017: https://www.unb.ca/cic/datasets/ids-2017.html
- UNSW-NB15: https://www.unsw.adfa.edu.au/unsw-canberra-cyber/cybersecurity/UNSW-NB15-Datasets/
- Edge-IIoT: https://www.kaggle.com/datasets/agungprabowo/edge-iiot-dataset

## Usage

### Step-by-Step Workflow

The scripts are numbered sequentially for a complete pipeline:

**Phase 1: Model Training (scripts/)**
```bash
cd scripts/
python 01_gan_wgan_cicids2017.py      # Train WGAN-GP for CIC-IDS2017
python 01_gan_wgan_unsw.py            # Train WGAN-GP for UNSW-NB15
python 01_gan_wgan_edgeiiot.py        # Train WGAN-GP for Edge-IIoT
```

**Phase 2: Generate Synthetic Data (scripts/)**
```bash
python 02_generate_synthetic_data_cicids2017.py   # Generate synthetic samples
python 02_generate_synthetic_data_unsw.py
python 02_generate_synthetic_data_edgeiiot.py
```

**Phase 3: Evaluation & Comparison (tests/)**
```bash
cd ../tests/
python 03_oversampling_cicids2017.py     # Run oversampling experiments
python 04_tstr_cicids2017.py             # TSTR evaluation
python 05_stress_test_nids.py            # Robustness testing
python 06_math_analysis.py               # Distribution analysis
python 07_load_test.py                   # Performance profiling
```

**Phase 4: Explainability Analysis (xai/)**
```bash
cd ../xai/
python 08_shap_generator_multi.py        # Feature importance (SHAP)
python 09_shap_rank_multi.py             # SHAP ranking overlap analysis
python 10_lime_soc_attribution.py        # Local interpretability (LIME)
python 11_tsne_multi.py                  # Dimensionality reduction comparison
```

### Configuration

Edit `.env` file to customize:
- Dataset paths
- Output directories
- Training hyperparameters
- GPU/resource settings
- Logging levels

### Example: Run complete pipeline for one dataset
```bash
# CIC-IDS2017
python scripts/01_gan_wgan_cicids2017.py
python scripts/02_generate_synthetic_data_cicids2017.py
python tests/03_oversampling_cicids2017.py
python tests/04_tstr_cicids2017.py
python tests/05_stress_test_nids.py
python tests/06_math_analysis.py
python xai/08_shap_generator_multi.py
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Dataset: CIC-IDS2017 from the Canadian Institute for Cybersecurity, UNSW-NB15, Edge-IIoT 2022.
- This repository is part of a publication, which is also part of the project "CiberIA: Investigación e Innovación para la Integración de Ciberseguridad e Inteligencia Artificial (Proyecto C079/23)", financed by "European Union NextGeneration-EU, the Recovery Plan, Transformation and Resilience", through INCIBE. It has also been partially supported by the project SecAI (PID2022-139268OB-I00) funded by the Spanish Ministerio de Ciencia e Innovacion, and Agencia Estatal de Investigacion.
