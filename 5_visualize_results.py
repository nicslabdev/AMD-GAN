"""
Script: Visualización de Results TSTR vs TRTR - Unificado
============================================================

Genera plots básicas + formato paper académico para múltiples datasets:
  - CICIDS2017: completo_v2_balanceado, completo_v2_uniforme, completo_v2_no_outliers
  - UNSW-NB15: unsw_balanceado, unsw_uniforme

Usage:
    python 5_visualize_results.py                          # todos los datasets
    python 5_visualize_results.py --dataset cicids2017     # solo CICIDS2017 (balanceado)
    python 5_visualize_results.py --dataset unsw           # solo UNSW (balanceado)
    python 5_visualize_results.py --dataset completo_v2_balanceado  # subdataset exacto
    python 5_visualize_results.py --list                   # listar datasets disponibles
    python 5_visualize_results.py --mode paper             # solo paper-quality
    python 5_visualize_results.py --mode basic             # solo plots exploratorias
    python 5_visualize_results.py --mode all               # ambos (por defecto)
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch, Rectangle
from matplotlib.lines import Line2D
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap

# ============================================================================
# DATASET REGISTRY
# ============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_REGISTRY = {
    # --- CICIDS2017 variants ---
    'completo_v2_balanceado': {
        'family': 'CICIDS2017',
        'label': 'CICIDS-2017 (Balanced)',
        'results_dir': os.path.join(BASE_DIR, 'results_tstr', 'completo_v2_balanceado'),
        'default_best_model': 'LightGBM',
        'default_basic_model': 'RandomForest',
    },
    'completo_v2_uniforme': {
        'family': 'CICIDS2017',
        'label': 'CICIDS-2017 (Uniform)',
        'results_dir': os.path.join(BASE_DIR, 'results_tstr', 'completo_v2_uniforme'),
        'default_best_model': 'LightGBM',
        'default_basic_model': 'RandomForest',
    },
    'completo_v2_no_outliers': {
        'family': 'CICIDS2017',
        'label': 'CICIDS-2017 (No Outliers)',
        'results_dir': os.path.join(BASE_DIR, 'results_tstr', 'completo_v2_no_outliers'),
        'default_best_model': 'LightGBM',
        'default_basic_model': 'RandomForest',
    },
    # --- UNSW-NB15 variants ---
    'unsw_balanceado': {
        'family': 'UNSW-NB15',
        'label': 'UNSW-NB15 (Balanced)',
        'results_dir': os.path.join(BASE_DIR, 'results_tstr_unsw', 'unsw_balanceado'),
        'default_best_model': 'LightGBM',
        'default_basic_model': 'RandomForest',
    },
    'unsw_uniforme': {
        'family': 'UNSW-NB15',
        'label': 'UNSW-NB15 (Uniform)',
        'results_dir': os.path.join(BASE_DIR, 'results_tstr_unsw', 'unsw_uniforme'),
        'default_best_model': 'LightGBM',
        'default_basic_model': 'RandomForest',
    },
    # --- Edge-IIoT 2022 variants ---
    'edgeiiot_balanceado': {
        'family': 'Edge-IIoT-2022',
        'label': 'Edge-IIoT 2022 (Balanced)',
        'results_dir': os.path.join(BASE_DIR, 'results_tstr_edgeiiot', 'edgeiiot_balanceado'),
        'default_best_model': 'LightGBM',
        'default_basic_model': 'RandomForest',
    },
    'edgeiiot_uniforme': {
        'family': 'Edge-IIoT-2022',
        'label': 'Edge-IIoT 2022 (Uniform)',
        'results_dir': os.path.join(BASE_DIR, 'results_tstr_edgeiiot', 'edgeiiot_uniforme'),
        'default_best_model': 'LightGBM',
        'default_basic_model': 'RandomForest',
    },
}

# Alias shortcuts
DATASET_ALIASES = {
    'cicids2017': 'completo_v2_balanceado',
    'cicids': 'completo_v2_balanceado',
    'unsw': 'unsw_balanceado',
    'unsw-nb15': 'unsw_balanceado',
    'unswnb15': 'unsw_balanceado',
    'edgeiiot': 'edgeiiot_balanceado',
    'edge-iiot': 'edgeiiot_balanceado',
    'edgeiiot2022': 'edgeiiot_balanceado',
}

# Models available across datasets
MODELOS_FULL = ['Dummy', 'LogisticReg', 'RandomForest', 'KNeighbors', 'LightGBM', 'XGBoost', 'Voting']
MODELOS_SHORT = ['Dum', 'LR', 'RF', 'KNN', 'LGBM', 'XGB', 'Vote']

# ============================================================================
# STYLE CONFIGURATION
# ============================================================================

def setup_academic_style():
    """Configure matplotlib for publication-quality figures."""
    plt.style.use('seaborn-v0_8-whitegrid')
    mpl.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif', 'Bitstream Vera Serif'],
        'font.size': 11,
        'axes.titlesize': 12,
        'axes.labelsize': 9,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'figure.titlesize': 13,
        'axes.linewidth': 1.0,
        'axes.edgecolor': 'black',
        'axes.spines.top': False,
        'axes.spines.right': False,
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        'xtick.major.size': 5,
        'ytick.major.size': 5,
        'xtick.minor.size': 3,
        'ytick.minor.size': 3,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linewidth': 0.5,
        'grid.linestyle': '--',
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,
        'figure.facecolor': 'white',
        'mathtext.fontset': 'stix',
    })


def setup_basic_style():
    """Configure matplotlib for quick exploratory plots."""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 11
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12


FONT_SCALE = 1.6

def scale_all_fonts(fig, scale=FONT_SCALE):
    """Scale all text sizes in a figure by the given factor."""
    for text in fig.findobj(match=mpl.text.Text):
        size = text.get_fontsize()
        if size is not None:
            text.set_fontsize(size * scale)


# Color palettes
COLORS = {
    'TSTR': '#1f77b4',
    'TRTR': '#ff7f0e',
    'accent1': '#2ca02c',
    'accent2': '#d62728',
    'positive': '#2ca02c',
    'negative': '#d62728',
    'neutral': '#7f7f7f',
    'highlight': '#9467bd',
}

COLORES_BASIC = {
    'TSTR': '#2ecc71',
    'TRTR': '#3498db',
    'gap_positive': '#27ae60',
    'gap_negative': '#e74c3c',
}

MODEL_COLORS = {
    'Dummy': '#d3d3d3',
    'LogisticReg': '#98df8a',
    'RandomForest': '#1f77b4',
    'KNeighbors': '#ff7f0e',
    'LightGBM': '#2ca02c',
    'XGBoost': '#d62728',
    'Voting': '#9467bd',
}

# ============================================================================
# DATA LOADING HELPERS
# ============================================================================

def cargar_matriz_confusion(filepath):
    """Load confusion matrix from CSV; returns None if not found."""
    if os.path.exists(filepath):
        return pd.read_csv(filepath, index_col=0)
    return None


def cargar_classification_report(filepath):
    """Load classification report from CSV; returns None if not found."""
    if os.path.exists(filepath):
        return pd.read_csv(filepath, index_col=0)
    return None


def cargar_results_globales(results_dir):
    """Load global comparison CSV. Tries comparative_global.csv, then todos_results.csv."""
    path = os.path.join(results_dir, 'comparative_global.csv')
    if os.path.exists(path):
        return pd.read_csv(path)
    path = os.path.join(results_dir, 'todos_results.csv')
    if os.path.exists(path):
        df = pd.read_csv(path)
        if 'Tipo' in df.columns and 'Clasificacion' in df.columns:
            df['Experimento'] = (df['Tipo'] + '_' +
                                 df['Clasificacion']
                                 .str.replace('Multiclass', 'Multi')
                                 .str.replace('Binaria', 'Binary'))
        return df
    return None


def detectar_classs_desde_cm(filepath):
    """Detect class names from a confusion matrix CSV."""
    if os.path.exists(filepath):
        df = pd.read_csv(filepath, index_col=0)
        classs = list(df.columns)
        classs_short = [c[:4] if len(c) > 4 else c for c in classs]
        return classs, classs_short
    return None, None


def detectar_models_disponibles(results_dir):
    """Detect which models actually have result files."""
    available = []
    for m in MODELOS_FULL:
        path = os.path.join(results_dir, f'TSTR_MULTI_{m}_confusion_matrix.csv')
        if os.path.exists(path):
            available.append(m)
    if not available:
        for m in MODELOS_FULL:
            path = os.path.join(results_dir, f'TSTR_BINARY_{m}_confusion_matrix.csv')
            if os.path.exists(path):
                available.append(m)
    return available if available else MODELOS_FULL


def get_best_model(results_dir, default='LightGBM'):
    """Determine the best model by highest F1_Macro in TSTR_Multi (excluding Dummy)."""
    df = cargar_results_globales(results_dir)
    if df is not None:
        df_tstr = df[df['Experimento'] == 'TSTR_Multi']
        if len(df_tstr) > 0:
            df_tstr_nodummy = df_tstr[df_tstr['Model'] != 'Dummy']
            if len(df_tstr_nodummy) > 0:
                return df_tstr_nodummy.loc[df_tstr_nodummy['F1_Macro'].idxmax(), 'Model']
    return default


# ============================================================================
# PAPER-QUALITY FIGURES (academic)
# ============================================================================

def fig1_confusion_matrices(results_dir, output_path, best_model='LightGBM', dataset_label=''):
    """Figure 1: Confusion matrices for multiclass classification (side by side)."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    cm_tstr_path = os.path.join(results_dir, f'TSTR_MULTI_{best_model}_confusion_matrix.csv')
    cm_trtr_path = os.path.join(results_dir, f'TRTR_MULTI_{best_model}_confusion_matrix.csv')

    cm_tstr_df = cargar_matriz_confusion(cm_tstr_path)
    cm_trtr_df = cargar_matriz_confusion(cm_trtr_path)

    if cm_tstr_df is None or cm_trtr_df is None:
        print(f"  [SKIP] Confusion matrices not found for {best_model}")
        plt.close()
        return

    classs = list(cm_tstr_df.columns)
    classs_short = [c[:4] if len(c) > 4 else c for c in classs]
    n_classes = len(classs)

    cm_tstr = cm_tstr_df.values
    cm_trtr = cm_trtr_df.values

    cm_tstr_norm = cm_tstr.astype('float') / cm_tstr.sum(axis=1)[:, np.newaxis] * 100
    cm_trtr_norm = cm_trtr.astype('float') / cm_trtr.sum(axis=1)[:, np.newaxis] * 100

    cmap = LinearSegmentedColormap.from_list('custom', ['#ffffff', '#1f77b4', '#0d3d6e'])

    for ax, cm_norm, panel, label in [
        (axes[0], cm_tstr_norm, 'a', f'TSTR - {best_model}'),
        (axes[1], cm_trtr_norm, 'b', f'TRTR (Baseline) - {best_model}'),
    ]:
        im = ax.imshow(cm_norm, cmap=cmap, vmin=0, vmax=100, aspect='auto')
        ax.set_xticks(range(n_classes))
        ax.set_yticks(range(n_classes))
        ax.set_xticklabels(classs_short, rotation=45, ha='right', fontsize=10)
        ax.set_yticklabels(classs_short, fontsize=10)
        ax.set_xlabel('Predicted Class', fontsize=11, fontweight='bold')
        ax.set_ylabel('True Class', fontsize=11, fontweight='bold')
        ax.set_title(f'({panel}) {label}', fontweight='bold', fontsize=12, pad=10)
        for i in range(n_classes):
            for j in range(n_classes):
                val = cm_norm[i, j]
                color = 'white' if val > 50 else 'black'
                fontweight = 'bold' if i == j else 'normal'
                ax.text(j, i, f'{val:.1f}', ha='center', va='center',
                        fontsize=9, color=color, fontweight=fontweight)

    fig.subplots_adjust(right=0.88)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Recall (%)', rotation=270, labelpad=20, fontsize=11, fontweight='bold')

    if dataset_label:
        fig.suptitle(dataset_label, fontsize=13, fontweight='bold', y=1.02)

    scale_all_fonts(fig)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  [OK] {os.path.basename(output_path)}")


def fig2_model_comparison(results_dir, output_path, dataset_label=''):
    """Figure 2: F1-Macro comparison across all models (Multiclass + Binary)."""
    df = cargar_results_globales(results_dir)
    if df is None:
        print("  [SKIP] Global results not found")
        return

    models_disponibles = [m for m in MODELOS_FULL if m in df['Model'].unique()]
    models_short = [MODELOS_SHORT[MODELOS_FULL.index(m)] for m in models_disponibles]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    x = np.arange(len(models_disponibles))
    width = 0.35

    for ax_idx, (ax, exp_tstr, exp_trtr, subtitle) in enumerate(zip(
        axes,
        ['TSTR_Multi', 'TSTR_Binary'],
        ['TRTR_Multi', 'TRTR_Binary'],
        ['(a) Multiclass', '(b) Binary'],
    )):
        df_tstr = df[df['Experimento'] == exp_tstr].set_index('Model')
        df_trtr = df[df['Experimento'] == exp_trtr].set_index('Model')

        tstr_vals = [df_tstr.loc[m, 'F1_Macro'] if m in df_tstr.index else 0 for m in models_disponibles]
        trtr_vals = [df_trtr.loc[m, 'F1_Macro'] if m in df_trtr.index else 0 for m in models_disponibles]

        bars1 = ax.bar(x - width/2, tstr_vals, width, label='TSTR (Synthetic)',
                       color=COLORS['TSTR'], edgecolor='black', linewidth=0.8)
        bars2 = ax.bar(x + width/2, trtr_vals, width, label='TRTR (Real)',
                       color=COLORS['TRTR'], edgecolor='black', linewidth=0.8)

        for bar, val in zip(bars1, tstr_vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=8, rotation=90)
        for bar, val in zip(bars2, trtr_vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=8, rotation=90)

        ax.set_ylabel('$F_1$-Macro Score', fontsize=11, fontweight='bold')
        ax.set_xlabel('Classification Model', fontsize=11, fontweight='bold')
        ax.set_title(subtitle, fontweight='bold', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(models_short, fontsize=10)
        ax.set_ylim(0, 1.15)
        ax.axhline(y=0.85, color='gray', linestyle='--', linewidth=1, alpha=0.7, label='Threshold (0.85)')
        ax.legend(loc='upper left', frameon=True, fancybox=True, shadow=True, fontsize=9)
        ax.grid(axis='y', alpha=0.3)

    if dataset_label:
        fig.suptitle(dataset_label, fontsize=13, fontweight='bold', y=1.02)

    plt.tight_layout()
    scale_all_fonts(fig)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  [OK] {os.path.basename(output_path)}")


def fig3_per_class_analysis(results_dir, output_path, best_model='LightGBM', dataset_label=''):
    """Figure 3: Per-class F1 + gap lollipop chart."""
    tstr_report = cargar_classification_report(
        os.path.join(results_dir, f'TSTR_MULTI_{best_model}_classification_report.csv'))
    trtr_report = cargar_classification_report(
        os.path.join(results_dir, f'TRTR_MULTI_{best_model}_classification_report.csv'))

    if tstr_report is None or trtr_report is None:
        print(f"  [SKIP] Reports not found for {best_model}")
        return

    classs = [c for c in tstr_report.index if c not in ['accuracy', 'macro avg', 'weighted avg']]
    classs_short = [c[:4] if len(c) > 4 else c for c in classs]
    n_classs = len(classs)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    x = np.arange(n_classs)
    width = 0.35

    # (a) F1-Score comparison
    ax = axes[0]
    tstr_f1 = [tstr_report.loc[c, 'f1-score'] if c in tstr_report.index else 0 for c in classs]
    trtr_f1 = [trtr_report.loc[c, 'f1-score'] if c in trtr_report.index else 0 for c in classs]

    bars1 = ax.bar(x - width/2, tstr_f1, width, label='TSTR',
                   color=COLORS['TSTR'], edgecolor='black', linewidth=0.8)
    bars2 = ax.bar(x + width/2, trtr_f1, width, label='TRTR',
                   color=COLORS['TRTR'], edgecolor='black', linewidth=0.8)

    for i, (t, r) in enumerate(zip(tstr_f1, trtr_f1)):
        if t < 0.5:
            ax.bar(x[i] - width/2, t, width, color=COLORS['negative'],
                   edgecolor='black', linewidth=0.8, alpha=0.7)

    ax.set_ylabel('$F_1$-Score', fontsize=11, fontweight='bold')
    ax.set_xlabel('Attack Class', fontsize=11, fontweight='bold')
    ax.set_title(f'(a) $F_1$-Score by Class ({best_model})', fontweight='bold', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(classs_short, rotation=45, ha='right', fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.axhline(y=0.85, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    ax.axhline(y=0.5, color='red', linestyle=':', linewidth=1, alpha=0.5)
    ax.legend(loc='lower right', frameon=True, fancybox=True, shadow=True)
    ax.grid(axis='y', alpha=0.3)

    # (b) Gap analysis lollipop
    ax = axes[1]
    gaps = [t - r for t, r in zip(tstr_f1, trtr_f1)]
    colors = [COLORS['positive'] if g >= 0 else COLORS['negative'] for g in gaps]
    y_pos = np.arange(n_classs)

    for i, gap in enumerate(gaps):
        ax.plot([0, gap], [i, i], color=colors[i], linewidth=2, zorder=1)
    ax.scatter(gaps, y_pos, c=colors, s=150, zorder=2, edgecolors='black', linewidth=1)
    ax.axvline(x=0, color='black', linewidth=1.5, zorder=0)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(classs_short, fontsize=10)
    ax.set_xlabel(r'$\Delta F_1$ (TSTR - TRTR)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Attack Class', fontsize=11, fontweight='bold')
    ax.set_title('(b) Performance Gap by Class', fontweight='bold', fontsize=12)

    gap_min = min(gaps) if gaps else -0.5
    gap_max = max(gaps) if gaps else 0.1
    ax.set_xlim(gap_min - 0.1, gap_max + 0.1)
    ax.grid(axis='x', alpha=0.3)

    for i, gap in enumerate(gaps):
        offset = -0.03 if gap < 0 else 0.03
        ha = 'right' if gap < 0 else 'left'
        ax.text(gap + offset, i, f'{gap:+.2f}', ha=ha, va='center', fontsize=9, fontweight='bold')

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['positive'],
               markersize=10, label='TSTR better'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['negative'],
               markersize=10, label='TRTR better'),
    ]
    ax.legend(handles=legend_elements, loc='lower left', frameon=True, fancybox=True)

    if dataset_label:
        fig.suptitle(dataset_label, fontsize=13, fontweight='bold', y=1.02)

    plt.tight_layout()
    scale_all_fonts(fig)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  [OK] {os.path.basename(output_path)}")


def _plot_precision_recall_axes(axes, results_dir, best_model='LightGBM', panel_labels=('a', 'b')):
    """Draw precision-recall scatter on given axes pair."""
    sample_report = cargar_classification_report(
        os.path.join(results_dir, f'TSTR_MULTI_{best_model}_classification_report.csv'))
    if sample_report is None:
        print(f"  [SKIP] Reports not found for {best_model}")
        return False

    classs = [c for c in sample_report.index if c not in ['accuracy', 'macro avg', 'weighted avg']]
    classs_short = [c[:4] if len(c) > 4 else c for c in classs]

    for idx, (exp_type, title) in enumerate([('TSTR', 'TSTR (Synthetic)'), ('TRTR', 'TRTR (Real)')]):
        ax = axes[idx]
        report = cargar_classification_report(
            os.path.join(results_dir, f'{exp_type}_MULTI_{best_model}_classification_report.csv'))
        if report is None:
            continue

        precision = [report.loc[c, 'precision'] if c in report.index else 0 for c in classs]
        recall = [report.loc[c, 'recall'] if c in report.index else 0 for c in classs]
        cmap_classes = plt.cm.tab10

        for i, class in enumerate(classs_short):
            ax.scatter(recall[i], precision[i], s=220, c=[cmap_classes(i)],
                       label=class, edgecolors='black', linewidth=1.5, zorder=3)

        for f1 in [0.2, 0.4, 0.6, 0.8, 0.9]:
            x_curve = np.linspace(0.01, 1, 100)
            y_curve = (f1 * x_curve) / (2 * x_curve - f1)
            mask = (y_curve >= 0) & (y_curve <= 1)
            ax.plot(x_curve[mask], y_curve[mask], '--', color='gray', alpha=0.4, linewidth=0.8)
            idx_label = np.argmin(np.abs(y_curve - x_curve))
            if 0 < idx_label < len(x_curve) - 10:
                ax.text(x_curve[idx_label] + 0.02, y_curve[idx_label] - 0.02,
                        f'$F_1$={f1}', fontsize=7, color='gray', rotation=45)

        ax.set_xlabel('Recall', fontsize=11, fontweight='bold')
        ax.set_ylabel('Precision', fontsize=11, fontweight='bold')
        ax.set_title(f'({panel_labels[idx]}) {title} - {best_model}', fontweight='bold', fontsize=12)
        ax.set_xlim(-0.05, 1.1)
        ax.set_ylim(-0.05, 1.1)
        ax.legend(loc='lower left', frameon=True, fancybox=True, shadow=True, ncol=2, fontsize=8)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
    return True


def _plot_summary_heatmaps_axes(fig, axes, results_dir, panel_labels=('c', 'd')):
    """Draw F1-Macro and Accuracy heatmaps on given axes pair."""
    df = cargar_results_globales(results_dir)
    if df is None:
        print("  [SKIP] Global results not found")
        return False

    models_ordenados = [m for m in MODELOS_FULL if m in df['Model'].unique()]
    n_models = len(models_ordenados)
    if n_models == 0:
        return False

    experimentos = ['TSTR_Multi', 'TRTR_Multi', 'TSTR_Binary', 'TRTR_Binary']
    exp_labels = ['TSTR\nMulti', 'TRTR\nMulti', 'TSTR\nBinary', 'TRTR\nBinary']

    for ax, metric, vmin, panel_label, cbar_label in [
        (axes[0], 'F1_Macro', 0.3, panel_labels[0], '$F_1$-Macro'),
        (axes[1], 'Accuracy', 0.5, panel_labels[1], 'Accuracy'),
    ]:
        data = np.zeros((n_models, 4))
        for i, m in enumerate(models_ordenados):
            for j, exp in enumerate(experimentos):
                val = df[(df['Model'] == m) & (df['Experimento'] == exp)][metric]
                data[i, j] = val.values[0] if len(val) > 0 else 0

        im = ax.imshow(data, cmap='RdYlGn', vmin=vmin, vmax=1.0, aspect='auto')
        ax.set_xticks(range(4))
        ax.set_yticks(range(n_models))
        ax.set_xticklabels(exp_labels, fontsize=10)
        ax.set_yticklabels([MODELOS_SHORT[MODELOS_FULL.index(m)] for m in models_ordenados], fontsize=10)
        ax.set_title(f'({panel_label}) {cbar_label} Heatmap', fontweight='bold', fontsize=12)

        thresh = 0.7 if metric == 'F1_Macro' else 0.8
        for i in range(n_models):
            for j in range(4):
                val = data[i, j]
                color = 'white' if val < thresh else 'black'
                ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                        fontsize=10, color=color, fontweight='bold')

        cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
        cbar.set_label(cbar_label, fontsize=10)

    return True


def fig4_precision_recall_heatmaps(results_dir, output_path, best_model='LightGBM', dataset_label=''):
    """Figure 4: Precision-Recall scatter (top) + Summary Heatmaps (bottom)."""
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 10.5))

    ok_pr = _plot_precision_recall_axes(axes[0], results_dir, best_model=best_model, panel_labels=('a', 'b'))
    ok_hm = _plot_summary_heatmaps_axes(fig, axes[1], results_dir, panel_labels=('c', 'd'))

    if not ok_pr or not ok_hm:
        plt.close()
        return

    if dataset_label:
        fig.suptitle(dataset_label, fontsize=13, fontweight='bold', y=1.01)

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.35)
    scale_all_fonts(fig)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  [OK] {os.path.basename(output_path)}")


def fig6_radar_chart(results_dir, output_path, dataset_label=''):
    """Figure 6: Radar chart comparing models across metrics."""
    df = cargar_results_globales(results_dir)
    if df is None:
        print("  [SKIP] Global results not found")
        return

    df_tstr = df[df['Experimento'] == 'TSTR_Multi'].set_index('Model')
    metrics = ['Accuracy', 'F1_Macro', 'F1_Weighted']
    metric_labels = ['Accuracy', '$F_1$-Macro', '$F_1$-Weighted']

    models_radar = [m for m in MODELOS_FULL if m != 'Dummy' and m in df_tstr.index]
    if len(models_radar) == 0:
        print("  [SKIP] No models for radar chart")
        return

    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    colors = plt.cm.Set2(np.linspace(0, 1, len(models_radar)))

    for idx, model in enumerate(models_radar):
        values = [df_tstr.loc[model, m] for m in metrics]
        values += values[:1]
        ax.plot(angles, values, 'o-', linewidth=2, label=model, color=colors[idx])
        ax.fill(angles, values, alpha=0.1, color=colors[idx])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_labels, fontsize=12)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=9)
    ax.legend(loc='lower right', bbox_to_anchor=(1.3, 0), frameon=True, fancybox=True)
    title = 'Model Comparison - TSTR Multiclass'
    if dataset_label:
        title += f'\n{dataset_label}'
    ax.set_title(title, fontweight='bold', fontsize=14, pad=20)

    plt.tight_layout()
    scale_all_fonts(fig)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  [OK] {os.path.basename(output_path)}")


def fig7_gap_analysis(results_dir, output_path, dataset_label=''):
    """Figure 7: Horizontal bar gap analysis (TSTR - TRTR)."""
    df = cargar_results_globales(results_dir)
    if df is None:
        print("  [SKIP] Global results not found")
        return

    models_ordenados = [m for m in MODELOS_FULL if m in df['Model'].unique() and m != 'Dummy']
    if not models_ordenados:
        print("  [SKIP] No models found")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    y_pos = np.arange(len(models_ordenados))

    for ax, exp_tstr, exp_trtr, subtitle in zip(
        axes,
        ['TSTR_Multi', 'TSTR_Binary'],
        ['TRTR_Multi', 'TRTR_Binary'],
        ['(a) Performance Gap - Multiclass', '(b) Performance Gap - Binary'],
    ):
        gaps = []
        for m in models_ordenados:
            tstr_val = df[(df['Model'] == m) & (df['Experimento'] == exp_tstr)]['F1_Macro']
            trtr_val = df[(df['Model'] == m) & (df['Experimento'] == exp_trtr)]['F1_Macro']
            gap = (tstr_val.values[0] - trtr_val.values[0]) if len(tstr_val) > 0 and len(trtr_val) > 0 else 0
            gaps.append(gap)

        colors = [COLORS['positive'] if g >= 0 else COLORS['negative'] for g in gaps]
        bars = ax.barh(y_pos, gaps, color=colors, edgecolor='black', linewidth=0.8, height=0.6)
        ax.axvline(x=0, color='black', linewidth=1.5)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([MODELOS_SHORT[MODELOS_FULL.index(m)] for m in models_ordenados], fontsize=11)
        ax.set_xlabel(r'$\Delta F_1$-Macro (TSTR - TRTR)', fontsize=11, fontweight='bold')
        ax.set_title(subtitle, fontweight='bold', fontsize=12)
        ax.grid(axis='x', alpha=0.3)

        for i, (bar, gap) in enumerate(zip(bars, gaps)):
            offset = 0.005 if gap >= 0 else -0.005
            ha = 'left' if gap >= 0 else 'right'
            ax.text(gap + offset, i, f'{gap:+.3f}', ha=ha, va='center', fontsize=10, fontweight='bold')

    legend_elements = [
        Patch(facecolor=COLORS['positive'], edgecolor='black', label='TSTR better'),
        Patch(facecolor=COLORS['negative'], edgecolor='black', label='TRTR better'),
    ]
    fig.legend(handles=legend_elements, loc='upper center', ncol=2,
               bbox_to_anchor=(0.5, 0.02), frameon=True, fancybox=True)

    if dataset_label:
        fig.suptitle(dataset_label, fontsize=13, fontweight='bold', y=1.02)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)
    scale_all_fonts(fig)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  [OK] {os.path.basename(output_path)}")


def fig8_binary_confusion(results_dir, output_path, best_model='LightGBM', dataset_label=''):
    """Figure 8: Binary confusion matrices."""
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))

    cm_tstr = cargar_matriz_confusion(
        os.path.join(results_dir, f'TSTR_BINARY_{best_model}_confusion_matrix.csv'))
    cm_trtr = cargar_matriz_confusion(
        os.path.join(results_dir, f'TRTR_BINARY_{best_model}_confusion_matrix.csv'))

    if cm_tstr is None or cm_trtr is None:
        print(f"  [SKIP] Binary confusion matrices not found for {best_model}")
        plt.close()
        return

    labels = list(cm_tstr.columns)
    cm_tstr = cm_tstr.values
    cm_trtr = cm_trtr.values

    cm_tstr_norm = cm_tstr.astype('float') / cm_tstr.sum(axis=1)[:, np.newaxis] * 100
    cm_trtr_norm = cm_trtr.astype('float') / cm_trtr.sum(axis=1)[:, np.newaxis] * 100

    cmap = LinearSegmentedColormap.from_list('custom', ['#ffffff', '#1f77b4', '#0d3d6e'])
    n = len(labels)

    for ax, cm_vals, cm_norm, title in [
        (axes[0], cm_tstr, cm_tstr_norm, f'(a) TSTR ({best_model})'),
        (axes[1], cm_trtr, cm_trtr_norm, f'(b) TRTR ({best_model})'),
    ]:
        im = ax.imshow(cm_norm, cmap=cmap, vmin=0, vmax=100, aspect='auto')
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(labels, fontsize=11)
        ax.set_yticklabels(labels, fontsize=11)
        ax.set_xlabel('Predicted', fontsize=11, fontweight='bold')
        ax.set_ylabel('True', fontsize=11, fontweight='bold')
        ax.set_title(title, fontweight='bold', fontsize=12)
        for i in range(n):
            for j in range(n):
                val = cm_vals[i, j]
                pct = cm_vals[i, j] / cm_vals.sum(axis=1)[i] * 100
                color = 'white' if pct > 50 else 'black'
                ax.text(j, i, f'{int(val):,}\n({pct:.1f}%)', ha='center', va='center',
                        fontsize=11, color=color, fontweight='bold')

    if dataset_label:
        fig.suptitle(dataset_label, fontsize=13, fontweight='bold', y=1.05)

    plt.tight_layout()
    scale_all_fonts(fig)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  [OK] {os.path.basename(output_path)}")


def generate_latex_table(results_dir, output_path, dataset_label=''):
    """Generate LaTeX table with TSTR vs TRTR comparison."""
    df = cargar_results_globales(results_dir)
    if df is None:
        print("  [SKIP] Global results not found for LaTeX table")
        return

    caption_ds = f' ({dataset_label})' if dataset_label else ''
    latex = rf"""\begin{{table}}[htbp]
\centering
\caption{{Performance comparison of TSTR vs TRTR across different models{caption_ds}.}}
\label{{tab:results}}
\small
\begin{{tabular}}{{llcccccc}}
\toprule
\textbf{{Task}} & \textbf{{Model}} & \multicolumn{{2}}{{c}}{{\textbf{{TSTR}}}} & \multicolumn{{2}}{{c}}{{\textbf{{TRTR}}}} & \multicolumn{{2}}{{c}}{{\textbf{{Gap}}}} \\
\cmidrule(lr){{3-4}} \cmidrule(lr){{5-6}} \cmidrule(lr){{7-8}}
& & Acc. & $F_1$-M & Acc. & $F_1$-M & $\Delta$Acc & $\Delta F_1$ \\
\midrule
"""

    models = [m for m in MODELOS_FULL if m in df['Model'].unique()]

    for task_label, exp_tstr, exp_trtr in [
        ('Multiclass', 'TSTR_Multi', 'TRTR_Multi'),
        ('Binary', 'TSTR_Binary', 'TRTR_Binary'),
    ]:
        for i, m in enumerate(models):
            tstr = df[(df['Experimento'] == exp_tstr) & (df['Model'] == m)]
            trtr = df[(df['Experimento'] == exp_trtr) & (df['Model'] == m)]
            if len(tstr) == 0 or len(trtr) == 0:
                continue
            tstr = tstr.iloc[0]
            trtr = trtr.iloc[0]
            abbr = MODELOS_SHORT[MODELOS_FULL.index(m)]
            task = task_label if i == 0 else ''
            delta_acc = tstr['Accuracy'] - trtr['Accuracy']
            delta_f1 = tstr['F1_Macro'] - trtr['F1_Macro']
            latex += (f"{task} & {abbr} & {tstr['Accuracy']:.3f} & {tstr['F1_Macro']:.3f} "
                      f"& {trtr['Accuracy']:.3f} & {trtr['F1_Macro']:.3f} "
                      f"& {delta_acc:+.3f} & {delta_f1:+.3f} \\\\\n")
        if task_label == 'Multiclass':
            latex += r"\midrule" + "\n"

    latex += r"""\bottomrule
\end{tabular}
\end{table}
"""
    with open(output_path, 'w') as f:
        f.write(latex)
    print(f"  [OK] {os.path.basename(output_path)}")


# ============================================================================
# BASIC (EXPLORATORY) FIGURES
# ============================================================================

def basic_confusion_matrices_comparison(results_dir, output_path, model='RandomForest',
                                         exp_type='MULTI', dataset_label=''):
    """Side-by-side confusion matrices (basic style)."""
    cm_tstr = cargar_matriz_confusion(
        os.path.join(results_dir, f'TSTR_{exp_type}_{model}_confusion_matrix.csv'))
    cm_trtr = cargar_matriz_confusion(
        os.path.join(results_dir, f'TRTR_{exp_type}_{model}_confusion_matrix.csv'))

    if cm_tstr is None or cm_trtr is None:
        print(f"  [SKIP] {exp_type} confusion matrices not found for {model}")
        return

    classs = list(cm_tstr.columns)
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    for ax, cm_df, cmap_name, label in [
        (axes[0], cm_tstr, 'Greens', 'TSTR\n(Train Synthetic, Test Real)'),
        (axes[1], cm_trtr, 'Blues', 'TRTR\n(Train Real, Test Real)'),
    ]:
        sns.heatmap(cm_df, annot=True, fmt='d', cmap=cmap_name, ax=ax,
                    xticklabels=classs, yticklabels=classs, annot_kws={'size': 9})
        tipo = exp_type.replace('MULTI', 'Multiclass').replace('BINARY', 'Binary')
        ax.set_title(f'{dataset_label} {tipo} - {label} ({model})', fontsize=12, fontweight='bold')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [OK] {os.path.basename(output_path)}")


def basic_comparative_models(results_dir, output_path, clasificacion='Multi', dataset_label=''):
    """Bar chart comparing TSTR vs TRTR across models (basic style)."""
    df = cargar_results_globales(results_dir)
    if df is None:
        print("  [SKIP] Global results not found")
        return

    df_filt = df[df['Experimento'].str.contains(clasificacion)].copy()
    df_filt['Tipo'] = df_filt['Experimento'].apply(lambda x: 'TSTR' if 'TSTR' in x else 'TRTR')

    models = df_filt['Model'].unique()
    x = np.arange(len(models))
    width = 0.35

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    metricas = ['Accuracy', 'F1_Macro', 'F1_Weighted']
    titulos = ['Accuracy', 'F1-Score Macro', 'F1-Score Weighted']

    for ax, metrica, titulo in zip(axes, metricas, titulos):
        tstr_vals = df_filt[df_filt['Tipo'] == 'TSTR'].set_index('Model')[metrica]
        trtr_vals = df_filt[df_filt['Tipo'] == 'TRTR'].set_index('Model')[metrica]

        bars1 = ax.bar(x - width/2, [tstr_vals.get(m, 0) for m in models],
                       width, label='TSTR', color=COLORES_BASIC['TSTR'], alpha=0.8)
        bars2 = ax.bar(x + width/2, [trtr_vals.get(m, 0) for m in models],
                       width, label='TRTR', color=COLORES_BASIC['TRTR'], alpha=0.8)
        ax.set_xlabel('Model')
        ax.set_ylabel(titulo)
        ax.set_title(titulo, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.legend()
        ax.set_ylim(0.4, 1.05)

        for bar in list(bars1) + list(bars2):
            h = bar.get_height()
            ax.annotate(f'{h:.3f}', xy=(bar.get_x() + bar.get_width()/2, h),
                        xytext=(0, 3), textcoords='offset points',
                        ha='center', va='bottom', fontsize=8, rotation=90)

    tipo_clf = 'Multiclass' if 'Multi' in clasificacion else 'Binary'
    fig.suptitle(f'{dataset_label} - TSTR vs TRTR ({tipo_clf})',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [OK] {os.path.basename(output_path)}")


def basic_gap_analysis(results_dir, output_path, dataset_label=''):
    """Horizontal bar gap chart (basic style)."""
    df = cargar_results_globales(results_dir)
    if df is None:
        print("  [SKIP] Global results not found")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, clas, titulo in zip(axes, ['Multi', 'Binary'], ['Multiclass', 'Binary']):
        df_c = df[df['Experimento'].str.contains(clas)].copy()
        tstr = df_c[df_c['Experimento'].str.contains('TSTR')].set_index('Model')['F1_Macro']
        trtr = df_c[df_c['Experimento'].str.contains('TRTR')].set_index('Model')['F1_Macro']
        models = tstr.index.tolist()
        gaps = [tstr[m] - trtr[m] for m in models]

        colors = [COLORES_BASIC['gap_positive'] if g >= 0 else COLORES_BASIC['gap_negative'] for g in gaps]
        bars = ax.barh(models, gaps, color=colors, alpha=0.8)
        ax.axvline(x=0, color='black', linewidth=1)
        ax.set_xlabel('Gap F1-Macro (TSTR - TRTR)')
        ax.set_title(titulo, fontweight='bold')

        for bar, gap in zip(bars, gaps):
            w = bar.get_width()
            ax.annotate(f'{gap:+.4f}',
                        xy=(w, bar.get_y() + bar.get_height()/2),
                        xytext=(5 if w >= 0 else -5, 0),
                        textcoords='offset points',
                        ha='left' if w >= 0 else 'right', va='center', fontsize=10)

    fig.suptitle(f'{dataset_label} - Gap Analysis (TSTR - TRTR)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [OK] {os.path.basename(output_path)}")


def basic_f1_por_class(results_dir, output_path, model='RandomForest', dataset_label=''):
    """F1 per-class bar chart (basic style)."""
    tstr_report = cargar_classification_report(
        os.path.join(results_dir, f'TSTR_MULTI_{model}_classification_report.csv'))
    trtr_report = cargar_classification_report(
        os.path.join(results_dir, f'TRTR_MULTI_{model}_classification_report.csv'))

    if tstr_report is None or trtr_report is None:
        print(f"  [SKIP] Reports not found for {model}")
        return

    classs = [c for c in tstr_report.index if c not in ['accuracy', 'macro avg', 'weighted avg']]
    tstr_f1 = [tstr_report.loc[c, 'f1-score'] if c in tstr_report.index else 0 for c in classs]
    trtr_f1 = [trtr_report.loc[c, 'f1-score'] if c in trtr_report.index else 0 for c in classs]

    x = np.arange(len(classs))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width/2, tstr_f1, width, label='TSTR', color=COLORES_BASIC['TSTR'], alpha=0.8)
    ax.bar(x + width/2, trtr_f1, width, label='TRTR', color=COLORES_BASIC['TRTR'], alpha=0.8)
    ax.axhline(y=0.85, color='red', linestyle='--', linewidth=1, label='Threshold 0.85')
    ax.set_xlabel('Class')
    ax.set_ylabel('F1-Score')
    ax.set_title(f'{dataset_label} - F1 per Class ({model})', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(classs, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, 1.1)

    for bars_data, sign in [(tstr_f1, -1), (trtr_f1, 1)]:
        offset = sign * width/2
        for i, val in enumerate(bars_data):
            ax.annotate(f'{val:.2f}', xy=(x[i] + offset, val),
                        xytext=(0, 3), textcoords='offset points',
                        ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [OK] {os.path.basename(output_path)}")


def basic_precision_recall_por_class(results_dir, output_path, model='RandomForest', dataset_label=''):
    """Precision vs Recall per class (basic style)."""
    report = cargar_classification_report(
        os.path.join(results_dir, f'TSTR_MULTI_{model}_classification_report.csv'))
    if report is None:
        print(f"  [SKIP] Report not found for {model}")
        return

    classs = [c for c in report.index if c not in ['accuracy', 'macro avg', 'weighted avg']]
    precision = [report.loc[c, 'precision'] if c in report.index else 0 for c in classs]
    recall = [report.loc[c, 'recall'] if c in report.index else 0 for c in classs]

    x = np.arange(len(classs))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width/2, precision, width, label='Precision', color='#9b59b6', alpha=0.8)
    ax.bar(x + width/2, recall, width, label='Recall', color='#f39c12', alpha=0.8)
    ax.set_xlabel('Class')
    ax.set_ylabel('Score')
    ax.set_title(f'{dataset_label} - Precision vs Recall (TSTR, {model})', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(classs, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, 1.1)

    for vals, sign in [(precision, -1), (recall, 1)]:
        offset = sign * width/2
        for i, val in enumerate(vals):
            ax.annotate(f'{val:.2f}', xy=(x[i] + offset, val),
                        xytext=(0, 3), textcoords='offset points',
                        ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [OK] {os.path.basename(output_path)}")


def basic_heatmap_all_results(results_dir, output_path, dataset_label=''):
    """Heatmap of all model results (basic style)."""
    df = cargar_results_globales(results_dir)
    if df is None:
        print("  [SKIP] Global results not found")
        return

    try:
        pivot_acc = df.pivot(index='Model', columns='Experimento', values='Accuracy')
        pivot_f1 = df.pivot(index='Model', columns='Experimento', values='F1_Macro')
    except Exception:
        print("  [SKIP] Cannot pivot results")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    sns.heatmap(pivot_acc, annot=True, fmt='.4f', cmap='YlGnBu', ax=axes[0],
                vmin=0.5, vmax=1.0, cbar_kws={'label': 'Accuracy'})
    axes[0].set_title(f'{dataset_label} - Accuracy', fontweight='bold')
    axes[0].set_xlabel('')
    axes[0].tick_params(axis='x', rotation=45)

    sns.heatmap(pivot_f1, annot=True, fmt='.4f', cmap='YlOrRd', ax=axes[1],
                vmin=0.0, vmax=1.0, cbar_kws={'label': 'F1-Macro'})
    axes[1].set_title(f'{dataset_label} - F1-Macro', fontweight='bold')
    axes[1].set_xlabel('')
    axes[1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [OK] {os.path.basename(output_path)}")


def basic_dashboard_summary(results_dir, output_path, model='RandomForest', dataset_label=''):
    """Summary dashboard (basic style)."""
    df_results = cargar_results_globales(results_dir)
    if df_results is None:
        print("  [SKIP] Global results not found")
        return

    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)

    # 1. Best F1-Macro per experiment
    ax1 = fig.add_subplot(gs[0, :2])
    experimentos = ['TSTR_Multi', 'TRTR_Multi', 'TSTR_Binary', 'TRTR_Binary']
    colores_exp = [COLORES_BASIC['TSTR'], COLORES_BASIC['TRTR'],
                   COLORES_BASIC['TSTR'], COLORES_BASIC['TRTR']]
    hatches = ['', '', '///', '///']

    for i, exp in enumerate(experimentos):
        df_exp = df_results[df_results['Experimento'] == exp]
        if len(df_exp) == 0:
            continue
        mejor = df_exp.loc[df_exp['F1_Macro'].idxmax()]
        ax1.bar(i, mejor['F1_Macro'], color=colores_exp[i], alpha=0.8,
                hatch=hatches[i], edgecolor='black', linewidth=1)
        ax1.annotate(f"{mejor['F1_Macro']:.4f}\n({mejor['Model']})",
                     xy=(i, mejor['F1_Macro']), xytext=(0, 5),
                     textcoords='offset points', ha='center', va='bottom', fontsize=9)

    ax1.set_xticks(range(4))
    ax1.set_xticklabels(['TSTR\nMulti', 'TRTR\nMulti', 'TSTR\nBinary', 'TRTR\nBinary'])
    ax1.set_ylabel('F1-Macro (Best Model)')
    ax1.set_title('Best F1-Macro per Experiment', fontweight='bold')
    ax1.set_ylim(0, 1.1)
    ax1.axhline(y=0.85, color='red', linestyle='--', alpha=0.5)

    # 2. Gap bar
    ax2 = fig.add_subplot(gs[0, 2])
    gaps = {}
    for label_g, exp_t, exp_r in [('Multiclass', 'TSTR_Multi', 'TRTR_Multi'),
                                    ('Binary', 'TSTR_Binary', 'TRTR_Binary')]:
        t_max = df_results[df_results['Experimento'] == exp_t]['F1_Macro'].max()
        r_max = df_results[df_results['Experimento'] == exp_r]['F1_Macro'].max()
        gaps[label_g] = t_max - r_max if pd.notna(t_max) and pd.notna(r_max) else 0

    gap_colors = [COLORES_BASIC['gap_negative'] if v < 0 else COLORES_BASIC['gap_positive']
                  for v in gaps.values()]
    bars = ax2.barh(list(gaps.keys()), list(gaps.values()), color=gap_colors, alpha=0.8)
    ax2.axvline(x=0, color='black', linewidth=1)
    ax2.set_xlabel('Gap (TSTR - TRTR)')
    ax2.set_title('Gap F1-Macro', fontweight='bold')
    for bar, (k, v) in zip(bars, gaps.items()):
        ax2.annotate(f'{v:+.4f}', xy=(v, bar.get_y() + bar.get_height()/2),
                     xytext=(5 if v >= 0 else -35, 0), textcoords='offset points',
                     ha='left', va='center', fontsize=11, fontweight='bold')

    # 3. F1 per class
    ax3 = fig.add_subplot(gs[1, :])
    tstr_report = cargar_classification_report(
        os.path.join(results_dir, f'TSTR_MULTI_{model}_classification_report.csv'))
    trtr_report = cargar_classification_report(
        os.path.join(results_dir, f'TRTR_MULTI_{model}_classification_report.csv'))

    if tstr_report is not None and trtr_report is not None:
        classs = [c for c in tstr_report.index if c not in ['accuracy', 'macro avg', 'weighted avg']]
        tstr_f1 = [tstr_report.loc[c, 'f1-score'] if c in tstr_report.index else 0 for c in classs]
        trtr_f1 = [trtr_report.loc[c, 'f1-score'] if c in trtr_report.index else 0 for c in classs]
        x = np.arange(len(classs))
        w = 0.35
        ax3.bar(x - w/2, tstr_f1, w, label='TSTR', color=COLORES_BASIC['TSTR'], alpha=0.8)
        ax3.bar(x + w/2, trtr_f1, w, label='TRTR', color=COLORES_BASIC['TRTR'], alpha=0.8)
        ax3.axhline(y=0.85, color='red', linestyle='--', alpha=0.7, label='Threshold 0.85')
        ax3.set_xticks(x)
        ax3.set_xticklabels(classs, rotation=45, ha='right')
        ax3.set_ylabel('F1-Score')
        ax3.set_title(f'F1 per Class - Multiclass ({model})', fontweight='bold')
        ax3.legend(loc='lower right')
        ax3.set_ylim(0, 1.1)

    # 4. Summary table
    ax4 = fig.add_subplot(gs[2, :])
    ax4.axis('off')
    table_data = []
    for exp in experimentos:
        df_exp = df_results[df_results['Experimento'] == exp]
        if len(df_exp) == 0:
            continue
        mejor = df_exp.loc[df_exp['F1_Macro'].idxmax()]
        table_data.append([
            exp.replace('_', ' '), mejor['Model'],
            f"{mejor['Accuracy']:.4f}", f"{mejor['F1_Macro']:.4f}", f"{mejor['F1_Weighted']:.4f}",
        ])

    if table_data:
        table = ax4.table(cellText=table_data,
                          colLabels=['Experiment', 'Best Model', 'Accuracy', 'F1-Macro', 'F1-Weighted'],
                          loc='center', cellLoc='center', colColours=['#ecf0f1']*5)
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1.2, 1.8)
        for i in range(len(table_data)):
            for j in range(5):
                cell = table[(i+1, j)]
                if 'TSTR' in table_data[i][0]:
                    cell.set_facecolor('#d5f5e3')
                else:
                    cell.set_facecolor('#d6eaf8')

    fig.suptitle(f'{dataset_label} - TSTR vs TRTR Dashboard', fontsize=16, fontweight='bold', y=0.98)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [OK] {os.path.basename(output_path)}")


# ============================================================================
# PIPELINE: run all figures for one dataset
# ============================================================================

def run_paper_figures(dataset_key, cfg):
    """Generate all paper-quality figures for one dataset."""
    results_dir = cfg['results_dir']
    label = cfg['label']
    output_dir = os.path.join(results_dir, 'graficas_paper')
    os.makedirs(output_dir, exist_ok=True)

    best_model = get_best_model(results_dir, cfg['default_best_model'])
    print(f"\n  Best model (auto-detected): {best_model}")

    for ext in ['pdf', 'png']:
        fig1_confusion_matrices(results_dir, os.path.join(output_dir, f'fig1_confusion_matrices.{ext}'),
                                best_model=best_model, dataset_label=label)
        fig2_model_comparison(results_dir, os.path.join(output_dir, f'fig2_model_comparison.{ext}'),
                              dataset_label=label)
        fig3_per_class_analysis(results_dir, os.path.join(output_dir, f'fig3_per_class_analysis.{ext}'),
                                best_model=best_model, dataset_label=label)
        fig4_precision_recall_heatmaps(results_dir, os.path.join(output_dir, f'fig4_precision_recall.{ext}'),
                                        best_model=best_model, dataset_label=label)
        fig6_radar_chart(results_dir, os.path.join(output_dir, f'fig6_radar_chart.{ext}'),
                         dataset_label=label)
        fig7_gap_analysis(results_dir, os.path.join(output_dir, f'fig7_gap_analysis.{ext}'),
                          dataset_label=label)
        fig8_binary_confusion(results_dir, os.path.join(output_dir, f'fig8_binary_confusion.{ext}'),
                              best_model=best_model, dataset_label=label)

    generate_latex_table(results_dir, os.path.join(output_dir, 'table_results.tex'),
                         dataset_label=label)


def run_basic_figures(dataset_key, cfg):
    """Generate all basic/exploratory figures for one dataset."""
    results_dir = cfg['results_dir']
    label = cfg['label']
    output_dir = os.path.join(results_dir, 'graficas')
    os.makedirs(output_dir, exist_ok=True)

    basic_model = cfg.get('default_basic_model', 'RandomForest')
    if not os.path.exists(os.path.join(results_dir, f'TSTR_MULTI_{basic_model}_confusion_matrix.csv')):
        basic_model = get_best_model(results_dir, cfg['default_best_model'])

    print(f"\n  Basic model: {basic_model}")

    basic_confusion_matrices_comparison(results_dir,
        os.path.join(output_dir, '01_confusion_comparison_multi.png'),
        model=basic_model, exp_type='MULTI', dataset_label=label)
    basic_confusion_matrices_comparison(results_dir,
        os.path.join(output_dir, '02_confusion_comparison_binary.png'),
        model=basic_model, exp_type='BINARY', dataset_label=label)
    basic_comparative_models(results_dir,
        os.path.join(output_dir, '03_comparative_models_multi.png'),
        clasificacion='Multi', dataset_label=label)
    basic_comparative_models(results_dir,
        os.path.join(output_dir, '04_comparative_models_binary.png'),
        clasificacion='Binary', dataset_label=label)
    basic_gap_analysis(results_dir,
        os.path.join(output_dir, '05_gap_analysis.png'), dataset_label=label)
    basic_f1_por_class(results_dir,
        os.path.join(output_dir, '06_f1_por_class.png'),
        model=basic_model, dataset_label=label)
    basic_precision_recall_por_class(results_dir,
        os.path.join(output_dir, '07_precision_recall_por_class.png'),
        model=basic_model, dataset_label=label)
    basic_heatmap_all_results(results_dir,
        os.path.join(output_dir, '08_heatmap_all_results.png'), dataset_label=label)
    basic_dashboard_summary(results_dir,
        os.path.join(output_dir, '09_dashboard_summary.png'),
        model=basic_model, dataset_label=label)


# ============================================================================
# ARGUMENT PARSING & MAIN
# ============================================================================

def resolve_datasets(dataset_arg):
    """Resolve --dataset argument to list of dataset keys."""
    if dataset_arg is None or dataset_arg.lower() == 'all':
        return list(DATASET_REGISTRY.keys())

    if dataset_arg in DATASET_REGISTRY:
        return [dataset_arg]

    alias_lower = dataset_arg.lower()
    if alias_lower in DATASET_ALIASES:
        return [DATASET_ALIASES[alias_lower]]

    matches = [k for k, v in DATASET_REGISTRY.items()
               if alias_lower in v['family'].lower() or alias_lower in k.lower()]
    if matches:
        return matches

    print(f"[ERROR] Unknown dataset: '{dataset_arg}'")
    print("Available datasets:")
    for k, v in DATASET_REGISTRY.items():
        print(f"  {k:<30} {v['label']}")
    print("\nAliases:", ', '.join(DATASET_ALIASES.keys()))
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description='TSTR vs TRTR Visualization - Unified multi-dataset script')
    parser.add_argument('--dataset', '-d', type=str, default='all',
                        help='Dataset to process (name, alias, family, or "all")')
    parser.add_argument('--mode', '-m', type=str, default='all',
                        choices=['basic', 'paper', 'all'],
                        help='Visualization mode: basic, paper, or all')
    parser.add_argument('--list', '-l', action='store_true',
                        help='List available datasets and exit')
    args = parser.parse_args()

    if args.list:
        print("\nAvailable datasets:")
        print(f"  {'Key':<30} {'Family':<15} {'Label'}")
        print("  " + "-"*70)
        for k, v in DATASET_REGISTRY.items():
            exists = 'OK' if os.path.isdir(v['results_dir']) else 'MISSING'
            print(f"  {k:<30} {v['family']:<15} {v['label']}  [{exists}]")
        print(f"\nAliases: {', '.join(DATASET_ALIASES.keys())}")
        return

    datasets = resolve_datasets(args.dataset)

    print("=" * 70)
    print("  TSTR vs TRTR VISUALIZATION - Unified Pipeline")
    print("=" * 70)
    print(f"  Mode     : {args.mode}")
    print(f"  Datasets : {len(datasets)}")
    for ds in datasets:
        print(f"    - {ds} ({DATASET_REGISTRY[ds]['label']})")
    print("=" * 70)

    for ds_key in datasets:
        cfg = DATASET_REGISTRY[ds_key]
        results_dir = cfg['results_dir']

        print(f"\n{'='*70}")
        print(f"  Processing: {cfg['label']}  ({ds_key})")
        print(f"  Results dir: {results_dir}")

        if not os.path.isdir(results_dir):
            print(f"  [SKIP] Results directory not found: {results_dir}")
            continue

        if args.mode in ('basic', 'all'):
            print(f"\n  --- Basic Figures ---")
            setup_basic_style()
            run_basic_figures(ds_key, cfg)

        if args.mode in ('paper', 'all'):
            print(f"\n  --- Paper-Quality Figures ---")
            setup_academic_style()
            run_paper_figures(ds_key, cfg)

        print(f"\n  Done: {ds_key}")

        for subdir in ['graficas', 'graficas_paper']:
            out_dir = os.path.join(results_dir, subdir)
            if os.path.isdir(out_dir):
                files = sorted(os.listdir(out_dir))
                if files:
                    print(f"\n  Output ({subdir}):")
                    for f in files:
                        size = os.path.getsize(os.path.join(out_dir, f)) / 1024
                        print(f"    {f:<45} ({size:.1f} KB)")

    print(f"\n{'='*70}")
    print("  ALL DONE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()