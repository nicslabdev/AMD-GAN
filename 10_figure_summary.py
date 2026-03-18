#!/usr/bin/env python3
"""
Genera 1 figura summary combinando CICIDS y UNSW-NB15.
Layout: 3 columnas (classs) × 4 rows (training + KDE per dataset).
Las KDE se recortan a 4 columnas × 4 rows de subplots.
"""

import os
import math
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE, "results_figures_summary")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── Configuration de datasets y classs representativas ─────────────────────
DATASETS = [
    {
        "name": "CIC-IDS2017",
        "dir": os.path.join(BASE, "outputs_wgan_multi_v2"),
        "classes": ["benign", "brute_force", "web_attack"],
        "labels": ["Benign", "Brute Force", "Web Attack"],
        "kde_cols": 4,
    },
    {
        "name": "UNSW-NB15",
        "dir": os.path.join(BASE, "outputs_wgan_unsw"),
        "classes": ["dos", "fuzzers", "reconnaissance"],
        "labels": ["DoS", "Fuzzers", "Reconnaissance"],
        "kde_cols": 4,
    },
]

KDE_KEEP_COLS = 4
KDE_KEEP_ROWS = 4


def crop_kde_grid(img_path, original_cols, keep_cols=4, keep_rows=4):
    """Recorta la imagen KDE a keep_rows × keep_cols subplots."""
    img = Image.open(img_path)
    w, h = img.size
    col_w = w / original_cols
    row_h = 450  # 3 inches * 150 dpi
    crop_w = int(col_w * keep_cols)
    crop_h = min(int(row_h * keep_rows), h)
    return np.array(img.crop((0, 0, crop_w, crop_h)))


def load_img(img_path):
    return np.array(Image.open(img_path))


def make_combined_figure():
    """
    Figura combinada: 3 columnas × 4 rows.
    Para cada dataset: fila de training curves + fila de KDE.
    """
    n_cols = 3  # classs por dataset

    # Cargar todas las imágenes
    all_data = []
    for ds in DATASETS:
        tc_imgs, kde_imgs = [], []
        for cls_name in ds["classes"]:
            cls_dir = os.path.join(ds["dir"], cls_name)
            tc_imgs.append(load_img(os.path.join(cls_dir, f"training_curves_{cls_name}.png")))
            kde_imgs.append(crop_kde_grid(
                os.path.join(cls_dir, f"kde_comparison_{cls_name}.png"),
                ds["kde_cols"], KDE_KEEP_COLS, KDE_KEEP_ROWS,
            ))
        all_data.append({"name": ds["name"], "labels": ds["labels"],
                         "tc": tc_imgs, "kde": kde_imgs})

    # Aspect ratios para calcular alturas relativas
    tc_h, tc_w = all_data[0]["tc"][0].shape[:2]
    kde_h, kde_w = all_data[0]["kde"][0].shape[:2]
    tc_ratio = tc_h / tc_w
    kde_ratio = kde_h / kde_w

    fig_width = 18
    col_w = fig_width / n_cols
    tc_fig_h = col_w * tc_ratio
    kde_fig_h = col_w * kde_ratio

    # 4 rows: tc1, kde1, tc2, kde2 + espaciado para títulos de dataset
    height_ratios = [tc_fig_h, kde_fig_h, tc_fig_h, kde_fig_h]
    fig_height = sum(height_ratios) + 2.5

    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = gridspec.GridSpec(
        4, n_cols, figure=fig,
        height_ratios=height_ratios,
        hspace=0.12, wspace=0.05,
        left=0.06, right=0.98, top=0.95, bottom=0.01,
    )

    row_labels = ["Training Curves", "KDE Comparison"]

    for ds_idx, ds_data in enumerate(all_data):
        row_tc = ds_idx * 2      # fila de training curves
        row_kde = ds_idx * 2 + 1  # fila de KDE

        for col_idx in range(n_cols):
            # Training curves
            ax_tc = fig.add_subplot(gs[row_tc, col_idx])
            ax_tc.imshow(ds_data["tc"][col_idx])
            ax_tc.axis("off")
            # Título de class solo en la primera fila del dataset
            ax_tc.set_title(ds_data["labels"][col_idx], fontsize=13, pad=6)

            # KDE
            ax_kde = fig.add_subplot(gs[row_kde, col_idx])
            ax_kde.imshow(ds_data["kde"][col_idx])
            ax_kde.axis("off")

        # Etiqueta del dataset a la izquierda (centrada entre sus 2 rows)
        # Posición Y media entre las 2 rows del dataset
        ax_mid = fig.add_subplot(gs[row_tc:row_kde + 1, 0])
        ax_mid.axis("off")
        ax_mid.text(
            -0.08, 0.5, ds_data["name"],
            transform=ax_mid.transAxes,
            fontsize=14, va="center", ha="right", rotation=90,
        )

    out_path = os.path.join(OUTPUT_DIR, "summary_combined_CICIDS_UNSW.pdf")
    fig.savefig(out_path, dpi=500, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"✓ Guardada: {out_path}")


# ─── Generar la figura combinada ────────────────────────────────────────────
if __name__ == "__main__":
    # Verificar archivos
    all_ok = True
    for ds in DATASETS:
        for cls_name in ds["classes"]:
            cls_dir = os.path.join(ds["dir"], cls_name)
            for fname in [f"training_curves_{cls_name}.png", f"kde_comparison_{cls_name}.png"]:
                fpath = os.path.join(cls_dir, fname)
                if not os.path.exists(fpath):
                    print(f"⚠  No encontrado: {fpath}")
                    all_ok = False

    if all_ok:
        make_combined_figure()
    else:
        print("✗ Faltan archivos, no se pudo generar la figura")

    print(f"\nResults en: {OUTPUT_DIR}")
