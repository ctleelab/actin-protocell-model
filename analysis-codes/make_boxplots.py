#!/usr/bin/env python3
"""
Generate boxplots of membrane extension for different conditions.
"""
import csv

import ctleelab_plothelper.plothelpers as ph
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import textwrap

import random
import pdb

def label_axes(ax, text):
    """Place a label at the center of an Axes, and remove the axis ticks."""
    ax.text(
        0.5,
        0.5,
        text,
        transform=ax.transAxes,
        horizontalalignment="center",
        verticalalignment="center",
    )
    ax.tick_params(bottom=False, labelbottom=False, left=False, labelleft=False)

plot_styles = [
    ("ctleelab_plothelper.light", ""),
    ("ctleelab_plothelper.dark", "_dark"),
]

def read_csv_rows(path):
    rows = []
    with open(path, 'r') as fh:
        reader = csv.DictReader(fh)
        for r in reader:
            rows.append(r)
    return rows


# Helper to parse membrane_extension values safely
def parse_ext(row):
    try:
        return float(row.get('membrane-extension','nan'))
    except Exception:
        return float('nan')


rows = read_csv_rows('../input+outputfiles/outputfiles/membrane-deformation-sterics.csv')

pVCA = [parse_ext(row) for row in rows if row.get('condition') == 'pVCA']
mDia1 = [parse_ext(row) for row in rows if row.get('condition') == 'mDia1']
pVCA_mDia1 = [parse_ext(row) for row in rows if row.get('condition') == 'pVCA + mDia1']


# Prepare plotting data and labels
labels = ['pVCA', 'mDia1', 'pVCA + mDia1']

# Some basic validation and fallback: if a group is empty, we will add a NaN so boxplot doesn't crash
def safe_list(lst):
    if not lst:
        return [float('nan')]
    return lst


plot_main = [safe_list(pVCA), safe_list(mDia1), safe_list(pVCA_mDia1)]


n = len(labels)
width = 0.6
offsets = [(-width / 2) + (i + 0.5) * (width / n) for i in range(n)]
wrapped = [textwrap.fill(lbl, width=5) for lbl in labels]

for i, (style, style_ext) in enumerate(plot_styles):
    with plt.style.context(["ctleelab_plothelper.base", style, "ctleelab_plothelper.transparent"]):

        fig, axs = ph.fixed_size_subplots(1, 1, subwidth=1.5, subheight=1.5)

        ax = axs 

        # Colors for boxes
        colors = ['blue', 'green', 'coral']

        box = ax.boxplot([np.array(data) for data in plot_main], 
                    patch_artist=True,
                    tick_labels=wrapped, #[l for l in labels],
                    widths=(width / n) * 1.1,
                    showfliers=False,
                    showmeans=True,
                    meanline=True,
                    meanprops={'color': 'black', 'linestyle': '-', 'linewidth': 0.9},
                    boxprops={'linewidth': 0.9},
                    whiskerprops={'linewidth': 0.9, 'color': 'black'},
                    capprops={'linewidth': 0.9, 'color': 'black'},
                    medianprops={'linewidth': 0, 'color': 'none'},
            )
        for patch, color in zip(box['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(1.0)
            patch.set_edgecolor('black')
            patch.set_linewidth(1.2)
        # Add jittered points
        for i, data in enumerate([np.array(d) for d in plot_main]):
            x = np.random.normal(i+1, 0.06, size=len(data))
            ax.scatter(x, data, color='black', alpha=0.45, s=30, edgecolors='none')
       
        ax.set_ylabel(r'Membrane extension ($\mathregular{\mu m}$)')
        ax.set_ylim(0, 0.035)


        fig.savefig(f"../input+outputfiles/sterics{style_ext}.svg", format="svg")
        # fig.savefig(f"../input+outputfiles/sterics{style_ext}.pdf", format="pdf")