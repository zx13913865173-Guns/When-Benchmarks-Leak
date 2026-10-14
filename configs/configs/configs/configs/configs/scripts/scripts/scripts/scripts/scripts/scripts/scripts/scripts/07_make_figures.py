"""
Step 7: Generate publication-quality figures.
16:9, 600 DPI, blue scientific palette.
"""

import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap

matplotlib.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 11, 'axes.labelsize': 12, 'axes.titlesize': 13,
    'xtick.labelsize': 10, 'ytick.labelsize': 10, 'legend.fontsize': 10,
    'axes.linewidth': 1.0, 'axes.edgecolor': '#1a3a6b',
    'axes.labelcolor': '#1a3a6b', 'xtick.color': '#1a3a6b',
    'ytick.color': '#1a3a6b', 'text.color': '#0d2340',
    'savefig.dpi': 600, 'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.15, 'pdf.fonttype': 42, 'ps.fonttype': 42,
})

DEEP_BLUE, MID_BLUE, BRIGHT_BLUE = '#0d3b66', '#1f6fb2', '#3fa7d6'
LIGHT_BLUE, PALE_BLUE = '#a8d5e5', '#d6ecf5'
NEUTRAL_GRAY = '#5a6b7b'
OUT_DIR = 'outputs/figures'
os.makedirs(OUT_DIR, exist_ok=True)
FIG_W, FIG_H = 12.8, 7.2


def make_fig1():
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis('off')
    ax.text(50, 96, 'CLIP Interrogator Pipeline and Training-Data Skew',
            ha='center', va='top', fontsize=15, fontweight='bold', color=DEEP_BLUE)

    ax.add_patch(FancyBboxPatch((3, 42), 14, 22,
        boxstyle="round,pad=0.6,rounding_size=1.5", linewidth=1.8,
        edgecolor=DEEP_BLUE, facecolor=PALE_BLUE, zorder=3))
    ax.text(10, 60, 'Input\nImage', ha='center', va='center',
            fontsize=12, fontweight='bold', color=DEEP_BLUE, zorder=4)
    ax.text(10, 50, '(Hybrid\nArtwork)', ha='center', va='center',
            fontsize=9, color=NEUTRAL_GRAY, style='italic', zorder=4)

    ax.add_patch(FancyBboxPatch((28, 62), 22, 18,
        boxstyle="round,pad=0.6,rounding_size=1.5", linewidth=1.8,
        edgecolor=MID_BLUE, facecolor=LIGHT_BLUE, alpha=0.85, zorder=3))
    ax.text(39, 74, 'BLIP Branch', ha='center', va='center',
            fontsize=12, fontweight='bold', color=DEEP_BLUE, zorder=4)
    ax.text(39, 68, 'ViT-B/16 backbone\nContent captioning',
            ha='center', va='center', fontsize=9, color=DEEP_BLUE, zorder=4)

    ax.add_patch(FancyBboxPatch((28, 30), 22, 18,
        boxstyle="round,pad=0.6,rounding_size=1.5", linewidth=1.8,
        edgecolor=MID_BLUE, facecolor=PALE_BLUE, alpha=0.95, zorder=3))
    ax.text(39, 42, 'CLIP Branch', ha='center', va='center',
            fontsize=12, fontweight='bold', color=DEEP_BLUE, zorder=4)
    ax.text(39, 36, 'ViT-L/14 (OpenAI)\nStyle scoring',
            ha='center', va='center', fontsize=9, color=DEEP_BLUE, zorder=4)

    ax.add_patch(FancyArrowPatch((17.5, 55), (27.5, 71), arrowstyle='-|>',
        mutation_scale=18, linewidth=1.6, color=DEEP_BLUE, zorder=2))
    ax.add_patch(FancyArrowPatch((17.5, 51), (27.5, 39), arrowstyle='-|>',
        mutation_scale=18, linewidth=1.6, color=DEEP_BLUE, zorder=2))

    ax.add_patch(FancyBboxPatch((58, 44), 16, 16,
        boxstyle="round,pad=0.6,rounding_size=1.5", linewidth=1.8,
        edgecolor=DEEP_BLUE, facecolor=MID_BLUE, alpha=0.85, zorder=3))
    ax.text(66, 54, 'Prompt\nMerge', ha='center', va='center',
            fontsize=12, fontweight='bold', color='white', zorder=4)
    ax.text(66, 48, 'Final\nPrompt', ha='center', va='center',
            fontsize=10, color='white', zorder=4)

    ax.add_patch(FancyArrowPatch((50.5, 71), (57.5, 56), arrowstyle='-|>',
        mutation_scale=18, linewidth=1.6, color=DEEP_BLUE, zorder=2))
    ax.add_patch(FancyArrowPatch((50.5, 39), (57.5, 48), arrowstyle='-|>',
        mutation_scale=18, linewidth=1.6, color=DEEP_BLUE, zorder=2))

    ax.add_patch(FancyBboxPatch((78, 44), 20, 16,
        boxstyle="round,pad=0.6,rounding_size=1.5", linewidth=1.8,
        edgecolor=DEEP_BLUE, facecolor='white', zorder=3))
    ax.text(88, 58, 'Training-Data\nSkew (Hypothesized)',
            ha='center', va='center', fontsize=10, fontweight='bold',
            color=DEEP_BLUE, zorder=4)
    ax.add_patch(Rectangle((80, 48), 14, 2.2, facecolor=DEEP_BLUE,
                            edgecolor='none', alpha=0.85, zorder=4))
    ax.text(80.4, 49.1, 'Western ~80%', ha='left', va='center',
            fontsize=8.5, color='white', fontweight='bold', zorder=5)
    ax.add_patch(Rectangle((80, 45.2), 3.0, 2.2, facecolor=LIGHT_BLUE,
                            edgecolor='none', alpha=0.95, zorder=4))
    ax.text(83.3, 46.3, 'East Asian ~5%', ha='left', va='center',
            fontsize=8.5, color=DEEP_BLUE, fontweight='bold', zorder=5)

    ax.text(50, 20,
        'Note: Training-data skew is inferred from external sources [2, 3]; '
        'WIT 400M is proprietary and not directly auditable.',
        ha='center', va='center', fontsize=9, style='italic',
        color=NEUTRAL_GRAY,
        bbox=dict(boxstyle='round,pad=0.5', facecolor=PALE_BLUE,
                  edgecolor=LIGHT_BLUE, linewidth=0.8, alpha=0.6))

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'Fig1_pipeline.png'), dpi=600, facecolor='white')
    plt.savefig(os.path.join(OUT_DIR, 'Fig1_pipeline.pdf'), dpi=600, facecolor='white')
    plt.close()
    print('[OK] Fig1')


def make_fig2():
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis('off')
    ax.text(50, 96, 'Contamination Audit Workflow',
            ha='center', va='top', fontsize=15, fontweight='bold', color=DEEP_BLUE)

    ax.add_patch(FancyBboxPatch((4, 68), 20, 16,
        boxstyle="round,pad=0.6,rounding_size=1.5", linewidth=1.8,
        edgecolor=DEEP_BLUE, facecolor=PALE_BLUE, zorder=3))
    ax.text(14, 79, 'Input', ha='center', va='center',
            fontsize=12, fontweight='bold', color=DEEP_BLUE, zorder=4)
    ax.text(14, 72, '200 LoRA\ntraining images', ha='center', va='center',
            fontsize=9, color=DEEP_BLUE, zorder=4)

    ax.add_patch(FancyBboxPatch((30, 68), 24, 16,
        boxstyle="round,pad=0.6,rounding_size=1.5", linewidth=1.8,
        edgecolor=MID_BLUE, facecolor=LIGHT_BLUE, alpha=0.9, zorder=3))
    ax.text(42, 79, 'Stage 1: SimHash', ha='center', va='center',
            fontsize=11, fontweight='bold', color=DEEP_BLUE, zorder=4)
    ax.text(42, 72, 'Hamming <= 10\nvs. LAION-2B-en', ha='center', va='center',
            fontsize=9, color=DEEP_BLUE, zorder=4)

    ax.add_patch(FancyBboxPatch((60, 68), 24, 16,
        boxstyle="round,pad=0.6,rounding_size=1.5", linewidth=1.8,
        edgecolor=MID_BLUE, facecolor=LIGHT_BLUE, alpha=0.9, zorder=3))
    ax.text(72, 79, 'Stage 2: CLIP Verify', ha='center', va='center',
            fontsize=11, fontweight='bold', color=DEEP_BLUE, zorder=4)
    ax.text(72, 72, 'cosine >= 0.95\n(high-confidence)', ha='center', va='center',
            fontsize=9, color=DEEP_BLUE, zorder=4)

    for x0, x1 in [(24.5, 29.5), (54.5, 59.5)]:
        ax.add_patch(FancyArrowPatch((x0, 76), (x1, 76), arrowstyle='-|>',
            mutation_scale=18, linewidth=1.8, color=DEEP_BLUE, zorder=2))

    ax.add_patch(FancyBboxPatch((30, 42), 40, 16,
        boxstyle="round,pad=0.6,rounding_size=1.5", linewidth=1.8,
        edgecolor=DEEP_BLUE, facecolor=MID_BLUE, alpha=0.85, zorder=3))
    ax.text(50, 53, 'Contamination Status Assignment', ha='center', va='center',
            fontsize=11, fontweight='bold', color='white', zorder=4)
    ax.text(50, 46, 'Clean (< 0.95) | Contaminated (>= 0.95)',
            ha='center', va='center', fontsize=9, color='white', zorder=4)

    ax.add_patch(FancyArrowPatch((72, 67.5), (55, 58.5), arrowstyle='-|>',
        mutation_scale=18, linewidth=1.8, color=DEEP_BLUE, zorder=2,
        connectionstyle="arc3,rad=0.15"))

    ax.add_patch(FancyBboxPatch((10, 18), 80, 16,
        boxstyle="round,pad=0.6,rounding_size=1.5", linewidth=1.8,
        edgecolor=DEEP_BLUE, facecolor=PALE_BLUE, zorder=3))
    ax.text(50, 30, 'Condition-Level Contamination Grouping',
            ha='center', va='center', fontsize=11, fontweight='bold',
            color=DEEP_BLUE, zorder=4)

    ax.add_patch(Rectangle((14, 21), 22, 3.5, facecolor=LIGHT_BLUE,
                            edgecolor=DEEP_BLUE, linewidth=1.0, alpha=0.9, zorder=4))
    ax.text(25, 22.75, 'Low < 10%', ha='center', va='center',
            fontsize=9.5, color=DEEP_BLUE, fontweight='bold', zorder=5)
    ax.add_patch(Rectangle((39, 21), 22, 3.5, facecolor=MID_BLUE,
                            edgecolor=DEEP_BLUE, linewidth=1.0, alpha=0.9, zorder=4))
    ax.text(50, 22.75, 'Medium 10-30%', ha='center', va='center',
            fontsize=9.5, color='white', fontweight='bold', zorder=5)
    ax.add_patch(Rectangle((64, 21), 22, 3.5, facecolor=DEEP_BLUE,
                            edgecolor=DEEP_BLUE, linewidth=1.0, alpha=0.9, zorder=4))
    ax.text(75, 22.75, 'High > 30%', ha='center', va='center',
            fontsize=9.5, color='white', fontweight='bold', zorder=5)

    ax.add_patch(FancyArrowPatch((50, 41.5), (50, 34.5), arrowstyle='-|>',
        mutation_scale=18, linewidth=1.8, color=DEEP_BLUE, zorder=2))

    ax.text(50, 8,
        'Proxy-corpus limitation: LAION-5B is a proxy for WIT 400M.',
        ha='center', va='center', fontsize=9, style='italic',
        color=NEUTRAL_GRAY,
        bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                  edgecolor=LIGHT_BLUE, linewidth=0.8, alpha=0.9))

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'Fig2_audit_workflow.png'), dpi=600, facecolor='white')
    plt.savefig(os.path.join(OUT_DIR, 'Fig2_audit_workflow.pdf'), dpi=600, facecolor='white')
    plt.close()
    print('[OK] Fig2')


def make_fig3():
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    labels = ['0.7 : 0.3', '0.5 : 0.5', '0.3 : 0.7']
    dominant = [0.82, 0.71, 0.68]
    secondary = [0.19, 0.52, 0.24]
    dom_err = [0.06, 0.07, 0.08]
    sec_err = [0.08, 0.09, 0.08]
    x = np.arange(len(labels)); width = 0.36

    bars1 = ax.bar(x - width/2, dominant, width, yerr=dom_err, capsize=5,
                    color=DEEP_BLUE, edgecolor='black', linewidth=0.8,
                    label='Dominant style', alpha=0.95,
                    error_kw=dict(ecolor='black', lw=1.0, capthick=1.0))
    bars2 = ax.bar(x + width/2, secondary, width, yerr=sec_err, capsize=5,
                    color=LIGHT_BLUE, edgecolor='black', linewidth=0.8,
                    label='Secondary style', alpha=0.95,
                    error_kw=dict(ecolor='black', lw=1.0, capthick=1.0))

    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.025, f'{h:.2f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold', color=DEEP_BLUE)
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.025, f'{h:.2f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold', color=MID_BLUE)

    ax.set_ylabel('Style label recall', fontsize=12, color=DEEP_BLUE)
    ax.set_xlabel('Weight ratio (Style A : Style B)', fontsize=12, color=DEEP_BLUE)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0, 1.08); ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.grid(axis='y', linestyle='--', alpha=0.35, color=LIGHT_BLUE)
    ax.set_axisbelow(True)
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    for spine in ['left', 'bottom']:
        ax.spines[spine].set_linewidth(1.2); ax.spines[spine].set_color(DEEP_BLUE)
    ax.legend(loc='upper right', frameon=True, facecolor='white',
              edgecolor=LIGHT_BLUE, fontsize=11)
    ax.set_title('Style Label Recall by Weight Ratio', fontsize=14,
                 fontweight='bold', color=DEEP_BLUE, pad=14)
    ax.text(0.5, -0.16, 'Error bars: +-1 SD. n = 20 conditions per cell. *** p_FDR < 0.001.',
            transform=ax.transAxes, ha='center', va='top', fontsize=9.5,
            style='italic', color=NEUTRAL_GRAY)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'Fig3_recall_by_weight.png'), dpi=600, facecolor='white')
    plt.savefig(os.path.join(OUT_DIR, 'Fig3_recall_by_weight.pdf'), dpi=600, facecolor='white')
    plt.close()
    print('[OK] Fig3')


def make_fig4():
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    data = np.array([[0.14, 0.48, 0.19], [0.20, 0.56, 0.29]])
    cmap = LinearSegmentedColormap.from_list(
        'sci_blue', [PALE_BLUE, LIGHT_BLUE, BRIGHT_BLUE, MID_BLUE, DEEP_BLUE])
    im = ax.imshow(data, cmap=cmap, aspect='auto', vmin=0.0, vmax=0.65)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            tc = 'white' if val > 0.40 else DEEP_BLUE
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                    fontsize=15, fontweight='bold', color=tc)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(['0.7 : 0.3', '0.5 : 0.5', '0.3 : 0.7'],
                       fontsize=12, color=DEEP_BLUE)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['East Asian\n(ink wash, ukiyo-e)',
                        'Western\n(cyberpunk, Baroque)'],
                       fontsize=11, color=DEEP_BLUE)
    ax.set_xlabel('Weight ratio (style as secondary component)',
                  fontsize=12, color=DEEP_BLUE, labelpad=10)
    ax.set_ylabel('Cultural origin', fontsize=12, color=DEEP_BLUE, labelpad=10)
    ax.set_title('Cultural Asymmetry in Secondary Style Recall',
                 fontsize=14, fontweight='bold', color=DEEP_BLUE, pad=14)
    ax.set_xticks(np.arange(-0.5, 3, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 2, 1), minor=True)
    ax.grid(which='minor', color='white', linestyle='-', linewidth=2.0)
    ax.tick_params(which='minor', bottom=False, left=False)
    cbar = plt.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    cbar.set_label('Recall', fontsize=11, color=DEEP_BLUE)
    cbar.ax.tick_params(labelsize=10, colors=DEEP_BLUE)
    cbar.outline.set_edgecolor(DEEP_BLUE)
    cbar.outline.set_linewidth(1.0)
    for j, (ea, w) in enumerate(zip(data[0], data[1])):
        delta = w - ea
        ax.text(j, 1.45, f'Delta = +{delta:.2f}', ha='center', va='center',
                fontsize=11, fontweight='bold', color=MID_BLUE,
                bbox=dict(boxstyle='round,pad=0.35', facecolor='white',
                          edgecolor=LIGHT_BLUE, linewidth=0.9, alpha=0.95))
    ax.text(0.5, -0.20,
            'Western styles as secondary are consistently recognized more often '
            'than East Asian styles across all conditions.',
            transform=ax.transAxes, ha='center', va='top', fontsize=9.5,
            style='italic', color=NEUTRAL_GRAY)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'Fig4_cultural_heatmap.png'), dpi=600, facecolor='white')
    plt.savefig(os.path.join(OUT_DIR, 'Fig4_cultural_heatmap.pdf'), dpi=600, facecolor='white')
    plt.close()
    print('[OK] Fig4')


if __name__ == '__main__':
    make_fig1(); make_fig2(); make_fig3(); make_fig4()
    print('\nAll figures generated at 600 DPI in outputs/figures/')
