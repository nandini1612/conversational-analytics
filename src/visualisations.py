"""
Person 2 — Models & Evaluation Visualisations
Produces all charts needed for the final report and presentation.

Run: python outputs/visualisations.py
Saves all figures to outputs/figures/
"""

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from scipy.stats import pearsonr

FIGURES_DIR = Path('outputs/figures')
METRICS_DIR = Path('outputs/metrics')
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ── Colour palette ────────────────────────────────────────────
C_RIDGE  = '#2196F3'   # blue
C_RF     = '#4CAF50'   # green
C_BERT   = '#FF9800'   # orange
C_ENS    = '#9C27B0'   # purple
C_TRUE   = '#607D8B'   # grey
C_POS    = '#43A047'
C_NEG    = '#E53935'

plt.rcParams.update({
    'font.family': 'sans-serif',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 150,
})


# ─────────────────────────────────────────────────────────────
# 1. Ridge alpha search
# ─────────────────────────────────────────────────────────────
def plot_alpha_search():
    df = pd.read_csv('outputs/metrics/ridge_alpha_search.csv')
    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax2 = ax1.twinx()

    ax1.semilogx(df['alpha'], df['mae'], 'o-', color=C_RIDGE, lw=2, label='Val MAE')
    ax2.semilogx(df['alpha'], df['pearson_r'], 's--', color=C_ENS, lw=2, label='Pearson r')

    best_idx = df['mae'].idxmin()
    ax1.axvline(df.loc[best_idx, 'alpha'], color='red', linestyle=':', alpha=0.7,
                label=f"Best α={df.loc[best_idx,'alpha']}")

    ax1.set_xlabel('Alpha (log scale)')
    ax1.set_ylabel('Validation MAE', color=C_RIDGE)
    ax2.set_ylabel('Pearson r', color=C_ENS)
    ax1.set_title('Ridge Regression — Alpha Search')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right', fontsize=9)

    plt.tight_layout()
    path = FIGURES_DIR / '1_ridge_alpha_search.png'
    plt.savefig(path)
    plt.close()
    print(f'  Saved: {path}')


# ─────────────────────────────────────────────────────────────
# 2. Ablation study — val and test side by side
# ─────────────────────────────────────────────────────────────
def plot_ablation():
    val_df  = pd.read_csv('outputs/metrics/ablation_val.csv')
    test_df = pd.read_csv('outputs/metrics/ablation_test.csv')

    groups = ['A_sentiment', 'B_structure', 'C_agent', 'D_metadata']
    labels = ['Remove\nGroup A\n(Sentiment)', 'Remove\nGroup B\n(Structure)',
              'Remove\nGroup C\n(Agent)', 'Remove\nGroup D\n(Metadata)']

    val_deltas  = []
    test_deltas = []
    for g in groups:
        row_v = val_df[val_df['features_removed'] == g]
        row_t = test_df[test_df['features_removed'] == g]
        val_deltas.append(float(row_v['mae_delta'].values[0]) if len(row_v) else 0)
        test_deltas.append(float(row_t['mae_delta'].values[0]) if len(row_t) else 0)

    x = np.arange(len(groups))
    w = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))

    bars_v = ax.bar(x - w/2, val_deltas,  w, label='Validation', color=C_RIDGE, alpha=0.85)
    bars_t = ax.bar(x + w/2, test_deltas, w, label='Test',       color=C_ENS,   alpha=0.85)

    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel('ΔMAE vs baseline (higher = group matters more)')
    ax.set_title('Ablation Study — Feature Group Contribution\n(Ridge Regression)')
    ax.legend()

    for bar in list(bars_v) + list(bars_t):
        h = bar.get_height()
        if abs(h) > 0.0001:
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.0003,
                    f'{h:+.4f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    path = FIGURES_DIR / '2_ablation_study.png'
    plt.savefig(path)
    plt.close()
    print(f'  Saved: {path}')


# ─────────────────────────────────────────────────────────────
# 3. Model comparison — test set metrics bar chart
# ─────────────────────────────────────────────────────────────
def plot_model_comparison():
    df = pd.read_csv('outputs/metrics/test_metrics_table.csv')
    models  = df['Model'].tolist()
    colours = [C_RIDGE, C_RF, C_BERT, C_ENS]
    metrics = ['MAE', 'RMSE', 'Pearson r', 'F1 (≥3.0)']

    fig, axes = plt.subplots(1, 4, figsize=(14, 5))
    fig.suptitle('Model Comparison — Test Set Metrics', fontsize=13, fontweight='bold')

    for ax, metric in zip(axes, metrics):
        vals = df[metric].tolist()
        # Replace NaN with 0 for plotting purposes
        vals_plot = [v if not (isinstance(v, float) and np.isnan(v)) else 0 for v in vals]
        bars = ax.bar(models, vals_plot, color=colours, alpha=0.85, edgecolor='white')
        ax.set_title(metric, fontsize=11)
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=20, ha='right', fontsize=8)

        # Highlight best (exclude NaN rows)
        valid = [(i, v) for i, v in enumerate(vals) if not (isinstance(v, float) and np.isnan(v))]
        if valid:
            if metric in ['MAE', 'RMSE']:
                best = min(valid, key=lambda x: x[1])[0]
            else:
                best = max(valid, key=lambda x: x[1])[0]
            bars[best].set_edgecolor('black')
            bars[best].set_linewidth(2)

        for bar, v in zip(bars, vals):
            label = f'{v:.4f}' if not (isinstance(v, float) and np.isnan(v)) else '—'
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    label, ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    path = FIGURES_DIR / '3_model_comparison.png'
    plt.savefig(path)
    plt.close()
    print(f'  Saved: {path}')


# ─────────────────────────────────────────────────────────────
# 4. Calibration scatter — predicted vs actual
# ─────────────────────────────────────────────────────────────
def plot_calibration():
    cal = json.load(open('outputs/metrics/calibration_data.json'))
    y_true   = np.array(cal['y_true'])
    ensemble = np.array(cal['ensemble'])
    ridge    = np.array(cal['ridge'])
    rf       = np.array(cal['rf'])

    fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=True)
    fig.suptitle('Calibration — Predicted vs Actual CSAT\n(Test Set, n=375)',
                 fontsize=12, fontweight='bold')

    pairs = [('Ridge', ridge, C_RIDGE), ('Random Forest', rf, C_RF),
             ('Ensemble', ensemble, C_ENS)]

    for ax, (name, preds, col) in zip(axes, pairs):
        ax.scatter(y_true, preds, alpha=0.25, s=12, color=col)
        lo, hi = 1.0, 5.0
        ax.plot([lo, hi], [lo, hi], 'k--', lw=1.2, label='Perfect calibration')
        r, _ = pearsonr(y_true, preds)
        mae  = np.mean(np.abs(y_true - preds))
        ax.set_title(f'{name}\nMAE={mae:.4f}  r={r:.4f}', fontsize=10)
        ax.set_xlabel('Actual CSAT')
        if ax == axes[0]:
            ax.set_ylabel('Predicted CSAT')
        ax.set_xlim(0.8, 5.2)
        ax.set_ylim(0.8, 5.2)
        ax.legend(fontsize=8)

    plt.tight_layout()
    path = FIGURES_DIR / '4_calibration_scatter.png'
    plt.savefig(path)
    plt.close()
    print(f'  Saved: {path}')


# ─────────────────────────────────────────────────────────────
# 5. Prediction distribution — all models vs true
# ─────────────────────────────────────────────────────────────
def plot_prediction_distributions():
    cal = json.load(open('outputs/metrics/calibration_data.json'))
    y_true   = np.array(cal['y_true'])
    ensemble = np.array(cal['ensemble'])
    ridge    = np.array(cal['ridge'])
    rf       = np.array(cal['rf'])

    fig, ax = plt.subplots(figsize=(9, 5))
    bins = np.linspace(1, 5, 30)

    ax.hist(y_true,   bins=bins, alpha=0.5, color=C_TRUE,  label=f'True CSAT  (std={y_true.std():.3f})',   density=True)
    ax.hist(ridge,    bins=bins, alpha=0.5, color=C_RIDGE, label=f'Ridge      (std={ridge.std():.3f})',    density=True)
    ax.hist(rf,       bins=bins, alpha=0.5, color=C_RF,    label=f'Random Forest (std={rf.std():.3f})',   density=True)
    ax.hist(ensemble, bins=bins, alpha=0.5, color=C_ENS,   label=f'Ensemble   (std={ensemble.std():.3f})', density=True)

    ax.set_xlabel('CSAT Score')
    ax.set_ylabel('Density')
    ax.set_title('Prediction Distribution vs True CSAT\n(Mean-compression visible in all models)')
    ax.legend(fontsize=9)
    ax.axvline(y_true.mean(), color=C_TRUE, linestyle='--', lw=1.5, alpha=0.8)

    plt.tight_layout()
    path = FIGURES_DIR / '5_prediction_distributions.png'
    plt.savefig(path)
    plt.close()
    print(f'  Saved: {path}')


# ─────────────────────────────────────────────────────────────
# 6. Feature importances — horizontal bar chart
# ─────────────────────────────────────────────────────────────
def plot_feature_importances():
    imp = json.load(open('outputs/metrics/feature_importances.json'))
    sorted_imp = imp['sorted']

    names  = [x[0] for x in sorted_imp]
    values = [x[1] for x in sorted_imp]

    # Colour by group
    group_colours = {
        'mean_sentiment': '#FF7043', 'last_20_sentiment': '#FF7043', 'std_sentiment': '#FF7043',
        'talk_time_ratio': '#42A5F5', 'avg_agent_words': '#42A5F5', 'avg_customer_words': '#42A5F5',
        'interruption_count': '#42A5F5', 'resolution_flag': '#42A5F5',
        'empathy_density': '#66BB6A', 'apology_count': '#66BB6A', 'transfer_count': '#66BB6A',
    }
    colours = [group_colours.get(n, '#AB47BC') for n in names]  # purple = metadata

    fig, ax = plt.subplots(figsize=(9, 8))
    y_pos = np.arange(len(names))
    ax.barh(y_pos, values, color=colours, alpha=0.85)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel('Feature Importance')
    ax.set_title('Random Forest Feature Importances\n(All 22 features, sorted descending)')

    legend_patches = [
        mpatches.Patch(color='#FF7043', label='Group A — Sentiment'),
        mpatches.Patch(color='#42A5F5', label='Group B — Structure'),
        mpatches.Patch(color='#66BB6A', label='Group C — Agent behaviour'),
        mpatches.Patch(color='#AB47BC', label='Group D — Metadata'),
    ]
    ax.legend(handles=legend_patches, fontsize=9, loc='lower right')

    plt.tight_layout()
    path = FIGURES_DIR / '6_feature_importances.png'
    plt.savefig(path)
    plt.close()
    print(f'  Saved: {path}')


# ─────────────────────────────────────────────────────────────
# 7. Residual plot — ensemble
# ─────────────────────────────────────────────────────────────
def plot_residuals():
    cal = json.load(open('outputs/metrics/calibration_data.json'))
    y_true   = np.array(cal['y_true'])
    ensemble = np.array(cal['ensemble'])
    residuals = y_true - ensemble

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Ensemble Residual Analysis (Test Set)', fontsize=12, fontweight='bold')

    # Residuals vs predicted
    axes[0].scatter(ensemble, residuals, alpha=0.25, s=12, color=C_ENS)
    axes[0].axhline(0, color='black', lw=1.2, linestyle='--')
    axes[0].set_xlabel('Predicted CSAT')
    axes[0].set_ylabel('Residual (Actual − Predicted)')
    axes[0].set_title('Residuals vs Predicted')

    # Residual distribution
    axes[1].hist(residuals, bins=30, color=C_ENS, alpha=0.8, edgecolor='white')
    axes[1].axvline(0, color='black', lw=1.2, linestyle='--')
    axes[1].axvline(residuals.mean(), color='red', lw=1.5, linestyle='-',
                    label=f'Mean={residuals.mean():.3f}')
    axes[1].set_xlabel('Residual')
    axes[1].set_ylabel('Count')
    axes[1].set_title(f'Residual Distribution\nstd={residuals.std():.3f}')
    axes[1].legend(fontsize=9)

    plt.tight_layout()
    path = FIGURES_DIR / '7_residuals.png'
    plt.savefig(path)
    plt.close()
    print(f'  Saved: {path}')


# ─────────────────────────────────────────────────────────────
# 8. Summary metrics table as figure (for slides/report)
# ─────────────────────────────────────────────────────────────
def plot_metrics_table():
    df = pd.read_csv('outputs/metrics/test_metrics_table.csv')

    fig, ax = plt.subplots(figsize=(9, 3))
    ax.axis('off')

    col_labels = ['Model', 'MAE ↓', 'RMSE ↓', 'Pearson r ↑', 'F1 (≥3.0) ↑']
    cell_text  = []
    for _, row in df.iterrows():
        pr = row['Pearson r']
        cell_text.append([
            row['Model'],
            f"{row['MAE']:.4f}",
            f"{row['RMSE']:.4f}",
            f"{pr:.4f}" if not (isinstance(pr, float) and np.isnan(pr)) else "—",
            f"{row['F1 (≥3.0)']:.4f}",
        ])

    table = ax.table(cellText=cell_text, colLabels=col_labels,
                     loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 2.0)

    # Colour header
    for j in range(len(col_labels)):
        table[0, j].set_facecolor('#37474F')
        table[0, j].set_text_props(color='white', fontweight='bold')

    # Highlight ensemble row
    model_col = df['Model'].tolist()
    ens_row = model_col.index('Ensemble') + 1
    for j in range(len(col_labels)):
        table[ens_row, j].set_facecolor('#EDE7F6')

    ax.set_title('Final Test Set Results (n=375, held-out)',
                 fontsize=12, fontweight='bold', pad=20)

    plt.tight_layout()
    path = FIGURES_DIR / '8_metrics_table.png'
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {path}')


# ─────────────────────────────────────────────────────────────
# Run all
# ─────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print('Generating visualisations...\n')
    plot_alpha_search()
    plot_ablation()
    plot_model_comparison()
    plot_calibration()
    plot_prediction_distributions()
    plot_feature_importances()
    plot_residuals()
    plot_metrics_table()
    print(f'\nAll figures saved to {FIGURES_DIR}/')
