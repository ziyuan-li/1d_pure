from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse, FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd
from pure_resmlp.evaluation.plot_style import FULL_WIDTH, SINGLE_WIDTH, apply_publication_style, model_label, model_style, save_publication_figure
from pure_resmlp.paths import PROJECT_ROOT
SCIENCE = ('friedman', 'concrete', 'california', 'energy', 'yacht', 'auto_mpg')
MODELS = ('mlp', 'resmlp', 'rv_pure', 'cv_mlp', 'cv_pure')
CONTEXT = ('mlp_ordinary', 'mlp_context', 'mlp_explicit_cross', 'dcnv2_context', 'rv_pure', 'cv_mlp', 'cv_pure')
MAP_MODELS = ('mlp_context', 'rv_pure', 'cv_pure')
TITLES = dict(zip(SCIENCE, ('Friedman', 'Concrete', 'California', 'Energy', 'Yacht', 'Auto MPG')))
TITLES.update(bike_sharing='Bike Sharing', parkinsons_telemonitoring='Parkinsons')
FEATURES = {'friedman': [f'$x_{{{i + 1}}}$' for i in range(10)], 'concrete': ['Cement', 'Slag', 'Fly ash', 'Water', 'SP', 'Coarse', 'Fine', 'Age'], 'california': ['Income', 'House age', 'Rooms', 'Bedrooms', 'Population', 'Occupancy', 'Latitude', 'Longitude']}
MAP_SPEC = (('bike_sharing', ['temp', 'atemp', 'hum', 'windspeed'], ['hr', 'mnth', 'season', 'weathersit', 'weekday', 'workingday', 'holiday', 'yr'], ['Temperature', 'Feels-like temp.', 'Humidity', 'Wind speed'], ['Hour', 'Month', 'Season', 'Weather', 'Weekday', 'Working day', 'Holiday', 'Year']), ('parkinsons_telemonitoring', ['Jitter(%)', 'Jitter(Abs)', 'Jitter:RAP', 'Jitter:PPQ5', 'Jitter:DDP', 'Shimmer', 'Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5', 'Shimmer:APQ11', 'Shimmer:DDA', 'NHR', 'HNR', 'RPDE', 'DFA', 'PPE'], ['subject', 'age', 'sex', 'test_time'], None, ['Subject', 'Age', 'Sex', 'Time']))
SOURCES = ('scientific_interactions/mean_maps.csv', 'context_interactions.csv', 'noise_summary.csv', 'learning_curves.csv')
STEMS = ('pure_real_block_vertical', 'pure_network_architectures_vertical', 'figure_scientific_interactions', 'figure2_context_interactions', 'figure3_noise_robustness', 'figure4_sample_efficiency')

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def title(axis, label):
    axis.set_title(label, loc='left', fontweight='bold', pad=8)

def curve(axis, frame, model, x, mean, std, *, small=False):
    values = frame.loc[frame.model == model].sort_values(x)
    if values.empty:
        raise ValueError(f'Missing curve: {model}')
    xx, yy, ss = (values[k].to_numpy(dtype=float) for k in (x, mean, std))
    if not np.isfinite(np.concatenate((xx, yy, ss))).all() or (ss < 0).any():
        raise ValueError(f'Invalid mean/SD: {model}')
    style = model_style(model)
    axis.fill_between(xx, yy - ss, yy + ss, color=style['color'], alpha=0.1, linewidth=0)
    axis.plot(xx, yy, **style, markeredgecolor='white', markeredgewidth=0.35, markersize=2.3 if small else 3.7, linewidth=0.85 if small else 1.15, label=model_label(model), zorder=3)

def legend(axis, models):
    axis.axis('off')
    handles = [Line2D([], [], **model_style(m), label=model_label(m), markersize=3.7, markeredgecolor='white', markeredgewidth=0.35) for m in models]
    rows = (len(handles) + 4) // 5
    handles = [handles[row * 5 + column] for column in range(5) for row in range(rows) if row * 5 + column < len(handles)]
    axis.legend(handles=handles, loc='center', ncol=5, columnspacing=1.2, handlelength=2.25, handletextpad=0.55, fontsize=7)

def finish(fig, out, stem):
    fig.canvas.draw()
    save_publication_figure(fig, out, stem)
    print(f'Exported {stem}', flush=True)

def arrow(axis, start, end, **kwargs):
    axis.add_patch(FancyArrowPatch(start, end, arrowstyle='-|>', mutation_scale=7, linewidth=0.8, color='#424A52', **kwargs))

def box(axis, x, y, w, h, text, *, color='#52606D', fill='#F5F7F9', size=8):
    axis.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle='round,pad=0.008,rounding_size=0.014', linewidth=0.8, edgecolor=color, facecolor=fill, zorder=2))
    axis.text(x, y, text, ha='center', va='center', fontsize=size, zorder=3)

def block_diagram(out):
    fig, axis = plt.subplots(figsize=(SINGLE_WIDTH, 4.9))
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.98)
    axis.set(xlim=(0, 1), ylim=(0, 1))
    axis.axis('off')
    center = 0.43
    axis.add_patch(FancyBboxPatch((0.08, 0.345), 0.7, 0.445, boxstyle='round,pad=0.008,rounding_size=0.02', linewidth=0.8, edgecolor='#E69F00', facecolor='#FFF7E7'))
    axis.text(0.107, 0.565, 'Product-unit mapping', ha='center', va='center', rotation=90, fontsize=7.5, fontweight='bold')
    axis.text(center, 0.965, '$\\mathbf{h}_{\\ell}$', ha='center', va='center', fontsize=10)
    nodes = [(0.86, 0.085, 'Affine map\n' + '$\\mathbf{a}=A_{\\ell}\\mathbf{h}_{\\ell}+\\mathbf{a}_{\\ell}$'), (0.695, 0.065, '$\\mathbf{s}=\\mathrm{ReLU}(\\mathbf{a})+\\mathbf{1}$'), (0.595, 0.06, '$\\log(\\mathbf{s})$'), (0.495, 0.06, '$W_{\\ell}\\log(\\mathbf{s})$'), (0.395, 0.06, '$\\mathbf{p}=\\exp(W_{\\ell}\\log(\\mathbf{s}))$'), (0.25, 0.085, 'Affine map\n' + '$B_{\\ell}\\mathbf{p}+\\mathbf{b}_{\\ell}$')]
    previous = 0.945
    for y, h, text in nodes:
        arrow(axis, (center, previous), (center, y + h / 2 + 0.008))
        box(axis, center, y, 0.57, h, text, fill='white', size=8)
        previous = y - h / 2 - 0.008
    arrow(axis, (center, previous), (center, 0.138))
    axis.add_patch(Ellipse((center, 0.12), 0.046, 0.033, facecolor='white', edgecolor='#424A52', linewidth=0.8))
    axis.text(center, 0.12, '+', ha='center', va='center', fontsize=11)
    axis.plot([center, 0.88, 0.88], [0.94, 0.94, 0.12], color='#424A52', linewidth=0.8)
    arrow(axis, (0.88, 0.12), (center + 0.023, 0.12))
    axis.text(0.915, 0.54, 'Identity shortcut', rotation=90, ha='center', va='center', fontsize=7.5)
    arrow(axis, (center, 0.102), (center, 0.043))
    axis.text(center, 0.022, '$\\mathbf{h}_{\\ell+1}$', ha='center', va='center', fontsize=10)
    finish(fig, out, STEMS[0])

def architecture_diagram(out):
    fig, axes = plt.subplots(1, 2, figsize=(SINGLE_WIDTH, 5.0))
    fig.subplots_adjust(left=0.025, right=0.975, bottom=0.015, top=0.965, wspace=0.15)
    for index, axis in enumerate(axes):
        complex_model = index == 1
        color = model_style('cv_pure' if complex_model else 'rv_pure')['color']
        fill = '#EDF5FA' if complex_model else '#FFF7E7'
        field = '\\mathbb{C}' if complex_model else '\\mathbb{R}'
        axis.set(xlim=(0, 1), ylim=(0, 1))
        axis.axis('off')
        axis.text(0.5, 1.01, '(b) CV-PURe' if complex_model else '(a) RV-PURe', ha='center', fontsize=8.5, fontweight='bold')
        axis.text(0.5, 0.956, '$\\mathbf{x}+\\mathrm{i}\\mathbf{0}$' if complex_model else '$\\mathbf{x}\\in\\mathbb{R}^{p}$', ha='center', va='center', fontsize=8.5)
        box(axis, 0.5, 0.86, 0.84, 0.085, 'Affine input\n' + f'${field}^p \\to {field}^h$', size=7.5)
        arrow(axis, (0.5, 0.932), (0.5, 0.91))
        if complex_model:
            arrow(axis, (0.5, 0.809), (0.5, 0.686))
            axis.text(0.54, 0.75, 'Direct to\nblock 1', fontsize=7, va='center', ha='left')
        else:
            arrow(axis, (0.5, 0.809), (0.5, 0.788))
            box(axis, 0.5, 0.755, 0.84, 0.05, 'ReLU', size=7.5)
            arrow(axis, (0.5, 0.721), (0.5, 0.686))
        for y, label in ((0.63, 'PURe block 1'), (0.48, 'PURe block 2')):
            box(axis, 0.5, y, 0.84, 0.095, label + '\n' + f'${field}^h \\to {field}^h$', color=color, fill=fill, size=7.5)
        arrow(axis, (0.5, 0.575), (0.5, 0.535))
        arrow(axis, (0.5, 0.425), (0.5, 0.385))
        box(axis, 0.5, 0.33, 0.84, 0.095, 'Affine output\n' + f'${field}^h \\to {field}$', size=7.5)
        if complex_model:
            arrow(axis, (0.5, 0.275), (0.5, 0.23))
            axis.text(0.57, 0.256, '$o$', fontsize=8, va='center')
            box(axis, 0.5, 0.17, 0.9, 0.1, 'Real-valued projection\n' + '$\\alpha\\Re(o)+\\beta\\Im(o)+b$', color=color, fill=fill, size=7)
            arrow(axis, (0.5, 0.112), (0.5, 0.06))
        else:
            arrow(axis, (0.5, 0.275), (0.5, 0.06))
            axis.text(0.54, 0.17, 'Direct real\nreadout', fontsize=7, va='center', ha='left')
        axis.text(0.5, 0.035, '$\\widehat{y}\\in\\mathbb{R}$', ha='center', fontsize=9)
    finish(fig, out, STEMS[1])

def scientific_maps(frame, out):
    fig = plt.figure(figsize=(FULL_WIDTH, 5.65), layout='constrained')
    grid = fig.add_gridspec(3, 6, width_ratios=[1, 1, 1, 1, 1, 0.075], hspace=0.1)
    for row, dataset in enumerate(SCIENCE[:3]):
        data = frame.loc[frame.dataset == dataset]
        labels = FEATURES[dataset]
        vmax = data.strength_mean.max()
        for column, model in enumerate(MODELS):
            axis = fig.add_subplot(grid[row, column])
            matrix = np.full((len(labels), len(labels)), np.nan)
            pairs = data.loc[data.model == model]
            assert len(pairs) == len(labels) * (len(labels) - 1) // 2
            for pair in pairs.itertuples():
                matrix[pair.i, pair.j] = matrix[pair.j, pair.i] = pair.strength_mean
            im = axis.imshow(matrix, vmin=0, vmax=vmax, cmap='cividis', interpolation='nearest')
            axis.set_xticks(range(len(labels)), labels, rotation=90, fontsize=6.3)
            axis.set_yticks(range(len(labels)), labels if column == 0 else [''] * len(labels), fontsize=6.3)
            axis.tick_params(length=0, pad=2)
            if row == 0:
                axis.set_title(model_label(model), fontsize=7.5, fontweight='bold', pad=10)
            if column == 0:
                axis.set_ylabel(f'({chr(97 + row)}) {TITLES[dataset]}', fontsize=8, fontweight='bold', labelpad=10)
            for spine in axis.spines.values():
                spine.set_visible(False)
        cax = fig.add_subplot(grid[row, 5])
        cax.set_box_aspect(1 / 0.075)
        bar = fig.colorbar(im, cax=cax)
        bar.set_label('Interaction strength', fontsize=7)
        bar.ax.tick_params(labelsize=6.3, length=2)
    finish(fig, out, STEMS[2])

def context_maps(frame, out):
    fig = plt.figure(figsize=(FULL_WIDTH, 5.8), layout='constrained')
    grid = fig.add_gridspec(2, 4, width_ratios=[1, 1, 1, 0.055], height_ratios=[1, 1.8], hspace=0.13)
    for row, (dataset, ordinary, context, ylabels, xlabels) in enumerate(MAP_SPEC):
        matrices = []
        for model in MAP_MODELS:
            data = frame.loc[(frame.dataset == dataset) & (frame.model == model)]
            assert data.sampling_seed.eq(0).all()
            grouped = data.groupby(['ordinary_feature', 'context_feature'])
            assert grouped.training_seed.apply(lambda x: sorted(x) == list(range(5))).all()
            matrix = grouped.interaction_strength.mean().unstack().loc[ordinary, context]
            matrices.append(matrix.to_numpy())
        vmax = max((m.max() for m in matrices))
        for column, (model, matrix) in enumerate(zip(MAP_MODELS, matrices)):
            axis = fig.add_subplot(grid[row, column])
            im = axis.imshow(matrix, vmin=0, vmax=vmax, cmap='cividis', interpolation='nearest', aspect='auto')
            axis.set_xticks(range(len(context)), xlabels, rotation=45, ha='right', rotation_mode='anchor', fontsize=7)
            axis.set_yticks(range(len(ordinary)), ylabels or ordinary if column == 0 else [''] * len(ordinary), fontsize=7)
            axis.tick_params(length=0, pad=3)
            title(axis, f'({chr(97 + row * 3 + column)}) {model_label(model)}')
            axis.set_xlabel('Context feature')
            if column == 0:
                axis.set_ylabel(TITLES[dataset] + '\nOrdinary feature', fontweight='bold', labelpad=9)
            for spine in axis.spines.values():
                spine.set_visible(False)
        cax = fig.add_subplot(grid[row, 3])
        bar = fig.colorbar(im, cax=cax)
        bar.set_label('Interaction strength', fontsize=7)
        bar.ax.tick_params(labelsize=7, length=2)
    finish(fig, out, STEMS[3])

def noise_curves(frame, out):
    datasets = (*SCIENCE, 'bike_sharing', 'parkinsons_telemonitoring')
    fig = plt.figure(figsize=(FULL_WIDTH, 4.8), layout='constrained')
    grid = fig.add_gridspec(3, 4, height_ratios=[1, 1, 0.23])
    for index, dataset in enumerate(datasets):
        axis = fig.add_subplot(grid[index // 4, index % 4])
        data = frame.loc[frame.dataset == dataset]
        assert data.training_seeds.eq(5).all()
        for model in MODELS if dataset in SCIENCE else CONTEXT:
            assert np.array_equal(np.sort(data.loc[data.model == model, 'noise_std'].unique()), [0, 0.02, 0.05, 0.1])
            curve(axis, data, model, 'noise_std', 'test_nrmse_mean', 'test_nrmse_std')
        title(axis, f'({chr(97 + index)}) {TITLES[dataset]}')
        axis.set_xticks([0, 0.02, 0.05, 0.1], ['0', '0.02', '0.05', '0.10'])
        axis.set_xlim(-0.004, 0.104)
        axis.grid(axis='y')
        axis.set_axisbelow(True)
        if index >= 4:
            axis.set_xlabel('Noise SD $\\sigma$')
        if index % 4 == 0:
            axis.set_ylabel('Normalized test RMSE')
    legend(fig.add_subplot(grid[2, :]), (*MODELS, *CONTEXT[:4]))
    finish(fig, out, STEMS[4])

def learning_curves(frame, out):
    fig = plt.figure(figsize=(FULL_WIDTH, 3.65), layout='constrained')
    grid = fig.add_gridspec(2, 3, height_ratios=[1, 0.24])
    for index, dataset in enumerate(('friedman', 'concrete', 'parkinsons_telemonitoring')):
        axis = fig.add_subplot(grid[0, index])
        data = frame.loc[frame.dataset == dataset].copy()
        assert data.test_rmse_count.eq(5).all()
        data['percent'] = data.fraction * 100
        models = MODELS if dataset in SCIENCE else CONTEXT
        for model in models:
            assert np.array_equal(np.sort(data.loc[data.model == model, 'percent'].unique()), [5, 10, 25, 50, 100])
            curve(axis, data, model, 'percent', 'test_rmse_mean', 'test_rmse_std')
        title(axis, f'({chr(97 + index)}) {TITLES[dataset]}')
        axis.set(xlim=(1, 104), xlabel='Training data (%)', ylabel=['Test RMSE', 'Test RMSE (MPa)', 'Test RMSE (UPDRS)'][index])
        axis.set_xticks([5, 10, 25, 50, 100])
        axis.grid(axis='y')
        axis.set_axisbelow(True)
        if dataset == 'parkinsons_telemonitoring':
            zoom = axis.inset_axes([0.38, 0.35, 0.59, 0.32])
            for model in CONTEXT[1:]:
                curve(zoom, data, model, 'percent', 'test_rmse_mean', 'test_rmse_std', small=True)
            zoom.set(xlim=(1, 104), ylim=(3.8, 7.6))
            zoom.set_xticks([10, 50, 100])
            zoom.set_yticks([4, 5, 6, 7])
            zoom.tick_params(labelsize=5.5, length=2, pad=1)
            zoom.set_title('Context models (zoom)', fontsize=6.2, pad=3)
            zoom.grid(axis='y', linewidth=0.3)
            zoom.set_axisbelow(True)
    legend(fig.add_subplot(grid[1, :]), (*MODELS, *CONTEXT[:4]))
    finish(fig, out, STEMS[5])
