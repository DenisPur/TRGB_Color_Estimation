import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import seaborn as sns


def get_raw_chart(data):
    fig, axs = plt.subplots(1, 2, layout='tight', figsize=[12,7])
    
    low = np.percentile(data['color_vi'].values, 1.5)
    upp = np.percentile(data['color_vi'].values, 98.5)
    
    norm=Normalize(vmin=low, vmax=upp, clip=False)
    # palette='RdYlBu_r'
    palette='rainbow'

    sns.scatterplot(data=data, x='x', y='y', 
                    hue='color_vi', palette=palette,
                    hue_norm=Normalize(vmin=low, vmax=upp, clip=False),
                    alpha=0.8, s=2, 
                    ax=axs[0])
    sm = plt.cm.ScalarMappable(cmap=palette, norm=norm)
    axs[0].figure.colorbar(sm, ax=axs[0], 
                           location='right',
                           fraction=0.03,
                           pad=0.01)
    axs[0].set_xlabel('x $_{[px]}$', size = 12)
    axs[0].set_ylabel('y $_{[px]}$', size = 12)
    axs[0].grid(linestyle=':')
    axs[0].get_legend().remove()
    axs[0].set_title('Instrument field (with (V-I) colors)')

    sns.scatterplot(data=data, x='color_vi', y='mag_i',
                    alpha=0.8, s=2, c='xkcd:teal',
                    ax=axs[1])
    axs[1].set_xlabel('V-I $_{[mag]}$', size = 12)
    axs[1].set_ylabel('$m_I$ $_{[mag]}$', size = 12)
    axs[1].invert_yaxis()
    axs[1].grid(linestyle=':')
    axs[1].set_title('CMD')
    fig.suptitle('Raw data')
    return fig


def gat_clearing_chart(clean, dirty):
    fig, axs = plt.subplots(1, 2, layout='tight', figsize=[12,7])
    sns.scatterplot(data=clean, x='x', y='y', 
                    alpha=0.8, s=2, c='xkcd:teal',
                    ax=axs[0])
    sns.scatterplot(data=dirty, x='x', y='y', 
                    alpha=0.4, s=2, c='red',
                    ax=axs[0])
    axs[0].set_xlabel('x $_{[px]}$', size = 12)
    axs[0].set_ylabel('y $_{[px]}$', size = 12)
    axs[0].grid(linestyle=':')
    axs[0].set_title('Instrument field')

    sns.scatterplot(data=clean, x='color_vi', y='mag_i', 
                    alpha=0.8, s=2, c='xkcd:teal',
                    ax=axs[1])
    sns.scatterplot(data=dirty, x='color_vi', y='mag_i', 
                    alpha=0.4, s=2, c='red',
                    ax=axs[1])
    axs[1].set_xlabel('V-I $_{[mag]}$', size = 12)
    axs[1].set_ylabel('$m_I$ $_{[mag]}$', size = 12)
    axs[1].invert_yaxis()
    axs[1].grid(linestyle=':')
    axs[1].set_title('CMD')

    selected = len(clean)
    of_all = len(dirty) + selected
    fig.suptitle(f'Clearing the data. Selected {selected} of {of_all} ({100*selected/of_all:1.4}%)')
    return fig


def get_masking_chart(data, mask, borders):
    x_left, x_right, y_bottom, y_top = borders

    fig, axs = plt.subplots(1, 2, layout='tight', figsize=[12,7])
    sns.scatterplot(data=data[mask], x='x', y='y', 
                    alpha=0.8, s=2, c='xkcd:teal',
                    ax=axs[0])
    sns.scatterplot(data=data[~mask], x='x', y='y', 
                    alpha=0.4, s=2, c='gray',
                    ax=axs[0])
    axs[0].plot([x_left, x_left, x_right, x_right, x_left], [y_bottom, y_top, y_top, y_bottom, y_bottom], 
                alpha=1.0, lw=1, c='black')
    axs[0].set_xlabel('x $_{[px]}$', size = 12)
    axs[0].set_ylabel('y $_{[px]}$', size = 12)
    axs[0].grid(linestyle=':')
    axs[0].set_title('Instrument field')

    sns.scatterplot(data=data[mask], x='color_vi', y='mag_i', 
                    alpha=0.8, s=2, c='xkcd:teal',
                    ax=axs[1])
    sns.scatterplot(data=data[~mask], x='color_vi', y='mag_i', 
                    alpha=0.4, s=2, c='red',
                    ax=axs[1])
    axs[1].set_xlabel('V-I $_{[mag]}$', size = 12)
    axs[1].set_ylabel('$m_I$ $_{[mag]}$', size = 12)
    axs[1].invert_yaxis()
    axs[1].grid(linestyle=':') 
    axs[1].set_title('CMD')

    selected = sum(mask)
    of_all = len(data)
    fig.suptitle(f'Cropping the field of view. Selected {selected} of {of_all} ({100*selected/of_all:1.4}%)', fontsize=10)
    return fig


def get_abs_mag_chart(data, dist, redshift, absorbtion, add_kde):
    fig, ax = plt.subplots(layout='tight', figsize=[9,8])
    
    data['color_vi_real'] = data['color_vi'] - redshift
    data['mag_i_real'] = data['mag_i'] - dist - absorbtion

    if add_kde:
        ax = sns.scatterplot(data=data, x='color_vi_real', y='mag_i_real', 
                             alpha=0.8, s=2)
        ax = sns.kdeplot(data=data, x='color_vi_real', y='mag_i_real', 
                         alpha=1.0, fill=False, cmap='plasma_r')
    else:
        ax = sns.scatterplot(data=data, x='color_vi_real', y='mag_i_real', 
                             alpha=0.9, color='xkcd:blue grey', s=2)
        ax = sns.histplot(data=data, x='color_vi_real', y='mag_i_real', 
                          alpha=0.4, cmap='cubehelix_r', cbar=True)

    ax.axhline(y=-4.0, xmin=data['color_vi_real'].min(), xmax=data['color_vi_real'].max(), 
               color='xkcd:red', alpha=0.9, linestyle='--')

    ax.set_xlabel('V-I (real) $_{[mag]}$', size = 12)
    ax.set_ylabel('$M_I$ $_{[mag]}$', size = 12)
    ax.invert_yaxis()
    ax.grid(linestyle=':') 
    ax.set_title(f'CMD in absolute values')
    return fig
