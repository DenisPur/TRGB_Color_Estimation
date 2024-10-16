import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib import ticker
import seaborn as sns
from scipy.stats import gaussian_kde


def get_overview_chart(data, point_size=2):
    fig, axs = plt.subplots(1, 2, layout='tight', figsize=[16,9])
    
    low = np.percentile(data['color_vi'].values, 1.5)
    upp = np.percentile(data['color_vi'].values, 98.5)
    
    norm=Normalize(vmin=low, vmax=upp, clip=False)
    # palette='RdYlBu_r'
    palette='rainbow'

    sns.scatterplot(data=data, x='x', y='y', 
                    hue='color_vi', palette=palette,
                    hue_norm=Normalize(vmin=low, vmax=upp, clip=False),
                    alpha=1.0, s=point_size, 
                    ax=axs[0])
    sm = plt.cm.ScalarMappable(cmap=palette, norm=norm)
    axs[0].figure.colorbar(sm, ax=axs[0], 
                           location='bottom',
                           fraction=0.03,
                           pad=0.10,
                           label='V-I $_{[mag]}$')
    axs[0].set_xlabel('x $_{[px]}$', size = 12)
    axs[0].set_ylabel('y $_{[px]}$', size = 12)
    axs[0].grid(linestyle=':')
    axs[0].get_legend().remove()
    axs[0].set_title('Instrument field (with (V-I) colors)')

    sns.scatterplot(data=data, x='color_vi', y='mag_i',
                    alpha=0.8, s=point_size, c='xkcd:teal',
                    ax=axs[1])
    axs[1].set_xlabel('V-I $_{[mag]}$', size = 12)
    axs[1].set_ylabel('$m_I$ $_{[mag]}$', size = 12)
    axs[1].invert_yaxis()
    axs[1].grid(linestyle=':')
    axs[1].set_title('CMD')
    return fig


def gat_clearing_chart(clean, dirty, point_size=2):
    fig, axs = plt.subplots(1, 2, layout='tight', figsize=[16,9])
    sns.scatterplot(data=dirty, x='x', y='y', 
                    alpha=0.8, s=point_size, c='red',
                    ax=axs[0])
    sns.scatterplot(data=clean, x='x', y='y', 
                    alpha=0.8, s=point_size, c='xkcd:teal',
                    ax=axs[0])
    axs[0].set_xlabel('x $_{[px]}$', size = 12)
    axs[0].set_ylabel('y $_{[px]}$', size = 12)
    axs[0].grid(linestyle=':')
    axs[0].set_title('Instrument field')

    sns.scatterplot(data=dirty, x='color_vi', y='mag_i', 
                    alpha=0.8, s=point_size, c='red',
                    ax=axs[1])
    sns.scatterplot(data=clean, x='color_vi', y='mag_i', 
                    alpha=0.8, s=point_size, c='xkcd:teal',
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


def get_masking_chart(data, mask, borders, point_size=2):
    x_left, x_right, y_bottom, y_top = borders

    fig, axs = plt.subplots(1, 2, layout='tight', figsize=[16,9])
    sns.scatterplot(data=data[~mask], x='x', y='y', 
                    alpha=0.8, s=point_size, c='gray',
                    ax=axs[0])
    sns.scatterplot(data=data[mask], x='x', y='y', 
                    alpha=0.8, s=point_size, c='xkcd:teal',
                    ax=axs[0])
    axs[0].plot([x_left, x_left, x_right, x_right, x_left], [y_bottom, y_top, y_top, y_bottom, y_bottom], 
                alpha=1.0, lw=1, c='black')
    axs[0].set_xlabel('x $_{[px]}$', size = 12)
    axs[0].set_ylabel('y $_{[px]}$', size = 12)
    axs[0].grid(linestyle=':')
    axs[0].set_title('Instrument field')

    sns.scatterplot(data=data[~mask], x='color_vi', y='mag_i', 
                    alpha=0.8, s=point_size, c='red',
                    ax=axs[1])
    sns.scatterplot(data=data[mask], x='color_vi', y='mag_i', 
                    alpha=0.8, s=point_size, c='xkcd:teal',
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


def get_kde(data, dist, redshift, absorbtion):
    x = data['color_vi'].to_numpy() - redshift
    y = data['mag_i'].to_numpy() - dist - absorbtion

    x_grid = np.linspace(min(x), max(x), num=100)
    y_grid = np.linspace(min(y), max(y), num=200)

    xy = np.vstack([x, y])
    kde = gaussian_kde(xy, bw_method=0.1)
    
    x_grid, y_grid = np.meshgrid(x_grid, y_grid)
    z_grid = kde(np.vstack([x_grid.ravel(), y_grid.ravel()])).reshape(x_grid.shape) * len(x)

    return (x_grid, y_grid, z_grid)


def get_abs_mag_chart(data, dist, redshift, absorbtion, kde=None, point_size=2):
    if kde:
        fig, ax = plt.subplots(layout='tight', figsize=[9,9])
    else:
        fig, ax = plt.subplots(layout='tight', figsize=[10,9])
    
    data['color_vi_real'] = data['color_vi'] - redshift
    data['mag_i_real'] = data['mag_i'] - dist - absorbtion

    if kde:
        x_grid, y_grid, z_grid = kde
        z_max = np.max(z_grid)
        levels = [100 * 2**i for i in range(int(np.log(z_max / 100)/np.log(2)) + 1)]

        ax = sns.scatterplot(data=data, x='color_vi_real', y='mag_i_real', 
                             alpha=0.8, s=point_size)
        contour = plt.contour(x_grid, y_grid, z_grid, 
                              cmap="plasma_r", alpha=0.8, 
                              levels=levels,
                              norm='log')
    else:
        ax = sns.scatterplot(data=data, x='color_vi_real', y='mag_i_real', 
                             alpha=0.9, color='xkcd:blue grey', s=point_size)
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
