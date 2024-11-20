import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import ticker
from matplotlib.colors import Normalize
from scipy.stats import gaussian_kde
from functools import lru_cache


def get_overview_chart(data, boundries_for_overview, point_size=2):
    fig, axs = plt.subplots(1, 2, layout='tight', figsize=[16,9])
    
    color_low, color_high, x_ax_low, x_ax_high, y_ax_low, y_ax_high = boundries_for_overview
    
    norm=Normalize(vmin=color_low, vmax=color_high, clip=False)
    palette='rainbow'

    sns.scatterplot(
        data=data, x='x', y='y', 
        hue='color_vi', palette=palette,
        hue_norm=Normalize(vmin=color_low, vmax=color_high, clip=False),
        alpha=1.0, s=point_size, 
        ax=axs[0]
    )
    sm = plt.cm.ScalarMappable(cmap=palette, norm=norm)
    axs[0].figure.colorbar(
        sm, ax=axs[0], 
        location='bottom',
        fraction=0.03,
        pad=0.10,
        label='V-I $_{[mag]}$'
        )
    axs[0].set_xlabel('x $_{[px]}$', size = 12)
    axs[0].set_ylabel('y $_{[px]}$', size = 12)
    axs[0].set_xlim((x_ax_low, x_ax_high))
    axs[0].set_ylim((y_ax_low, y_ax_high))
    axs[0].grid(linestyle=':')
    axs[0].get_legend().remove()
    axs[0].set_title('Instrument field (with (V-I) colors)')

    sns.scatterplot(
        data=data, x='color_vi', y='mag_i',
        alpha=0.8, s=point_size, c='xkcd:sea blue',
        ax=axs[1]
    )
    axs[1].set_xlabel('V-I $_{[mag]}$', size = 12)
    axs[1].set_ylabel('$m_I$ $_{[mag]}$', size = 12)
    axs[1].invert_yaxis()
    axs[1].grid(linestyle=':')
    axs[1].set_title('CMD')
    return fig


def gat_clearing_chart(clean, dirty, point_size=2):
    fig, axs = plt.subplots(1, 2, layout='tight', figsize=[16,9])
    sns.scatterplot(
        data=dirty, x='x', y='y', 
        alpha=0.8, s=point_size, c='xkcd:bright red',
        ax=axs[0]
    )
    sns.scatterplot(
        data=clean, x='x', y='y', 
        alpha=0.8, s=point_size, c='xkcd:teal',
        ax=axs[0]
    )
    axs[0].set_xlabel('x $_{[px]}$', size = 12)
    axs[0].set_ylabel('y $_{[px]}$', size = 12)
    axs[0].grid(linestyle=':')
    axs[0].set_title('Instrument field')

    sns.scatterplot(
        data=dirty, x='color_vi', y='mag_i', 
        alpha=0.8, s=point_size, c='xkcd:bright red',
        ax=axs[1]
    )
    sns.scatterplot(
        data=clean, x='color_vi', y='mag_i', 
        alpha=0.8, s=point_size, c='xkcd:teal',
        ax=axs[1]
    )
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
    sns.scatterplot(
        data=data[~mask], x='x', y='y', 
        alpha=0.8, s=point_size, c='xkcd:light red',
        ax=axs[0]
    )
    sns.scatterplot(
        data=data[mask], x='x', y='y', 
        alpha=0.8, s=point_size, c='xkcd:sea blue',
        ax=axs[0]
    )
    axs[0].plot(
        [x_left, x_left, x_right, x_right, x_left], 
        [y_bottom, y_top, y_top, y_bottom, y_bottom], 
        alpha=0.9, lw=1, ls=':', c='black'
    )
    axs[0].set_xlabel('x $_{[px]}$', size = 12)
    axs[0].set_ylabel('y $_{[px]}$', size = 12)
    axs[0].grid(linestyle=':')
    axs[0].set_title('Instrument field')

    sns.scatterplot(
        data=data[~mask], x='color_vi', y='mag_i', 
        alpha=0.4, s=point_size, c='xkcd:light red',
        ax=axs[1]
    )
    sns.scatterplot(
        data=data[mask], x='color_vi', y='mag_i', 
        alpha=0.8, s=point_size, c='xkcd:sea blue',
        ax=axs[1]
    )
    axs[1].set_xlabel('V-I $_{[mag]}$', size = 12)
    axs[1].set_ylabel('$m_I$ $_{[mag]}$', size = 12)
    axs[1].invert_yaxis()
    axs[1].grid(linestyle=':') 
    axs[1].set_title('CMD')

    selected = sum(mask)
    of_all = len(data)
    fig.suptitle(f'Cropping the field of view. Selected {selected} of {of_all} ({100*selected/of_all:1.4}%)', fontsize=10)
    return fig


def get_cells_chart(data, number_of_cells, point_size=2):
    fig, ax = plt.subplots(layout='tight', figsize=[10,9])
    sns.scatterplot(
        data=data, x='x', y='y', 
        alpha=0.8, s=point_size, color='xkcd:sea blue'
    )
    x_grid = np.linspace(start=data['x'].min(), stop=data['x'].max(), num=number_of_cells+1)
    y_grid = np.linspace(start=data['y'].min(), stop=data['y'].max(), num=number_of_cells+1)
    for x in x_grid:
        ax.plot(
            [x, x], [data['y'].min(), data['y'].max()],
            alpha=0.8, ls=':', lw=1, color='black' 
        )
    for y in y_grid:
        ax.plot(
            [data['x'].min(), data['x'].max()], [y, y],
            alpha=0.8, ls=':', lw=1, color='black'
        )
    sns.histplot(
        data=data, x='x', y='y', 
        bins=[x_grid, y_grid], 
        alpha=0.4, cmap='cubehelix_r', 
        stat='count', cbar=True
    )
    ax.set_xlabel('x $_{[px]}$', size = 12)
    ax.set_ylabel('y $_{[px]}$', size = 12)
    ax.grid(linestyle=':')
    ax.set_title('Instrument field')
    return fig


def get_masked_cells_chart(data, number_of_cells, bool_mask, point_size=2):
    fig, axs = plt.subplots(1, 2, layout='tight', figsize=[16,9])
    sns.scatterplot(
        data=data[bool_mask], x='x', y='y', 
        alpha=0.8, s=point_size, color='xkcd:sea blue',
        ax=axs[0],
    )
    sns.scatterplot(
        data=data[~ bool_mask], x='x', y='y', 
        alpha=0.8, s=point_size, color='xkcd:light red',
        ax=axs[0]
    )
    x_grid = np.linspace(start=data['x'].min(), stop=data['x'].max(), num=number_of_cells+1)
    y_grid = np.linspace(start=data['y'].min(), stop=data['y'].max(), num=number_of_cells+1)
    for x in x_grid:
        axs[0].plot(
            [x, x], [data['y'].min(), data['y'].max()],
            alpha=0.8, ls=':', lw=1, color='black' 
        )
    for y in y_grid:
        axs[0].plot(
            [data['x'].min(), data['x'].max()], [y, y],
            alpha=0.8, ls=':', lw=1, color='black'
        )
    axs[0].set_xlabel('x $_{[px]}$', size = 12)
    axs[0].set_ylabel('y $_{[px]}$', size = 12)
    axs[0].grid(linestyle=':')
    axs[0].set_title('Instrument field')
    
    sns.scatterplot(
        data=data[~ bool_mask], x='color_vi', y='mag_i', 
        alpha=0.4, s=point_size, c='xkcd:light red',
        ax=axs[1]
    )
    sns.scatterplot(
        data=data[bool_mask], x='color_vi', y='mag_i', 
        alpha=0.8, s=point_size, c='xkcd:sea blue',
        ax=axs[1]
    )
    axs[1].set_xlabel('V-I $_{[mag]}$', size = 12)
    axs[1].set_ylabel('$m_I$ $_{[mag]}$', size = 12)
    axs[1].invert_yaxis()
    axs[1].grid(linestyle=':') 
    axs[1].set_title('CMD')

    selected = sum(bool_mask)
    of_all = len(data)
    fig.suptitle(f'Cropping the field of view. Selected {selected} of {of_all} ({100*selected/of_all:1.4}%)', fontsize=10)
    return fig


def get_kde(data, dist, redshift, absorbtion):
    x_raw = data['color_vi'].to_numpy(dtype=np.float32)
    y_raw = data['mag_i'].to_numpy(dtype=np.float32)
    x_grid, y_grid, z_grid = get_kde_computation_part(tuple(x_raw), tuple(y_raw))
    
    x_grid = x_grid - redshift
    y_grid = y_grid - dist - absorbtion
    return (x_grid, y_grid, z_grid)


@lru_cache(maxsize=16, typed=False)
def get_kde_computation_part(x, y):
    x = np.array(x)
    y = np.array(y)
    xy = np.vstack([x, y])

    x_grid = np.linspace(min(x), max(x), num=150)
    y_grid = np.linspace(min(y), max(y), num=250)
    x_grid, y_grid = np.meshgrid(x_grid, y_grid)

    kde = gaussian_kde(xy, bw_method=0.1)
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

        sns.scatterplot(
            data=data, x='color_vi_real', y='mag_i_real', 
            alpha=0.9, s=point_size, color='xkcd:blue grey'
        )
        contour = plt.contour(
            x_grid, y_grid, z_grid, 
            cmap="plasma_r", alpha=0.8, 
            levels=levels,
            norm='log'
        )
    else:
        sns.scatterplot(
            data=data, x='color_vi_real', y='mag_i_real', 
            alpha=0.9, s=point_size, color='xkcd:blue grey'
        )
        sns.histplot(
            data=data, x='color_vi_real', y='mag_i_real', 
            alpha=0.4, cmap='cubehelix_r', cbar=True
        )

    ax.axhline(
        y=-4.0, xmin=data['color_vi_real'].min(), xmax=data['color_vi_real'].max(), 
        alpha=0.9, linestyle='--', color='xkcd:red'
    )

    ax.set_xlabel('V-I (real) $_{[mag]}$', size = 12)
    ax.set_ylabel('$M_I$ $_{[mag]}$', size = 12)
    ax.invert_yaxis()
    ax.grid(linestyle=':') 
    ax.set_title(f'CMD in absolute values')
    return fig
