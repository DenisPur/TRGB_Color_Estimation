import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import ticker
from matplotlib.colors import Normalize


def get_overview_chart(
        data: pd.DataFrame, 
        boundries_for_overview: list[int], 
        point_size: float=2) -> plt.Figure:
    fig, axs = plt.subplots(1, 2, layout='tight', figsize=[16,9])
    
    color_low, color_high, x_ax_low, x_ax_high, y_ax_low, y_ax_high = boundries_for_overview
    norm=Normalize(vmin=color_low, vmax=color_high, clip=False)
    palette='rainbow'
    sm = plt.cm.ScalarMappable(cmap=palette, norm=norm)

    sns.scatterplot(
        data=data, x='x', y='y', 
        hue='color_vi', palette=palette,
        hue_norm=Normalize(vmin=color_low, vmax=color_high, clip=False),
        alpha=1.0, s=point_size, 
        ax=axs[0])
    axs[0].figure.colorbar(
        sm, ax=axs[0], 
        location='bottom',
        fraction=0.03,
        pad=0.10,
        label='V-I $_{[mag]}$')
        
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
        ax=axs[1])

    axs[1].set_xlabel('V-I $_{[mag]}$', size = 12)
    axs[1].set_ylabel('$m_I$ $_{[mag]}$', size = 12)
    axs[1].invert_yaxis()
    axs[1].grid(linestyle=':')
    axs[1].set_title('CMD')
    return fig


def gat_clearing_chart(
        clean: pd.DataFrame, 
        dirty: pd.DataFrame, 
        point_size: float=2) -> plt.Figure:
    fig, axs = plt.subplots(1, 2, layout='tight', figsize=[16,9])

    sns.scatterplot(
        data=dirty, x='x', y='y', 
        alpha=0.8, s=point_size, c='xkcd:bright red',
        ax=axs[0])
    sns.scatterplot(
        data=clean, x='x', y='y', 
        alpha=0.8, s=point_size, c='xkcd:teal',
        ax=axs[0])

    axs[0].set_xlabel('x $_{[px]}$', size = 12)
    axs[0].set_ylabel('y $_{[px]}$', size = 12)
    axs[0].grid(linestyle=':')
    axs[0].set_title('Instrument field')

    sns.scatterplot(
        data=dirty, x='color_vi', y='mag_i', 
        alpha=0.8, s=point_size, c='xkcd:bright red',
        ax=axs[1])
    sns.scatterplot(
        data=clean, x='color_vi', y='mag_i', 
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


def get_masking_chart(
        data: pd.DataFrame, 
        mask: pd.DataFrame, 
        borders: list[int], 
        point_size: float=2) -> plt.Figure:
    fig, axs = plt.subplots(1, 2, layout='tight', figsize=[16,9])

    sns.scatterplot(
        data=data[~mask], x='x', y='y', 
        alpha=0.8, s=point_size, c='xkcd:light red',
        ax=axs[0])
    sns.scatterplot(
        data=data[mask], x='x', y='y', 
        alpha=0.8, s=point_size, c='xkcd:sea blue',
        ax=axs[0])
    
    x_left, x_right, y_bottom, y_top = borders
    axs[0].plot(
        [x_left, x_left, x_right, x_right, x_left], 
        [y_bottom, y_top, y_top, y_bottom, y_bottom], 
        alpha=0.9, lw=1, ls=':', c='black')

    axs[0].set_xlabel('x $_{[px]}$', size = 12)
    axs[0].set_ylabel('y $_{[px]}$', size = 12)
    axs[0].grid(linestyle=':')
    axs[0].set_title('Instrument field')

    sns.scatterplot(
        data=data[~mask], x='color_vi', y='mag_i', 
        alpha=0.4, s=point_size, c='xkcd:light red',
        ax=axs[1])
    sns.scatterplot(
        data=data[mask], x='color_vi', y='mag_i', 
        alpha=0.8, s=point_size, c='xkcd:sea blue',
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


def get_cells_chart(
        data: pd.DataFrame, 
        cells_number: int, 
        point_size: float=2) -> plt.Figure:
    fig, ax = plt.subplots(layout='tight', figsize=[10,9])
    
    sns.scatterplot(
        data=data, x='x', y='y', 
        alpha=0.8, s=point_size, color='xkcd:sea blue')
    
    x_grid = np.linspace(start=data['x'].min(), stop=data['x'].max(), num=cells_number+1)
    y_grid = np.linspace(start=data['y'].min(), stop=data['y'].max(), num=cells_number+1)
    for x in x_grid:
        ax.plot(
            [x, x], [data['y'].min(), data['y'].max()],
            alpha=0.8, ls=':', lw=1, color='black')
    for y in y_grid:
        ax.plot(
            [data['x'].min(), data['x'].max()], [y, y],
            alpha=0.8, ls=':', lw=1, color='black')
    
    sns.histplot(
        data=data, x='x', y='y', 
        bins=[x_grid, y_grid], 
        alpha=0.4, cmap='cubehelix_r', 
        stat='count', cbar=True)

    ax.set_xlabel('x $_{[px]}$', size = 12)
    ax.set_ylabel('y $_{[px]}$', size = 12)
    ax.grid(linestyle=':')
    ax.set_title('Instrument field')
    return fig


def get_masked_cells_chart(
        data: pd.DataFrame, 
        cells_number: int, 
        bool_mask: pd.Series, 
        point_size: float=2) -> plt.Figure:
    fig, axs = plt.subplots(1, 2, layout='tight', figsize=[16,9])

    sns.scatterplot(
        data=data[bool_mask], x='x', y='y', 
        alpha=0.8, s=point_size, color='xkcd:sea blue',
        ax=axs[0])
    sns.scatterplot(
        data=data[~ bool_mask], x='x', y='y', 
        alpha=0.8, s=point_size, color='xkcd:light red',
        ax=axs[0])

    x_grid = np.linspace(start=data['x'].min(), stop=data['x'].max(), num=cells_number+1)
    y_grid = np.linspace(start=data['y'].min(), stop=data['y'].max(), num=cells_number+1)
    for x in x_grid:
        axs[0].plot(
            [x, x], [data['y'].min(), data['y'].max()],
            alpha=0.8, ls=':', lw=1, color='black')
    for y in y_grid:
        axs[0].plot(
            [data['x'].min(), data['x'].max()], [y, y],
            alpha=0.8, ls=':', lw=1, color='black')
    
    axs[0].set_xlabel('x $_{[px]}$', size = 12)
    axs[0].set_ylabel('y $_{[px]}$', size = 12)
    axs[0].grid(linestyle=':')
    axs[0].set_title('Instrument field')
    
    sns.scatterplot(
        data=data[~ bool_mask], x='color_vi', y='mag_i', 
        alpha=0.4, s=point_size, c='xkcd:light red',
        ax=axs[1])
    sns.scatterplot(
        data=data[bool_mask], x='color_vi', y='mag_i', 
        alpha=0.8, s=point_size, c='xkcd:sea blue',
        ax=axs[1])

    axs[1].set_xlabel('V-I $_{[mag]}$', size = 12)
    axs[1].set_ylabel('$m_I$ $_{[mag]}$', size = 12)
    axs[1].invert_yaxis()
    axs[1].grid(linestyle=':') 
    axs[1].set_title('CMD')

    selected = sum(bool_mask)
    of_all = len(data)
    fig.suptitle(f'Cropping the field of view. Selected {selected} of {of_all} ({100*selected/of_all:1.4}%)', fontsize=10)
    return fig


def get_abs_mag_chart(
        data: pd.DataFrame, 
        dist: float, 
        extinction: float, 
        absorbtion: float, 
        kde: None | tuple[np.array, np.array, np.array]=None, 
        point_size: float=2) -> plt.Figure:
    fig, ax = plt.subplots(layout='tight', figsize=[9,9] if kde else [10,9])
    
    data['color_vi_real'] = data['color_vi'] - extinction
    data['mag_i_real'] = data['mag_i'] - dist - absorbtion

    sns.scatterplot(
        data=data, x='color_vi_real', y='mag_i_real', 
        alpha=0.9, s=point_size, color='xkcd:blue grey')

    if kde:
        x_grid, y_grid, z_grid = kde
        z_max = np.max(z_grid)
        levels = [100 * 2**i for i in range(int(np.log(z_max / 100)/np.log(2)) + 1)]

        contour = plt.contour(
            x_grid, y_grid, z_grid, 
            cmap="plasma_r", alpha=0.8, 
            levels=levels, norm='log')
    else:
        sns.histplot(
            data=data, x='color_vi_real', y='mag_i_real', 
            alpha=0.4, cmap='cubehelix_r', cbar=True)

    ax.axhline(
        y=-4.0, xmin=data['color_vi_real'].min(), xmax=data['color_vi_real'].max(), 
        alpha=0.9, linestyle='--', color='xkcd:red')

    ax.set_xlabel('V-I (real) $_{[mag]}$', size = 12)
    ax.set_ylabel('$M_I$ $_{[mag]}$', size = 12)
    ax.invert_yaxis()
    ax.grid(linestyle=':') 
    ax.set_title(f'CMD in absolute values')
    return fig
