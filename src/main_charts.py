import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import ticker
from matplotlib.colors import Normalize


def layout_cmd(ax: plt.Axes):
    ax.set_xlabel('V-I $_{[mag]}$', size = 12)
    ax.set_ylabel('$m_I$ $_{[mag]}$', size = 12)
    ax.invert_yaxis()
    ax.grid(linestyle=':')
    ax.set_title('CMD')


def layout_field(ax: plt.Axes):
    ax.set_xlabel('x $_{[px]}$', size = 12)
    ax.set_ylabel('y $_{[px]}$', size = 12)
    ax.grid(linestyle=':')
    ax.set_title('Instrument field')


def scatter_main_accent(
        ax: plt.Axes, 
        data: pd.DataFrame, 
        x: str, y:str):
    sns.scatterplot(
        data=data, x=x, y=y, 
        alpha=0.8, s=2, c='xkcd:sea blue',
        ax=ax)


def scatter_negative_accent_1(
        ax: plt.Axes, 
        data: pd.DataFrame, 
        x: str, y:str):
    sns.scatterplot(
        data=data, x=x, y=y, 
        alpha=0.9, s=2, c='xkcd:red',
        ax=ax)


def scatter_negative_accent_2(
        ax: plt.Axes, 
        data: pd.DataFrame, 
        x: str, y:str):
    sns.scatterplot(
        data=data, x=x, y=y, 
        alpha=0.4, s=2, c='xkcd:light red',
        ax=ax)


def cells_grid(
        ax: plt.Axes,
        x_grid: np.array, 
        y_grid: np.array):
    for x in x_grid:
        ax.plot(
            [x, x], [y_grid.min(), y_grid.max()],
            alpha=0.8, ls=':', lw=1, color='black')
    for y in y_grid:
        ax.plot(
            [x_grid.min(), x_grid.max()], [y, y],
            alpha=0.8, ls=':', lw=1, color='black')


def overview_cmd_field_chart(
        data: pd.DataFrame, 
        boundries_for_overview: list[int]) -> plt.Figure:
    fig, axs = plt.subplots(1, 2, layout='tight', figsize=[16,9])
    
    color_low, color_high, x_ax_low, x_ax_high, y_ax_low, y_ax_high = boundries_for_overview
    norm=Normalize(vmin=color_low, vmax=color_high, clip=False)
    palette='rainbow'
    sm = plt.cm.ScalarMappable(cmap=palette, norm=norm)

    sns.scatterplot(
        data=data, x='x', y='y', 
        hue='color_vi', palette=palette,
        hue_norm=Normalize(vmin=color_low, vmax=color_high, clip=False),
        alpha=1.0, s=2, 
        ax=axs[0])
    axs[0].figure.colorbar(
        sm, ax=axs[0], 
        location='bottom',
        fraction=0.03,
        pad=0.10,
        label='V-I $_{[mag]}$')
        
    layout_field(axs[0])
    axs[0].set_xlim((x_ax_low, x_ax_high))
    axs[0].set_ylim((y_ax_low, y_ax_high))
    axs[0].get_legend().remove()

    scatter_main_accent(axs[1], data, 'color_vi', 'mag_i')
    layout_cmd(axs[1])
    return fig


def clearing_chart(
        clean: pd.DataFrame, 
        dirty: pd.DataFrame) -> plt.Figure:
    fig, axs = plt.subplots(1, 2, layout='tight', figsize=[16,9])

    scatter_negative_accent_1(axs[0], dirty, 'x', 'y')
    scatter_main_accent(axs[0], clean, 'x', 'y')
    layout_field(axs[0])

    scatter_negative_accent_1(axs[1], dirty, 'color_vi', 'mag_i')
    scatter_main_accent(axs[1], clean, 'color_vi', 'mag_i')
    layout_cmd(axs[1])

    selected = len(clean)
    of_all = len(dirty) + selected
    fig.suptitle(f'Clearing the data. Selected {selected} of {of_all} ({100*selected/of_all:1.4}%)')
    return fig


def rectangular_masking_chart(
        data: pd.DataFrame, 
        mask: pd.DataFrame, 
        borders: list[int]) -> plt.Figure:
    fig, axs = plt.subplots(1, 2, layout='tight', figsize=[16,9])

    scatter_main_accent(axs[0], data[mask], 'x', 'y')
    scatter_negative_accent_1(axs[0], data[~mask], 'x', 'y')

    x_left, x_right, y_bottom, y_top = borders
    axs[0].plot(
        [x_left, x_left, x_right, x_right, x_left], 
        [y_bottom, y_top, y_top, y_bottom, y_bottom], 
        alpha=0.9, lw=1, ls=':', c='black')
    layout_field(axs[0])

    scatter_negative_accent_2(axs[1], data[~mask], 'color_vi', 'mag_i')
    scatter_main_accent(axs[1], data[mask], 'color_vi', 'mag_i')
    layout_cmd(axs[1])

    selected = sum(mask)
    of_all = len(data)
    fig.suptitle(f'Cropping the field of view. Selected {selected} of {of_all} ({100*selected/of_all:1.4}%)', fontsize=10)
    return fig


def cells_hist_chart(
        data: pd.DataFrame, 
        cells_number: int) -> plt.Figure:
    fig, ax = plt.subplots(layout='tight', figsize=[10,9])

    scatter_main_accent(ax, data, 'x', 'y')

    x_grid = np.linspace(start=data['x'].min(), stop=data['x'].max(), num=cells_number+1)
    y_grid = np.linspace(start=data['y'].min(), stop=data['y'].max(), num=cells_number+1)
    cells_grid(ax, x_grid, y_grid)
    sns.histplot(
        data=data, x='x', y='y', 
        bins=[x_grid, y_grid], 
        alpha=0.4, cmap='cubehelix_r', 
        stat='count', cbar=True)
    layout_field(ax)
    return fig


def cells_masking_chart(
        data: pd.DataFrame, 
        cells_number: int, 
        bool_mask: pd.Series) -> plt.Figure:
    fig, axs = plt.subplots(1, 2, layout='tight', figsize=[16,9])

    scatter_negative_accent_1(axs[0], data[~bool_mask], 'x', 'y')
    scatter_main_accent(axs[0], data[bool_mask], 'x', 'y')

    x_grid = np.linspace(start=data['x'].min(), stop=data['x'].max(), num=cells_number+1)
    y_grid = np.linspace(start=data['y'].min(), stop=data['y'].max(), num=cells_number+1)
    cells_grid(axs[0], x_grid, y_grid)
    layout_field(axs[0])
    
    scatter_negative_accent_2(axs[1], data[~bool_mask], 'color_vi', 'mag_i')
    scatter_main_accent(axs[1], data[bool_mask], 'color_vi', 'mag_i')
    layout_cmd(axs[1])

    selected = sum(bool_mask)
    of_all = len(data)
    fig.suptitle(f'Cropping the field of view. Selected {selected} of {of_all} ({100*selected/of_all:1.4}%)', fontsize=10)
    return fig


def abs_mag_cmd_chart(
        data: pd.DataFrame, 
        kde: None | tuple[np.array, np.array, np.array]=None) -> plt.Figure:
    fig, ax = plt.subplots(layout='tight', figsize=[9,9] if kde else [10,9])
    
    sns.scatterplot(
        data=data, x='abs_color_vi', y='abs_mag_i', 
        alpha=0.9, s=2, color='xkcd:blue grey')

    if kde:
        x_grid, y_grid, z_grid = kde
        z_max = np.max(z_grid)
        levels = [100 * 2**i for i in range(int(np.log(z_max / 100)/np.log(2)) + 1)]
        plt.contour(
            x_grid, y_grid, z_grid, 
            cmap="plasma_r", alpha=0.8, 
            levels=levels, norm='log')
    else:
        sns.histplot(
            data=data, x='abs_color_vi', y='abs_mag_i', 
            alpha=0.4, cmap='cubehelix_r', cbar=True)

    ax.axhline(
        y=-4.0, xmin=data['abs_color_vi'].min(), xmax=data['abs_color_vi'].max(), 
        alpha=0.9, linestyle='--', color='xkcd:red')

    layout_cmd(ax)
    ax.set_title(f'CMD in absolute values')
    return fig
