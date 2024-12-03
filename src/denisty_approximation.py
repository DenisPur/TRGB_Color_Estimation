import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from src.infra import simple_1d_kde

def choosing_low_density_regions(
        data: pd.DataFrame, 
        params: dict) -> tuple[pd.Series, float, float, float, float]:
    data['color_vi_real'] = data['color_vi'] - params['extinction']
    data['mag_i_real'] = data['mag_i'] - params['dist'] - params['absorbtion']

    delta_i_taken_for_error_sample = 0.1
    error_sample = data[(data['color_vi_real'] >= params['vi_left']) 
                        & (data['color_vi_real'] <= params['vi_right'])
                        & (data['mag_i_real'] >= params['i_level'] - delta_i_taken_for_error_sample) 
                        & (data['mag_i_real'] <= params['i_level'] + delta_i_taken_for_error_sample)]

    mean_v_error = error_sample['err_v'].mean()
    mean_i_error = error_sample['err_i'].mean()
    mean_color_error = np.sqrt(mean_v_error**2+mean_i_error**2)

    i_level_low = params['i_level'] - params['s_scaler'] * (params['d_minus'] + mean_i_error)
    i_level_high = params['i_level'] + params['s_scaler'] * (params['d_plus'] + mean_i_error)

    chosen_bool = ((data['color_vi_real'] >= params['vi_left']) 
                   & (data['color_vi_real'] <= params['vi_right'])
                   & (data['mag_i_real'] >= i_level_low) 
                   & (data['mag_i_real'] <= i_level_high))
    return chosen_bool, i_level_low, i_level_high, mean_i_error, mean_color_error


def slice_density_graph(
        data: pd.DataFrame, 
        params: dict, 
        smoothing_bw: float) -> plt.Figure:
    fig, axs = plt.subplots(2, 1, sharex=True, layout='tight', figsize=[9,9])

    i_level_low = params['i_level_low']
    i_level_high = params['i_level_high']
    vi_left = params['vi_left']
    vi_right = params['vi_right']

    chosen = data[params['chosen_bool']]
    non_chosen = data[~params['chosen_bool']]

    sns.scatterplot(
        data=non_chosen, 
        x='color_vi_real', y='mag_i_real', 
        alpha=0.6, s=5, 
        ax=axs[0])
    sns.scatterplot(
        data=chosen, 
        x='color_vi_real', y='mag_i_real', 
        alpha=0.8, s=5, color='xkcd:royal blue',
        ax=axs[0])
    axs[0].plot(
        [vi_left, vi_left, vi_right, vi_right, vi_left],
        [i_level_low, i_level_high, i_level_high, i_level_low, i_level_low],
        lw=1, ls=':', color='black')

    axs[0].errorbar(
        (vi_left + vi_right)/2, 
        params['i_level'], 
        xerr=params['mean_color_error'], 
        yerr=params['mean_i_error'],
        color='xkcd:light red',
        label=f'$\Delta$I    : {params["mean_i_error"]:1.3f} \n$\Delta$V-I : {params["mean_color_error"]:1.3f}')

    vertical_offset = 0.5
    horizontal_offset = 0.1
    axs[0].set_xlim(vi_left - horizontal_offset, vi_right + horizontal_offset)
    axs[0].set_ylim(i_level_low - vertical_offset, i_level_high + vertical_offset)
    axs[0].set_xlabel('V-I (real) $_{[mag]}$', size=12)
    axs[0].set_ylabel('$M_I$ $_{[mag]}$', size=12)
    axs[0].grid(linestyle=':')
    axs[0].legend(loc='upper right', fontsize='medium')
    axs[0].invert_yaxis()

    hist_bins = 40
    sns.histplot(
        data=chosen, x='color_vi_real', 
        stat='density', bins=hist_bins,
        alpha=0.5, 
        ax=axs[1])

    x_points_num = int((params['vi_right'] - params['vi_left']) * 200)
    x_linspace = np.linspace(start=params['vi_left'], stop=params['vi_right'], num=x_points_num)
    y_kde = simple_1d_kde(
        data_points=chosen['color_vi_real'].values,
        std_error=(params['mean_color_error'] + smoothing_bw),
        eval_points=x_linspace)

    x_max_y_value = np.argmax(y_kde) / (x_points_num - 1) * (params['vi_right'] - params['vi_left']) + params['vi_left']

    axs[1].plot(
        x_linspace, y_kde, 
        alpha=0.9, ls='-', lw=1, color='xkcd:red')
    axs[1].axvline(
        x_max_y_value, 
        ls='-.', lw=1, 
        color='xkcd:dark red', 
        label=f'kde max : {x_max_y_value:1.3f}')

    mean = chosen['color_vi_real'].mean()
    median = chosen['color_vi_real'].median()

    axs[1].axvline(
        mean, 
        ls='--', lw=1, 
        color='xkcd:dark olive', 
        label=f'$\mu$   : {mean:1.3f}')
    axs[1].axvline(
        median, 
        ls='-.', lw=1, 
        color='xkcd:deep blue', 
        label=f'$^1/_2$ : {median:1.3f}')

    axs[1].set_ylim(0, y_kde.max()+0.3)
    axs[1].set_ylabel('Density', size = 12)
    axs[1].set_xlabel('V-I $_{[mag]}$', size = 12)
    axs[1].grid(axis='x', linestyle=':')
    axs[1].legend(fontsize='large')
    return fig
