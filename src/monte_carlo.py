import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
import seaborn as sns

from src.denisty_approximation import density_choosing_region


def randomly_stir_data(data):
    noice_v = np.random.normal(loc=0.0, scale=1.0, size=len(data))
    noice_i = np.random.normal(loc=0.0, scale=1.0, size=len(data))

    d_mag_v = data['err_v'].values
    d_mag_i = data['err_i'].values
    
    data_stirred = data.copy()
    data_stirred['mag'] += d_mag_i * noice_i
    data_stirred['color'] += d_mag_v * noice_v - d_mag_i * noice_i
    return data_stirred


def crop_data(data, params):
    chosen_bool = ((data['color'] >= params['vi_left']) & 
                   (data['color'] <= params['vi_right']) & 
                   (data['mag'] >= params['i_level_low']) & 
                   (data['mag'] <= params['i_level_high']))

    colors = data[chosen_bool]['color'].values
    return colors


def ax_add_histogram(ax, colors, params, std_err):
    bw = 2 * std_err / (max(colors) - min(colors))
    kde = gaussian_kde(colors, bw_method=bw)
    x = np.linspace(min(colors), max(colors), num=200)
    y = kde.evaluate(x)
    x_max = x[np.argmax(y)]
    
    ax.hist(colors, bins=21, color='xkcd:peach', density=True, alpha=0.8)
    ax.plot(x, y, color='xkcd:red')
    ax.axvline(x_max, c='xkcd:red', ls='--', lw=1)
    ax.plot(x_max, max(y), marker='x', color='black')
    ax.text(x_max, max(y)+0.05, f'{x_max:1.4}', fontsize=12)


def plot_histogrm_3x3(data, params):
    fig, axes = plt.subplots(3, 3, sharex=True, sharey=True, figsize=[8,8])

    data_s = data[['mag_v', 'mag_i', 'err_v', 'err_i']].copy()
    data_s['color'] = data_s['mag_v'] - data_s['mag_i'] - params['redshift']
    data_s['mag'] = data_s['mag_i'] - params['dist'] - params['absorbtion']

    for row in axes:
        for ax in row:
            data_stirred = randomly_stir_data(data_s)
            colors = crop_data(data_stirred, params)
            ax_add_histogram(ax, colors, params, params['mean_color_error'])
    fig.set_layout_engine('tight')
    return fig


def iterate_over_n_experiments(data, params, n):
    data_s = data[['mag_v', 'mag_i', 'err_v', 'err_i']].copy()
    data_s['color'] = data_s['mag_v'] - data_s['mag_i'] - params['redshift']
    data_s['mag'] = data_s['mag_i'] - params['dist'] - params['absorbtion']
    
    x_linspace = np.linspace(params['vi_left'], params['vi_right'], num=100)
    bw = 2 * params['mean_color_error'] / (params['vi_right'] - params['vi_left'])
    
    results = list()
    iter_num = 0

    while(iter_num <= n):
        data_stirred = randomly_stir_data(data_s)
        colors = crop_data(data_stirred, params)
        if len(data_stirred) == 0: 
            continue
        iter_num += 1

        kde = gaussian_kde(colors, bw_method=bw)
        y = kde.evaluate(x_linspace)
        x_max = x_linspace[np.argmax(y)]
        results.append(x_max)
    return np.array(results)


def plot_monte_carlo_results(experiments_results, mean_color, std_err, n):
    def gauss(t, mean, std):
        t = np.array(t)
        y = 1 / (std * np.sqrt(2*np.pi)) * np.exp(- 0.5 * (t - mean)**2 / std**2)
        return y

    fig, ax = plt.subplots(figsize=[8,8])

    ax.hist(experiments_results, bins=30, color='xkcd:light teal', density=True, alpha=0.8)

    x_left, x_right = plt.gca().get_xlim()
    y_bottom, y_top = plt.gca().get_ylim()
    x = np.linspace(x_left, x_right, num=100)
    y = gauss(x, mean_color, std_err)

    ax.plot(x, y, color='xkcd:deep blue', ls=':')
    ax.text(x_left, y_top / 2, f'{n=}\n{mean_color=:1.3f}\n{std_err=:1.3f}', fontsize=15)
    ax.set_xlabel('V-I $_{[mag]}$', size=12)
    fig.set_layout_engine('tight')
    return fig
