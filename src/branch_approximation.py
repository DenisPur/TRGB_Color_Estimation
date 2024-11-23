import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def get_marking_and_approx_function(
        data: pd.DataFrame, 
        params: dict) -> tuple[pd.Series, pd.Series, callable, callable]:
    data['color_vi_real'] = data['color_vi'] - params['extinction']
    data['mag_i_real'] = data['mag_i'] - params['dist'] - params['absorbtion']
    
    chosen_bool = ((data['color_vi_real'] >= params['vi_left']) 
                   & (data['color_vi_real'] <= params['vi_right'])
                   & (data['mag_i_real'] >= params['i_left'])
                   & (data['mag_i_real'] <= params['i_right']))
    
    chosen = data[chosen_bool]
    non_chosen = data[~chosen_bool]

    def parabola(
            i: float, alpha: float, 
            beta: float, gamma:float) -> float:
        return alpha * i**2 + beta * i + gamma   

    x_data_0 = chosen['mag_i_real'].values
    y_data_0 = chosen['color_vi_real'].values
    
    curve_fit_result_0 = curve_fit(parabola, x_data_0, y_data_0)
    alpha_0, beta_0, gamma_0 = curve_fit_result_0[0]

    x_linspace = np.linspace(
        start=params['i_left'], stop=params['i_right'], num=50)

    for index in chosen.index:
        chosen.loc[index, 'error'] = np.abs(
            chosen.loc[index, 'color_vi_real'] - parabola(chosen.loc[index, 'mag_i_real'], alpha_0, beta_0, gamma_0))
    percent_to_take = params['p_chosen']
    threshold = np.percentile(chosen['error'].values, percent_to_take)

    inliers_bool = chosen['error'] <= threshold
    inliers = chosen[inliers_bool]

    x_data = inliers['mag_i_real'].values
    y_data = inliers['color_vi_real'].values

    curve_fit_result = curve_fit(parabola, x_data, y_data)
    alpha, beta, gamma = curve_fit_result[0]
    covariance = curve_fit_result[1]

    def f_approx(x: float) -> float: 
        return parabola(x, alpha, beta, gamma)

    def f_std(i: float) -> float:
        disp = np.array([[i**2, i, 1]]) @ covariance @ np.array([[i**2, i, 1]]).T
        return np.sqrt(disp).flatten()[0]
    
    return chosen_bool, inliers_bool, f_approx, f_std


def get_branch_approx_chart(
        data: pd.DataFrame, 
        params: dict, 
        chosen_bool: pd.Series, 
        inliers_bool: pd.Series, 
        f_approx: callable, 
        f_std: callable) -> plt.Figure:
    fig, axs = plt.subplots(1, 2, layout='tight', figsize=[16,9])
       
    chosen = data[chosen_bool]
    non_chosen = data[~chosen_bool]

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
        [params['vi_left'], params['vi_left'], params['vi_right'], params['vi_right'], params['vi_left']],
        [params['i_left'], params['i_right'], params['i_right'], params['i_left'], params['i_left']],
        lw=1, ls=':', color='black')
    
    axs[0].set_xlabel('V-I (real) $_{[mag]}$', size = 12)
    axs[0].set_ylabel('$M_I$ $_{[mag]}$', size = 12)
    axs[0].grid(linestyle=':')
    axs[0].invert_yaxis()

    inliers = chosen[inliers_bool]
    outliers = chosen[~inliers_bool]

    x_linspace = np.linspace(
        start=params['i_left'], stop=params['i_right'], num=50)
    y = [f_approx(x) for x in x_linspace]
    y_err = np.array([f_std(x) for x in x_linspace])

    d_m = params['d_minus']
    d_p = params['d_plus']
    i_level = params['i_level']

    estimate = f_approx(i_level)
    estimate_up = f_approx(i_level+d_p) - f_std(i_level+d_p)
    estimate_low = f_approx(i_level-d_m) + f_std(i_level-d_m)

    sns.scatterplot(
        data=inliers, 
        x='mag_i_real', y='color_vi_real', 
        alpha=0.6, c='xkcd:teal',
        ax=axs[1])
    sns.scatterplot(
        data=outliers,
        x='mag_i_real', y='color_vi_real', 
        alpha=0.3, c='xkcd:red',
        ax=axs[1])
    axs[1].plot(x_linspace, y, c='xkcd:forest green', alpha=0.9)
    axs[1].fill_between(x_linspace, y-y_err, y+y_err, alpha=0.3, color='xkcd:teal')

    axs[1].plot(
        [i_level, i_level], 
        [params['vi_left'], params['vi_right']],
        lw=1, ls=':', color='black')
    axs[1].plot(
        [i_level+d_p, i_level+d_p], 
        [params['vi_left'], params['vi_right']],
        lw=1, ls=':', color='black')
    axs[1].plot(
        [i_level-d_m, i_level-d_m], 
        [params['vi_left'], params['vi_right']],
        lw=1, ls=':', color='black')

    axs[1].plot(
        [params['i_left'], params['i_right']], 
        [estimate, estimate],
        lw=1, ls=':', color='black')
    axs[1].plot(
        [params['i_left'], params['i_right']],
        [estimate_up, estimate_up],
        lw=1, ls=':', color='black')
    axs[1].plot(
        [params['i_left'], params['i_right']], 
        [estimate_low, estimate_low],
        lw=1, ls=':', color='black')

    axs[1].set_xlabel('$M_I$ $_{[mag]}$', size = 12)
    axs[1].set_ylabel('V-I (real) $_{[mag]}$', size = 12)
    axs[1].grid(linestyle=':')
    return fig
