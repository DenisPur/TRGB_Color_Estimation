import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def get_raw_chart(data):
    fig, axs = plt.subplots(1, 2, layout='tight', figsize=[12,6])
    sns.scatterplot(data=data, x='x', y='y', alpha=0.8, s=3, ax=axs[0])
    axs[0].set_xlabel('x $_{[px]}$', size = 12)
    axs[0].set_ylabel('y $_{[px]}$', size = 12)
    axs[0].grid(linestyle=':')

    sns.scatterplot(data=data, x='color_vi', y='mag_i', alpha=0.8, s=3, ax=axs[1])
    axs[1].set_xlabel('V-I $_{[mag]}$', size = 12)
    axs[1].set_ylabel('$m_I$ $_{[mag]}$', size = 12)
    axs[1].invert_yaxis()
    axs[1].grid(linestyle=':')
    plt.show()


def gat_clearing_chart(clean, dirty):
    fig, axs = plt.subplots(1, 2, layout='tight', figsize=[12,6])
    sns.scatterplot(data=clean, x='x', y='y', 
                    alpha=0.6, s=3, c='xkcd:teal',
                    ax=axs[0])
    sns.scatterplot(data=dirty, x='x', y='y', 
                    alpha=0.3, s=3, c='red',
                    ax=axs[0])
    axs[0].set_xlabel('x $_{[px]}$', size = 12)
    axs[0].set_ylabel('y $_{[px]}$', size = 12)
    axs[0].grid(linestyle=':')

    sns.scatterplot(data=clean, x='color_vi', y='mag_i', 
                    alpha=0.6, s=3, c='xkcd:teal',
                    ax=axs[1])
    sns.scatterplot(data=dirty, x='color_vi', y='mag_i', 
                    alpha=0.3, s=3, c='red',
                    ax=axs[1])
    axs[1].set_xlabel('V-I $_{[mag]}$', size = 12)
    axs[1].set_ylabel('$m_I$ $_{[mag]}$', size = 12)
    axs[1].invert_yaxis()
    axs[1].grid(linestyle=':')
    return fig


def get_masking_chart(data, mask, borders):
    x_left, x_right, y_bottom, y_top = borders

    fig, axs = plt.subplots(1, 2, layout='tight', figsize=[12,6])
    sns.scatterplot(data=data[mask], x='x', y='y', 
                    alpha=0.6, s=3, c='xkcd:teal',
                    ax=axs[0])
    sns.scatterplot(data=data[~mask], x='x', y='y', 
                    alpha=0.3, s=3, c='gray',
                    ax=axs[0])
    axs[0].plot([x_left, x_left, x_right, x_right, x_left], [y_bottom, y_top, y_top, y_bottom, y_bottom], 
                alpha=1.0, lw=1, c='black')
    axs[0].set_xlabel('x $_{[px]}$', size = 12)
    axs[0].set_ylabel('y $_{[px]}$', size = 12)
    axs[0].grid(linestyle=':')

    sns.scatterplot(data=data[mask], x='color_vi', y='mag_i', 
                    alpha=0.6, s=3, c='xkcd:teal',
                    ax=axs[1])
    sns.scatterplot(data=data[~mask], x='color_vi', y='mag_i', 
                    alpha=0.3, s=3, c='red',
                    ax=axs[1])
    axs[1].set_xlabel('V-I $_{[mag]}$', size = 12)
    axs[1].set_ylabel('$m_I$ $_{[mag]}$', size = 12)
    axs[1].invert_yaxis()
    axs[1].grid(linestyle=':') 
    return fig


def get_abs_mag_chart(data, dist, redshift, absorbtion):
    fig, ax = plt.subplots(layout='tight', figsize=[8,8])
    
    data['color_vi_real'] = data['color_vi'] - redshift
    data['mag_i_real'] = data['mag_i'] - dist - absorbtion

    ax = sns.scatterplot(data=data, x='color_vi_real', y='mag_i_real', 
                         alpha=0.5, s=3)

    ax.axhline(y=-3.5, xmin=data['color_vi_real'].min(), xmax=data['color_vi_real'].max(), 
               color='xkcd:red', alpha=0.9, linestyle='--')

    ax.set_xlabel('V-I (real) $_{[mag]}$', size = 12)
    ax.set_ylabel('$M_I$ $_{[mag]}$', size = 12)
    ax.invert_yaxis()
    ax.grid(linestyle=':') 
    return fig
