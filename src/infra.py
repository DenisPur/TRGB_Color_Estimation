import pandas as pd
import numpy as np
import io
from fpdf import FPDF
from PIL import Image
import matplotlib.pyplot as plt
from functools import lru_cache
from scipy.stats import gaussian_kde


def read_file(filename: str) -> pd.DataFrame:
    data = pd.read_csv(filename, sep=',')
    mandatory_column_names = ['x', 'y', 'mag_v', 'err_v', 'mag_i', 'err_i']
    for name in mandatory_column_names:
        if name not in data.columns:
            raise pd.errors.ParserError
    data['color_vi'] = data['mag_v'] - data['mag_i']
    return data


def check_available_columns(data: pd.DataFrame) -> dict:
    column_names = {'type':['type',], 
                    'mag':['mag_v', 'mag_i'],
                    'snr':['snr_v', 'snr_i'],
                    'sharp':['sharp_v', 'sharp_i'],
                    'flag':['flag_v', 'flag_i'],
                    'crowd':['crowd_v', 'crowd_i']}
    columns_available = dict()
    for col in column_names.keys():
        columns_available[col] = all([c in data.columns for c in column_names[col]])
    return columns_available


def mask_based_on_cells_density(
        data: pd.DataFrame, 
        cells_num: int, 
        threshold: float) -> np.array:
    x = np.array(data['x'].values)
    y = np.array(data['y'].values)
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    
    cells_count = np.zeros(shape=(cells_num, cells_num))
    nx = np.zeros(len(data), dtype=np.int8)
    ny = np.zeros(len(data), dtype=np.int8)
    eps = 1e-8

    for i, (x_i, y_i) in enumerate(zip(x, y)):
        nx[i] = int((cells_num) * (x_i - x_min) / (x_max + eps - x_min))
        ny[i] = int((cells_num) * (y_i - y_min) / (y_max + eps - y_min))
        cells_count[nx[i], ny[i]] += 1

    max_count = cells_count.max()
    count_limit = threshold * max_count

    bool_mask = np.zeros(shape=len(data), dtype=bool)
    for i, (nx_i, ny_i) in enumerate(zip(nx, ny)):
        bool_mask[i] = (cells_count[nx_i, ny_i] <= count_limit)
    return bool_mask


def create_pdf_out_of_figures(fig_list: list[plt.Figure]) -> FPDF:
    ''' This code made by AI. Its wierd but working good. '''
    pdf = FPDF()
    for fig in fig_list:
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)

        img = Image.open(buf)

        k = 0.264583  # 1 px = 0.264583 mm

        width, height = img.size
        width_mm = width *  k
        height_mm = height * k
        pdf.add_page(orientation='P', format=(width_mm, height_mm))

        pdf.image(buf, x=1, y=1, w=width_mm * 0.99)  # Adjust positioning and size as needed
    
        buf.close()
    return pdf


def get_kde(
        data: pd.DataFrame, 
        dist: float, 
        extinction: float, 
        absorbtion: float) -> tuple[np.array, np.array, np.array]:
    x_raw = data['color_vi'].to_numpy(dtype=np.float32)
    y_raw = data['mag_i'].to_numpy(dtype=np.float32)
    x_grid, y_grid, z_grid = get_kde_long_running_part(tuple(x_raw), tuple(y_raw))
    
    x_grid = x_grid - extinction
    y_grid = y_grid - dist - absorbtion
    return (x_grid, y_grid, z_grid)


@lru_cache(maxsize=16, typed=False)
def get_kde_long_running_part(
        x: tuple[int, ...], 
        y: tuple[int, ...]) -> tuple[np.array, np.array, np.array]:
    x = np.array(x)
    y = np.array(y)
    xy = np.vstack([x, y])

    x_grid = np.linspace(min(x), max(x), num=150)
    y_grid = np.linspace(min(y), max(y), num=250)
    x_grid, y_grid = np.meshgrid(x_grid, y_grid)

    kde = gaussian_kde(xy, bw_method=0.1)
    z_grid = kde(np.vstack([x_grid.ravel(), y_grid.ravel()])).reshape(x_grid.shape) * len(x)
    return (x_grid, y_grid, z_grid)
