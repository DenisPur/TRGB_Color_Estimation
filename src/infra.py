import pandas as pd
import io
from fpdf import FPDF
from PIL import Image

def read_file(filename):
    data = pd.read_csv(filename, sep=',')
    mandatory_column_names = ['x', 'y', 'mag_v', 'err_v', 'mag_i', 'err_i']
    for name in mandatory_column_names:
        if name not in data.columns:
            raise pd.errors.ParserError
    data['color_vi'] = data['mag_v'] - data['mag_i']
    return data


def check_available_columns(data):
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


def create_pdf_out_of_figures(fig_list):
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
