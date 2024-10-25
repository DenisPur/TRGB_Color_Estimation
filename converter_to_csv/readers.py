from pathlib import Path
import pandas as pd
import numpy as np


def define_type(file: Path | str) -> str:
    """Rough approximation of a file type. Types correspond
    to readers defines in edd_reader. 
    """

    with open(file) as f:
        lines = f.readlines()
        if lines[0][0:9] == 'extension':
            data = pd.read_csv(file, delimiter='\s+', header=[0,], nrows=10)
            if 'mag_mag1' in data.columns:
                return 'type_1'
            elif 'mag_V' in data.columns:
                return 'type_2'
        elif lines[1][0] == 'e':
            data = pd.read_csv(file, delimiter='\s+', skiprows=[0,], nrows=10)
            return 'type_3'
        elif lines[1][0:3] == 'chp':
            data = pd.read_csv(file, delimiter='\s+', skiprows=[0,], nrows=10)
            return 'type_4'
        else:
            data = pd.read_csv(file, delimiter='\s+', header=None, nrows=10)
            if len(data.columns) == 69:
                return 'type_5'
            else:
                return 'type_6'


##########################################################################


def read_type1(file : str | Path) -> pd.DataFrame:
    DICT_1 = {'mag_mag1' : 'mag_v', 'mag_mag2' : 'mag_i',
          'snr_mag1' : 'snr_v', 'snr_mag2' : 'snr_i', 
          'flag_mag1' : 'flag_v', 'flag_mag2' : 'flag_i', 
          'crowd_mag1' : 'crowd_v', 'crowd_mag2' : 'crowd_i',
          'sharp_mag1' : 'sharp_v', 'sharp_mag2' : 'sharp_i',
          'mag1_err' : 'err_v', 'mag2_err' : 'err_i'}
    data = pd.read_csv(file, header=0, delim_whitespace=True, index_col=False)
    data = data[['extension', 'x', 'y',
                 'mag_mag1', 'mag_mag2', 'type', 
                 'snr_mag1', 'snr_mag2', 
                 'flag_mag1', 'flag_mag2',
                 'crowd_mag1', 'crowd_mag2',
                 'sharp_mag1', 'sharp_mag2',
                 'mag1_err', 'mag2_err']]
    return data.rename(columns=DICT_1)


def read_type2(file : str | Path) -> pd.DataFrame:
    DICT_2 = {'mag_V' : 'mag_v', 'mag_I' : 'mag_i',
          'snr_V' : 'snr_v', 'snr_I' : 'snr_i', 
          'flag_V' : 'flag_v', 'flag_I' : 'flag_i',
          'crowd_V' : 'crowd_v', 'crowd_I' : 'crowd_i',
          'sharp_V' : 'sharp_v', 'sharp_I' : 'sharp_i',
          'Verr' : 'err_v', 'Ierr' : 'err_i'}
    data = pd.read_csv(file, header=0, delim_whitespace=True, index_col=False)
    data = data[['extension', 'x', 'y',
                 'mag_V', 'mag_I', 'type', 
                 'snr_V', 'snr_I',
                 'flag_V', 'flag_I',
                 'crowd_V', 'crowd_I',
                 'sharp_V', 'sharp_I',
                 'Verr', 'Ierr']]
    return data.rename(columns=DICT_2)


def read_type3(file : str | Path) -> pd.DataFrame:
    DICT_3 = {'e': 'extension', 'T' : 'type',
          'X' : 'x', 'Y' : 'y',
          'm_V' : 'mag_v', 'm_I' : 'mag_i', 
          'SN.1' : 'snr_v', 'SN.2' : 'snr_i', 
          'Eflag' : 'flag_v', 'Eflag.1' : 'flag_i',
          'Crowd.1' : 'crowd_v', 'Crowd.2' : 'crowd_i',
          'Sharp.1' : 'sharp_v', 'Sharp.2' : 'sharp_i',
          'm_err' : 'err_v', 'm_err.1' : 'err_i'}
    data = pd.read_csv(file, header=1, delim_whitespace=True, index_col=False)
    data = data.rename(columns={'ex' : 'e', 'X' : 'x', 'Y' : 'y'})

    if 'Crowd.2' in data.columns:
        data = data[['e', 'x', 'y',
                     'm_V', 'm_I', 'T', 
                     'SN.1', 'SN.2',
                     'Eflag', 'Eflag.1',
                     'Crowd.1', 'Crowd.2',
                     'Sharp.1', 'Sharp.2',
                     'm_err', 'm_err.1']]
        return data.rename(columns=DICT_3)

    data = data[['e', 'x', 'y',
                 'm_V', 'm_I', 'T', 
                 'SN.1', 'SN.2',
                 'Eflag', 'Eflag.1',
                 'Sharp.1', 'Sharp.2',
                 'm_err', 'm_err.1']]
    return data.rename(columns=DICT_3)


def read_type4(file : str | Path) -> pd.DataFrame:
    DICT_4 = {'chp': 'extension', 'T' : 'type',
        'X' : 'x', 'Y' : 'y',
        'm_V' : 'mag_v', 'm_I' : 'mag_i', 
        'SN.1' : 'snr_v', 'SN.2' : 'snr_i', 
        'Eflag' : 'flag_v', 'Eflag.1' : 'flag_i',
        'Crowd.1' : 'crowd_v', 'Crowd.2' : 'crowd_i',
        'Sharp.1' : 'sharp_v', 'Sharp.2' : 'sharp_i',
        'Round.1' : 'round_v', 'Round.2' : 'round_i',
        'm_err' : 'err_v', 'm_err.1' : 'err_i'}
    data = pd.read_csv(file, header=1, delim_whitespace=True, index_col=False)
    columns = [col for col in list(DICT_4.keys()) if col in data.columns]
    data = data[columns]
    return data.rename(columns=DICT_4)


def read_type5(file : str | Path) -> pd.DataFrame:
    data = pd.read_csv(file, header=None, delim_whitespace=True, index_col=False)
    data = data.iloc[:, [0, 1, 2, 8, 
                         12, 13, 15, 16, 17, 18, 
                         22, 23, 25, 26, 27, 28]]
    data.columns = ['extension', 'x', 'y', 'type', 
                    'mag_v', 'err_v', 'snr_v', 'sharp_v', 'round_v', 'crowd_v',
                    'mag_i', 'err_i', 'snr_i', 'sharp_i', 'round_i', 'crowd_i']
    return data


def read_type6(file : str | Path) -> pd.DataFrame:
    data = pd.read_csv(file, header=None, delim_whitespace=True, index_col=False)
    data = data.iloc[:, [0, 2, 3, 10, 
                         16, 17, 19, 20, 21, 22, 23, 
                         29, 30, 32, 33, 34, 35, 36]]
    data.columns = ['extension', 'x', 'y', 'type', 
                    'mag_v', 'err_v', 'snr_v', 'sharp_v', 'round_v', 'crowd_v', 'flag_v',
                    'mag_i', 'err_i', 'snr_i', 'sharp_i', 'round_i', 'crowd_i', 'flag_i']
    return data


##########################################################################


def synch_coordinates_wfcs(phot):
    x_array = phot['x'].to_numpy()
    y_array = phot['y'].to_numpy()

    x_rotated = x_array.copy()
    y_rotated = y_array.copy()

    kinda_size_of_a_chip = 750

    for ind, ext in enumerate(phot['extension']):
        if ext == 0:
            x_rotated[ind] = kinda_size_of_a_chip + x_array[ind] * 0.5 - 75
            y_rotated[ind] = kinda_size_of_a_chip + y_array[ind] * 0.5 - 75
        elif ext == 1:
            x_rotated[ind] = kinda_size_of_a_chip - y_array[ind]
            y_rotated[ind] = kinda_size_of_a_chip + x_array[ind] - 100
        elif ext == 2:
            x_rotated[ind] = kinda_size_of_a_chip - x_array[ind]
            y_rotated[ind] = kinda_size_of_a_chip - y_array[ind]
        elif ext == 3:
            x_rotated[ind] = kinda_size_of_a_chip + y_array[ind] - 100
            y_rotated[ind] = kinda_size_of_a_chip - x_array[ind]

    phot['x'] = x_rotated
    phot['y'] = y_rotated

    return phot


def synch_coordinates_acs(phot):
    phot['y'] = phot['y'] + 2100 * (phot['extension'] == 2)
    return phot


def file_has_broken_bites(filename : str | Path) -> bool:
    with open(filename, newline='', encoding='utf-8') as f:
        return '\0' in f.read()


##########################################################################

# another possible readers

def read_type7(file : str | Path) -> pd.DataFrame:
    data = pd.read_csv(file, header=None, delim_whitespace=True, index_col=False)
    data = data.iloc[:, [0, 1, 2, 6, 
                         10, 11, 13, 14, 
                         18, 19, 21, 22]]
    data.columns = ['extension', 'x', 'y', 'type', 
                    'mag_v', 'err_v', 'snr_v', 'sharp_v',
                    'mag_i', 'err_i', 'snr_i', 'sharp_i']
    return data


def read_type8(file : str | Path) -> pd.DataFrame:
    data = pd.read_csv(file, header=None, delim_whitespace=True, index_col=False)
    data = data.iloc[:, [0, 2, 3, 10, 
                         14, 15, 17, 18, 19, 20, 21, 
                         30, 31, 33, 34, 35, 36, 37]]
    data.columns = ['extension', 'x', 'y', 'type', 
                    'mag_v', 'err_v', 'snr_v', 'sharp_v', 'round_v', 'crowd_v', 'flag_v',
                    'mag_i', 'err_i', 'snr_i', 'sharp_i', 'round_i', 'crowd_i', 'flag_i']
    return data


def read_type9(file : str | Path) -> pd.DataFrame:
    DICT = {'Chp': 'extension', 'Type' : 'type',
        'Xpos' : 'x', 'Ypos' : 'y',
        'm_V' : 'mag_v', 'm_I' : 'mag_i', 
        'S/N.1' : 'snr_v', 'S/N.2' : 'snr_i', 
        'Eflag' : 'flag_v', 'Eflag.1' : 'flag_i',
        'Crowd.1' : 'crowd_v', 'Crowd.2' : 'crowd_i',
        'Sharp.1' : 'sharp_v', 'Sharp.2' : 'sharp_i',
        'm_err' : 'err_v', 'm_err.1' : 'err_i'}
    data = pd.read_csv(file, header=1, delim_whitespace=True, index_col=False)
    return data.rename(columns=DICT)


def read_type9(file : str | Path) -> pd.DataFrame:
    data = pd.read_csv(file, header=None, delim_whitespace=True, index_col=False)
    data = data.iloc[:, [0, 1, 2, 8, 
                         12, 13, 15, 16, 17,
                         22, 23, 25, 26, 27]]
    data.columns = ['extension', 'x', 'y', 'type', 
                    'mag_v', 'err_v', 'snr_v', 'sharp_v', 'round_v',
                    'mag_i', 'err_i', 'snr_i', 'sharp_i', 'round_i']
    return data


def read_type10(file : str | Path) -> pd.DataFrame:
    data = pd.read_csv(file, header=None, delim_whitespace=True, index_col=False)
    data = data.iloc[:, [0, 2, 3, 10, 
                         15, 17, 19, 20, 21, 22, 23, 
                         28, 30, 32, 33, 34, 35, 36]]
    data.columns = ['extension', 'x', 'y', 'type', 
                    'mag_v', 'err_v', 'snr_v', 'sharp_v', 'round_v', 'crowd_v', 'flag_v',
                    'mag_i', 'err_i', 'snr_i', 'sharp_i', 'round_i', 'crowd_i', 'flag_i']
    return data
