import geopandas as gpd
import glob
import jal.utils.config as cf
import numpy as np
import pandas as pd
import subprocess

city_dct = {'09':'09_ciudaddemexico', '11':'11_guanajuato', '14':'14_jalisco'}

def load_mza(city=None):
    city = f'{city:02d}'
    filedir = str(cf.DATA_DIR)+'/03_surveys/marco_geoestadistico/'+city_dct[city]+'/conjunto_de_datos/'+city+'m.shp'
    print(filedir)
    mza = gpd.read_file(filedir)
    mza.rename({'CVEGEO': 'MZA_ID'}, axis=1, inplace=True)
    mza.set_index('MZA_ID', inplace=True, drop=True)
    mza = mza.to_crs(epsg=4326)
    return mza

def load_censo(year=None, city=None):
    year = str(year)
    city = f'{city:02d}'
    filedir = str(cf.DATA_DIR) + '/03_surveys/censo/' + year + '/' + city_dct[city] + '/'
    censo, censo_path = feather_load(filedir, raw_format='csv', feather_write=False)

    if 'feather' not in censo_path:
        # remove all non block rows (AGEB, localidad, ...)
        censo = censo[censo.MZA != '000']
        censo.reset_index(inplace=True, drop=True)
        # Generate AGEBs and MZA IDs
        censo['MUN'] = censo['MUN'].apply(lambda x: x.zfill(3))
        censo['LOC'] = censo['LOC'].apply(lambda x: x.zfill(4))
        censo['AGEB'] = censo['AGEB'].apply(lambda x: x.zfill(4))
        censo['MZA'] = censo['MZA'].apply(lambda x: x.zfill(3))
        censo['LOC_ID'] = censo['ENTIDAD'] + censo['MUN'] + censo['LOC']
        censo['AGEB_ID'] = censo['LOC_ID'] + censo['AGEB']
        censo['MZA_ID'] = censo['AGEB_ID'] + censo['MZA']
        censo['POBTOT'] = to_float(censo['POBTOT'])
        censo_pob_loc = censo[['LOC_ID', 'POBTOT']].groupby(['LOC_ID']).sum()
        censo= censo.join(censo_pob_loc, on='LOC_ID', how='left', rsuffix='_loc')

        censo_path = censo_path.split('.')[0] + '.feather'
        censo.to_feather(censo_path)
    return censo

def load_zap(city=None):
    city = f'{city:02d}'
    filedir = str(cf.DATA_DIR) + '/03_surveys/ZAP/' + city_dct[city] + '/'
    zap, zap_path = feather_load(filedir, raw_format='xlsx', feather_write=False)
    zap = zap[zap['CLAVE DE ENTIDAD FEDERATIVA'] == city]
    zap.reset_index(inplace=True, drop=True)
    zap.rename({'CLAVE DE AGEB': 'AGEB_ID'}, inplace=True, axis=1)
    zap['AGEB_ID'] = zap['AGEB_ID'].astype('str')
    zap_path = zap_path.split('.')[0] + '.feather'
    zap.to_feather(zap_path)
    return zap

def load_master(year=None, city=None):
    mza = load_mza(city=city)
    zap = load_zap(city=city)
    zap.set_index('AGEB_ID', inplace=True, drop=True)
    censo = load_censo(year=year, city=city)
    master = gpd.GeoDataFrame(censo)
    master = master.join(mza, on='MZA_ID', how='left')
    master = master.join(zap, on='AGEB_ID', how='left')
    return master

def feather_load(filedir, raw_format='csv', feather_write=False):
    feather_in_folder = sorted(glob.glob(f'{filedir}/*.feather'))
    read_raw_format = False
    try:
        filepath = feather_in_folder[-1]
    except:
        print("No feather file found in folder: " + filedir)
        read_raw_format = True
    # We obtain the data of last modification of feather file and folder and write it as an integer for comparison
    if not read_raw_format:
        ls_call = subprocess.run(["date", "-r", filepath, "+%Y%m%d%H%M%S"], stdout=subprocess.PIPE, text=True)
        feather_mod = int(ls_call.stdout)
        filepath_load = cf.SRC_DIR / 'utils/load_database.py'
        ls_call = subprocess.run(["date", "-r", filepath_load, "+%Y%m%d%H%M%S"], stdout=subprocess.PIPE, text=True)
        load_mod = int(ls_call.stdout)
        if load_mod > feather_mod:
            print("The load_files.py file has been updated recently so an updated feather file will be generated.")
            read_raw_format = True

    if read_raw_format:
        format_in_folder = sorted(glob.glob(f'{filedir}/*.' + raw_format))
        try:
            filepath = format_in_folder[-1]
        except:
            print('No files with format ' + raw_format + ' found in folder: ' + filedir)
            database, filepath = None, None

        if filepath != None:
            print('Reading newest ' + raw_format +' : ' + filepath + ': ', end='')
            if raw_format == 'csv':
                database = pd.read_csv(filepath, low_memory=False, dtype=str)
                print('done')
            elif raw_format == 'xlsx':
                database = pd.read_excel(filepath, dtype=str, sheet_name=1)
                print('done')
            elif raw_format == 'feather':
                database, filepath = None, None
                print('done')
            else:
                print('Cannot read this format')
                exit()

        if feather_write:
            filepath = filepath.split('.')[0] + '.feather'
            database.to_feather(filepath)
    else:
        print("Reading feather file: " + filepath + ": ", end='')
        database = pd.read_feather(filepath)
        print("done")
    return database, filepath

def to_float(col):
    if type(col[0]) == str:
        col = col.str.replace('*', '-2')
        col = col.str.replace('N/D', '-1')
        col = col.astype(float)
    col.replace({-1:np.nan}, inplace=True)
    return col