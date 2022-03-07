import geopandas as gpd
import matplotlib.pyplot as plt
from shapely import wkt
import pandas as pd
import numpy as np
import datetime


# --------------COMPUTE  INDEX --------------------
def create_index_metric(df, weight_dict, metric_name='installed_cap_index'):
    for col in weight_dict.keys():
        df[col] = df[col].replace(np.inf, 0)
    total_weight = 0
    df[metric_name] = 0

    for i, col in enumerate(weight_dict):
        total_weight += weight_dict[col]['weight']
        df[metric_name] += df[col] * weight_dict[col]['indicator_direction'] * weight_dict[col]['weight']
    df[metric_name] = df[metric_name] / total_weight
    df.loc[(df[metric_name] > df[metric_name].quantile(.95)), metric_name] = df[metric_name].quantile(.95)
    # Srtandarization
    # TODO delete this if statement and treat special cases
    metric_range = df[metric_name].max() - df[metric_name].min()
    if metric_range == 0:
        print('\tMetric Range equals 0')
        df[metric_name] = 0
    else:
        df[metric_name] = (df[metric_name] - df[metric_name].min()) / (df[metric_name].max() - df[metric_name].min())
    print(
        f'--------------------------------metric_name: {metric_name}\n\t total_weight: {total_weight} \n\t min: {df[metric_name].min()} \n\t max: {df[metric_name].max()}\n\t mean: {df[metric_name].mean()}')
    return df


def cut_variables(df, col, x):
    qc = df[col].quantile(x)
    qc_range = df[col] > qc
    changed_obs = np.sum(qc_range)
    if changed_obs < len(df) * x:
        df.loc[qc_range, col] = qc
        print(f'\n\tpercentile: {x} \n\tchanged_obs: {changed_obs} \n\tvalue:{qc} ')
    return df


# -------------- CARTO AND COBERTURE--------------------
def areaoverlay_capacity_assignment(df1, df2, df1_key, df2_key, capacity_col):
    ''' Returns the proportion of capacity_col that each df1(municipalities, neighborhood) entry
    covers for all the different df2 entries (influence area).
    Example:
    ----------
    df1  =  Neighborhoods with N neighborhoods
    df2 = Influence area of a particular center with C capacity
    Then we create a new variable capacity_area_assignment inside df1 with
    the capacity C*n_i where n_i is the proportion of the influence area
    covered by the nth neighborhood.

    #*res_union contains all the pieces of the intersections in this case
    #?    - Matches of df1 entry 1 with df2 entry 1
    #?    - Matches of df1 entry 1 with df2 entry 2
    #?    - Matches of df1 entry 2 with df2 entry 2
    #?    - Unmatched  entry 1 of df 1
    #?    - Unmatched  entry 2 of df 1
    #?    - Unmatched  entry 1 of df 2
    #?    - Unmatched  entry 2 of df 2
    #* df_intersection = gpd.overlay(df1_o, df2_o, how='intersection')
    #?    - Matches of df1 entry 1 with df2 entry 1
    #?    - Matches of df1 entry 1 with df2 entry 2
    #?    - Matches of df1 entry 2 with df2 entry 2

    '''
    # It cannot be the case that if a particular center doesn't report cpaacity then
    # we don't share it's influence area. That's why we imputed 1 if the re not reported value
    # or if the capacity it's equal to 0
    df2.loc[df2[capacity_col] <= 0, capacity_col] = 1
    df1_o = df1[[df1_key, 'geometry']].copy()
    df2_o = df2[[df2_key, 'geometry']].copy()  # Influence area with capacity

    # We want to distribute the initial influence area into the differnet neighborhoods that overposition it
    df_intersection = gpd.overlay(df1_o, df2_o, how='intersection')
    df_intersection['intersection_area'] = df_intersection.area
    # A new influence area must be calculated since not all the influence areas match when we calculate the gdf.overlay
    df_intersection_newinfarea = df_intersection.groupby(df2_key).sum().reset_index()
    df_intersection_newinfarea = df_intersection_newinfarea.rename(columns={'intersection_area': 'new_influence_area'})

    # We are passiing the new_influence area at a df2_key (influence area 'id') aggregated level
    df_intersection = df_intersection.merge(df_intersection_newinfarea[[df2_key, 'new_influence_area']], on=df2_key)
    # Calaculate the proportions on the new influence area, this assignment should sum 1
    df_intersection['proportion_newinfluence_area'] = df_intersection['intersection_area'] / df_intersection[
        'new_influence_area']

    # Now we are just passing the total capacity of the influence area to calculate the assigned capacity to the neighborhood
    df_intersection_capacity = df_intersection.merge(df2[[df2_key, capacity_col]], on=df2_key).reset_index()
    df_intersection_capacity['capacity_area_assignment'] = df_intersection_capacity['proportion_newinfluence_area'] * \
                                                           df_intersection_capacity[capacity_col]

    # Check that the capacities match
    total_capacity_assignment = df_intersection_capacity['capacity_area_assignment'].sum()
    total_capacity_df2 = df2[capacity_col].sum()
    diff = np.abs((total_capacity_df2 - total_capacity_assignment) / total_capacity_assignment)
    if diff > .000001:
        # Something wrong happened while distributing the influence areas into the neighborhood or municipalities
        raise ValueError(
            'ERROR: geo_functions/areaoverlay_capacity_assignment: Total capacity assigment do not match initial capacity')

    del df_intersection_capacity[capacity_col]
    df_intersection_capacity = df_intersection_capacity.rename(columns={'capacity_area_assignment': capacity_col})
    # Now we just want to return a single observation for each df1_key (municipality/neighborhood)
    df_out = df_intersection_capacity.groupby(df1_key)[capacity_col].sum().reset_index()
    # raise ValueError()
    return df_out


# --------------FEATURE ENGINEERING --------------------
def process_quotients(df, df_transform):
    for i in df_transform.index:
        new_var = df_transform.loc[i, 'variable_processed']
        denominator = df_transform.loc[i, 'indicator_variable']
        numerator = df_transform.loc[i, 'raw_name']
        print(
            f'\n------------------\nprocess_quotients -> new_var: {new_var} \n\tnumerator: {numerator} \n\tdenominator: {denominator}')
        if denominator == '1':
            df[new_var] = df[numerator]
        else:
            df_n0 = df[df[denominator] > 0]
            max_cover = np.max(df_n0[numerator] / df_n0[denominator])

            df[new_var] = 0
            df.loc[(df[denominator] == 0) & (df[numerator] > 0), new_var] = max_cover
            df.loc[(df[denominator].isna()) & ((df[numerator] > 0)), new_var] = max_cover
            df.loc[df[numerator].isna(), new_var] = 0
            df.loc[(df[denominator] > 0) & (~df[numerator].isna()), new_var] = df_n0[numerator] / df_n0[denominator]

            if denominator in ['TVIVPARHAB', 'P_3YMAS', 'P_8A14', 'P_18YMAS_F']:
                outliers = df[new_var] > 1
                outliers_n = outliers.sum()
                outliers_pct = outliers_n / len(df)
                df.loc[outliers, new_var] = 1
                if outliers_pct > .3:
                    # This scenario only happes for viviendas habitadas (TVIVPARHAB) since it's subreported in Censo Poblacion y vivienda 2020
                    raise ValueError(f'ERROR: More than 15 pct of percentage values are above 1 {new_var}')

            else:
                outliers = df[new_var] > 1
                outliers_n = outliers.sum()
                outliers_pct = outliers_n / len(df)
                df.loc[(outliers), new_var] = 1
                if outliers_n > 1:
                    # This scenario only happes for viviendas habitadas (TVIVPARHAB) since it's subreported in Censo Poblacion y vivienda 2020
                    raise ValueError(f'ERROR: There are incorrect pct values for {new_var}  ')

            print(f'\toutliers: {outliers_n} \t toutliers pct: {outliers_pct}')

            # q95 = df[new_var].quantile(.95)
            # if q95>0:
            #    df.loc[(df[new_var] > df[new_var].quantile(.95)), new_var] = q95
            # df[ new_var ] = (df[ new_var ] - df[ new_var ].min())/df[ new_var ].max()
            print(f'\tmax_cover: {max_cover} \n\tmean: {df[new_var].mean()}')

    df_out = df.copy()
    return df_out


def process_any(df, transform_dict):
    for k in transform_dict.keys():
        new_var = k
        selectedcol = transform_dict[k][0]
        selectedvalue = transform_dict[k][1]
        print(
            f'\n------------------\nprocess_any -> new_var: {new_var} \nselectedcol: {selectedcol} \n\tselectedvalue: {selectedvalue}')
        df[new_var] = 0
        df.loc[(df[selectedcol] == selectedvalue), new_var] = 1
    df_out = df.copy()
    return df_out


def process_sum(df, transform_dict):
    for k in transform_dict.keys():
        new_var = k
        sumcolumns = transform_dict[k]
        print(f'\n------------------\nprocess_sum -> new_var: {new_var} \n\tsumcolumns: {sumcolumns}')
        df[new_var] = 0
        for col in sumcolumns:
            df[new_var] = df[new_var] + df[col]
    df_out = df.copy()
    return df_out


def process_diff(df, df_transform):
    for i in df_transform.index:
        new_var = df_transform.loc[i, 'variable_processed']
        diff_variable_1 = df_transform.loc[i, 'diff_variable_1']
        diff_variable_2 = df_transform.loc[i, 'diff_variable_2']

        print(
            f'\n------------------\nprocess_diff -> new_var: {new_var} \n\tdiff_variable_1: {diff_variable_1} \n\t diff_variable_2: {diff_variable_2}')
        df[new_var] = df[diff_variable_1] - df[diff_variable_2]
    df_out = df.copy()
    return df_out


def review_data(df, name):
    df_desc = df.describe()
    print(
        f'\n\n\n--------------------------------------------\ndf -> {name}\n\tdf.shape: {df.shape}\n\tdf.describe: {df_desc}\n\tdf.head: \n{df.head(2)}')


def create_id(df, cols, new_id_name):
    '''
    Generates a new column that serves as a new index of the dataframe.
    The column name is 'new_id_name' of the dataframe 'df.
    The cols_dict keys contains the column names of df to concatenate as strings in order to generate the index.
    The value of the keys contains the number of zeros to the left to generate the index in each column.
    '''
    # Remove pointer from database.
    df = df.copy()
    
    # Zeros to the left to generate index
    index_cols_zeros_dict = {'ENT': 2, 'MUN': 3, 'LOC': 4, 'AGEB': 4, 'MZA': 3, 'id_mun':5, 'id_loc':8,}
    df_cols = list(df.columns)
    
    df[new_id_name] = ''
    for col in cols:
        nl = index_cols_zeros_dict[col]
        df[new_id_name] += df[col].astype(str).str.zfill(nl)

    cols_reorder = [new_id_name] + df_cols
    df = df[cols_reorder]
    return df



def pivot_testing(df, index, columns, values='ones'):
    # Pivot table with margins to undesrtand distribution across categories.
    test = pd.pivot_table(df, \
                          values=values, \
                          index=index, \
                          columns=columns, \
                          aggfunc=np.sum, fill_value=0, \
                          margins=True)
    return (test)


def get_unpairing_obs(df1, df2, merge_key):
    df = df1.merge(df2, on=merge_key, how='outer', indicator=True)
    onl = np.sum(df._merge == 'left_only')
    onr = np.sum(df._merge == 'right_only')
    both = np.sum(df._merge == 'both')
    print(
        f'Only left elements: {onl}, \nOnly right elements: {onr} \nBoth sides elements: {both} \ndf merge shape: {df.shape} \ndf left shape: {df1.shape} \ndf right shape: {df2.shape}')
    return df


# def to_gdf(df):
#     df_c = df.copy()
#     df_c['geometry'] = df_c['geometry'].apply(wkt.loads)
#     df_c = gpd.GeoDataFrame(df_c, crs="EPSG:4326", geometry='geometry')
#     return df_c


def print_values(df):
    for col in df.columns:
        print(col)
        unique_vals = df[col].unique()
        if len(unique_vals) < 20:
            print('\t', unique_vals)
        else:
            print('total unique values: ', len(unique_vals))


def non_repeated_columns(df1: pd.DataFrame, df2: pd.DataFrame, keep_columns: list = [], exclude_columns: list = [],
                         include_geometry: bool = False):
    '''
    Returns a list with the names of the columns that apear in two dataframes
    with the same name. This is useful for merging with sjoin that dont have the
    pd.merge() functionality

    keep_columns (list):
    include_geometry (bool) : If True include 'geometry column for sjoin'

    '''
    t1 = df1.columns
    t2 = df2.columns
    c2_n = [x for x in t2 if x not in t1]
    if include_geometry:
        c2_n.append('geometry')

    for k in keep_columns:
        if k in c2_n:
            continue
        else:
            c2_n.append(k)

    for e in exclude_columns:
        if e in c2_n:
            c2_n.remove(e)

    return c2_n


def shw(gdf):
    cdisplay(gdf.head(2), \
             gdf.shape, \
             gdf.plot())


def get_notmatching(gdf1, gdf2, col='neighborhood_key'):
    '''
    col (str): must be a column that exists only in gdf2 so it returns NaN
    whenever we are unable to find an sjoin match for the associated
    geometry variable in gf2
    '''
    gdf_join = gpd.sjoin(gdf1, gdf2, how='left', op='within')

    return gdf_join[gdf_join[col].isna()]


# shw(lostcaipi_neig_gdf)
def plot_maps(gdf1, gdf2, gdf3=None, gdf4=None, gdf5=None):
    fig, ax1 = plt.subplots(figsize=(5, 3.5))
    gdf1.plot(ax=ax1)
    try:
        gdf2.plot(ax=ax1, color='red')
    except Exception as e:
        print('')
    try:
        gdf3.plot(ax=ax1, color='green')
    except Exception as e:
        print('')

    try:
        gdf4.plot(ax=ax1, color='purple')
    except Exception as e:
        print('')

    try:
        gdf5.plot(ax=ax1, color='orange')
    except Exception as e:
        print('')


def replace_nonreported(df):
    for col in df.columns:
        # No mostrada por privacidad entre 1 y 3.
        df[col] = df[col].replace('*', -1)
        # No mostrada porque no se pudieron recopilar datos.
        df[col] = df[col].replace('N/D', -1)
    return df

# Zona Metrpolitana municipio id's.
zm_id_mun = ['14039', '14120', '14098', '14101', '14097', '14070', '14044', '14051', '14124', '14002']
