import numpy as np
from scipy.signal import savgol_filter

def watt_to_dbm(df, target_column = 'power'):
    df['dbm'] = ((10*np.log10(df[target_column]) + 30)) # convert watts to dbm

    # df['dbm'].replace(np.nan, 0.0, inplace=True)
    min_without_inf = np.nanmin(df['dbm'][df['dbm'] != -np.inf])
    df['dbm'].replace(-np.inf, min_without_inf, inplace=True)

# rolling zscore function
def rolling_zscore(df, window = 5):
    std = df.rolling(window = window).std()
    mean = df.rolling(window = window).mean()
    z = (df - mean) / std
    return z

def smooth(df, window = 5, threshold = 1.0, target_column = 'dbm'):
    # std = df.rolling(window = window).std()
    # mean = df.rolling(window = window).mean()
    # print(std, mean)
 
    z = df.apply(lambda x: 0 if np.std(x) == 0 else rolling_zscore(x, window = window))

    print(df['id'].iloc[0], ' - ', df['tx'].iloc[0], '- z_min:', z[target_column].min(), 'z_max:', z[target_column].max())

    mask = (z[target_column] < -threshold) | (z[target_column] > threshold)
    df[f'{target_column}_copy'] = df[target_column]
    df.loc[mask, f'{target_column}_copy'] = np.nan
    df['interpolated'] = df[f'{target_column}_copy'].interpolate(method = 'cubic').astype(float)
    df['interpolated'].fillna(df[target_column], inplace = True)
    df['smooth'] = df[['interpolated']].apply(savgol_filter, window_length=101, polyorder=1)
    df['smooth'].fillna(df['interpolated'], inplace = True)
    df.drop([f'{target_column}_copy', 'interpolated'], axis = 1, inplace = True)
    return df
