# Author
# Juliane Oliveira
# Date: May 14, 2025
# packs

import pandas as pd
import pandas.testing as tm
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import statsmodels.api as sm
import statsmodels.discrete.discrete_model as dm

from patsy import dmatrices
import statsmodels.graphics.tsaplots as tsa


from scipy.fft import fft, ifft, fftfreq

import statsmodels.formula.api as smf
from statsmodels.tsa.stattools import acf

import itertools
from itertools import combinations, chain

from scipy.stats import pearsonr

import re

from datetime import datetime

import pymannkendall as mk

import math

from scipy.stats import friedmanchisquare

from sklearn.metrics import r2_score

from pmdarima.preprocessing import FourierFeaturizer
from pmdarima.datasets import load_wineind
from sklearn.linear_model import LinearRegression


import warnings
warnings.filterwarnings("ignore")


# Set initial functions

def lst_dfs_cities(df, city_code_col='co_ibge', epiweek_date_col='year_week_str', serie_col='unidades_gripal'):
    """
    Create a list of DataFrames for each unique city in the input DataFrame,
    with additional rolling mean and lag features.

    Parameters:
    df (pd.DataFrame): Input DataFrame containing the necessary columns.
    city_code_col (str): Column name for city codes. Default is 'co_ibge'.
    epiweek_date_col (str): Column name for epidemiological week dates. Default is 'year_week_str'.
    serie_col (str): Column name for the series. Default is 'unidades_gripal'.

    Returns:
    list: A list of DataFrames, each corresponding to a unique city, with additional computed columns:
        - 4-week rolling mean.
        - Lagged 1 to 4 weeks of the series.
    """
    lst_dfs_cities = []

    for muni in df[city_code_col].unique():
        # Select and sort data for a specific municipality
        set_muni = df[df[city_code_col] == muni].sort_values(by=epiweek_date_col).copy()

        # Compute rolling mean
        rolling_col_name = f"{serie_col}_4"
        set_muni[rolling_col_name] = set_muni[serie_col].rolling(window=4, min_periods=1).mean()

        # Compute lags 1 to 4
        for lag in range(1, 5):
            lag_col_name = f"{serie_col}_lag_{lag}"
            set_muni[lag_col_name] = set_muni[serie_col].shift(lag)

        lst_dfs_cities.append(set_muni)

    return lst_dfs_cities



############


def final_kendall_negbi(lst_dfs_cities, serie = 'unidades_gripal_4'):
    """
    Perform Mann-Kendall trend analysis and fit Negative Binomial regression for series across multiple DataFrames.

    Parameters:
    - lst_dfs_cities (list of pd.DataFrame): List of city-level DataFrames.
    - serie (str): Column name for the second series (default: 'unidades_gripal_4').

    Returns:
    - pd.DataFrame: Combined DataFrame with Mann-Kendall results and Negative Binomial regression outputs.
    """
    lst_results = []  # Initialize an empty list to store the processed DataFrames

    for dta in lst_dfs_cities:
        # Add a time trend column
        dta = dta.assign(time_trend=np.arange(len(dta)))

        def fit_negative_binomial(series_name):
            if dta[series_name].isnull().all() or np.all(dta[series_name] == 0):
                return {f'coef_negbi_{series_name}': 0,
                        f'std_err_negbi_{series_name}': 0,
                        f'z_negbi_{series_name}': 0,
                        f'p_values_negbi_{series_name}': 1,
                        f'IC_low_negbi_{series_name}': 0,
                        f'IC_high_negbi_{series_name}': 0,
                        f'trend_line_negbi_{series_name}': np.zeros(len(dta))
                        }
            else:
                model = smf.glm(
                formula=f'{series_name} ~ time_trend',
                data=dta,
                family=sm.families.NegativeBinomial(alpha=1)
                ).fit()
                
                return {f'coef_negbi_{series_name}': model.params['time_trend'],
                        f'std_err_negbi_{series_name}': model.bse['time_trend'],
                        f'z_negbi_{series_name}': model.tvalues['time_trend'],
                        f'p_values_negbi_{series_name}': model.pvalues['time_trend'],
                        f'IC_low_negbi_{series_name}': model.conf_int().loc['time_trend', 0],
                        f'IC_high_negbi_{series_name}': model.conf_int().loc['time_trend', 1],
                        f'trend_line_negbi_{series_name}': model.predict(dta)
                        }


        # Fit Negative Binomial regression for both series and update DataFrame
        negbi_serie = fit_negative_binomial(serie)

        dta = dta.assign(**negbi_serie)

        # Append the processed DataFrame to the list
        lst_results.append(dta)

    # Combine all results into a single DataFrame
    final_kendall_negbi = pd.concat(lst_results, ignore_index=True)

    return final_kendall_negbi

#######

def friedman_test(x, freq=None):
    """
    Conducts a Friedman rank test for seasonality in a time series.
    
    Parameters:
    - x: pandas Series or numpy array, the time series data.
    - freq: int, the frequency of the time series.
    
    Returns:
    - dict: Contains test statistic, p-value, and additional information.
    """
    
    # Reshape into matrix form for Friedman test
    rows = len(x) // freq
    if rows < 2:
        raise ValueError("Not enough data points for the specified frequency.")
    
    x = x[:rows * freq]  # Truncate to fit matrix dimensions
    data_matrix = np.reshape(x, (rows, freq))
    
    # Perform Friedman test
    test_stat, p_value = friedmanchisquare(*data_matrix.T)
    
    # Output results
    return {
        "test_stat": test_stat,
        "p_value": p_value,
        "rows": rows,
        "columns": freq
    }

