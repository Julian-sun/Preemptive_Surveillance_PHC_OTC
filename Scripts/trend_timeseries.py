# Read packedges

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

import functions

from datetime import datetime

import pymannkendall as mk

import math

# Lets start

##############################################################################################
##############################################################################################
##############################################################################################

def summary_trend_kendall(lst_dfs_cities, serie1='atend_ivas_4', serie2='unidades_gripal_4'): 
    """
    Perform Mann-Kendall trend analysis for two series across multiple DataFrames and summarize the results.

    Parameters:
    - lst_dfs_cities (list of pd.DataFrame): List of DataFrames, each representing city-level data.
    - serie1 (str): Column name for the first time series (default: 'atend_ivas_4').
    - serie2 (str): Column name for the second time series (default: 'unidades_gripal_4').

    Returns:
    - result_kendall (pd.DataFrame): Combined DataFrame of trend analysis results for all cities.
    - summary_table (pd.DataFrame): Summary table with counts and percentages of trend classifications.
    """
    lst1 = []  # Initialize an empty list to store the results
    
    for dta in lst_dfs_cities:
        # Extract metadata from the first row
        uf = dta.co_uf.iloc[0]
        nome_uf = dta.nm_uf.iloc[0]
        co_ibge7 = dta['co_ibge7'].iloc[0]
        muni_name = dta['nm_municipio'].iloc[0]
        co_ibge = dta.co_ibge.iloc[0]

        # Perform Mann-Kendall tests
        resultado_serie1 = mk.original_test(dta[serie1].to_numpy())
        resultado_serie2 = mk.original_test(dta[serie2].to_numpy())

        # Create a dictionary of results
        data = {
            'co_uf': uf,
            'nm_uf': nome_uf,
            'co_ibge7': co_ibge7,
            'muni_name': muni_name,
            'co_ibge': co_ibge,
            f'trend_{serie1}': resultado_serie1.trend,
            f'trend_p_value_{serie1}': resultado_serie1.p,
            f'trend_z_{serie1}': resultado_serie1.z,
            f'trend_slope_{serie1}': resultado_serie1.slope,
            f'trend_{serie2}': resultado_serie2.trend,
            f'trend_p_value_{serie2}': resultado_serie2.p,
            f'trend_z_{serie2}': resultado_serie2.z,
            f'trend_slope_{serie2}': resultado_serie2.slope
        }

        # Append the result to the list
        lst1.append(data)

    # Combine all results into a single DataFrame
    result_kendall = pd.DataFrame(lst1)

    # Define significance level for trends
    significance_level = 0.05

    # Initialize counts for trend classifications
    counts = {
        f'{serie1} Trend Only': 0,
        f'{serie2} Trend Only': 0,
        "Both Significant": 0,
        "Both Increase Together": 0,
        "Both Decrease Together": 0,
        "Increase in One, Decrease in Other": 0,
        "No Significant Trend in Both": 0,
    }

    # Classify trends for each city
    for _, row in result_kendall.iterrows():
        serie1_significant = row[f"trend_p_value_{serie1}"] < significance_level
        serie2_significant = row[f"trend_p_value_{serie2}"] < significance_level

        serie1_direction = row[f"trend_{serie1}"]
        serie2_direction = row[f"trend_{serie2}"]

        if serie1_significant and not serie2_significant:
            counts[f"{serie1} Trend Only"] += 1
        elif serie2_significant and not serie1_significant:
            counts[f"{serie2} Trend Only"] += 1
        elif serie1_significant and serie2_significant:
            counts["Both Significant"] += 1
            if serie1_direction == "increasing" and serie2_direction == "increasing":
                counts["Both Increase Together"] += 1
            elif serie1_direction == "decreasing" and serie2_direction == "decreasing":
                counts["Both Decrease Together"] += 1
            else:
                counts["Increase in One, Decrease in Other"] += 1
        else:
            counts["No Significant Trend in Both"] += 1

    # Total number of rows
    total = len(result_kendall)
    counts["Total"] = total

    # Calculate percentages
    percentages = {key: (value / total) * 100 for key, value in counts.items() if key != "Total"}

    # Create a summary table
    summary_table = pd.DataFrame({
        "Trend Analysis": list(counts.keys()),
        "Count": list(counts.values()),
        "% of Total": [percentages.get(key, 100) for key in counts.keys()],
    })

    # Descriptive text
    text = (
        "Description of Categories:\n"
        "Series 1 Trend Only: Cases where only the PHC time series has a significant trend.\n"
        "OTC Trend Only: Cases where only the OTC time series has a significant trend.\n"
        "Series 1 Significant: Cases where both series have significant trends.\n"
        "Both Increase Together: Cases where both PHC and OTC have significant increasing trends.\n"
        "Both Decrease Together: Cases where both PHC and OTC have significant decreasing trends.\n"
        "Increase in One, Decrease in Other: Cases where the trends are significant but in opposite directions.\n"
        "No Significant Trend in Both: Cases where neither time series has a significant trend."
    )
    
    return result_kendall, summary_table, text

#############################################################################################

def summary_trend_kendall_imed(lst_dfs_cities,serie1='atend_ivas_4', serie2='unum_otc_ivas_4'): 
    """
    Perform Mann-Kendall trend analysis for two series across multiple DataFrames and summarize the results.

    Parameters:
    - lst_dfs_cities (list of pd.DataFrame): List of DataFrames, each representing city-level data.
    - serie1 (str): Column name for the first time series (default: 'atend_ivas_4').
    - serie2 (str): Column name for the second time series (default: 'unidades_gripal_4').

    Returns:
    - result_kendall (pd.DataFrame): Combined DataFrame of trend analysis results for all cities.
    - summary_table (pd.DataFrame): Summary table with counts and percentages of trend classifications.
    """
    lst1 = []  # Initialize an empty list to store the results
    
    for dta in lst_dfs_cities:
        # Extract metadata from the first row
        uf = dta.co_uf.iloc[0]
        nome_uf = dta.nm_uf.iloc[0]
        co_imed = dta['co_imed'].iloc[0]
        Nome_imed = dta['Nome_imed'].iloc[0]

        # Perform Mann-Kendall tests
        resultado_serie1 = mk.original_test(dta[serie1].to_numpy())
        resultado_serie2 = mk.original_test(dta[serie2].to_numpy())

        # Create a dictionary of results
        data = {
            'co_uf': uf,
            'nm_uf': nome_uf,
            'co_imed': co_imed,
            'Nome_imed': Nome_imed,
            f'trend_{serie1}': resultado_serie1.trend,
            f'trend_p_value_{serie1}': resultado_serie1.p,
            f'trend_z_{serie1}': resultado_serie1.z,
            f'trend_slope_{serie1}': resultado_serie1.slope,
            f'trend_{serie2}': resultado_serie2.trend,
            f'trend_p_value_{serie2}': resultado_serie2.p,
            f'trend_z_{serie2}': resultado_serie2.z,
            f'trend_slope_{serie2}': resultado_serie2.slope
        }

        # Append the result to the list
        lst1.append(data)

    # Combine all results into a single DataFrame
    result_kendall = pd.DataFrame(lst1)

    # Define significance level for trends
    significance_level = 0.05

    # Initialize counts for trend classifications
    counts = {
        f'{serie1} Trend Only': 0,
        f'{serie2} Trend Only': 0,
        "Both Significant": 0,
        "Both Increase Together": 0,
        "Both Decrease Together": 0,
        "Increase in One, Decrease in Other": 0,
        "No Significant Trend in Both": 0,
    }

    # Classify trends for each city
    for _, row in result_kendall.iterrows():
        serie1_significant = row[f"trend_p_value_{serie1}"] < significance_level
        serie2_significant = row[f"trend_p_value_{serie2}"] < significance_level

        serie1_direction = row[f"trend_{serie1}"]
        serie2_direction = row[f"trend_{serie2}"]

        if serie1_significant and not serie2_significant:
            counts[f"{serie1} Trend Only"] += 1
        elif serie2_significant and not serie1_significant:
            counts[f"{serie2} Trend Only"] += 1
        elif serie1_significant and serie2_significant:
            counts["Both Significant"] += 1
            if serie1_direction == "increasing" and serie2_direction == "increasing":
                counts["Both Increase Together"] += 1
            elif serie1_direction == "decreasing" and serie2_direction == "decreasing":
                counts["Both Decrease Together"] += 1
            else:
                counts["Increase in One, Decrease in Other"] += 1
        else:
            counts["No Significant Trend in Both"] += 1

    # Total number of rows
    total = len(result_kendall)
    counts["Total"] = total

    # Calculate percentages
    percentages = {key: (value / total) * 100 for key, value in counts.items() if key != "Total"}

    # Create a summary table
    summary_table = pd.DataFrame({
        "Trend Analysis": list(counts.keys()),
        "Count": list(counts.values()),
        "% of Total": [percentages.get(key, 100) for key in counts.keys()],
    })

    # Descriptive text
    text = (
        "Description of Categories:\n"
        "Series 1 Trend Only: Cases where only the PHC time series has a significant trend.\n"
        "OTC Trend Only: Cases where only the OTC time series has a significant trend.\n"
        "Series 1 Significant: Cases where both series have significant trends.\n"
        "Both Increase Together: Cases where both PHC and OTC have significant increasing trends.\n"
        "Both Decrease Together: Cases where both PHC and OTC have significant decreasing trends.\n"
        "Increase in One, Decrease in Other: Cases where the trends are significant but in opposite directions.\n"
        "No Significant Trend in Both: Cases where neither time series has a significant trend."
    )
    
    return result_kendall, summary_table, text



##############################################################################################
##############################################################################################
##############################################################################################

def final_kendall_negbi(lst_dfs_cities, serie1='atend_ivas_4', serie2='unidades_gripal_4'):
    """
    Perform Mann-Kendall trend analysis and fit Negative Binomial regression for two series across multiple DataFrames.

    Parameters:
    - lst_dfs_cities (list of pd.DataFrame): List of city-level DataFrames.
    - serie1 (str): Column name for the first series (default: 'atend_ivas_4').
    - serie2 (str): Column name for the second series (default: 'unidades_gripal_4').

    Returns:
    - pd.DataFrame: Combined DataFrame with Mann-Kendall results and Negative Binomial regression outputs.
    """
    lst_results = []  # Initialize an empty list to store the processed DataFrames

    for dta in lst_dfs_cities:
        # Add a time trend column
        dta = dta.assign(time_trend=np.arange(len(dta)))

        # Run Mann-Kendall tests for both series
        resultado_serie1 = mk.original_test(dta[serie1].to_numpy())
        resultado_serie2 = mk.original_test(dta[serie2].to_numpy())

        # Add Mann-Kendall results to the DataFrame
        dta = dta.assign(
            trend_kendall_serie1=resultado_serie1.trend,
            trend_p_value_kendall_serie1=resultado_serie1.p,
            trend_z_kendall_serie1=resultado_serie1.z,
            trend_slope_kendall_serie1=resultado_serie1.slope,
            trend_line_kendall_serie1=resultado_serie1.intercept + resultado_serie1.slope * dta['time_trend'],
            trend_kendall_serie2=resultado_serie2.trend,
            trend_p_value_kendall_serie2=resultado_serie2.p,
            trend_z_kendall_serie2=resultado_serie2.z,
            trend_slope_kendall_serie2=resultado_serie2.slope,
            trend_line_kendall_serie2=resultado_serie2.intercept + resultado_serie2.slope * dta['time_trend']
        )

        # Function to handle Negative Binomial regression
        def fit_negative_binomial(series_name):
            if dta[series_name].isnull().all() or np.all(dta[series_name] == 0):
                return {
                    f'coef_negbi_{series_name}': 0,
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
                return {
                    f'coef_negbi_{series_name}': model.params['time_trend'],
                    f'std_err_negbi_{series_name}': model.bse['time_trend'],
                    f'z_negbi_{series_name}': model.tvalues['time_trend'],
                    f'p_values_negbi_{series_name}': model.pvalues['time_trend'],
                    f'IC_low_negbi_{series_name}': model.conf_int().loc['time_trend', 0],
                    f'IC_high_negbi_{series_name}': model.conf_int().loc['time_trend', 1],
                    f'trend_line_negbi_{series_name}': model.predict()
                }

        # Fit Negative Binomial regression for both series and update DataFrame
        negbi_serie1 = fit_negative_binomial(serie1)
        negbi_serie2 = fit_negative_binomial(serie2)

        dta = dta.assign(**negbi_serie1, **negbi_serie2)

        # Append the processed DataFrame to the list
        lst_results.append(dta)

    # Combine all results into a single DataFrame
    final_kendall_negbi = pd.concat(lst_results, ignore_index=True)

    return final_kendall_negbi


    