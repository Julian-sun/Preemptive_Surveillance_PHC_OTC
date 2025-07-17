# Author
# Juliane Oliveira


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


########### Start the game

def year_week(y, w):
    """
    Convert year and week number to a datetime object representing the first day of that week.

    Parameters:
    y (int or str): The year.
    w (int or str): The ISO week number.

    Returns:
    datetime: A datetime object for the first day of the specified week.
    """
    return datetime.strptime(f'{y} {w} 1', '%G %V %w')


def year_week_ts(df, epiweek_col='epiweek', year_col='ano'):
    """
    Add columns to the DataFrame for ISO year-week, a datetime object for the first day of that week,
    and a string representation of the year-week.

    Parameters:
    df (pd.DataFrame): The input DataFrame containing year and week columns.
    epiweek_col (str): The name of the column in df representing the week number. Default is 'epiweek'.
    year_col (str): The name of the column in df representing the year. Default is 'ano'.

    Returns:
    pd.DataFrame: The DataFrame with additional columns:
        - 'year_week': ISO year-week in 'YYYY-WW' format.
        - 'year_week_ts': datetime object for the first day of the specified week.
        - 'year_week_str': String representation of the year-week in 'YYYY-WW' format.
    """
    # Create the 'year_week' column in the format 'YYYY-WW'
    df['year_week'] = df.apply(
        lambda row: f"{int(row[year_col])}-{int(row[epiweek_col]):02d}", axis=1
    )

    # Create the 'year_week_ts' column using the year_week function
    df['year_week_ts'] = df.apply(
        lambda row: year_week(row[year_col], row[epiweek_col]), axis=1
    )

    # Create the 'year_week_str' column as a string in 'YYYY-WW' format
    df['year_week_str'] = df['year_week_ts'].apply(
        lambda x: x.strftime('%G-%V')
    )

    return df


# Create list of cities dataframe
def lst_dfs_cities(df, city_code_col='co_ibge', epiweek_date_col='year_week_str', ili_col='atend_ivas', otc_col='unidades_gripal'):
    """
    Create a list of DataFrames for each unique city in the input DataFrame,
    with additional rolling mean and lag features.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame containing the necessary columns.
    city_code_col (str): Column name for city codes. Default is 'co_ibge'.
    epiweek_date_col (str): Column name for epidemiological week dates. Default is 'year_week_str'.
    ili_col (str): Column name for ILI (Influenza-Like Illness) counts. Default is 'atend_ivas'.
    otc_col (str): Column name for OTC (Over-The-Counter) medication counts. Default is 'unidades_gripal'.

    Returns:
    list: A list of DataFrames, each corresponding to a unique city, with additional computed columns:
        - Rolling mean and log-transformed columns for 'ili_col' and 'otc_col' over a 4-week window.
        - Lagged values and their logarithms for 'ili_col' and 'otc_col' for lags 1 to 4 weeks.
    """
    lst_dfs_cities = []
    
    for muni in df[city_code_col].unique():
        # Select data for a specific municipality
        set_muni = df[df[city_code_col] == muni]

        # Sort by the epidemiological week date column
        set_muni = set_muni.sort_values(by=epiweek_date_col)

        # Calculate 4-week rolling means and rename columns
        ili_4_col = f'{ili_col}_4'
        otc_4_col = f'{otc_col}_4'
        set_muni = set_muni.assign(ili_4=set_muni[ili_col].rolling(window=4).mean())
        set_muni = set_muni.rename(columns={'ili_4': ili_4_col})
        set_muni = set_muni.assign(otc_4=set_muni[otc_col].rolling(window=4).mean())
        set_muni = set_muni.rename(columns={'otc_4': otc_4_col})
        
        # Calculate 8-week rolling means and rename columns
        ili_8_col = f'{ili_col}_8'
        otc_8_col = f'{otc_col}_8'
        set_muni = set_muni.assign(ili_8=set_muni[ili_col].rolling(window=8).mean())
        set_muni = set_muni.rename(columns={'ili_8': ili_8_col})
        set_muni = set_muni.assign(otc_8=set_muni[otc_col].rolling(window=8).mean())
        set_muni = set_muni.rename(columns={'otc_8': otc_8_col})

        # Calculate logarithms of rolling means and rename columns
        #set_muni = set_muni.assign(ili_4_log=np.log(set_muni[ili_4_col]))
        #set_muni = set_muni.rename(columns={'ili_4_log': f'{ili_col}_4_log'})
        #set_muni = set_muni.assign(otc_4_log=np.log(set_muni[otc_4_col]))
        #set_muni = set_muni.rename(columns={'otc_4_log': f'{otc_col}_4_log'})

        # Calculate lagged values and their logarithms
        for lag in range(1, 5):
            set_muni = set_muni.assign(**{f'{otc_col}_4_lag_{lag}': set_muni[otc_4_col].shift(lag)})
            #set_muni = set_muni.assign(**{f'{otc_col}_4_log_lag_{lag}': np.log(set_muni[f'{otc_col}_4_lag_{lag}'])})
            
            set_muni = set_muni.assign(**{f'{ili_col}_4_lag_{lag}': set_muni[ili_4_col].shift(lag)})
            #set_muni = set_muni.assign(**{f'{ili_col}_4_log_lag_{lag}': np.log(set_muni[f'{ili_col}_4_lag_{lag}'])})

        # Fill NaN values with 0
        set_muni = set_muni.fillna(0)

        # Append the DataFrame to the list
        lst_dfs_cities.append(set_muni)
        
    return lst_dfs_cities

#Harmonic terms

# Function to perform FFT and reconstruct the time series
def best_fft_reconstruction(data_values, period=52):
    # Convert to numpy array for FFT
    S = np.array(data_values)

    # Perform FFT
    N = period
    T = 1.0  # Assuming the time step is 1 (weekly data)
    yf = fft(S)

    best_S_reconstructed = None
    best_tsig = None
    yf_filtered_best = None

    for tsig in [0.15, 0.14, 0.13, 0.12, 0.11, 0.1, 0.09, 0.08]: #0.07, 0.06, 0.05
        # Threshold for significant components (can be adjusted)
        threshold = np.max(np.abs(yf)) * tsig

        # Zero out insignificant components
        yf_filtered = yf.copy()
        yf_filtered[np.abs(yf) < threshold] = 0

        # Perform inverse FFT to reconstruct the time series
        S_reconstructed = ifft(yf_filtered).real

        # Update the best reconstruction if variance is greater than 0
        if S_reconstructed.var() > 0.001:
            best_S_reconstructed = S_reconstructed
            best_tsig = tsig
            yf_filtered_best = yf_filtered
            break

    return best_S_reconstructed, best_tsig, yf_filtered_best

# Define the Fourier series function based on the extracted coefficients
def fourier_series(t, a0, an, bn, N):
    result = a0
    for n, (a, b) in enumerate(zip(an, bn), start=1):
        result += a * np.cos(2 * np.pi * n * t / N) + b * np.sin(2 * np.pi * n * t / N)
    return result

def harmonic(list_dfs,col):
    """
    Add harmonic features to each DataFrame in the input list.
    
    Parameters:
    list_dfs (list): List of DataFrames to be processed.
    
    Returns:
    list: The input list with each DataFrame containing additional harmonic features.
    """
    lst_dfs = []
    
    for df in list_dfs:
        # Time series data
        data_values = df[col][0:52].to_numpy()

        # Get the best reconstructed time series and the corresponding tsig
        best_S_reconstructed, best_tsig, yf_filtered_best = best_fft_reconstruction(data_values, period=52)
        
        # Check if yf_filtered_best is None and skip processing if so
        if yf_filtered_best is None:
            print("Skipping DataFrame due to yf_filtered_best being None",  df['co_ibge'].iloc[0])
            continue

        # Extract the significant coefficients from the best filtered yf
        N = 52
        a0 = np.real(yf_filtered_best[0]) / N
        an = 2.0 / N * np.real(yf_filtered_best[1:N//2])
        bn = -2.0 / N * np.imag(yf_filtered_best[1:N//2])

        # Set the desired number of decimal places
        decimal_places = 6
        

        # Construct the Fourier series equation with more precision
        equation = f"S(t) = {a0:.{decimal_places}f}"
        for n, (a, b) in enumerate(zip(an, bn), start=1):
            if np.abs(a) >= 1e-6 or np.abs(b) >= 1e-6:  # Exclude small coefficients
                equation += f" + ({a:.{decimal_places}f}) * cos(2π * {n} * t / {N}) + ({b:.{decimal_places}f}) * sin(2π * {n} * t / {N})"
        
        # Compute the values of the Fourier series for the same time points
        time_points = np.arange(len(data_values))
        fourier_series_values = fourier_series(time_points, a0, an, bn, N)


        data_values = df[col].to_numpy()
        # Compute the values of the Fourier series for the same time points
        time_points = np.arange(len(data_values))
        fourier_series_values = fourier_series(time_points, a0, an, bn, N)

        #set_muni = set_muni.assign(harmonics = fourier_series_values - a0)
        df = df.assign(S_t_fou = fourier_series_values)

        t = np.arange(len(df))

        # Define a threshold for significant terms
        threshold = 1e-6


        # Add columns for each significant Fourier term
        df['a0'] = a0
        for n, (a, b) in enumerate(zip(an, bn), start=1):
            if np.abs(a) >= threshold:
                df[f'cos_{n}'] = a * np.cos(2 * np.pi * n * t / N)
            if np.abs(b) >= threshold:
                df[f'sin_{n}'] = b * np.sin(2 * np.pi * n * t / N)

        # Sum up the significant Fourier terms to get the reconstructed series
        df['Reconstructed'] = df['a0'] + df.filter(like='cos').sum(axis=1) + df.filter(like='sin').sum(axis=1)
        
        df = df.assign(Reconstructed_log = np.log(df.Reconstructed))
        
        df.Reconstructed_log = df.Reconstructed_log.fillna(0)
        
        v1 = df.loc[~df['Reconstructed_log'].isin([np.inf,-np.inf]),'Reconstructed_log'].max()
        v2 = df.loc[~df['Reconstructed_log'].isin([np.inf,-np.inf]),'Reconstructed_log'].min()
        df['Reconstructed_log'] = df['Reconstructed_log'].replace([np.inf,-np.inf], [v1,v2])
        
        lst_dfs.append(df)

    return lst_dfs


def reg(dta, formulas):
    """
    Fits Poisson and Negative Binomial regression models to the provided DataFrame for each formula,
    selects the best model based on AIC, and returns a list of DataFrames with model results.

    Parameters:
    dta (pd.DataFrame): Input DataFrame containing the necessary columns.
    formulas (list): List of formula strings for the regression models.

    Returns:
    list: A list of DataFrames, each corresponding to a unique city, with additional computed columns:
        - Model parameters, p-values, and confidence intervals for the best model.
        - AIC value of the best model.
        - Best model type ('Poisson' or 'Negative Binomial').
        - Municipality code and name.
        - Correlation and p-value between actual and predicted ILI values.
        - Largest OTC lag value.
    """
    #lst_dfs = []

    # Initialize the list to store the results
    results = []

    for f in formulas:
        formula = f

        try:
            # Poisson model
            poisson_model = smf.glm(formula=formula, data=dta.iloc[:-4], family=sm.families.Poisson()).fit()

            # Negative Binomial model
            pearson_chi2 = sum(poisson_model.resid_pearson**2)
            df_resid = poisson_model.df_resid
            alpha = pearson_chi2 / df_resid
            negative_binomial_model = smf.glm(formula=formula, data=dta.iloc[:-4], family=sm.families.NegativeBinomial(alpha=alpha)).fit()

            # Select best model
            best_model = poisson_model if poisson_model.aic <= negative_binomial_model.aic else negative_binomial_model

            # Determine which model was selected
            best_model_type = 'Poisson' if poisson_model.aic <= negative_binomial_model.aic else 'Negative Binomial'

            # Collect parameters
            params_df = best_model.params.reset_index().transpose()
            params_df = params_df.rename(columns=params_df.iloc[0]).iloc[1:]
            params_df = params_df.rename(columns={col: col + '_par' for col in params_df.columns})

            # Add model p-values
            pvalues_df = best_model.pvalues.reset_index().transpose()
            pvalues_df = pvalues_df.rename(columns=pvalues_df.iloc[0]).iloc[1:]
            pvalues_df = pvalues_df.rename(columns={col: col + '_pvalues' for col in pvalues_df.columns})

            # Add confidence intervals
            conf_int_df = best_model.conf_int()
            conf_int_low = conf_int_df[0].reset_index().transpose()
            conf_int_low = conf_int_low.rename(columns=conf_int_low.iloc[0]).iloc[1:]
            conf_int_low = conf_int_low.rename(columns={col: col + '_IC_low' for col in conf_int_low.columns})

            conf_int_high = conf_int_df[1].reset_index().transpose()
            conf_int_high = conf_int_high.rename(columns=conf_int_high.iloc[0]).iloc[1:]
            conf_int_high = conf_int_high.rename(columns={col: col + '_IC_high' for col in conf_int_high.columns})

            # Combine all results
            results_df = pd.concat([params_df, pvalues_df, conf_int_low, conf_int_high], axis=1)
            
            # Add AIC value
            results_df = results_df.assign(aic=best_model.aic)

            # Add formula
            results_df = results_df.assign(form = formula)

            # Add best model type
            results_df = results_df.assign(best_model=best_model_type)

            # Add munic code and name
            results_df = results_df.assign(co_ibge = dta['co_ibge'].iloc[0])
            results_df = results_df.assign(municipio = dta['municipio'].iloc[0])

            # Add the predictions of ILI from drug sales were evaluated by Spearman correlation coefficients
            n1 = best_model.predict(dta.iloc[:-4]).to_numpy()
            n2 = best_model.predict(dta.iloc[-4:]).to_numpy()
            n3 = np.concatenate((n1, n2), axis=None)
            correlation, p_value = pearsonr(dta['atend_ivas_4'].to_numpy(), n3)
            
                
            results_df = results_df.assign(cor_pred_real = correlation)
            results_df = results_df.assign(p_value_pred_real = p_value)

            # Append the result to the list
            results.append(results_df)
        except Exception as e:
            print(f"Error fitting model for formula {formula}: {e}")
            continue

    if results:
        # Convert the list of results to a DataFrame
        results_models = pd.concat(results)

        # Define the list of criteria to be checked sequentially
        criteria_list = [
            "otc_4_lag_4_pvalues",
            "otc_4_lag_3_pvalues",
            "otc_4_lag_2_pvalues",
            "otc_4_lag_1_pvalues",
            "otc_4_lag_0_pvalues"
        ]

        select1 = pd.DataFrame()
        for criterion in criteria_list:
            select1 = results_models[(results_models[criterion] <= 0.05) & (results_models['aic'] == results_models['aic'].min())]
            if not select1.empty:
                break

        if not select1.empty:
            # Extract the formula for the best model
            formula = select1['form'].iloc[0]

            # Extract all OTC lag terms using a regular expression
            otc_lag_terms = re.findall(r'otc_4_lag_(\d+)', formula)

            # Convert the extracted lag values to integers
            otc_lag_values = [int(lag) for lag in otc_lag_terms]

            # Find the largest OTC lag value
            max_otc_lag = max(otc_lag_values)

            # Add the largest OTC lag value to the DataFrame
            dta = dta.assign(otc_lag = max_otc_lag)
        else:
            print("No valid model selected for the given criteria.")
            dta['otc_lag'] = np.nan  # Assign NaN if no valid model
    else:
        print("No valid models were fit.")
        dta['otc_lag'] = np.nan  # Assign NaN if no models were fit

    #lst_dfs.append(dta)

    return dta



def sen_spe(df,col):
    # Calculate True Positives, False Negatives, True Negatives, and False Positives
    TP = ((df['warning_local_health'] == 1) & (df[col] == 1)).sum()
    FN = ((df['warning_local_health'] == 1) & (df[col] == 0)).sum()
    TN = ((df['warning_local_health'] == 0) & (df[col] == 0)).sum()
    FP = ((df['warning_local_health'] == 0) & (df[col] == 1)).sum()

    # Calculate Sensitivity and Specificity
    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
    
    return sensitivity, specificity









