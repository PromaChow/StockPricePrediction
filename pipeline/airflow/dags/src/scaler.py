import pandas as pd
import numpy as np
import yfinance as yf
import requests
import glob
import sys
import os
import matplotlib.pyplot as plt
import seaborn as sns
from fredapi import Fred
from datetime import datetime
from sklearn.preprocessing import StandardScaler

parent_path = os.path.abspath(os.path.dirname(__file__))

root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(parent_path))))
sys.path.append(root_path)
from dags.src.download_data import (
    get_yfinance_data,
    get_fama_french_data,
    get_ads_index_data,
    get_sp500_data,
    get_fred_data,
    merge_data,
)
from dags.src.convert_column_dtype import convert_type_of_columns
from dags.src.keep_latest_data import keep_latest_data
from dags.src.remove_weekend_data import remove_weekends
from dags.src.handle_missing import fill_missing_values
from dags.src.correlation import plot_correlation_matrix, removing_correlated_variables
from dags.src.lagged_features import add_lagged_features
from dags.src.feature_interactions import add_feature_interactions
from dags.src.technical_indicators import add_technical_indicators


def scaler(data: pd.DataFrame, mean=None, variance=None):
    """
    if training data, mean and variance should be None
    if test data, mean and variance should be provided, calculated from training data
    """
    scaler = StandardScaler()
    data_scaling = data.drop(columns=["date"])
    if mean is not None and variance is not None:
        scaler = scaler.fit(data_scaling)
        scaler.mean_ = mean
        scaler.var_ = variance
        scaled_data = scaler.transform(data_scaling)
    else:
        scaled_data = scaler.fit_transform(data_scaling)
    # add date column back to the scaled data
    final_scaled_data = pd.concat(
        [data["date"].reset_index(drop=True), pd.DataFrame(scaled_data, columns=data_scaling.columns)], axis=1
    )

    return final_scaled_data


if __name__ == "__main__":
    ticker_symbol = "GOOGL"
    data = merge_data(ticker_symbol)
    data = convert_type_of_columns(data)
    filtered_data = keep_latest_data(data, 10)
    removed_weekend_data = remove_weekends(filtered_data)
    filled_data = fill_missing_values(removed_weekend_data)
    removed_correlated_data = removing_correlated_variables(filled_data)
    lagged_data = add_lagged_features(removed_correlated_data)
    interaction_data = add_feature_interactions(lagged_data)
    technical_data = add_technical_indicators(interaction_data)
    scaled_data = scaler(technical_data)
    # breakpoint()
    print(scaled_data)
