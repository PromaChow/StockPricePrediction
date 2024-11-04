import pandas as pd
import yfinance as yf
import requests
from fredapi import Fred
import glob
from datetime import datetime
import sys
import os
import matplotlib.pyplot as plt
import seaborn as sns

# parent_path = os.path.abspath(os.path.dirname(__file__))

# root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(parent_path))))
# sys.path.append(root_path)
sys.path.append(os.path.abspath("pipeline/airflow"))
sys.path.append(os.path.abspath("."))

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


def add_feature_interactions(data: pd.DataFrame) -> pd.DataFrame:
    """
    Adds interaction features between specified columns in the DataFrame.
    Includes product and ratio interactions for 'close' with 'SP500' and 'VIXCLS',
    as well as interactions between 'SP500', 'VIXCLS'

    Returns:
    - DataFrame with additional interaction features.
    """
    # Initialize a copy of the DataFrame to store interaction features
    # data = data.copy()

    # Interaction between 'close' price and 'SP500' (product and ratio)
    data["close_SP500_product"] = data["close"] * data["SP500"]
    data["close_SP500_ratio"] = data["close"] / (data["SP500"] + 1e-5)  # Avoid division by zero

    # Interaction between 'close' price and 'VIXCLS' (volatility)
    data["close_VIXCLS_product"] = data["close"] * data["VIXCLS"]
    data["close_VIXCLS_ratio"] = data["close"] / (data["VIXCLS"] + 1e-5)

    # Interaction between 'SP500' and 'VIXCLS'
    data["SP500_VIXCLS_product"] = data["SP500"] * data["VIXCLS"]
    data["SP500_VIXCLS_ratio"] = data["SP500"] / (data["VIXCLS"] + 1e-5)

    return data


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
    # breakpoint()
    print(interaction_data)
