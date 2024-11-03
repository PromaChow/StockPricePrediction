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


def plot_correlation_matrix(data: pd.DataFrame):
    plt.figure(figsize=(25, 18))
    sns.heatmap(data.corr(), annot=True, fmt=".2f")
    plt.title("Correlation Matrix")

    ## make folder if not exist
    if not os.path.exists("artifacts"):
        os.makedirs("artifacts")
    plt.savefig("artifacts/correlation_matrix_original_dataset.png")


def removing_correlated_variables(data: pd.DataFrame) -> pd.DataFrame:

    df_corr = data.corr()
    # Removing variables with NaN correlation with every other variable (0 value)
    NaN_columns = [col for col in df_corr.columns if df_corr[col].isna().all()]
    df__withoutNaN = data.drop(columns=NaN_columns)
    print("NaN correlation/ 0 valued columns dropped: ", NaN_columns)

    corr_matrix_cleaned = df__withoutNaN.corr()

    columns_to_drop = set()

    for i in range(len(corr_matrix_cleaned.columns)):
        for j in range(i):
            if abs(corr_matrix_cleaned.iloc[i, j]) >= 0.95:
                colname = corr_matrix_cleaned.columns[i]

                if colname not in ["open", "high", "low", "close", "volume", "SP500"]:
                    columns_to_drop.add(colname)

    # Dropping the highly correlated columns (excluding yfinance columns)
    df_final = df__withoutNaN.drop(columns=columns_to_drop)
    # print("Columns dropped: ", columns_to_drop)

    # plot correlation matrix after removing correlated features
    plt.figure(figsize=(25, 18))
    sns.heatmap(df_final.corr(), annot=True, fmt=".2f")
    plt.title("Correlation Matrix")
    # plt.savefig("assets/correlation_matrix_after_removing_correlated_features.png")
    ## make folder if not exist
    if not os.path.exists("artifacts"):
        os.makedirs("artifacts")
    plt.savefig("artifacts/correlation_matrix_after_removing_correlated_features.png")

    return df_final


if __name__ == "__main__":
    ticker_symbol = "GOOGL"
    data = merge_data(ticker_symbol)
    data = convert_type_of_columns(data)
    filtered_data = keep_latest_data(data, 10)
    removed_weekend_data = remove_weekends(filtered_data)
    filled_data = fill_missing_values(removed_weekend_data)
    removed_correlated_data = removing_correlated_variables(filled_data)
    # breakpoint()
    print(removed_correlated_data)
