import pandas as pd
import yfinance as yf
import requests
from fredapi import Fred
import glob
from datetime import datetime
import sys
import os
import matplotlib.pyplot as plt

parent_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(parent_path))))
sys.path.append(root_path)
from pipeline.airflow.dags.src.download_data import (
    get_yfinance_data,
    get_fama_french_data,
    get_ads_index_data,
    get_sp500_data,
    get_fred_data,
    merge_data,
)
from pipeline.airflow.dags.src.convert_column_dtype import convert_type_of_columns
from pipeline.airflow.dags.src.keep_latest_data import keep_latest_data
from pipeline.airflow.dags.src.remove_weekend_data import remove_weekends
from pipeline.airflow.dags.src.handle_missing import fill_missing_values


def plot_yfinance_time_series(data: pd.DataFrame):

    data.set_index("date", inplace=True)

    yfinance_columns = ["open", "high", "low", "close", "volume"]

    # Filtering to include only yfinance columns
    yfinance_data = data[yfinance_columns]

    num_rows = yfinance_data.shape[1]

    fig, axs = plt.subplots(num_rows, 1, figsize=(15, 5 * num_rows))
    axs = axs.flatten()

    for i, column in enumerate(yfinance_data.columns):
        axs[i].plot(yfinance_data.index, yfinance_data[column], label=column)
        axs[i].set_title(column)
        axs[i].set_xlabel("Date")
        axs[i].set_ylabel(column)
        axs[i].legend()

    plt.tight_layout()
    plt.savefig("assets/yfinance_time_series.png")
    # plt.show()


if __name__ == "__main__":
    ticker_symbol = "GOOGL"
    data = merge_data(ticker_symbol)
    data = convert_type_of_columns(data)
    filtered_data = keep_latest_data(data, 10)
    removed_weekend_data = remove_weekends(filtered_data)
    filled_data = fill_missing_values(removed_weekend_data)
    plot_yfinance_time_series(filled_data)
