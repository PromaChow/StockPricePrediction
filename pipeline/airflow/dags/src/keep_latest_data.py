import pandas as pd
import yfinance as yf
import requests
from fredapi import Fred
import glob
from datetime import datetime
import sys
import os

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


def keep_latest_data(data: pd.DataFrame, num_years: int) -> pd.DataFrame:
    current_date = datetime.now()
    cutoff_date = current_date - pd.DateOffset(years=num_years)

    filtered_data = data[data["date"] >= cutoff_date]

    return filtered_data


if __name__ == "__main__":
    ticker_symbol = "GOOGL"
    data = merge_data(ticker_symbol)
    data = convert_type_of_columns(data)
    filtered_data = keep_latest_data(data, 10)
    print(filtered_data)
