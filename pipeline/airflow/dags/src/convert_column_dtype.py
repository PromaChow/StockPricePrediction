import pandas as pd
import yfinance as yf
from fredapi import Fred
import glob
from datetime import datetime
import sys
import os

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


def convert_type_of_columns(data: pd.DataFrame) -> pd.DataFrame:
    # convert 'date' to datetime
    data["date"] = pd.to_datetime(data["date"], errors="coerce")

    # convert all numerical columns to float
    object_columns = data.select_dtypes(include=["object"]).columns

    for col in object_columns:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    return data


if __name__ == "__main__":
    ticker_symbol = "GOOGL"
    data = merge_data(ticker_symbol)
    data = convert_type_of_columns(data)
    print(data.dtypes)
