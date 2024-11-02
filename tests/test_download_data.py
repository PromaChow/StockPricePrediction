## append path
import sys
import os
import pandas as pd


from pipeline.airflow.dags.src.download_data import (
    get_yfinance_data,
    get_fama_french_data,
    get_ads_index_data,
    get_sp500_data,
    get_fred_data,
    merge_data,
)


def test_get_yfinance_data_columns():
    # Test the get_yfinance_data function
    result = get_yfinance_data("GOOGL")
    expected_columns = ["open", "high", "low", "close", "volume", "dividends", "stock_splits"]
    assert result.columns.tolist() == expected_columns


def test_get_yfinance_data_len():
    # Test the get_yfinance_data function
    result = get_yfinance_data("GOOGL")
    expected_data_len = result.shape[0]
    assert expected_data_len > 1000
