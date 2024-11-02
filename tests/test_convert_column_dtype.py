import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add the src directory to the Python path
sys.path.append(os.path.abspath('pipeline/airflow/dags/src'))

from convert_column_dtype import convert_type_of_columns

def test_convert_type_of_columns():
    # Create a sample DataFrame with mixed types
    sample_data = pd.DataFrame({
        'date': ['2023-01-01', '2023-01-02', '2023-01-03'],
        'float_col': ['1.1', '2.2', '3.3'],
        'object_col': ['a','b','c']
    })

    result = convert_type_of_columns(sample_data)

    # Check if date column is converted to datetime
    assert pd.api.types.is_datetime64_any_dtype(result['date'])

    # Check if numeric columns are converted to float
    assert result['float_col'].dtype == 'float64'
    assert result['object_col'].dtype == 'float64'

    # Check if all values are preserved
    assert result['float_col'].tolist() == [1.1, 2.2, 3.3]
