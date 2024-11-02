## append path
import sys
import os
import pandas as pd


from pipeline.airflow.dags.src.correlation import removing_correlated_variables


def test_correlation():
    # Test the correlation function
    df = pd.DataFrame(
        {
            "A": [1, 2, 3, 4, 5],
            "B": [1, 2, 3, 4, 5],
            "C": [1, 2, 3, 4, 5],
        }
    )
    result = removing_correlated_variables(df)

    expected = pd.DataFrame(
        {
            "A": [1, 2, 3, 4, 5],
        }
    )
    assert result.equals(expected)
