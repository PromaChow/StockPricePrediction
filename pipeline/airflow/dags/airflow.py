from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow import configuration as conf

from src.download_data import (
    get_yfinance_data,
    get_fama_french_data,
    get_ads_index_data,
    get_sp500_data,
    get_fred_data,
    merge_data,
)
from src.convert_column_dtype import convert_type_of_columns
from src.keep_latest_data import keep_latest_data
from src.remove_weekend_data import remove_weekends
from src.handle_missing import fill_missing_values
from src.plot_yfinance_time_series import plot_yfinance_time_series
from src.correlation import removing_correlated_variables, plot_correlation_matrix


# Enable pickle support for XCom, allowing data to be passed between tasks
conf.set("core", "enable_xcom_pickling", "True")
conf.set("core", "enable_parquet_xcom", "True")

# Define default arguments for your DAG
default_args = {
    "owner": "group_10",
    "start_date": datetime(2024, 10, 29),
    "retries": 0,  # Number of retries in case of task failure
    "retry_delay": timedelta(minutes=5),  # Delay before retries
}

# Create a DAG instance named 'datapipeline' with the defined default arguments
dag = DAG(
    "datapipeline",
    default_args=default_args,
    description="Airflow DAG for the datapipeline",
    schedule_interval=None,  # Set the schedule interval or use None for manual triggering
    catchup=False,
)

# Define PythonOperators for each function

# Task to download data from source, calls the 'download_data' Python function
download_data_task = PythonOperator(
    task_id="download_data_task",
    python_callable=merge_data,
    op_args=["GOOGL"],
    dag=dag,
)

# Task to convert column data types, calls the 'convert_type_of_columns' Python function
convert_type_data_task = PythonOperator(
    task_id="convert_data_task",
    python_callable=convert_type_of_columns,
    op_args=[download_data_task.output],
    dag=dag,
)


# Task to keep the latest data, calls the 'keep_latest_data' Python function
keep_latest_data_task = PythonOperator(
    task_id="keep_latest_data_task",
    python_callable=keep_latest_data,
    op_args=[convert_type_data_task.output, 10],
    dag=dag,
)

# Task to remove weekend data, calls the 'remove_weekends' Python function
remove_weekend_data_task = PythonOperator(
    task_id="remove_weekend_data_task",
    python_callable=remove_weekends,
    op_args=[keep_latest_data_task.output],
    dag=dag,
)

# Task to handle missing values, calls the 'fill_missing_values' Python function
handle_missing_values_task = PythonOperator(
    task_id="handle_missing_values_task",
    python_callable=fill_missing_values,
    op_args=[remove_weekend_data_task.output],
    dag=dag,
)

# Task to plot the time series data, calls the 'plot_yfinance_time_series' Python function
plot_time_series_task = PythonOperator(
    task_id="plot_time_series_task",
    python_callable=plot_yfinance_time_series,
    op_args=[handle_missing_values_task.output],
    dag=dag,
)

# Task to plot the correlation matrix, calls the 'removing_correlated_variables' Python function
removing_correlated_variables_task = PythonOperator(
    task_id="removing_correlated_variables_task",
    python_callable=removing_correlated_variables,
    op_args=[handle_missing_values_task.output],
    dag=dag,
)


# Set task dependencies
(
    download_data_task
    >> convert_type_data_task
    >> keep_latest_data_task
    >> remove_weekend_data_task
    >> handle_missing_values_task
    >> plot_time_series_task
    >> removing_correlated_variables_task
)


# If this script is run directly, allow command-line interaction with the DAG
if __name__ == "__main__":
    dag.cli()
