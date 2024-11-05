# DataPipeline Assignment Phase

This section explains dags pipeline implemented using Apache Airflow for workflow orchestration. The approach focus on stock market analysis, combining data from various sources including FRED (Federal Reserve Economic Data), Fama-French factors. The pipeline is organized with clear separation of concerns - data storage in the 'data' directory, processing scripts in 'src', and output artifacts for visualization. The source code includes essential ML preprocessing steps like handling missing values, feature engineering (including technical indicators and lagged features), dimensionality reduction through PCA, and correlation analysis. 

## Table of Contents
- [Directory Structure](#directory-structure)
- [Pipeline Components](#pipeline-components)
- [Setup and Usage](#setup-and-usage)
- [Data Sources](#data-sources)

---

### Directory Structure

```
.
├── airflow
│   ├── artifacts                     # Output Visualization files
│   │   ├── correlation_matrix_after_removing_correlated_features.png
│   │   ├── pca_components.png
│   │   └── yfinance_time_series.png
│   ├── dags                          # Contains Airflow DAGs
│   │   ├── airflow.py                # Main DAG for executing the pipeline
│   │   ├── data                      # Data sources 
│   │   │   ├── ADS_index.csv
│   │   │   ├── fama_french.csv
│   │   │   ├── FRED_Variables        
│   │   │   └── merged_original_dataset.csv
│   │   └── src                       # Scripts for various tasks
│   │       ├── convert_column_dtype.py
│   │       ├── correlation.py
│   │       ├── download_data.py
│   │       ├── feature_interactions.py
│   │       ├── handle_missing.py
│   │       ├── keep_latest_data.py
│   │       ├── lagged_features.py
│   │       ├── pca.py
│   │       ├── plot_yfinance_time_series.py
│   │       ├── remove_weekend_data.py
│   │       ├── scaler.py
│   │       └── technical_indicators.py
│   ├── docker-compose.yaml           # Docker configuration for running Airflow
│   ├── dvc.yaml                      # DVC configuration for data version control
│   ├── logs                          # Airflow logs
│   ├── plugins                      
│   └── working_data                  
└── pipelinetree.txt                  # Airflow working structure
```

### Pipeline Components

1. **Data Extraction**:
   - `download_data.py`: Downloads datasets from various financial sources and saves them in the `data` directory.

2. **Data Preprocessing**:
   - `convert_column_dtype.py`: Converts data types for efficient processing.
   - `remove_weekend_data.py`: Filters out weekend data to maintain consistency in time-series analyses.
   - `handle_missing.py`: Handles missing data by imputation or removal.

3. **Feature Engineering**:
   - `correlation.py`: Generates a correlation matrix to identify highly correlated features.
   - `pca.py`: Performs Principal Component Analysis to reduce dimensionality and capture key components.
   - `lagged_features.py`: Creates lagged versions of features for time-series analysis.
   - `technical_indicators.py`: Calculates common technical indicators used in financial analyses.

4. **Data Scaling**:
   - `scaler.py`: Scales features to a standard range for better model performance.

5. **Analysis and Visualization**:
   - `plot_yfinance_time_series.py`: Plots time-series data.
   - `feature_interactions.py`: Generates interaction terms between features.

### Setup and Usage

To set up and run the pipeline:

   > Refer to [Environment Setup](https://github.com/IE7374-MachineLearningOperations/StockPricePrediction/tree/main?tab=readme-ov-file#environment-setup) 

   > Refer to [Running Pipeline](https://github.com/IE7374-MachineLearningOperations/StockPricePrediction/tree/main?tab=readme-ov-file#running-the-pipeline)

### Data Sources

1. **ADS Index**: Tracks economic trends and business cycles.
2. **Fama-French Factors**: Provides historical data for financial research.
3. **FRED Variables**: Includes various economic indicators, such as AMERIBOR, NIKKEI 225, and VIX.
4. **YFinance**: Pulls historical stock data ('GOOGL') for financial time-series analysis. 
