# Model Deployment Phase

## Table of Contents

- [Prerequisites](#prerequisites)
- [Model Serving and Deployment](#model-serving-and-deployment)
- [Monitoring and Maintenance](#monitoring-and-maintenance)
- [Notifications](#notifications)
- [UI Dashboard for Stock Price Prediction](#ui-dashboard-for-stock-price-predictor)
---

## Prerequisites
Steps for Replication

#### 1. **Environment Setup**
    - Access to a Google Cloud Platform (GCP) account with billing enabled.
    - Vertex AI and Cloud Build services activated within your GCP project.
    - The `gcloud` CLI tool installed and configured with the appropriate permissions.
    - A GitHub repository with access to GitHub Actions for automation.
    - Required IAM roles for deploying models to Vertex AI and managing Cloud Build resources.

![GCP Billing Dashboard](https://github.com/IE7374-MachineLearningOperations/StockPricePrediction/blob/257dc9560a2e3741e7e383bd63d00340c193c9c0/assets/GCP%20billing%20dashbgoard.png)

#### 2. **Running Deployment Automation**
   - Push changes to the main branch of the GitHub repository.
   - GitHub Actions automatically triggers the CI/CD pipeline to initiate deployment using Cloud Build.
   - The pipeline executes pre-defined steps, ensuring the model is correctly deployed to Vertex AI.
   - Confirm that all dependencies are installed locally or in the CI/CD pipeline environment.

#### 3. **Verifying Deployment**
   - Access the Vertex AI console to verify the deployment.
   - Test the deployed model endpoint to confirm successful deployment and validate model predictions.
   - Review monitoring dashboards to ensure no issues with prediction outputs or feature drift.

![Drift Detection Logging](https://github.com/IE7374-MachineLearningOperations/StockPricePrediction/blob/v2.1/assets/Drift%20Detection%20logging.png)

> Please refer to [Data drift Notebook](https://github.com/IE7374-MachineLearningOperations/StockPricePrediction/blob/c79c65ec5044c5935e2e3052730dac3a9778c51a/src/Datadrift_detection_updated.ipynb)

- **The drift detection** process identifies significant shifts in data patterns by analyzing key statistical metrics across features such as Volume, RSI, MACD, MA20, and SP500_VIXCLS_ratio. Metrics like mean, variance, and percentile differences reveal substantial deviations, indicating changes in data distribution. For example, Volume shows a mean difference of `1.083e+16` and variance difference of `0.591`, while RSI highlights deviations in its 50th percentile and variance. MACD and MA20 exhibit notable shifts in percentiles, suggesting changes in trend-related features, and SP500_VIXCLS_ratio reveals variability in market volatility. These findings emphasize the need to monitor data sources, adjust preprocessing pipelines, and potentially retrain models to maintain prediction accuracy and ensure model reliability in dynamic environments.

---

## Model Serving and Deployment

Workflows and setups for managing machine learning pipelines on Vertex AI in Google Cloud are as follows:

1. **Jupyter Notebooks in Vertex AI Workbench**:
   - The setup includes instances like `group10-test-vy` and `mlops-group10`, both configured for NumPy/SciPy and scikit-learn environments. These notebooks are GPU-enabled, optimizing their utility for intensive ML operations.

2. **Training Pipelines**:
   - Multiple training pipelines are orchestrated on Vertex AI, such as `mlops-group10` and `group10-model-train`. These are primarily custom training pipelines aimed at tasks like hyperparameter tuning, training, and validation, leveraging the scalability of Google Cloud's infrastructure.

3. **Metadata Management**:
   - Metadata tracking is managed through Vertex AI Metadata Store, with records such as `vertex_dataset`. This ensures reproducibility and streamlined monitoring of all artifacts produced during ML workflows.

4. **Model Registry**:
   - Deployed models like `mlops-group10-deploy` and `group10-model` are maintained in the model registry. The registry supports versioning and deployment tracking for consistency and monitoring.

5. **Endpoints for Online Prediction**:
   - Various endpoints, such as `mlops-group10-deploy` and `testt`, are active and ready for predictions. The setup is optimized for real-time online predictions, and monitoring can be enabled for anomaly detection or drift detection.

### Steps for Deployment of Trained Models
1. **Model Registration**: Once a model is trained, register it in Vertex AI's Model Registry. Specify the model name, version, and any relevant metadata.

![Vertex AI Jupyter Notebooks](https://github.com/IE7374-MachineLearningOperations/StockPricePrediction/blob/v3.1/assets/Vertex%20AI%20jupyter%20notebooks.png)

![Model Serving](https://github.com/IE7374-MachineLearningOperations/StockPricePrediction/blob/v2.1/assets/Model%20serving.png)

![Vertex AI Model Registry](https://github.com/IE7374-MachineLearningOperations/StockPricePrediction/blob/v3.1/assets/Vertex%20AI%20model%20registry.png)

2. **Create an Endpoint**: 
   - In Vertex AI, create an endpoint. This endpoint will act as the interface for serving predictions.
   - Navigate to Vertex AI > Online prediction > Endpoints > Create.
   - Assign a name and select the appropriate region.

![Vertex AI Endpoints](https://github.com/IE7374-MachineLearningOperations/StockPricePrediction/blob/v2.1/assets/Vertex%20Ai%20endpoints.png)

3. **Deploy the Model to an Endpoint**:
   - Select the registered model and choose "Deploy to Endpoint".
   - Configure the deployment settings such as machine type, traffic splitting among model versions, and whether to enable logging or monitoring.
   - Confirm deployment which will make the model ready to serve predictions.

![Vertex AI Model Development Training](https://github.com/IE7374-MachineLearningOperations/StockPricePrediction/blob/v3.1/assets/Vertex%20AI%20model%20development%20training.png)

### Model Versioning
- **Manage Versions**: In Vertex AI, each model can have multiple versions allowing easy rollback and version comparison.
- **Update Versions**: Upload new versions of the model to the Model Registry and adjust the endpoint configurations to direct traffic to the newer version.

### Deployment Automation
#### Continuous Integration and Deployment Pipeline
- **Automate Deployments**: Use GitHub Actions and Google Cloud Build to automate the deployment of new model versions from a repository.
- **CI/CD Pipeline Configuration**:
   - **GitHub Actions**: Configure workflows in `.github/workflows/` directory to automate testing, building, and deploying models.
   - **Cloud Build**: Create a `cloudbuild.yaml` file specifying steps to build, test, and deploy models based on changes in the repository.

![GitHub Actions CI/CD](https://github.com/IE7374-MachineLearningOperations/StockPricePrediction/blob/v2.1/assets/Github%20Actions%20CICD.png)

---

#### Automated Deployment Scripts
- **Script Functions**:
  - **Pull the Latest Model**: Scripts should fetch the latest model version from Vertex AI Model Registry or a specified repository.
  - **Deploy or Update Model**: Automate the deployment of the model to the configured Vertex AI endpoint.
  - **Monitor and Log**: Set up logging for deployment status to ensure visibility and troubleshooting capabilities.

---
### Workflows Overview

#### 1. **Airflow DAG Trigger Workflow**
   - **File:** `airflowtrigeer.yaml`
   - **Purpose:** Initializes Airflow, starts required services, triggers a DAG (`Group10_DataPipeline_MLOps`), and monitors its execution.
   - **Key Steps:**
     - Sets up Python and dependencies.
     - Initializes and starts Airflow services via Docker Compose.
     - Triggers the specified Airflow DAG and monitors its status until completion.
     - Stops services and removes containers after execution.
   - **Triggers:** On push to `main` 

![Airflow DAG Trigger Workflow](https://github.com/IE7374-MachineLearningOperations/StockPricePrediction/blob/v3.1/assets/Trigger%20airflow%20action.png)

#### 2. **Model Training Workflow**
   - **File:** `model.yml`
   - **Purpose:** Packages the trainer code, uploads it to GCS, and triggers a Vertex AI Custom Job for model training.
   - **Key Steps:**
     - Prepares a Python trainer package (`trainer/task.py`) and packages it.
     - Uploads the package to a GCS bucket.
     - Triggers a Vertex AI Custom Job with specified training arguments (e.g., dataset path, epochs, batch size).
     - Notifies upon completion.

![Model Training Workflow](https://github.com/IE7374-MachineLearningOperations/StockPricePrediction/blob/v3.1/assets/Train%20deploy%20github%20action.png)

#### 3. **Deploy and Monitor Workflow**
   - **File:** `deploy_monitor.yaml`
   - **Purpose:** Deploys the trained model to Vertex AI, creates an endpoint, and sets up monitoring for model performance and data drift.
   - **Key Steps:**
     - Deploys the model to Vertex AI.
     - Configures a new or existing endpoint for predictions.
     - Enables monitoring for performance and feature/data drift.
   - **Triggers:** On push to `main` 

![Deploy and Monitor Workflow](https://github.com/IE7374-MachineLearningOperations/StockPricePrediction/blob/v3.1/assets/Deploy%20and%20monitor%20action.png)

#### 4. **Train and Deploy Workflow**
   - **File:** `train_deploy.yaml`
   - **Purpose:** Trains the ML model and deploys it to Vertex AI for online predictions.
   - **Key Steps:**
     - Builds a Docker image of the training script and pushes it to GCP Artifact Registry.
     - Triggers model training.
     - Deploys the trained model to a new or existing endpoint.
   - **Triggers:** On push to `main`

![Train Workflow](https://github.com/IE7374-MachineLearningOperations/StockPricePrediction/blob/v3.1/assets/Train%20deploy%20github%20action.png)

#### 5. **Pytest Workflow**
   - **File:** Not explicitly listed (part of repository tests).
   - **Purpose:** Runs automated tests on codebase changes.
   - **Key Steps:**
     - Installs project dependencies.
     - Executes tests using `pytest` to ensure the codebase is robust and ready for deployment.
   - **Triggers:** On push to `main`

![Pytest Workflow](https://github.com/IE7374-MachineLearningOperations/StockPricePrediction/blob/v3.1/assets/Pytest%20action.png)

#### **6. `syncgcp.yaml`**
- **Purpose**: Synchronizes local artifacts and Airflow DAGs with a Google Cloud Storage bucket. Workflow to build Docker images, upload artifacts to GCP, and deploy to endpoints.
- **Steps**:
  - **Environment setup**: Installs the Google Cloud CLI and authenticates with a service account.
  - **File uploads**:
    - Uploads specific artifacts and files to predefined GCP bucket locations.
    - Synchronizes repository content with the bucket directory structure.
  - **Verification**: Lists uploaded files to confirm the sync.

![Upload and Dockerize Workflow](https://github.com/IE7374-MachineLearningOperations/StockPricePrediction/blob/v3.1/assets/syncgcp%20githubaction.png)

---

![GitHub Actions](https://github.com/IE7374-MachineLearningOperations/StockPricePrediction/blob/v2.1/assets/Github%20Workflows.png)


#### Summary
These YAML workflows automate various aspects of an ML lifecycle:
1. **`airflowtrigger.yaml`**: Airflow DAG management.
2. **`train_deploy.yaml`**: Trains the ML model and deploys it to Vertex AI for online predictions.
3. **`model.yml`**: Training pipeline and GCS uploads.
4. **`PyTest.yaml`**: Testing and reporting.
5. **`syncgcp.yaml`**: Artifact and DAG synchronization with GCP.
6. **`deploy_monitor.yaml`**: Deploys the trained model, monitoring for model performance and data drift.

- Each workflow is tailored for a specific task in CI/CD for ML operations, leveraging GitHub Actions and Google Cloud services.
---

## Monitoring and Maintenance

1. **Monitoring**:
   - Vertex AI provides dashboards to monitor model performance and data drift.
   - Alerts are configured to notify stakeholders when anomalies, such as feature attribution drift, are detected.

![Monitor Feature Detection](https://github.com/IE7374-MachineLearningOperations/StockPricePrediction/blob/v2.1/assets/Monitor%20feature%20detection.png)

![Monitor Drift Detection](https://github.com/IE7374-MachineLearningOperations/StockPricePrediction/blob/v2.1/assets/Monitor%20drift%20detection.png)

![Model Monitoring Anomalies](https://github.com/IE7374-MachineLearningOperations/StockPricePrediction/blob/v2.1/assets/Model%20monitoring%20Anomolies.png)

The provided images highlight the active setup and management of a Vertex AI model monitoring system. Files like `anomalies.json` and `anomalies.textproto` document identified issues in the input data. The structure also includes folders such as `baseline`, `logs`, and `metrics`, which organize monitoring data effectively for future analysis. A notification email confirming the creation of a model monitoring job for a specific Vertex AI endpoint. This email provides essential details, such as the endpoint name, monitoring job link, and the GCS bucket path where statistics and anomalies will be saved. 

2. **Maintenance**:
   - Pre-configured thresholds for model performance trigger retraining or redeployment of updated models.
   - Logs and alerts from Vertex AI and Cloud Build ensure the system remains reliable and scalable.

![Monitor Details](https://github.com/IE7374-MachineLearningOperations/StockPricePrediction/blob/v2.1/assets/Monitor%20details.png)

![Logging Dashboard](https://github.com/IE7374-MachineLearningOperations/StockPricePrediction/blob/v2.1/assets/Logging%20Dashboard.png)


---

## Notifications

### Vertex AI Failure Notification
This notification is sent when a Vertex AI Model Monitoring job partially fails. For example, if the model encounters a drift detection issue or missing explanation specifications, an email alert is sent.

![Vertex AI Failure Notification](https://github.com/IE7374-MachineLearningOperations/StockPricePrediction/blob/v3.1/assets/VertexAI%20failure%20notification.png)  

### Vertex AI Anomaly and Data Drift Notification
This email notification is triggered when data drift or anomalies are detected in the deployed model. It provides links to detailed logs and monitoring jobs for debugging.

![Vertex AI Anomaly and Data Drift Notification](https://github.com/IE7374-MachineLearningOperations/StockPricePrediction/blob/v3.1/assets/Vertex%20AI%20Anomoly%20and%20Datadrift%20notify.png)  


#### **Partial Failure Notification**
This email is sent when a model monitoring job partially fails, providing details about the type of failure and suggestions for resolving the issue.

![Partial Failure Notification](https://github.com/IE7374-MachineLearningOperations/StockPricePrediction/blob/v3.1/assets/VertexAI%20failure%20notification.png)  

#### **Anomaly and Data Drift Detection Notification**
This email is triggered when the Vertex AI monitoring job detects data drift or anomalies in the deployed model. It provides links for further investigation.

![Anomaly and Data Drift Detection Notification](https://github.com/IE7374-MachineLearningOperations/StockPricePrediction/blob/v3.1/assets/Vertex%20AI%20Anomoly%20and%20Datadrift%20notify.png)  

## UI Dashboard for Stock Price Predictor

The **Stock Price Predictor UI Dashboard** is a user-friendly interface designed to analyzing stock trends and making informed financial decisions. This dashboard, running locally at [http://127.0.0.1:5001](http://127.0.0.1:5001), provides a good experience for users to explore stock predictions and insights.

### Running the Dashboard

To launch the application:

1. Run the following command:
   ```bash
   python app/app.py
   ```

2. Access the dashboard at:
   [http://127.0.0.1:5001](http://127.0.0.1:5001)

This dashboard bridges the gap between raw stock data and actionable investment strategies, making financial decision-making easier and more efficient.

### Key Features

1. **Homepage**: 
   - Provides an overview of the tool's capabilities and purpose.
   - Highlights the integration of machine learning with financial decision-making.

   ![Homepage](https://github.com/IE7374-MachineLearningOperations/StockPricePrediction/blob/63ef3b06820c954454afe53fd669159ebb54bf34/assets/UI_Dashboard_homepage.png)

2. **Prediction Page**:
   - Displays the **current stock price** and the **predicted price**.
   - Includes options to "Buy" or "Sell" based on the predicted price trend.
   - Offers a gamified experience to engage users in testing their predictions.

   ![Prediction Dashboard](https://github.com/IE7374-MachineLearningOperations/StockPricePrediction/blob/63ef3b06820c954454afe53fd669159ebb54bf34/assets/UI_Dashboard_predict.png)

3. **Visualization Dashboard**:
   - Features dynamic and interactive charts to compare actual vs. predicted stock prices.
   - Provides insights into stock trends.

   ![Visualization Dashboard](https://github.com/IE7374-MachineLearningOperations/StockPricePrediction/blob/63ef3b06820c954454afe53fd669159ebb54bf34/assets/UI_Dashboard_visuals.png)

---