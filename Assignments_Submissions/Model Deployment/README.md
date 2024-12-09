# Model Deployment Phase

## Table of Contents

- [Prerequisites](#prerequisites)
- [Model Serving and Deployment](#model-serving-and-deployment)
- [Monitoring and Maintenance](#monitoring-and-maintenance)

---

## Prerequisites
Steps for Replication

#### 1. **Environment Setup**
    - Access to a Google Cloud Platform (GCP) account with billing enabled.
    - Vertex AI and Cloud Build services activated within your GCP project.
    - The `gcloud` CLI tool installed and configured with the appropriate permissions.
    - A GitHub repository with access to GitHub Actions for automation.
    - Required IAM roles for deploying models to Vertex AI and managing Cloud Build resources.

![GCP Billing Dashboard](https://github.com/IE7374-MachineLearningOperations/StockPricePrediction/blob/104a48ddf826520ccc31374002d8df92f2015796/assets/GCP%20billing%20dashboard.png)

#### 2. **Running Deployment Automation**
   - Push changes to the main branch of the GitHub repository.
   - GitHub Actions automatically triggers the CI/CD pipeline to initiate deployment using Cloud Build.
   - The pipeline executes pre-defined steps, ensuring the model is correctly deployed to Vertex AI.
   - Confirm that all dependencies are installed locally or in the CI/CD pipeline environment.

#### 3. **Verifying Deployment**
   - Access the Vertex AI console to verify the deployment.
   - Test the deployed model endpoint to confirm successful deployment and validate model predictions.
   - Review monitoring dashboards to ensure no issues with prediction outputs or feature drift.

![Drift Detection Logging](https://github.com/IE7374-MachineLearningOperations/StockPricePrediction/blob/104a48ddf826520ccc31374002d8df92f2015796/assets/Drift%20Detection%20logging.png)

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

![Vertex AI Jupyter Notebooks](https://github.com/IE7374-MachineLearningOperations/StockPricePrediction/blob/104a48ddf826520ccc31374002d8df92f2015796/assets/Vertex%20Ai%20jupyter%20notebooks.png)

![Model Serving](https://github.com/IE7374-MachineLearningOperations/StockPricePrediction/blob/104a48ddf826520ccc31374002d8df92f2015796/assets/Model%20serving.png)

![Vertex AI Model Registry](https://github.com/IE7374-MachineLearningOperations/StockPricePrediction/blob/104a48ddf826520ccc31374002d8df92f2015796/assets/Vertex%20Ai%20model%20registry.png)

2. **Create an Endpoint**: 
   - In Vertex AI, create an endpoint. This endpoint will act as the interface for serving predictions.
   - Navigate to Vertex AI > Online prediction > Endpoints > Create.
   - Assign a name and select the appropriate region.

![Vertex AI Endpoints](https://github.com/IE7374-MachineLearningOperations/StockPricePrediction/blob/104a48ddf826520ccc31374002d8df92f2015796/assets/Vertex%20Ai%20endpoints.png)

3. **Deploy the Model to an Endpoint**:
   - Select the registered model and choose "Deploy to Endpoint".
   - Configure the deployment settings such as machine type, traffic splitting among model versions, and whether to enable logging or monitoring.
   - Confirm deployment which will make the model ready to serve predictions.

![Vertex AI Model Development Training](https://github.com/IE7374-MachineLearningOperations/StockPricePrediction/blob/104a48ddf826520ccc31374002d8df92f2015796/assets/Vertex%20Ai%20model%20development%20training.png)

### Model Versioning
- **Manage Versions**: In Vertex AI, each model can have multiple versions allowing easy rollback and version comparison.
- **Update Versions**: Upload new versions of the model to the Model Registry and adjust the endpoint configurations to direct traffic to the newer version.

### Deployment Automation
#### Continuous Integration and Deployment Pipeline
- **Automate Deployments**: Use GitHub Actions and Google Cloud Build to automate the deployment of new model versions from a repository.
- **CI/CD Pipeline Configuration**:
   - **GitHub Actions**: Configure workflows in `.github/workflows/` directory to automate testing, building, and deploying models.
   - **Cloud Build**: Create a `cloudbuild.yaml` file specifying steps to build, test, and deploy models based on changes in the repository.

![GitHub Actions CI/CD](https://github.com/IE7374-MachineLearningOperations/StockPricePrediction/blob/104a48ddf826520ccc31374002d8df92f2015796/assets/Github%20Actions%20CICD.png)

---

#### Automated Deployment Scripts
- **Script Functions**:
  - **Pull the Latest Model**: Scripts should fetch the latest model version from Vertex AI Model Registry or a specified repository.
  - **Deploy or Update Model**: Automate the deployment of the model to the configured Vertex AI endpoint.
  - **Monitor and Log**: Set up logging for deployment status to ensure visibility and troubleshooting capabilities.

---
#### **1. `airflowtrigger.yaml`**
- **Purpose**: Triggers and manages Apache Airflow DAG workflows.
- **Steps**:
  - **Set up environment**: Installs Python, dependencies, and Docker Compose.
  - **Airflow initialization**: Starts Airflow services and checks their status.
  - **DAG management**: Lists, triggers, and monitors DAG execution (success or failure).
  - **Cleanup**: Stops Airflow services and removes unnecessary files.

---

#### **2. `deploy.yaml`**
- **Purpose**: Deploys and monitors a machine learning model on Vertex AI.
- **Steps**:
  - **Environment setup**: Configures Google Cloud SDK using secrets.
  - **Model deployment**: Deploys a trained model to Vertex AI endpoints.
  - **Monitoring**: Fetches the latest model and endpoint IDs and sets them for further monitoring.

---

#### **3. `model.yml`**
- **Purpose**: Handles training and packaging a machine learning model for deployment.
- **Steps**:
  - **Trainer creation**: Builds a Python package (`trainer`) for model training.
  - **Package upload**: Uploads the trainer package to Google Cloud Storage.
  - **Training job**: Triggers a Vertex AI custom training job using the uploaded package.
  - **Notification**: Indicates the completion of the training process.

---

#### **4. `PyTest.yaml`**
- **Purpose**: Runs Python unit tests and generates test coverage reports.
- **Steps**:
  - **Environment setup**: Installs dependencies and Google Cloud CLI.
  - **Testing**: Runs tests with pytest, generates coverage reports, and uploads them as artifacts.
  - **Upload results**: Saves coverage reports to a GCP bucket for review.

---

#### **5. `syncgcp.yaml`**
- **Purpose**: Synchronizes local artifacts and Airflow DAGs with a Google Cloud Storage bucket.
- **Steps**:
  - **Environment setup**: Installs the Google Cloud CLI and authenticates with a service account.
  - **File uploads**:
    - Uploads specific artifacts and files to predefined GCP bucket locations.
    - Synchronizes repository content with the bucket directory structure.
  - **Verification**: Lists uploaded files to confirm the sync.
---

![GitHub Actions](https://github.com/IE7374-MachineLearningOperations/StockPricePrediction/blob/104a48ddf826520ccc31374002d8df92f2015796/assets/Github%20Workflows.png)


#### Summary
These YAML workflows automate various aspects of an ML lifecycle:
1. **`airflowtrigger.yaml`**: Airflow DAG management.
2. **`deploy.yaml`**: Vertex AI deployment and monitoring.
3. **`model.yml`**: Training pipeline and GCS uploads.
4. **`PyTest.yaml`**: Testing and reporting.
5. **`syncgcp.yaml`**: Artifact and DAG synchronization with GCP.

Each workflow is tailored for a specific task in CI/CD for ML operations, leveraging GitHub Actions and Google Cloud services.
---

## Monitoring and Maintenance

1. **Monitoring**:
   - Vertex AI provides dashboards to monitor model performance and data drift.
   - Alerts are configured to notify stakeholders when anomalies, such as feature attribution drift, are detected.

![Model Monitoring Notification](https://github.com/IE7374-MachineLearningOperations/StockPricePrediction/blob/104a48ddf826520ccc31374002d8df92f2015796/assets/Model%20Monitoring%20notification.png)


![Model Monitoring Anomalies](https://github.com/IE7374-MachineLearningOperations/StockPricePrediction/blob/104a48ddf826520ccc31374002d8df92f2015796/assets/Model%20monitoring%20Anomolies.png)

The provided images highlight the active setup and management of a Vertex AI model monitoring system. Files like `anomalies.json` and `anomalies.textproto` document identified issues in the input data. The structure also includes folders such as `baseline`, `logs`, and `metrics`, which organize monitoring data effectively for future analysis. A notification email confirming the creation of a model monitoring job for a specific Vertex AI endpoint. This email provides essential details, such as the endpoint name, monitoring job link, and the GCS bucket path where statistics and anomalies will be saved. 

2. **Maintenance**:
   - Pre-configured thresholds for model performance trigger retraining or redeployment of updated models.
   - Logs and alerts from Vertex AI and Cloud Build ensure the system remains reliable and scalable.

![Monitor Details](https://github.com/IE7374-MachineLearningOperations/StockPricePrediction/blob/104a48ddf826520ccc31374002d8df92f2015796/assets/Monitor%20details.png)

![Logging Dashboard](https://github.com/IE7374-MachineLearningOperations/StockPricePrediction/blob/104a48ddf826520ccc31374002d8df92f2015796/assets/Logging%20Dashboard.png)

![Monitor Feature Detection](https://github.com/IE7374-MachineLearningOperations/StockPricePrediction/blob/104a48ddf826520ccc31374002d8df92f2015796/assets/Monitor%20feature%20detection.png)

![Monitor Drift Detection](https://github.com/IE7374-MachineLearningOperations/StockPricePrediction/blob/104a48ddf826520ccc31374002d8df92f2015796/assets/Monitor%20drift%20detection.png)


---


