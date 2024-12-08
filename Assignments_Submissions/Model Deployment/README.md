# Model Deployment 

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
#### 2. **Running Deployment Automation**
   - Push changes to the main branch of the GitHub repository.
   - GitHub Actions automatically triggers the CI/CD pipeline to initiate deployment using Cloud Build.
   - The pipeline executes pre-defined steps, ensuring the model is correctly deployed to Vertex AI.
   - Confirm that all dependencies are installed locally or in the CI/CD pipeline environment.
#### 3. **Verifying Deployment**
   - Access the Vertex AI console to verify the deployment.
   - Test the deployed model endpoint to confirm successful deployment and validate model predictions.
   - Review monitoring dashboards to ensure no issues with prediction outputs or feature drift.

---

## Model Serving and Deployment
To serve models using Vertex AI by deploying trained models, managing versions, and automating deployments through CI/CD pipelines.

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

#### Automated Deployment Scripts
- **Script Functions**:
  - **Pull the Latest Model**: Scripts should fetch the latest model version from Vertex AI Model Registry or a specified repository.
  - **Deploy or Update Model**: Automate the deployment of the model to the configured Vertex AI endpoint.
  - **Monitor and Log**: Set up logging for deployment status to ensure visibility and troubleshooting capabilities.

---


## Monitoring and Maintenance

1. **Monitoring**:
   - Vertex AI provides dashboards to monitor model performance and data drift.
   - Alerts are configured to notify stakeholders when anomalies, such as feature attribution drift, are detected.

![Model Monitoring Notification](https://github.com/IE7374-MachineLearningOperations/StockPricePrediction/blob/104a48ddf826520ccc31374002d8df92f2015796/assets/Model%20Monitoring%20notification.png)


![Model Monitoring Anomalies](https://github.com/IE7374-MachineLearningOperations/StockPricePrediction/blob/104a48ddf826520ccc31374002d8df92f2015796/assets/Model%20monitoring%20Anomolies.png)

2. **Maintenance**:
   - Pre-configured thresholds for model performance trigger retraining or redeployment of updated models.
   - Logs and alerts from Vertex AI and Cloud Build ensure the system remains reliable and scalable.

![Monitor Details](https://github.com/IE7374-MachineLearningOperations/StockPricePrediction/blob/104a48ddf826520ccc31374002d8df92f2015796/assets/Monitor%20details.png)


![Monitor Feature Detection](https://github.com/IE7374-MachineLearningOperations/StockPricePrediction/blob/104a48ddf826520ccc31374002d8df92f2015796/assets/Monitor%20feature%20detection.png)


![Monitor Drift Detection](https://github.com/IE7374-MachineLearningOperations/StockPricePrediction/blob/104a48ddf826520ccc31374002d8df92f2015796/assets/Monitor%20drift%20detection.png)

---


