#!/bin/bash

# Variables
KEY_FILE="/home/manormanore/Documents/Git_Hub/StockPricePrediction/GCP/striped-graph-440017-d7-c8fdb42bc8ba.json"
LOCAL_REPO="/home/manormanore/Documents/Git_Hub/StockPricePrediction/"
BUCKET_NAME="stock_price_prediction_dataset"
BUCKET_PATH="Data"

# Authenticate Google Cloud using service account key
gcloud auth activate-service-account --key-file="$KEY_FILE"

# Create a temporary directory to store filtered files
TEMP_DIR=$(mktemp -d)

# Upload only files not excluded by .gitignore and matching the extensions (.csv and .png)
cd "$LOCAL_REPO"
git ls-files --cached --others --exclude-standard | grep -E "\.csv$|\.png$" | while read file; do
    mkdir -p "$TEMP_DIR/$(dirname "$file")"
    cp "$file" "$TEMP_DIR/$file"
done

# Sync the filtered files to GCS, preserving folder structure
gsutil -m rsync -r "$TEMP_DIR" "gs://$BUCKET_NAME/$BUCKET_PATH/"

# Clean up the temporary directory
rm -rf "$TEMP_DIR"

echo "Selected CSV and PNG files uploaded to gs://$BUCKET_NAME/$BUCKET_PATH/ excluding .gitignore entries"
