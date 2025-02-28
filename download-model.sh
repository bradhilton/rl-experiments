#! /bin/bash

# Check if model name argument is provided
if [ $# -eq 0 ]; then
    echo "Error: Model name argument is required."
    echo "Usage: $0 <model_name>"
    exit 1
fi

MODEL_NAME=$1

# Create the local directory if it doesn't exist
mkdir -p ./experiments/models/$MODEL_NAME

# Download the model from GCS to the local directory
gsutil -m rsync -r gs://atreides/openpipe/models/$MODEL_NAME ./experiments/models/$MODEL_NAME
