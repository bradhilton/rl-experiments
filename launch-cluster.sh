#!/bin/bash

# Check for unstaged changes
if ! git diff --quiet; then
    echo "Warning: You have unstaged changes. Unstaged changes will be discarded from the cluster working directory."
fi

# Check for uncommitted changes
if ! git diff --cached --quiet; then
    echo "Warning: You have uncommitted changes. Uncommitted changes will be discarded from the cluster working directory."
fi

# Pull latest changes
echo "Pulling latest changes..."
if ! git pull; then
    echo "Error: Failed to pull latest changes."
    exit 1
fi

# Launch the cluster
sky launch cluster.yaml -c openpipe --env-file .env -y "$@"