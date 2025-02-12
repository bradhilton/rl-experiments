#!/bin/bash

# Check for unstaged changes
if ! git diff --quiet; then
    echo "Error: You have unstaged changes. Please commit or stash them before launching the cluster."
    exit 1
fi

# Check for uncommitted changes
if ! git diff --cached --quiet; then
    echo "Error: You have uncommitted changes. Please commit them before launching the cluster."
    exit 1
fi

# Pull latest changes
echo "Pulling latest changes..."
if ! git pull; then
    echo "Error: Failed to pull latest changes."
    exit 1
fi

# Launch the cluster
sky launch cluster.yaml -c openpipe --env-file .env -y "$@"