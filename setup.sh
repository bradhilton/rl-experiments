#!/bin/bash

# Export environment variables from .env, skipping commented lines
while IFS= read -r line; do
  # Skip empty lines and comments
  if [[ ! -z "$line" ]] && [[ ! "$line" =~ ^[[:space:]]*# ]]; then
    export "$line"
  fi
done < .env
git config --global user.name "$GIT_USER_NAME"
git config --global user.email "$GIT_USER_EMAIL"
gcloud auth activate-service-account --key-file .gcloud.json
sudo snap install --classic astral-uv
uv sync
uv remove torchtune
uv add git+https://github.com/pytorch/torchtune --rev 4b6877a6ef31a1f987c27594eaf8fe467b5ab785