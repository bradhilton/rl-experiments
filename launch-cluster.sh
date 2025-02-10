#!/bin/bash
export $(cat .env | xargs) && sky launch cluster.yaml -y --name openpipe-rl "$@"