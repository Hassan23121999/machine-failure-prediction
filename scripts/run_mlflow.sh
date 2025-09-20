#!/bin/bash
echo "Starting MLflow UI..."
echo "Navigate to http://localhost:5000"
mlflow ui --backend-store-uri sqlite:///mlruns.db --port 5000