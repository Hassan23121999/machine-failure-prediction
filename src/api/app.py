"""
FastAPI application for serving predictive maintenance models
"""
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import numpy as np
import pandas as pd
import joblib
import json
import logging
from datetime import datetime
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.data_processor import DataProcessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Predictive Maintenance API",
    description="Machine failure prediction for manufacturing equipment",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model and preprocessing
MODEL = None
PROCESSOR = None
MODEL_METADATA = None
FEATURE_NAMES = None


class SensorData(BaseModel):
    """Input schema for sensor data"""
    air_temperature: float = Field(..., description="Air temperature in Kelvin")
    process_temperature: float = Field(..., description="Process temperature in Kelvin")
    rotational_speed: int = Field(..., description="Rotational speed in rpm")
    torque: float = Field(..., description="Torque in Nm")
    tool_wear: int = Field(..., description="Tool wear in minutes")
    product_type: str = Field(..., description="Product type (L, M, or H)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "air_temperature": 298.1,
                "process_temperature": 308.6,
                "rotational_speed": 1551,
                "torque": 42.8,
                "tool_wear": 0,
                "product_type": "M"
            }
        }


class BatchSensorData(BaseModel):
    """Input schema for batch prediction"""
    data: List[SensorData]


class PredictionResponse(BaseModel):
    """Response schema for predictions"""
    prediction: int
    failure_probability: float
    risk_level: str
    confidence: float
    recommended_action: str
    timestamp: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "prediction": 0,
                "failure_probability": 0.15,
                "risk_level": "LOW",
                "confidence": 0.85,
                "recommended_action": "Continue normal operation",
                "timestamp": "2024-01-20T10:30:00"
            }
        }


class BatchPredictionResponse(BaseModel):
    """Response schema for batch predictions"""
    predictions: List[PredictionResponse]
    summary: Dict


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    model_info: Optional[Dict]
    timestamp: str


def prepare_features(sensor_data: SensorData) -> pd.DataFrame:
    """Prepare features from sensor data"""
    
    # Create base features
    features = {
        'Air temperature [K]': sensor_data.air_temperature,
        'Process temperature [K]': sensor_data.process_temperature,
        'Rotational speed [rpm]': sensor_data.rotational_speed,
        'Torque [Nm]': sensor_data.torque,
        'Tool wear [min]': sensor_data.tool_wear,
        'Temperature Difference [K]': sensor_data.process_temperature - sensor_data.air_temperature,
        'Power [W]': (sensor_data.torque * sensor_data.rotational_speed * 2 * np.pi) / 60,
        'Tool Wear Rate': sensor_data.tool_wear / 100,  # Normalized
        'Torque Speed Ratio': sensor_data.torque / (sensor_data.rotational_speed + 1),
        'Temp Stress': ((sensor_data.process_temperature - 308) / 5 + 
                       (sensor_data.air_temperature - 298) / 5)
    }
    
    # One-hot encode product type
    features['Type_H'] = 1 if sensor_data.product_type == 'H' else 0
    features['Type_L'] = 1 if sensor_data.product_type == 'L' else 0
    features['Type_M'] = 1 if sensor_data.product_type == 'M' else 0
    
    # Tool wear categories
    if sensor_data.tool_wear <= 50:
        features['Tool_Wear_Low'] = 1
        features['Tool_Wear_Medium'] = 0
        features['Tool_Wear_High'] = 0
    elif sensor_data.tool_wear <= 150:
        features['Tool_Wear_Low'] = 0
        features['Tool_Wear_Medium'] = 1
        features['Tool_Wear_High'] = 0
    else:
        features['Tool_Wear_Low'] = 0
        features['Tool_Wear_Medium'] = 0
        features['Tool_Wear_High'] = 1
    
    # Create DataFrame with correct column order
    df = pd.DataFrame([features])
    
    # Ensure all expected features are present
    if FEATURE_NAMES:
        # Reorder columns to match training data
        missing_cols = set(FEATURE_NAMES) - set(df.columns)
        for col in missing_cols:
            df[col] = 0
        df = df[FEATURE_NAMES]
    
    return df


def get_risk_level(probability: float) -> str:
    """Determine risk level based on failure probability"""
    if probability < 0.3:
        return "LOW"
    elif probability < 0.6:
        return "MEDIUM"
    elif probability < 0.8:
        return "HIGH"
    else:
        return "CRITICAL"


def get_recommended_action(risk_level: str, probability: float) -> str:
    """Get recommended action based on risk level"""
    actions = {
        "LOW": "Continue normal operation",
        "MEDIUM": "Schedule maintenance in next cycle",
        "HIGH": "Plan immediate maintenance",
        "CRITICAL": "STOP - Immediate maintenance required"
    }
    return actions.get(risk_level, "Review with maintenance team")


@app.on_event("startup")
async def load_model():
    """Load model on startup"""
    global MODEL, PROCESSOR, MODEL_METADATA, FEATURE_NAMES
    
    try:
        # Load model metadata
        with open("models/model_metadata.json", "r") as f:
            MODEL_METADATA = json.load(f)
        
        # Load the best model
        model_name = MODEL_METADATA.get("model_name", "xgboost")
        model_path = f"models/best_model_{model_name}.pkl"
        
        MODEL = joblib.load(model_path)
        
        # Initialize processor
        PROCESSOR = DataProcessor()
        
        # Get feature names from a dummy processing run
        dummy_data = pd.DataFrame({
            'Air temperature [K]': [298.0],
            'Process temperature [K]': [308.0],
            'Rotational speed [rpm]': [1500],
            'Torque [Nm]': [40.0],
            'Tool wear [min]': [100],
            'Type': ['M'],
            'Target': [0]
        })
        
        dummy_data = PROCESSOR.create_features(dummy_data)
        dummy_data = PROCESSOR.encode_categorical(dummy_data)
        X, _ = PROCESSOR.prepare_features(dummy_data)
        FEATURE_NAMES = X.columns.tolist()
        
        logger.info(f"Model loaded successfully: {model_name}")
        logger.info(f"Model performance - ROC AUC: {MODEL_METADATA['roc_auc_score']:.4f}")
        
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise


@app.get("/", response_model=Dict)
async def root():
    """Root endpoint"""
    return {
        "message": "Predictive Maintenance API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "batch_predict": "/batch_predict",
            "docs": "/docs"
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if MODEL is not None else "unhealthy",
        model_loaded=MODEL is not None,
        model_info=MODEL_METADATA if MODEL else None,
        timestamp=datetime.now().isoformat()
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(sensor_data: SensorData):
    """Single prediction endpoint"""
    
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Prepare features
        features_df = prepare_features(sensor_data)
        
        # Scale features (if processor has a scaler)
        if hasattr(PROCESSOR, 'scaler') and PROCESSOR.scaler:
            # Get numerical columns
            num_cols = [col for col in features_df.columns 
                       if not col.startswith('Type_') and not col.startswith('Tool_Wear_')]
            features_df[num_cols] = PROCESSOR.scaler.transform(features_df[num_cols])
        
        # Make prediction
        prediction = MODEL.predict(features_df)[0]
        probability = MODEL.predict_proba(features_df)[0, 1]
        
        # Determine risk level and action
        risk_level = get_risk_level(probability)
        action = get_recommended_action(risk_level, probability)
        
        # Calculate confidence (distance from 0.5)
        confidence = abs(probability - 0.5) * 2
        
        return PredictionResponse(
            prediction=int(prediction),
            failure_probability=float(probability),
            risk_level=risk_level,
            confidence=float(confidence),
            recommended_action=action,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/batch_predict", response_model=BatchPredictionResponse)
async def batch_predict(batch_data: BatchSensorData):
    """Batch prediction endpoint"""
    
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    predictions = []
    risk_levels = []
    
    for sensor_data in batch_data.data:
        try:
            # Prepare features
            features_df = prepare_features(sensor_data)
            
            # Scale features
            if hasattr(PROCESSOR, 'scaler') and PROCESSOR.scaler:
                num_cols = [col for col in features_df.columns 
                          if not col.startswith('Type_') and not col.startswith('Tool_Wear_')]
                features_df[num_cols] = PROCESSOR.scaler.transform(features_df[num_cols])
            
            # Make prediction
            prediction = MODEL.predict(features_df)[0]
            probability = MODEL.predict_proba(features_df)[0, 1]
            
            # Determine risk level and action
            risk_level = get_risk_level(probability)
            action = get_recommended_action(risk_level, probability)
            confidence = abs(probability - 0.5) * 2
            
            predictions.append(PredictionResponse(
                prediction=int(prediction),
                failure_probability=float(probability),
                risk_level=risk_level,
                confidence=float(confidence),
                recommended_action=action,
                timestamp=datetime.now().isoformat()
            ))
            
            risk_levels.append(risk_level)
            
        except Exception as e:
            logger.error(f"Batch prediction error for item: {str(e)}")
            predictions.append(None)
    
    # Calculate summary statistics
    valid_predictions = [p for p in predictions if p is not None]
    
    summary = {
        "total_items": len(batch_data.data),
        "successful_predictions": len(valid_predictions),
        "failed_predictions": len(predictions) - len(valid_predictions),
        "failure_rate": sum(p.prediction for p in valid_predictions) / len(valid_predictions) if valid_predictions else 0,
        "risk_distribution": {
            "LOW": risk_levels.count("LOW"),
            "MEDIUM": risk_levels.count("MEDIUM"),
            "HIGH": risk_levels.count("HIGH"),
            "CRITICAL": risk_levels.count("CRITICAL")
        }
    }
    
    return BatchPredictionResponse(
        predictions=valid_predictions,
        summary=summary
    )


@app.get("/model_info", response_model=Dict)
async def model_info():
    """Get model information"""
    
    if MODEL_METADATA is None:
        raise HTTPException(status_code=503, detail="Model metadata not loaded")
    
    return {
        "model_name": MODEL_METADATA.get("model_name"),
        "metrics": MODEL_METADATA.get("metrics"),
        "timestamp": MODEL_METADATA.get("timestamp"),
        "features_expected": len(FEATURE_NAMES) if FEATURE_NAMES else None,
        "feature_names": FEATURE_NAMES
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)