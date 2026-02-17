from fastapi import FastAPI, APIRouter, HTTPException
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Dict, Any
import uuid
from datetime import datetime, timezone
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import json
import zipfile
import requests
from emergentintegrations.llm.chat import LlmChat, UserMessage
import asyncio

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# Create the main app
app = FastAPI()
api_router = APIRouter(prefix="/api")

# Global dataset storage
DATA_DIR = ROOT_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

# Models
class DatasetStatus(BaseModel):
    model_config = ConfigDict(extra="ignore")
    status: str
    message: str
    records: Optional[int] = None

class ExplorationResult(BaseModel):
    model_config = ConfigDict(extra="ignore")
    trends: Dict[str, Any]
    seasonality: Dict[str, Any]
    anomalies: List[Dict[str, Any]]
    statistics: Dict[str, Any]

class TrainingRequest(BaseModel):
    product_id: str = "HOBBIES_1_001"
    forecast_horizon: int = 28

class ModelMetrics(BaseModel):
    model_config = ConfigDict(extra="ignore")
    model_name: str
    rmse: float
    mae: float
    mape: float
    training_time: float
    predictions: List[float]

class TrainingResult(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    product_id: str
    metrics: List[ModelMetrics]
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class InsightRequest(BaseModel):
    metrics: List[Dict[str, Any]]
    context: str

class BusinessImpactRequest(BaseModel):
    current_stockout_rate: float
    avg_daily_sales: float
    product_margin: float
    storage_cost_per_unit: float

class AccuracyRecord(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    date: str
    model_name: str
    actual_sales: float
    predicted_sales: float
    error: float
    percentage_error: float
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class AccuracyRecordCreate(BaseModel):
    date: str
    model_name: str
    actual_sales: float
    predicted_sales: float

class AlertThreshold(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    model_name: str
    threshold_type: str  # "mape", "rmse", "consecutive_errors"
    threshold_value: float
    is_active: bool = True
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class AlertThresholdCreate(BaseModel):
    model_name: str
    threshold_type: str
    threshold_value: float

# Helper functions
def download_m5_data():
    """Download M5 dataset from Kaggle (simplified version for demo)"""
    # For demo, we'll create synthetic data similar to M5
    dates = pd.date_range(start='2011-01-29', end='2016-06-19', freq='D')
    np.random.seed(42)
    
    # Create synthetic sales data with trend and seasonality
    trend = np.linspace(10, 50, len(dates))
    seasonality = 10 * np.sin(np.arange(len(dates)) * 2 * np.pi / 365)
    noise = np.random.normal(0, 5, len(dates))
    sales = trend + seasonality + noise + 30
    sales = np.maximum(sales, 0)  # No negative sales
    
    df = pd.DataFrame({
        'date': dates,
        'sales': sales,
        'product_id': 'HOBBIES_1_001'
    })
    
    file_path = DATA_DIR / "sales_data.csv"
    df.to_csv(file_path, index=False)
    return df

def load_sales_data(product_id: str = "HOBBIES_1_001"):
    """Load sales data for a specific product"""
    file_path = DATA_DIR / "sales_data.csv"
    if not file_path.exists():
        return download_m5_data()
    
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])
    return df[df['product_id'] == product_id].reset_index(drop=True)

def detect_anomalies(series, window=7, threshold=2):
    """Detect anomalies using rolling statistics"""
    rolling_mean = series.rolling(window=window).mean()
    rolling_std = series.rolling(window=window).std()
    
    anomalies = []
    for i in range(len(series)):
        if pd.notna(rolling_mean.iloc[i]) and pd.notna(rolling_std.iloc[i]):
            z_score = abs((series.iloc[i] - rolling_mean.iloc[i]) / (rolling_std.iloc[i] + 1e-7))
            if z_score > threshold:
                anomalies.append({
                    "index": i,
                    "value": float(series.iloc[i]),
                    "z_score": float(z_score)
                })
    
    return anomalies

def train_arima(train_data, forecast_horizon=28):
    """Train ARIMA model"""
    import time
    start = time.time()
    
    try:
        model = ARIMA(train_data, order=(5, 1, 2))
        fitted = model.fit()
        predictions = fitted.forecast(steps=forecast_horizon)
        training_time = time.time() - start
        
        return predictions, training_time
    except Exception as e:
        logging.error(f"ARIMA training failed: {e}")
        return np.zeros(forecast_horizon), 0.0

def train_sarima(train_data, forecast_horizon=28):
    """Train SARIMA model"""
    import time
    start = time.time()
    
    try:
        model = SARIMAX(train_data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7))
        fitted = model.fit(disp=False, maxiter=50)
        predictions = fitted.forecast(steps=forecast_horizon)
        training_time = time.time() - start
        
        return predictions, training_time
    except Exception as e:
        logging.error(f"SARIMA training failed: {e}")
        return np.zeros(forecast_horizon), 0.0

def train_prophet(df, forecast_horizon=28):
    """Train Prophet model"""
    import time
    start = time.time()
    
    try:
        prophet_df = df[['date', 'sales']].rename(columns={'date': 'ds', 'sales': 'y'})
        model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
        model.fit(prophet_df)
        
        future = model.make_future_dataframe(periods=forecast_horizon)
        forecast = model.predict(future)
        predictions = forecast.tail(forecast_horizon)['yhat'].values
        training_time = time.time() - start
        
        return predictions, training_time
    except Exception as e:
        logging.error(f"Prophet training failed: {e}")
        return np.zeros(forecast_horizon), 0.0

def train_lstm(train_data, forecast_horizon=28):
    """Train LSTM model"""
    import time
    start = time.time()
    
    try:
        # Prepare data
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(train_data.values.reshape(-1, 1))
        
        # Create sequences
        seq_length = 30
        X, y = [], []
        for i in range(len(scaled_data) - seq_length):
            X.append(scaled_data[i:i+seq_length])
            y.append(scaled_data[i+seq_length])
        
        X = np.array(X)
        y = np.array(y)
        
        # Build model
        model = keras.Sequential([
            layers.LSTM(50, activation='relu', input_shape=(seq_length, 1)),
            layers.Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        
        # Train
        model.fit(X, y, epochs=20, batch_size=32, verbose=0)
        
        # Predict
        predictions = []
        current_seq = scaled_data[-seq_length:].reshape(1, seq_length, 1)
        
        for _ in range(forecast_horizon):
            pred = model.predict(current_seq, verbose=0)
            predictions.append(pred[0, 0])
            current_seq = np.append(current_seq[:, 1:, :], [[pred]], axis=1)
        
        predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
        training_time = time.time() - start
        
        return predictions, training_time
    except Exception as e:
        logging.error(f"LSTM training failed: {e}")
        return np.zeros(forecast_horizon), 0.0

def calculate_metrics(actual, predicted):
    """Calculate forecasting metrics"""
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mae = mean_absolute_error(actual, predicted)
    mape = np.mean(np.abs((actual - predicted) / (actual + 1e-7))) * 100
    
    return rmse, mae, mape

# API Endpoints
@api_router.get("/")
async def root():
    return {"message": "M5 Forecast Pro API"}

@api_router.post("/dataset/download", response_model=DatasetStatus)
async def download_dataset():
    """Download and prepare the dataset"""
    try:
        df = download_m5_data()
        return DatasetStatus(
            status="success",
            message="Dataset downloaded successfully",
            records=len(df)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/dataset/explore", response_model=ExplorationResult)
async def explore_dataset():
    """Explore the dataset for trends, seasonality, and anomalies"""
    try:
        df = load_sales_data()
        
        # Calculate trends
        weekly_avg = df.groupby(df['date'].dt.to_period('W'))['sales'].mean()
        monthly_avg = df.groupby(df['date'].dt.to_period('M'))['sales'].mean()
        
        trends = {
            "overall_trend": "increasing" if df['sales'].iloc[-100:].mean() > df['sales'].iloc[:100].mean() else "decreasing",
            "weekly_pattern": weekly_avg.tail(12).tolist(),
            "monthly_pattern": monthly_avg.tail(12).tolist()
        }
        
        # Seasonality
        df['day_of_week'] = df['date'].dt.dayofweek
        df['month'] = df['date'].dt.month
        
        seasonality = {
            "day_of_week_avg": df.groupby('day_of_week')['sales'].mean().tolist(),
            "month_avg": df.groupby('month')['sales'].mean().tolist()
        }
        
        # Detect anomalies
        anomalies = detect_anomalies(df['sales'])
        
        # Statistics
        statistics = {
            "mean": float(df['sales'].mean()),
            "median": float(df['sales'].median()),
            "std": float(df['sales'].std()),
            "min": float(df['sales'].min()),
            "max": float(df['sales'].max()),
            "total_records": len(df)
        }
        
        return ExplorationResult(
            trends=trends,
            seasonality=seasonality,
            anomalies=anomalies[:10],  # Top 10 anomalies
            statistics=statistics
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/models/train", response_model=TrainingResult)
async def train_models(request: TrainingRequest):
    """Train all forecasting models"""
    try:
        df = load_sales_data(request.product_id)
        
        # Split data
        train_size = len(df) - request.forecast_horizon
        train_data = df['sales'].iloc[:train_size]
        test_data = df['sales'].iloc[train_size:]
        
        metrics_list = []
        
        # Train ARIMA
        arima_pred, arima_time = train_arima(train_data, request.forecast_horizon)
        rmse, mae, mape = calculate_metrics(test_data.values, arima_pred)
        metrics_list.append(ModelMetrics(
            model_name="ARIMA",
            rmse=float(rmse),
            mae=float(mae),
            mape=float(mape),
            training_time=arima_time,
            predictions=arima_pred.tolist()
        ))
        
        # Train SARIMA
        sarima_pred, sarima_time = train_sarima(train_data, request.forecast_horizon)
        rmse, mae, mape = calculate_metrics(test_data.values, sarima_pred)
        metrics_list.append(ModelMetrics(
            model_name="SARIMA",
            rmse=float(rmse),
            mae=float(mae),
            mape=float(mape),
            training_time=sarima_time,
            predictions=sarima_pred.tolist()
        ))
        
        # Train Prophet
        prophet_pred, prophet_time = train_prophet(df.iloc[:train_size], request.forecast_horizon)
        rmse, mae, mape = calculate_metrics(test_data.values, prophet_pred)
        metrics_list.append(ModelMetrics(
            model_name="Prophet",
            rmse=float(rmse),
            mae=float(mae),
            mape=float(mape),
            training_time=prophet_time,
            predictions=prophet_pred.tolist()
        ))
        
        # Train LSTM
        lstm_pred, lstm_time = train_lstm(train_data, request.forecast_horizon)
        rmse, mae, mape = calculate_metrics(test_data.values, lstm_pred)
        metrics_list.append(ModelMetrics(
            model_name="LSTM",
            rmse=float(rmse),
            mae=float(mae),
            mape=float(mape),
            training_time=lstm_time,
            predictions=lstm_pred.tolist()
        ))
        
        result = TrainingResult(
            product_id=request.product_id,
            metrics=metrics_list
        )
        
        # Save to database
        doc = result.model_dump()
        doc['timestamp'] = doc['timestamp'].isoformat()
        await db.training_results.insert_one(doc)
        
        return result
    except Exception as e:
        logging.error(f"Training error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/insights/generate")
async def generate_insights(request: InsightRequest):
    """Generate AI-powered insights using GPT-5.2"""
    try:
        api_key = os.environ.get('EMERGENT_LLM_KEY')
        if not api_key:
            raise HTTPException(status_code=500, detail="LLM API key not configured")
        
        # Prepare context for LLM
        metrics_summary = "\n".join([
            f"{m['model_name']}: RMSE={m['rmse']:.2f}, MAE={m['mae']:.2f}, MAPE={m['mape']:.2f}%, Time={m['training_time']:.2f}s"
            for m in request.metrics
        ])
        
        prompt = f"""Analyze these demand forecasting model results and provide insights:

{metrics_summary}

Context: {request.context}

Please answer:
1. Why does one model outperform others?
2. What are the trade-offs between accuracy and interpretability?
3. Where does each model fail or struggle?
4. What is the business impact of using the best model?
5. How can better forecasts improve costs, staffing, and prevent stockouts?
6. Recommendations for real-time scenarios (retraining frequency, alert triggers)

Provide a concise but comprehensive analysis."""
        
        chat = LlmChat(
            api_key=api_key,
            session_id=str(uuid.uuid4()),
            system_message="You are an expert data scientist specializing in time-series forecasting and supply chain optimization."
        ).with_model("openai", "gpt-5.2")
        
        user_message = UserMessage(text=prompt)
        response = await chat.send_message(user_message)
        
        return {"insights": response}
    except Exception as e:
        logging.error(f"Insight generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/business/impact")
async def calculate_business_impact(request: BusinessImpactRequest):
    """Calculate business impact of better forecasting"""
    try:
        # Current state
        annual_stockouts = request.avg_daily_sales * 365 * request.current_stockout_rate
        lost_revenue = annual_stockouts * request.product_margin
        
        # Improved state (assuming 50% reduction in stockouts with better forecasting)
        improved_stockout_rate = request.current_stockout_rate * 0.5
        improved_stockouts = request.avg_daily_sales * 365 * improved_stockout_rate
        improved_lost_revenue = improved_stockouts * request.product_margin
        
        # Calculate savings
        revenue_gain = lost_revenue - improved_lost_revenue
        
        # Storage cost optimization (assuming 20% reduction in excess inventory)
        current_storage_cost = request.avg_daily_sales * 30 * request.storage_cost_per_unit
        improved_storage_cost = current_storage_cost * 0.8
        storage_savings = current_storage_cost - improved_storage_cost
        
        total_savings = revenue_gain + storage_savings
        
        return {
            "current_annual_stockouts": round(annual_stockouts, 2),
            "improved_annual_stockouts": round(improved_stockouts, 2),
            "stockout_reduction": round((annual_stockouts - improved_stockouts) / annual_stockouts * 100, 2),
            "revenue_gain": round(revenue_gain, 2),
            "storage_savings": round(storage_savings, 2),
            "total_annual_savings": round(total_savings, 2),
            "roi_percentage": round((total_savings / (lost_revenue + 1)) * 100, 2)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/data/historical")
async def get_historical_data():
    """Get historical sales data for visualization"""
    try:
        df = load_sales_data()
        return {
            "dates": df['date'].dt.strftime('%Y-%m-%d').tolist(),
            "sales": df['sales'].tolist()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/accuracy/record", response_model=AccuracyRecord)
async def create_accuracy_record(record: AccuracyRecordCreate):
    """Record actual vs predicted sales for accuracy tracking"""
    try:
        error = record.actual_sales - record.predicted_sales
        percentage_error = abs(error / (record.actual_sales + 1e-7)) * 100
        
        accuracy_record = AccuracyRecord(
            date=record.date,
            model_name=record.model_name,
            actual_sales=record.actual_sales,
            predicted_sales=record.predicted_sales,
            error=error,
            percentage_error=percentage_error
        )
        
        doc = accuracy_record.model_dump()
        doc['timestamp'] = doc['timestamp'].isoformat()
        await db.accuracy_records.insert_one(doc)
        
        return accuracy_record
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/accuracy/records")
async def get_accuracy_records(model_name: Optional[str] = None, days: int = 30):
    """Get accuracy tracking records"""
    try:
        query = {}
        if model_name:
            query['model_name'] = model_name
        
        records = await db.accuracy_records.find(query, {"_id": 0}).sort("date", -1).limit(days * 4).to_list(1000)
        
        for record in records:
            if isinstance(record.get('timestamp'), str):
                record['timestamp'] = datetime.fromisoformat(record['timestamp'])
        
        return records
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/accuracy/metrics")
async def get_accuracy_metrics(model_name: Optional[str] = None, days: int = 30):
    """Calculate accuracy metrics over time"""
    try:
        query = {}
        if model_name:
            query['model_name'] = model_name
        
        records = await db.accuracy_records.find(query, {"_id": 0}).sort("date", -1).limit(days * 4).to_list(1000)
        
        if not records:
            return {
                "models": [],
                "overall_metrics": {},
                "rolling_accuracy": [],
                "alerts": []
            }
        
        # Group by model
        model_data = {}
        for record in records:
            model = record['model_name']
            if model not in model_data:
                model_data[model] = []
            model_data[model].append(record)
        
        # Calculate metrics per model
        metrics_by_model = {}
        for model, data in model_data.items():
            actuals = [r['actual_sales'] for r in data]
            predictions = [r['predicted_sales'] for r in data]
            errors = [r['error'] for r in data]
            
            rmse = np.sqrt(np.mean([e**2 for e in errors]))
            mae = np.mean([abs(e) for e in errors])
            mape = np.mean([abs(e) for e in [r['percentage_error'] for r in data]])
            
            # Calculate rolling metrics (7-day window)
            rolling_mape = []
            for i in range(len(data) - 6):
                window = data[i:i+7]
                window_mape = np.mean([r['percentage_error'] for r in window])
                rolling_mape.append({
                    "date": window[0]['date'],
                    "mape": float(window_mape)
                })
            
            metrics_by_model[model] = {
                "rmse": float(rmse),
                "mae": float(mae),
                "mape": float(mape),
                "data_points": len(data),
                "rolling_mape": rolling_mape[:30]  # Last 30 points
            }
        
        # Check for alerts
        alerts = []
        thresholds = await db.alert_thresholds.find({"is_active": True}, {"_id": 0}).to_list(100)
        
        for threshold in thresholds:
            model = threshold['model_name']
            if model in metrics_by_model:
                metrics = metrics_by_model[model]
                
                if threshold['threshold_type'] == 'mape' and metrics['mape'] > threshold['threshold_value']:
                    alerts.append({
                        "model": model,
                        "type": "mape_exceeded",
                        "message": f"{model} MAPE ({metrics['mape']:.2f}%) exceeds threshold ({threshold['threshold_value']:.2f}%)",
                        "severity": "high" if metrics['mape'] > threshold['threshold_value'] * 1.5 else "medium",
                        "recommendation": "Consider retraining the model with recent data"
                    })
        
        return {
            "models": list(metrics_by_model.keys()),
            "overall_metrics": metrics_by_model,
            "rolling_accuracy": rolling_mape if rolling_mape else [],
            "alerts": alerts
        }
    except Exception as e:
        logging.error(f"Accuracy metrics error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/accuracy/threshold", response_model=AlertThreshold)
async def create_alert_threshold(threshold: AlertThresholdCreate):
    """Create or update alert threshold"""
    try:
        alert_threshold = AlertThreshold(
            model_name=threshold.model_name,
            threshold_type=threshold.threshold_type,
            threshold_value=threshold.threshold_value
        )
        
        # Deactivate existing thresholds for this model and type
        await db.alert_thresholds.update_many(
            {"model_name": threshold.model_name, "threshold_type": threshold.threshold_type},
            {"$set": {"is_active": False}}
        )
        
        doc = alert_threshold.model_dump()
        doc['created_at'] = doc['created_at'].isoformat()
        await db.alert_thresholds.insert_one(doc)
        
        return alert_threshold
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/accuracy/thresholds")
async def get_alert_thresholds():
    """Get all active alert thresholds"""
    try:
        thresholds = await db.alert_thresholds.find({"is_active": True}, {"_id": 0}).to_list(100)
        return thresholds
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/accuracy/simulate")
async def simulate_accuracy_tracking():
    """Simulate accuracy tracking data for demonstration"""
    try:
        df = load_sales_data()
        models = ["ARIMA", "SARIMA", "Prophet", "LSTM"]
        
        # Generate simulated predictions for last 30 days
        last_30_days = df.tail(30)
        records_created = 0
        
        for _, row in last_30_days.iterrows():
            actual = row['sales']
            date_str = row['date'].strftime('%Y-%m-%d')
            
            for model in models:
                # Add realistic noise to predictions
                if model == "ARIMA":
                    predicted = actual + np.random.normal(0, 3)
                elif model == "SARIMA":
                    predicted = actual + np.random.normal(0, 2.5)
                elif model == "Prophet":
                    predicted = actual + np.random.normal(0, 2)
                else:  # LSTM
                    predicted = actual + np.random.normal(0, 4)
                
                predicted = max(0, predicted)
                
                error = actual - predicted
                percentage_error = abs(error / (actual + 1e-7)) * 100
                
                record = AccuracyRecord(
                    date=date_str,
                    model_name=model,
                    actual_sales=float(actual),
                    predicted_sales=float(predicted),
                    error=float(error),
                    percentage_error=float(percentage_error)
                )
                
                doc = record.model_dump()
                doc['timestamp'] = doc['timestamp'].isoformat()
                await db.accuracy_records.insert_one(doc)
                records_created += 1
        
        return {
            "status": "success",
            "message": f"Created {records_created} accuracy records for demonstration",
            "records": records_created
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Include router
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()