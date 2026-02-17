import React, { useState } from 'react';
import axios from 'axios';
import { API } from '../App';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { Brain, Play, CheckCircle, Loader } from 'lucide-react';

const ModelTraining = () => {
  const [loading, setLoading] = useState(false);
  const [trainingResults, setTrainingResults] = useState(null);
  const [productId, setProductId] = useState('HOBBIES_1_001');
  const [forecastHorizon, setForecastHorizon] = useState(28);
  const [selectedModel, setSelectedModel] = useState('all');

  const trainModels = async () => {
    try {
      setLoading(true);
      const response = await axios.post(`${API}/models/train`, {
        product_id: productId,
        forecast_horizon: forecastHorizon
      });
      setTrainingResults(response.data);
    } catch (error) {
      console.error('Error training models:', error);
      alert('Error training models. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const getModelColor = (modelName) => {
    const colors = {
      'ARIMA': '#22d3ee',
      'SARIMA': '#a855f7',
      'Prophet': '#facc15',
      'LSTM': '#f472b6'
    };
    return colors[modelName] || '#22d3ee';
  };

  const prepareChartData = () => {
    if (!trainingResults) return [];
    
    const maxLength = Math.max(...trainingResults.metrics.map(m => m.predictions.length));
    const data = [];
    
    for (let i = 0; i < maxLength; i++) {
      const point = { day: i + 1 };
      trainingResults.metrics.forEach(metric => {
        if (selectedModel === 'all' || selectedModel === metric.model_name) {
          point[metric.model_name] = metric.predictions[i]?.toFixed(2);
        }
      });
      data.push(point);
    }
    
    return data;
  };

  return (
    <div className="p-8">
      <div className="mb-8">
        <h1 className="section-title text-white mb-2" data-testid="training-title">Model Training</h1>
        <p className="text-zinc-400 text-sm">Train and compare ARIMA, SARIMA, Prophet, and LSTM forecasting models</p>
      </div>

      {/* Training Controls */}
      <div className="glass-card rounded-xl p-6 mb-8" data-testid="training-controls">
        <h2 className="text-lg font-heading font-semibold text-white mb-4">Training Configuration</h2>
        
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
          <div>
            <label className="block text-sm text-zinc-400 mb-2">Product ID</label>
            <input
              type="text"
              value={productId}
              onChange={(e) => setProductId(e.target.value)}
              className="w-full bg-zinc-900/50 border border-white/10 rounded-lg px-4 py-2 text-white focus:outline-none focus:ring-2 focus:ring-cyan-400"
              data-testid="product-id-input"
            />
          </div>
          
          <div>
            <label className="block text-sm text-zinc-400 mb-2">Forecast Horizon (days)</label>
            <input
              type="number"
              value={forecastHorizon}
              onChange={(e) => setForecastHorizon(parseInt(e.target.value))}
              className="w-full bg-zinc-900/50 border border-white/10 rounded-lg px-4 py-2 text-white focus:outline-none focus:ring-2 focus:ring-cyan-400"
              data-testid="forecast-horizon-input"
            />
          </div>
          
          <div className="flex items-end">
            <button
              onClick={trainModels}
              disabled={loading}
              className="btn-primary w-full flex items-center justify-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed"
              data-testid="train-button"
            >
              {loading ? (
                <>
                  <Loader className="animate-spin" size={18} />
                  Training...
                </>
              ) : (
                <>
                  <Play size={18} />
                  Train All Models
                </>
              )}
            </button>
          </div>
        </div>

        {loading && (
          <div className="bg-cyan-400/10 border border-cyan-400/30 rounded-lg p-4">
            <div className="flex items-center gap-3">
              <Loader className="animate-spin text-cyan-400" size={20} />
              <div>
                <div className="text-sm font-semibold text-white">Training in Progress</div>
                <div className="text-xs text-zinc-400">This may take 30-60 seconds. Training ARIMA, SARIMA, Prophet, and LSTM...</div>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Training Results */}
      {trainingResults && (
        <>
          {/* Metrics Cards */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
            {trainingResults.metrics.map((metric, idx) => (
              <div
                key={idx}
                className="glass-card rounded-xl p-6 hover:scale-105 transition-transform cursor-pointer"
                style={{ borderColor: `${getModelColor(metric.model_name)}40` }}
                onClick={() => setSelectedModel(selectedModel === metric.model_name ? 'all' : metric.model_name)}
                data-testid={`model-card-${metric.model_name.toLowerCase()}`}
              >
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-lg font-heading font-semibold" style={{ color: getModelColor(metric.model_name) }}>
                    {metric.model_name}
                  </h3>
                  {selectedModel === metric.model_name && <CheckCircle className="text-green-400" size={20} />}
                </div>
                
                <div className="space-y-2">
                  <div className="flex justify-between items-center">
                    <span className="text-xs text-zinc-400">RMSE</span>
                    <span className="data-value text-sm text-white">{metric.rmse.toFixed(3)}</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-xs text-zinc-400">MAE</span>
                    <span className="data-value text-sm text-white">{metric.mae.toFixed(3)}</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-xs text-zinc-400">MAPE</span>
                    <span className="data-value text-sm text-white">{metric.mape.toFixed(2)}%</span>
                  </div>
                  <div className="flex justify-between items-center pt-2 border-t border-white/10">
                    <span className="text-xs text-zinc-400">Training Time</span>
                    <span className="data-value text-sm text-cyan-400">{metric.training_time.toFixed(2)}s</span>
                  </div>
                </div>
              </div>
            ))}
          </div>

          {/* Forecast Visualization */}
          <div className="glass-card rounded-xl p-6 mb-8" data-testid="forecast-chart">
            <div className="flex items-center justify-between mb-4">
              <div>
                <h2 className="text-xl font-heading font-semibold text-white">Forecast Predictions</h2>
                <p className="text-xs text-zinc-400 mt-1">Click on a model card to isolate its forecast</p>
              </div>
              {selectedModel !== 'all' && (
                <button
                  onClick={() => setSelectedModel('all')}
                  className="text-sm text-cyan-400 hover:text-cyan-300"
                  data-testid="show-all-button"
                >
                  Show All Models
                </button>
              )}
            </div>
            
            <ResponsiveContainer width="100%" height={400}>
              <LineChart data={prepareChartData()}>
                <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
                <XAxis 
                  dataKey="day" 
                  stroke="#71717a" 
                  style={{ fontSize: '12px' }}
                  label={{ value: 'Days', position: 'insideBottom', offset: -5, fill: '#71717a' }}
                />
                <YAxis 
                  stroke="#71717a" 
                  style={{ fontSize: '12px' }}
                  label={{ value: 'Sales', angle: -90, position: 'insideLeft', fill: '#71717a' }}
                />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#18181b',
                    border: '1px solid rgba(255,255,255,0.1)',
                    borderRadius: '8px',
                    color: '#f4f4f5'
                  }}
                />
                <Legend />
                {trainingResults.metrics.map((metric) => {
                  if (selectedModel !== 'all' && selectedModel !== metric.model_name) return null;
                  return (
                    <Line
                      key={metric.model_name}
                      type="monotone"
                      dataKey={metric.model_name}
                      stroke={getModelColor(metric.model_name)}
                      strokeWidth={2}
                      dot={false}
                      activeDot={{ r: 5 }}
                    />
                  );
                })}
              </LineChart>
            </ResponsiveContainer>
          </div>

          {/* Model Insights */}
          <div className="glass-card rounded-xl p-6 bg-gradient-to-br from-cyan-400/5 to-purple-400/5 border border-cyan-400/20">
            <h3 className="text-lg font-heading font-semibold text-white mb-3">Training Summary</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm text-zinc-300">
              <div>
                <p className="mb-2">• All four models trained successfully on {forecastHorizon}-day forecast horizon</p>
                <p className="mb-2">• ARIMA and SARIMA are traditional statistical models with high interpretability</p>
                <p className="mb-2">• Prophet handles seasonality and holidays automatically</p>
              </div>
              <div>
                <p className="mb-2">• LSTM is a deep learning model that captures complex patterns</p>
                <p className="mb-2">• Lower RMSE and MAE indicate better forecast accuracy</p>
                <p className="mb-2">• MAPE shows percentage error - useful for business understanding</p>
              </div>
            </div>
          </div>
        </>
      )}

      {!trainingResults && !loading && (
        <div className="glass-card rounded-xl p-12 text-center">
          <Brain className="text-zinc-600 mx-auto mb-4" size={48} />
          <h3 className="text-xl font-heading font-semibold text-zinc-400 mb-2">No Training Results Yet</h3>
          <p className="text-sm text-zinc-500 mb-6">Configure your settings above and click "Train All Models" to begin</p>
        </div>
      )}
    </div>
  );
};

export default ModelTraining;