import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { API } from '../App';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, BarChart, Bar } from 'recharts';
import { Target, AlertTriangle, TrendingDown, CheckCircle, RefreshCw, Bell } from 'lucide-react';

const AccuracyTracker = () => {
  const [loading, setLoading] = useState(true);
  const [metrics, setMetrics] = useState(null);
  const [records, setRecords] = useState([]);
  const [thresholds, setThresholds] = useState([]);
  const [selectedModel, setSelectedModel] = useState('all');
  const [simulating, setSimulating] = useState(false);
  const [newThreshold, setNewThreshold] = useState({
    model_name: 'ARIMA',
    threshold_type: 'mape',
    threshold_value: 10
  });

  useEffect(() => {
    loadAccuracyData();
  }, []);

  const loadAccuracyData = async () => {
    try {
      setLoading(true);
      
      const [metricsRes, recordsRes, thresholdsRes] = await Promise.all([
        axios.get(`${API}/accuracy/metrics?days=30`),
        axios.get(`${API}/accuracy/records?days=30`),
        axios.get(`${API}/accuracy/thresholds`)
      ]);
      
      setMetrics(metricsRes.data);
      setRecords(recordsRes.data);
      setThresholds(thresholdsRes.data);
    } catch (error) {
      console.error('Error loading accuracy data:', error);
    } finally {
      setLoading(false);
    }
  };

  const simulateData = async () => {
    try {
      setSimulating(true);
      await axios.post(`${API}/accuracy/simulate`);
      await loadAccuracyData();
    } catch (error) {
      console.error('Error simulating data:', error);
      alert('Error generating simulation data');
    } finally {
      setSimulating(false);
    }
  };

  const createThreshold = async () => {
    try {
      await axios.post(`${API}/accuracy/threshold`, newThreshold);
      await loadAccuracyData();
      alert('Alert threshold created successfully!');
    } catch (error) {
      console.error('Error creating threshold:', error);
      alert('Error creating threshold');
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

  const prepareActualVsPredictedChart = () => {
    if (!records || records.length === 0) return [];
    
    // Group by date
    const dataByDate = {};
    records.forEach(record => {
      if (!dataByDate[record.date]) {
        dataByDate[record.date] = { date: new Date(record.date).toLocaleDateString('en-US', { month: 'short', day: 'numeric' }) };
      }
      if (selectedModel === 'all' || selectedModel === record.model_name) {
        dataByDate[record.date][`${record.model_name}_pred`] = Math.round(record.predicted_sales);
        dataByDate[record.date]['actual'] = Math.round(record.actual_sales);
      }
    });
    
    return Object.values(dataByDate).slice(-20);
  };

  const prepareErrorChart = () => {
    if (!metrics || !metrics.overall_metrics) return [];
    
    return Object.keys(metrics.overall_metrics).map(model => ({
      model,
      mape: metrics.overall_metrics[model].mape,
      mae: metrics.overall_metrics[model].mae,
      color: getModelColor(model)
    }));
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-cyan-400 mx-auto mb-4"></div>
          <p className="text-zinc-400">Loading accuracy tracker...</p>
        </div>
      </div>
    );
  }

  const hasData = metrics && metrics.models && metrics.models.length > 0;

  return (
    <div className="p-8">
      <div className="mb-8">
        <h1 className="section-title text-white mb-2" data-testid="accuracy-tracker-title">Forecast Accuracy Tracker</h1>
        <p className="text-zinc-400 text-sm">Monitor actual vs predicted sales and track model performance over time</p>
      </div>

      {/* Alerts Section */}
      {hasData && metrics.alerts && metrics.alerts.length > 0 && (
        <div className="mb-8 space-y-3">
          {metrics.alerts.map((alert, idx) => (
            <div 
              key={idx}
              className={`glass-card rounded-xl p-4 border-2 ${
                alert.severity === 'high' ? 'border-red-400/50 bg-red-400/10' : 'border-yellow-400/50 bg-yellow-400/10'
              }`}
              data-testid={`alert-${idx}`}
            >
              <div className="flex items-start gap-3">
                <AlertTriangle className={alert.severity === 'high' ? 'text-red-400' : 'text-yellow-400'} size={24} />
                <div className="flex-1">
                  <div className="text-sm font-semibold text-white mb-1">{alert.type.replace('_', ' ').toUpperCase()}</div>
                  <div className="text-sm text-zinc-300 mb-2">{alert.message}</div>
                  <div className="text-xs text-zinc-400 flex items-center gap-2">
                    <RefreshCw size={12} />
                    {alert.recommendation}
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Metrics Cards */}
      {hasData && (
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-8">
          {Object.keys(metrics.overall_metrics).map((model, idx) => {
            const modelMetrics = metrics.overall_metrics[model];
            return (
              <div
                key={idx}
                className={`metric-card rounded-xl p-5 cursor-pointer transition-all ${
                  selectedModel === model ? 'ring-2 ring-cyan-400' : ''
                }`}
                style={{ borderColor: `${getModelColor(model)}40` }}
                onClick={() => setSelectedModel(selectedModel === model ? 'all' : model)}
                data-testid={`accuracy-card-${model.toLowerCase()}`}
              >
                <div className="flex items-center justify-between mb-3">
                  <h3 className="text-lg font-heading font-semibold" style={{ color: getModelColor(model) }}>
                    {model}
                  </h3>
                  {modelMetrics.mape < 10 ? (
                    <CheckCircle className="text-green-400" size={20} />
                  ) : (
                    <TrendingDown className="text-yellow-400" size={20} />
                  )}
                </div>
                
                <div className="space-y-2">
                  <div className="flex justify-between items-center">
                    <span className="text-xs text-zinc-400">MAPE</span>
                    <span className="data-value text-sm text-white">{modelMetrics.mape.toFixed(2)}%</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-xs text-zinc-400">MAE</span>
                    <span className="data-value text-sm text-white">{modelMetrics.mae.toFixed(2)}</span>
                  </div>
                  <div className="flex justify-between items-center pt-2 border-t border-white/10">
                    <span className="text-xs text-zinc-400">Data Points</span>
                    <span className="data-value text-sm text-cyan-400">{modelMetrics.data_points}</span>
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      )}

      {/* Actual vs Predicted Chart */}
      {hasData && (
        <div className="glass-card rounded-xl p-6 mb-8" data-testid="actual-vs-predicted-chart">
          <div className="flex items-center justify-between mb-4">
            <div>
              <h2 className="text-xl font-heading font-semibold text-white">Actual vs Predicted Sales</h2>
              <p className="text-xs text-zinc-400 mt-1">Last 20 days - Click model card to filter</p>
            </div>
            {selectedModel !== 'all' && (
              <button
                onClick={() => setSelectedModel('all')}
                className="text-sm text-cyan-400 hover:text-cyan-300"
                data-testid="show-all-models-btn"
              >
                Show All Models
              </button>
            )}
          </div>
          
          <ResponsiveContainer width="100%" height={400}>
            <LineChart data={prepareActualVsPredictedChart()}>
              <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
              <XAxis 
                dataKey="date" 
                stroke="#71717a" 
                style={{ fontSize: '11px' }}
                angle={-45}
                textAnchor="end"
                height={80}
              />
              <YAxis stroke="#71717a" style={{ fontSize: '11px' }} />
              <Tooltip
                contentStyle={{
                  backgroundColor: '#18181b',
                  border: '1px solid rgba(255,255,255,0.1)',
                  borderRadius: '8px',
                  color: '#f4f4f5'
                }}
              />
              <Legend />
              <Line 
                type="monotone" 
                dataKey="actual" 
                stroke="#ffffff" 
                strokeWidth={3}
                dot={{ fill: '#ffffff', r: 4 }}
                name="Actual Sales"
              />
              {metrics.models.map((model) => {
                if (selectedModel !== 'all' && selectedModel !== model) return null;
                return (
                  <Line
                    key={model}
                    type="monotone"
                    dataKey={`${model}_pred`}
                    stroke={getModelColor(model)}
                    strokeWidth={2}
                    strokeDasharray="5 5"
                    dot={false}
                    name={`${model} Predicted`}
                  />
                );
              })}
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Error Metrics Chart */}
      {hasData && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
          <div className="glass-card rounded-xl p-6" data-testid="mape-chart">
            <div className="flex items-center gap-2 mb-4">
              <Target className="text-cyan-400" size={20} />
              <h3 className="text-lg font-heading font-semibold text-white">MAPE by Model (%)</h3>
            </div>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={prepareErrorChart()}>
                <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
                <XAxis dataKey="model" stroke="#71717a" style={{ fontSize: '11px' }} />
                <YAxis stroke="#71717a" style={{ fontSize: '11px' }} />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#18181b',
                    border: '1px solid rgba(255,255,255,0.1)',
                    borderRadius: '8px'
                  }}
                />
                <Bar dataKey="mape" radius={[8, 8, 0, 0]}>
                  {prepareErrorChart().map((entry, index) => (
                    <bar key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>

          <div className="glass-card rounded-xl p-6" data-testid="mae-comparison">
            <div className="flex items-center gap-2 mb-4">
              <TrendingDown className="text-purple-400" size={20} />
              <h3 className="text-lg font-heading font-semibold text-white">MAE by Model</h3>
            </div>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={prepareErrorChart()}>
                <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
                <XAxis dataKey="model" stroke="#71717a" style={{ fontSize: '11px' }} />
                <YAxis stroke="#71717a" style={{ fontSize: '11px' }} />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#18181b',
                    border: '1px solid rgba(255,255,255,0.1)',
                    borderRadius: '8px'
                  }}
                />
                <Bar dataKey="mae" radius={[8, 8, 0, 0]}>
                  {prepareErrorChart().map((entry, index) => (
                    <bar key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {/* Alert Configuration */}
      <div className="glass-card rounded-xl p-6 mb-8" data-testid="alert-config">
        <div className="flex items-center gap-2 mb-4">
          <Bell className="text-yellow-400" size={24} />
          <h3 className="text-lg font-heading font-semibold text-white">Alert Configuration</h3>
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-4">
          <div>
            <label className="block text-sm text-zinc-400 mb-2">Model</label>
            <select
              value={newThreshold.model_name}
              onChange={(e) => setNewThreshold({...newThreshold, model_name: e.target.value})}
              className="w-full bg-zinc-900/50 border border-white/10 rounded-lg px-4 py-2 text-white focus:outline-none focus:ring-2 focus:ring-cyan-400"
              data-testid="threshold-model-select"
            >
              <option value="ARIMA">ARIMA</option>
              <option value="SARIMA">SARIMA</option>
              <option value="Prophet">Prophet</option>
              <option value="LSTM">LSTM</option>
            </select>
          </div>
          
          <div>
            <label className="block text-sm text-zinc-400 mb-2">Metric</label>
            <select
              value={newThreshold.threshold_type}
              onChange={(e) => setNewThreshold({...newThreshold, threshold_type: e.target.value})}
              className="w-full bg-zinc-900/50 border border-white/10 rounded-lg px-4 py-2 text-white focus:outline-none focus:ring-2 focus:ring-cyan-400"
              data-testid="threshold-type-select"
            >
              <option value="mape">MAPE (%)</option>
              <option value="rmse">RMSE</option>
              <option value="mae">MAE</option>
            </select>
          </div>
          
          <div>
            <label className="block text-sm text-zinc-400 mb-2">Threshold Value</label>
            <input
              type="number"
              value={newThreshold.threshold_value}
              onChange={(e) => setNewThreshold({...newThreshold, threshold_value: parseFloat(e.target.value)})}
              className="w-full bg-zinc-900/50 border border-white/10 rounded-lg px-4 py-2 text-white focus:outline-none focus:ring-2 focus:ring-cyan-400"
              data-testid="threshold-value-input"
            />
          </div>
          
          <div className="flex items-end">
            <button
              onClick={createThreshold}
              className="btn-primary w-full"
              data-testid="create-threshold-btn"
            >
              Set Alert
            </button>
          </div>
        </div>

        {thresholds.length > 0 && (
          <div className="mt-4">
            <h4 className="text-sm font-semibold text-zinc-400 mb-3">Active Thresholds</h4>
            <div className="space-y-2">
              {thresholds.map((threshold, idx) => (
                <div key={idx} className="bg-zinc-900/50 rounded-lg p-3 flex items-center justify-between" data-testid={`threshold-${idx}`}>
                  <div className="flex items-center gap-3">
                    <div className="w-3 h-3 rounded-full" style={{ backgroundColor: getModelColor(threshold.model_name) }}></div>
                    <span className="text-sm text-white font-medium">{threshold.model_name}</span>
                    <span className="text-xs text-zinc-400">
                      {threshold.threshold_type.toUpperCase()} &lt; {threshold.threshold_value}
                    </span>
                  </div>
                  <CheckCircle className="text-green-400" size={16} />
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* No Data State */}
      {!hasData && (
        <div className="glass-card rounded-xl p-12 text-center">
          <Target className="text-zinc-600 mx-auto mb-4" size={48} />
          <h3 className="text-xl font-heading font-semibold text-zinc-400 mb-2">No Accuracy Data Yet</h3>
          <p className="text-sm text-zinc-500 mb-6">
            Generate simulation data to see accuracy tracking in action
          </p>
          <button
            onClick={simulateData}
            disabled={simulating}
            className="btn-primary flex items-center justify-center gap-2 mx-auto disabled:opacity-50"
            data-testid="simulate-data-btn"
          >
            {simulating ? (
              <>
                <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-black"></div>
                Generating...
              </>
            ) : (
              <>
                <RefreshCw size={18} />
                Generate Simulation Data
              </>
            )}
          </button>
        </div>
      )}

      {/* Production Recommendations */}
      {hasData && (
        <div className="glass-card rounded-xl p-6 bg-gradient-to-br from-cyan-400/5 to-purple-400/5 border border-cyan-400/20">
          <h3 className="text-lg font-heading font-semibold text-white mb-4">Production Deployment Guide</h3>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <h4 className="text-sm font-semibold text-cyan-400 mb-2">Retraining Strategy</h4>
              <ul className="text-sm text-zinc-300 space-y-1">
                <li>• Retrain models weekly for stable patterns</li>
                <li>• Trigger immediate retraining if MAPE &gt; 15%</li>
                <li>• Use rolling 90-day window for training</li>
                <li>• A/B test new models before deployment</li>
              </ul>
            </div>
            
            <div>
              <h4 className="text-sm font-semibold text-purple-400 mb-2">Alert Configuration</h4>
              <ul className="text-sm text-zinc-300 space-y-1">
                <li>• Set MAPE threshold at 10% for critical products</li>
                <li>• Monitor 3 consecutive days of errors</li>
                <li>• Alert stakeholders via email/Slack</li>
                <li>• Review alerts daily during business hours</li>
              </ul>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default AccuracyTracker;
