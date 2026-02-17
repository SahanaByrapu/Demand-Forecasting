import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { API } from '../App';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import { Trophy, TrendingUp, Clock, Target, Brain, Sparkles } from 'lucide-react';

const ModelComparison = () => {
  const [loading, setLoading] = useState(false);
  const [comparisonData, setComparisonData] = useState(null);
  const [insights, setInsights] = useState('');
  const [generatingInsights, setGeneratingInsights] = useState(false);

  const loadLatestTraining = async () => {
    try {
      setLoading(true);
      // For demo, train models to get fresh comparison data
      const response = await axios.post(`${API}/models/train`, {
        product_id: 'HOBBIES_1_001',
        forecast_horizon: 28
      });
      setComparisonData(response.data);
    } catch (error) {
      console.error('Error loading comparison data:', error);
    } finally {
      setLoading(false);
    }
  };

  const generateInsights = async () => {
    if (!comparisonData) return;
    
    try {
      setGeneratingInsights(true);
      const metricsForLLM = comparisonData.metrics.map(m => ({
        model_name: m.model_name,
        rmse: m.rmse,
        mae: m.mae,
        mape: m.mape,
        training_time: m.training_time
      }));
      
      const response = await axios.post(`${API}/insights/generate`, {
        metrics: metricsForLLM,
        context: 'Walmart M5 demand forecasting for retail stockout prevention'
      });
      
      setInsights(response.data.insights);
    } catch (error) {
      console.error('Error generating insights:', error);
      setInsights('Failed to generate insights. Please try again.');
    } finally {
      setGeneratingInsights(false);
    }
  };

  useEffect(() => {
    loadLatestTraining();
  }, []);

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-cyan-400 mx-auto mb-4"></div>
          <p className="text-zinc-400">Loading model comparison...</p>
        </div>
      </div>
    );
  }

  const getModelColor = (modelName) => {
    const colors = {
      'ARIMA': '#22d3ee',
      'SARIMA': '#a855f7',
      'Prophet': '#facc15',
      'LSTM': '#f472b6'
    };
    return colors[modelName] || '#22d3ee';
  };

  const getBestModel = (metric) => {
    if (!comparisonData) return null;
    return comparisonData.metrics.reduce((best, current) => 
      current[metric] < best[metric] ? current : best
    );
  };

  const prepareMetricChart = (metricName) => {
    if (!comparisonData) return [];
    return comparisonData.metrics.map(m => ({
      model: m.model_name,
      value: parseFloat(m[metricName].toFixed(3)),
      color: getModelColor(m.model_name)
    }));
  };

  return (
    <div className="p-8">
      <div className="mb-8">
        <h1 className="section-title text-white mb-2" data-testid="comparison-title">Model Comparison</h1>
        <p className="text-zinc-400 text-sm">Analyze performance metrics and understand model trade-offs</p>
      </div>

      {comparisonData && (
        <>
          {/* Best Model Highlight */}
          <div className="glass-card rounded-xl p-6 mb-8 bg-gradient-to-br from-cyan-400/10 to-purple-400/10 border border-cyan-400/30 glow-cyan" data-testid="best-model-card">
            <div className="flex items-center gap-3 mb-4">
              <Trophy className="text-yellow-400" size={32} />
              <div>
                <h2 className="text-2xl font-heading font-bold text-white">Best Overall Model</h2>
                <p className="text-sm text-zinc-400">Based on lowest RMSE (Root Mean Squared Error)</p>
              </div>
            </div>
            <div className="flex items-baseline gap-4">
              <div className="text-4xl font-heading font-bold" style={{ color: getModelColor(getBestModel('rmse').model_name) }}>
                {getBestModel('rmse').model_name}
              </div>
              <div className="text-zinc-400 text-sm">
                RMSE: <span className="data-value text-white">{getBestModel('rmse').rmse.toFixed(3)}</span>
              </div>
            </div>
          </div>

          {/* Metrics Comparison Grid */}
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-8">
            {/* RMSE Chart */}
            <div className="glass-card rounded-xl p-6" data-testid="rmse-chart">
              <div className="flex items-center gap-2 mb-4">
                <Target className="text-cyan-400" size={20} />
                <h3 className="text-lg font-heading font-semibold text-white">RMSE Comparison</h3>
              </div>
              <ResponsiveContainer width="100%" height={250}>
                <BarChart data={prepareMetricChart('rmse')}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
                  <XAxis dataKey="model" stroke="#71717a" style={{ fontSize: '11px' }} angle={-15} textAnchor="end" height={60} />
                  <YAxis stroke="#71717a" style={{ fontSize: '11px' }} />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: '#18181b',
                      border: '1px solid rgba(255,255,255,0.1)',
                      borderRadius: '8px'
                    }}
                  />
                  <Bar dataKey="value" radius={[8, 8, 0, 0]}>
                    {prepareMetricChart('rmse').map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>

            {/* MAE Chart */}
            <div className="glass-card rounded-xl p-6" data-testid="mae-chart">
              <div className="flex items-center gap-2 mb-4">
                <TrendingUp className="text-purple-400" size={20} />
                <h3 className="text-lg font-heading font-semibold text-white">MAE Comparison</h3>
              </div>
              <ResponsiveContainer width="100%" height={250}>
                <BarChart data={prepareMetricChart('mae')}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
                  <XAxis dataKey="model" stroke="#71717a" style={{ fontSize: '11px' }} angle={-15} textAnchor="end" height={60} />
                  <YAxis stroke="#71717a" style={{ fontSize: '11px' }} />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: '#18181b',
                      border: '1px solid rgba(255,255,255,0.1)',
                      borderRadius: '8px'
                    }}
                  />
                  <Bar dataKey="value" radius={[8, 8, 0, 0]}>
                    {prepareMetricChart('mae').map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>

            {/* MAPE Chart */}
            <div className="glass-card rounded-xl p-6" data-testid="mape-chart">
              <div className="flex items-center gap-2 mb-4">
                <Target className="text-yellow-400" size={20} />
                <h3 className="text-lg font-heading font-semibold text-white">MAPE Comparison (%)</h3>
              </div>
              <ResponsiveContainer width="100%" height={250}>
                <BarChart data={prepareMetricChart('mape')}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
                  <XAxis dataKey="model" stroke="#71717a" style={{ fontSize: '11px' }} angle={-15} textAnchor="end" height={60} />
                  <YAxis stroke="#71717a" style={{ fontSize: '11px' }} />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: '#18181b',
                      border: '1px solid rgba(255,255,255,0.1)',
                      borderRadius: '8px'
                    }}
                  />
                  <Bar dataKey="value" radius={[8, 8, 0, 0]}>
                    {prepareMetricChart('mape').map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Detailed Metrics Table */}
          <div className="glass-card rounded-xl p-6 mb-8 overflow-x-auto" data-testid="metrics-table">
            <h3 className="text-lg font-heading font-semibold text-white mb-4">Detailed Metrics</h3>
            <table className="w-full">
              <thead>
                <tr className="border-b border-white/10">
                  <th className="text-left text-sm text-zinc-400 pb-3 font-medium">Model</th>
                  <th className="text-right text-sm text-zinc-400 pb-3 font-medium">RMSE</th>
                  <th className="text-right text-sm text-zinc-400 pb-3 font-medium">MAE</th>
                  <th className="text-right text-sm text-zinc-400 pb-3 font-medium">MAPE (%)</th>
                  <th className="text-right text-sm text-zinc-400 pb-3 font-medium">Training Time (s)</th>
                  <th className="text-center text-sm text-zinc-400 pb-3 font-medium">Interpretability</th>
                </tr>
              </thead>
              <tbody>
                {comparisonData.metrics.map((metric, idx) => {
                  const interpretability = {
                    'ARIMA': 'High',
                    'SARIMA': 'High',
                    'Prophet': 'Medium',
                    'LSTM': 'Low'
                  };
                  
                  return (
                    <tr key={idx} className="border-b border-white/5" data-testid={`table-row-${metric.model_name.toLowerCase()}`}>
                      <td className="py-3">
                        <div className="flex items-center gap-2">
                          <div className="w-3 h-3 rounded-full" style={{ backgroundColor: getModelColor(metric.model_name) }}></div>
                          <span className="font-semibold text-white">{metric.model_name}</span>
                        </div>
                      </td>
                      <td className="text-right data-value text-white">{metric.rmse.toFixed(3)}</td>
                      <td className="text-right data-value text-white">{metric.mae.toFixed(3)}</td>
                      <td className="text-right data-value text-white">{metric.mape.toFixed(2)}%</td>
                      <td className="text-right data-value text-cyan-400">{metric.training_time.toFixed(2)}s</td>
                      <td className="text-center">
                        <span className={`inline-block px-3 py-1 rounded-full text-xs font-medium ${
                          interpretability[metric.model_name] === 'High' ? 'bg-green-400/20 text-green-400' :
                          interpretability[metric.model_name] === 'Medium' ? 'bg-yellow-400/20 text-yellow-400' :
                          'bg-red-400/20 text-red-400'
                        }`}>
                          {interpretability[metric.model_name]}
                        </span>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>

          {/* AI-Powered Insights */}
          <div className="glass-card rounded-xl p-6 border-2 border-purple-400/30" data-testid="insights-section">
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center gap-2">
                <Brain className="text-purple-400" size={24} />
                <h3 className="text-lg font-heading font-semibold text-white">AI-Powered Insights</h3>
                <Sparkles className="text-yellow-400" size={16} />
              </div>
              <button
                onClick={generateInsights}
                disabled={generatingInsights}
                className="btn-secondary flex items-center gap-2 disabled:opacity-50"
                data-testid="generate-insights-btn"
              >
                {generatingInsights ? (
                  <>
                    <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white"></div>
                    Analyzing...
                  </>
                ) : (
                  <>
                    <Brain size={16} />
                    Generate Insights
                  </>
                )}
              </button>
            </div>
            
            {insights ? (
              <div className="bg-zinc-900/50 rounded-lg p-4 border border-purple-400/20" data-testid="insights-content">
                <pre className="text-sm text-zinc-300 whitespace-pre-wrap font-sans leading-relaxed">{insights}</pre>
              </div>
            ) : (
              <div className="bg-zinc-900/30 rounded-lg p-8 text-center border border-dashed border-zinc-700">
                <p className="text-zinc-500 text-sm">Click "Generate Insights" to get AI-powered analysis of model performance, trade-offs, and business recommendations</p>
              </div>
            )}
          </div>
        </>
      )}

      {!comparisonData && !loading && (
        <div className="glass-card rounded-xl p-12 text-center">
          <TrendingUp className="text-zinc-600 mx-auto mb-4" size={48} />
          <h3 className="text-xl font-heading font-semibold text-zinc-400 mb-2">No Comparison Data</h3>
          <p className="text-sm text-zinc-500 mb-6">Train models first to see comparison metrics</p>
          <button onClick={loadLatestTraining} className="btn-primary" data-testid="load-data-btn">
            Load Training Data
          </button>
        </div>
      )}
    </div>
  );
};

export default ModelComparison;