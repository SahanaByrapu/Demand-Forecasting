import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { API } from '../App';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { TrendingUp, Package, AlertCircle, Activity } from 'lucide-react';

const Dashboard = () => {
  const [loading, setLoading] = useState(true);
  const [stats, setStats] = useState(null);
  const [historicalData, setHistoricalData] = useState([]);

  useEffect(() => {
    loadData();
  }, []);

  const loadData = async () => {
    try {
      setLoading(true);
      
      // Download dataset first with timeout
      await axios.post(`${API}/dataset/download`, {}, { timeout: 10000 });
      
      // Get exploration stats
      const exploreRes = await axios.get(`${API}/dataset/explore`, { timeout: 10000 });
      setStats(exploreRes.data);
      
      // Get historical data
      const histRes = await axios.get(`${API}/data/historical`, { timeout: 10000 });
      const chartData = histRes.data.dates.slice(-90).map((date, idx) => ({
        date: new Date(date).toLocaleDateString('en-US', { month: 'short', day: 'numeric' }),
        sales: Math.round(histRes.data.sales[histRes.data.sales.length - 90 + idx])
      }));
      setHistoricalData(chartData);
    } catch (error) {
      console.error('Error loading data:', error);
      // Set default data on error to prevent infinite loading
      setStats({ 
        statistics: { mean: 0, median: 0, std: 0, min: 0, max: 0 },
        trends: { overall_trend: 'unknown' },
        anomalies: []
      });
      setHistoricalData([]);
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-cyan-400 mx-auto mb-4"></div>
          <p className="text-zinc-400">Loading dashboard...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="p-8">
      <div className="mb-8">
        <h1 className="section-title text-white mb-2" data-testid="dashboard-title">Demand Forecasting Dashboard</h1>
        <p className="text-zinc-400 text-sm">M5 Walmart Sales - Real-time insights and predictions</p>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-8">
        <div className="metric-card rounded-xl p-6 glow-cyan" data-testid="metric-avg-sales">
          <div className="flex items-center justify-between mb-3">
            <Package className="text-cyan-400" size={24} />
            <TrendingUp className="text-green-400" size={16} />
          </div>
          <div className="data-value text-3xl text-white mb-1">
            {stats?.statistics?.mean?.toFixed(0) || 0}
          </div>
          <div className="text-zinc-400 text-sm">Avg Daily Sales</div>
        </div>

        <div className="metric-card rounded-xl p-6">
          <div className="flex items-center justify-between mb-3">
            <Activity className="text-purple-400" size={24} />
          </div>
          <div className="data-value text-3xl text-white mb-1">
            {stats?.statistics?.std?.toFixed(1) || 0}
          </div>
          <div className="text-zinc-400 text-sm">Std Deviation</div>
        </div>

        <div className="metric-card rounded-xl p-6">
          <div className="flex items-center justify-between mb-3">
            <AlertCircle className="text-yellow-400" size={24} />
          </div>
          <div className="data-value text-3xl text-white mb-1">
            {stats?.anomalies?.length || 0}
          </div>
          <div className="text-zinc-400 text-sm">Anomalies Detected</div>
        </div>

        <div className="metric-card rounded-xl p-6">
          <div className="flex items-center justify-between mb-3">
            <TrendingUp className="text-green-400" size={24} />
          </div>
          <div className="data-value text-3xl text-white mb-1 capitalize">
            {stats?.trends?.overall_trend || 'N/A'}
          </div>
          <div className="text-zinc-400 text-sm">Overall Trend</div>
        </div>
      </div>

      {/* Historical Sales Chart */}
      <div className="glass-card rounded-xl p-6 mb-8" data-testid="historical-chart">
        <h2 className="text-xl font-heading font-semibold text-white mb-4">Historical Sales (Last 90 Days)</h2>
        <ResponsiveContainer width="100%" height={400}>
          <LineChart data={historicalData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
            <XAxis 
              dataKey="date" 
              stroke="#71717a" 
              style={{ fontSize: '12px' }}
              angle={-45}
              textAnchor="end"
              height={80}
            />
            <YAxis stroke="#71717a" style={{ fontSize: '12px' }} />
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
              dataKey="sales" 
              stroke="#22d3ee" 
              strokeWidth={2}
              dot={{ fill: '#22d3ee', r: 2 }}
              activeDot={{ r: 5 }}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Quick Actions */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="glass-card rounded-xl p-6">
          <h3 className="text-lg font-heading font-semibold text-white mb-2">Next Steps</h3>
          <p className="text-zinc-400 text-sm mb-4">Start exploring data patterns and train forecasting models</p>
          <button 
            onClick={() => window.location.href = '/explore'}
            className="btn-primary w-full"
            data-testid="explore-data-btn"
          >
            Explore Data
          </button>
        </div>

        <div className="glass-card rounded-xl p-6">
          <h3 className="text-lg font-heading font-semibold text-white mb-2">Model Training</h3>
          <p className="text-zinc-400 text-sm mb-4">Train ARIMA, SARIMA, Prophet, and LSTM models</p>
          <button 
            onClick={() => window.location.href = '/train'}
            className="btn-secondary w-full"
            data-testid="train-models-btn"
          >
            Train Models
          </button>
        </div>

        <div className="glass-card rounded-xl p-6">
          <h3 className="text-lg font-heading font-semibold text-white mb-2">Business Impact</h3>
          <p className="text-zinc-400 text-sm mb-4">Calculate ROI and savings from better forecasts</p>
          <button 
            onClick={() => window.location.href = '/impact'}
            className="btn-primary w-full"
            data-testid="calculate-impact-btn"
          >
            Calculate Impact
          </button>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;