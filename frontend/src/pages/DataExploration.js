import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { API } from '../App';
import { BarChart, Bar, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ScatterChart, Scatter } from 'recharts';
import { TrendingUp, Calendar, AlertTriangle } from 'lucide-react';

const DataExploration = () => {
  const [loading, setLoading] = useState(true);
  const [data, setData] = useState(null);

  useEffect(() => {
    loadExplorationData();
  }, []);

  const loadExplorationData = async () => {
    try {
      setLoading(true);
      const response = await axios.get(`${API}/dataset/explore`);
      setData(response.data);
    } catch (error) {
      console.error('Error loading exploration data:', error);
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-cyan-400 mx-auto mb-4"></div>
          <p className="text-zinc-400">Analyzing data patterns...</p>
        </div>
      </div>
    );
  }

  const dayOfWeekData = data?.seasonality?.day_of_week_avg?.map((val, idx) => ({
    day: ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'][idx],
    sales: Math.round(val)
  })) || [];

  const monthData = data?.seasonality?.month_avg?.map((val, idx) => ({
    month: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][idx],
    sales: Math.round(val)
  })) || [];

  return (
    <div className="p-8">
      <div className="mb-8">
        <h1 className="section-title text-white mb-2" data-testid="exploration-title">Data Exploration</h1>
        <p className="text-zinc-400 text-sm">Discover trends, seasonality, and anomalies in the dataset</p>
      </div>

      {/* Statistics Cards */}
      <div className="grid grid-cols-1 md:grid-cols-5 gap-4 mb-8">
        <div className="glass-card rounded-xl p-5" data-testid="stat-mean">
          <div className="text-sm text-zinc-400 mb-1">Mean Sales</div>
          <div className="data-value text-2xl text-cyan-400">{data?.statistics?.mean?.toFixed(2)}</div>
        </div>
        <div className="glass-card rounded-xl p-5" data-testid="stat-median">
          <div className="text-sm text-zinc-400 mb-1">Median</div>
          <div className="data-value text-2xl text-purple-400">{data?.statistics?.median?.toFixed(2)}</div>
        </div>
        <div className="glass-card rounded-xl p-5" data-testid="stat-std">
          <div className="text-sm text-zinc-400 mb-1">Std Dev</div>
          <div className="data-value text-2xl text-yellow-400">{data?.statistics?.std?.toFixed(2)}</div>
        </div>
        <div className="glass-card rounded-xl p-5" data-testid="stat-min">
          <div className="text-sm text-zinc-400 mb-1">Min</div>
          <div className="data-value text-2xl text-green-400">{data?.statistics?.min?.toFixed(2)}</div>
        </div>
        <div className="glass-card rounded-xl p-5" data-testid="stat-max">
          <div className="text-sm text-zinc-400 mb-1">Max</div>
          <div className="data-value text-2xl text-red-400">{data?.statistics?.max?.toFixed(2)}</div>
        </div>
      </div>

      {/* Trend Analysis */}
      <div className="glass-card rounded-xl p-6 mb-8" data-testid="trend-card">
        <div className="flex items-center gap-2 mb-4">
          <TrendingUp className="text-cyan-400" size={24} />
          <h2 className="text-xl font-heading font-semibold text-white">Trend Analysis</h2>
        </div>
        <div className="bg-zinc-900/50 rounded-lg p-4 border border-cyan-400/20">
          <div className="flex items-center gap-3">
            <div className="text-4xl">{data?.trends?.overall_trend === 'increasing' ? '📈' : '📉'}</div>
            <div>
              <div className="text-lg font-semibold text-white capitalize">{data?.trends?.overall_trend} Trend</div>
              <div className="text-sm text-zinc-400">Overall sales pattern shows a {data?.trends?.overall_trend} trajectory over time</div>
            </div>
          </div>
        </div>
      </div>

      {/* Seasonality Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
        <div className="glass-card rounded-xl p-6" data-testid="day-seasonality-chart">
          <div className="flex items-center gap-2 mb-4">
            <Calendar className="text-purple-400" size={20} />
            <h2 className="text-lg font-heading font-semibold text-white">Day of Week Seasonality</h2>
          </div>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={dayOfWeekData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
              <XAxis dataKey="day" stroke="#71717a" style={{ fontSize: '12px' }} />
              <YAxis stroke="#71717a" style={{ fontSize: '12px' }} />
              <Tooltip
                contentStyle={{
                  backgroundColor: '#18181b',
                  border: '1px solid rgba(255,255,255,0.1)',
                  borderRadius: '8px',
                  color: '#f4f4f5'
                }}
              />
              <Bar dataKey="sales" fill="#a855f7" radius={[8, 8, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div className="glass-card rounded-xl p-6" data-testid="month-seasonality-chart">
          <div className="flex items-center gap-2 mb-4">
            <Calendar className="text-cyan-400" size={20} />
            <h2 className="text-lg font-heading font-semibold text-white">Monthly Seasonality</h2>
          </div>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={monthData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
              <XAxis dataKey="month" stroke="#71717a" style={{ fontSize: '12px' }} />
              <YAxis stroke="#71717a" style={{ fontSize: '12px' }} />
              <Tooltip
                contentStyle={{
                  backgroundColor: '#18181b',
                  border: '1px solid rgba(255,255,255,0.1)',
                  borderRadius: '8px',
                  color: '#f4f4f5'
                }}
              />
              <Line type="monotone" dataKey="sales" stroke="#22d3ee" strokeWidth={3} dot={{ fill: '#22d3ee', r: 4 }} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Anomalies Detection */}
      <div className="glass-card rounded-xl p-6" data-testid="anomalies-card">
        <div className="flex items-center gap-2 mb-4">
          <AlertTriangle className="text-yellow-400" size={24} />
          <h2 className="text-xl font-heading font-semibold text-white">Anomalies Detected</h2>
        </div>
        <div className="text-sm text-zinc-400 mb-4">
          Identified {data?.anomalies?.length || 0} unusual patterns in the sales data (Z-score > 2)
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-3">
          {data?.anomalies?.slice(0, 10).map((anomaly, idx) => (
            <div key={idx} className="bg-zinc-900/50 rounded-lg p-4 border border-yellow-400/20" data-testid={`anomaly-${idx}`}>
              <div className="text-xs text-zinc-500 mb-1">Index {anomaly.index}</div>
              <div className="data-value text-lg text-yellow-400">{anomaly.value.toFixed(1)}</div>
              <div className="text-xs text-zinc-400">Z-score: {anomaly.z_score.toFixed(2)}</div>
            </div>
          ))}
        </div>
      </div>

      {/* Insights */}
      <div className="mt-8 glass-card rounded-xl p-6 bg-gradient-to-br from-cyan-400/5 to-purple-400/5 border border-cyan-400/20">
        <h3 className="text-lg font-heading font-semibold text-white mb-3">Key Insights</h3>
        <ul className="space-y-2 text-sm text-zinc-300">
          <li className="flex items-start gap-2">
            <span className="text-cyan-400 mt-1">•</span>
            <span>Sales show a <strong className="text-white">{data?.trends?.overall_trend}</strong> trend over the historical period</span>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-cyan-400 mt-1">•</span>
            <span>Weekly patterns reveal consistent seasonality with variations across days</span>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-cyan-400 mt-1">•</span>
            <span>Detected <strong className="text-white">{data?.anomalies?.length || 0}</strong> anomalies requiring attention</span>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-cyan-400 mt-1">•</span>
            <span>Monthly patterns suggest seasonal effects that should be captured in forecasting models</span>
          </li>
        </ul>
      </div>
    </div>
  );
};

export default DataExploration;