import React, { useState } from 'react';
import { BrowserRouter, Routes, Route, Link, useLocation } from 'react-router-dom';
import '@/App.css';
import Dashboard from './pages/Dashboard';
import DataExploration from './pages/DataExploration';
import ModelTraining from './pages/ModelTraining';
import ModelComparison from './pages/ModelComparison';
import BusinessImpact from './pages/BusinessImpact';
import AccuracyTracker from './pages/AccuracyTracker';
import { BarChart3, TrendingUp, Brain, DollarSign, Database, Target } from 'lucide-react';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
export const API = `${BACKEND_URL}/api`;

const Sidebar = () => {
  const location = useLocation();
  
  const navItems = [
    { path: '/', icon: BarChart3, label: 'Dashboard' },
    { path: '/explore', icon: Database, label: 'Data Exploration' },
    { path: '/train', icon: Brain, label: 'Model Training' },
//{ path: '/compare', icon: TrendingUp, label: 'Comparison' } 
    { path: '/accuracy', icon: Target, label: 'Accuracy Tracker' },
    { path: '/impact', icon: DollarSign, label: 'Business Impact' },
  ];
  
  return (
    <div className="w-64 min-h-screen bg-zinc-900/50 backdrop-blur-md border-r border-white/10 fixed left-0 top-0">
      <div className="p-6">
        <h1 className="text-2xl font-heading font-bold text-cyan-400" data-testid="app-title">M5 Forecast Pro</h1>
        <p className="text-xs text-zinc-500 mt-1 font-mono">Demand Forecasting</p>
      </div>
      
      <nav className="px-3 space-y-1">
        {navItems.map((item) => {
          const Icon = item.icon;
          const isActive = location.pathname === item.path;
          
          return (
            <Link
              key={item.path}
              to={item.path}
              data-testid={`nav-${item.label.toLowerCase().replace(' ', '-')}`}
              className={`flex items-center gap-3 px-3 py-2.5 rounded-lg transition-all ${
                isActive
                  ? 'bg-cyan-400/10 text-cyan-400 border border-cyan-400/30'
                  : 'text-zinc-400 hover:bg-white/5 hover:text-zinc-200'
              }`}
            >
              <Icon size={20} />
              <span className="font-medium text-sm">{item.label}</span>
            </Link>
          );
        })}
      </nav>
    </div>
  );
};

const AppContent = () => {
  return (
    <div className="flex">
      <Sidebar />
      <div className="ml-64 flex-1 min-h-screen grid-background">
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/explore" element={<DataExploration />} />
          <Route path="/train" element={<ModelTraining />} />
          <Route path="/compare" element={<ModelComparison />} />
          <Route path="/accuracy" element={<AccuracyTracker />} />
          <Route path="/impact" element={<BusinessImpact />} />
        </Routes>
      </div>
    </div>
  );
};

function App() {
  return (
    <div className="App">
      <BrowserRouter>
        <AppContent />
      </BrowserRouter>
    </div>
  );
}

export default App;