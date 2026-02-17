import React, { useState } from 'react';
import axios from 'axios';
import { API } from '../App';
import { DollarSign, TrendingUp, Package, Users, AlertCircle, Calculator } from 'lucide-react';

const BusinessImpact = () => {
  const [loading, setLoading] = useState(false);
  const [impact, setImpact] = useState(null);
  
  const [inputs, setInputs] = useState({
    currentStockoutRate: 15,
    avgDailySales: 50,
    productMargin: 25,
    storageCostPerUnit: 2
  });

  const calculateImpact = async () => {
    try {
      setLoading(true);
      const response = await axios.post(`${API}/business/impact`, {
        current_stockout_rate: inputs.currentStockoutRate / 100,
        avg_daily_sales: inputs.avgDailySales,
        product_margin: inputs.productMargin,
        storage_cost_per_unit: inputs.storageCostPerUnit
      });
      setImpact(response.data);
    } catch (error) {
      console.error('Error calculating impact:', error);
      alert('Error calculating impact. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const handleInputChange = (field, value) => {
    setInputs(prev => ({ ...prev, [field]: parseFloat(value) || 0 }));
  };

  return (
    <div className="p-8">
      <div className="mb-8">
        <h1 className="section-title text-white mb-2" data-testid="impact-title">Business Impact Calculator</h1>
        <p className="text-zinc-400 text-sm">Quantify the financial benefits of better demand forecasting</p>
      </div>

      {/* Hero Section with Background Image */}
      <div 
        className="glass-card rounded-xl p-8 mb-8 relative overflow-hidden" 
        style={{
          backgroundImage: 'url(https://images.unsplash.com/photo-1606824722920-4c652a70f348?crop=entropy&cs=srgb&fm=jpg&ixid=M3w3NTY2NzZ8MHwxfHNlYXJjaHwxfHxzdXBlcm1hcmtldCUyMHNoZWx2ZXMlMjByZXRhaWx8ZW58MHx8fHwxNzcxMzU0NTM5fDA&ixlib=rb-4.1.0&q=85)',
          backgroundSize: 'cover',
          backgroundPosition: 'center'
        }}
        data-testid="hero-section"
      >
        <div className="absolute inset-0 bg-zinc-950/90 backdrop-blur-sm"></div>
        <div className="relative z-10 text-center">
          <DollarSign className="text-green-400 mx-auto mb-4" size={48} />
          <h2 className="text-3xl font-heading font-bold text-white mb-3">Transform Forecasts into Profit</h2>
          <p className="text-zinc-300 max-w-2xl mx-auto">
            Better demand forecasting reduces stockouts, optimizes inventory, and improves customer satisfaction.
            Calculate your potential savings below.
          </p>
        </div>
      </div>

      {/* Input Form */}
      <div className="glass-card rounded-xl p-6 mb-8" data-testid="calculator-form">
        <div className="flex items-center gap-2 mb-6">
          <Calculator className="text-cyan-400" size={24} />
          <h3 className="text-xl font-heading font-semibold text-white">Input Parameters</h3>
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
          <div>
            <label className="block text-sm text-zinc-400 mb-2 flex items-center gap-2">
              <AlertCircle size={16} />
              Current Stockout Rate (%)
            </label>
            <input
              type="number"
              value={inputs.currentStockoutRate}
              onChange={(e) => handleInputChange('currentStockoutRate', e.target.value)}
              className="w-full bg-zinc-900/50 border border-white/10 rounded-lg px-4 py-3 text-white focus:outline-none focus:ring-2 focus:ring-cyan-400"
              data-testid="stockout-rate-input"
            />
            <p className="text-xs text-zinc-500 mt-1">Percentage of days with stockouts</p>
          </div>
          
          <div>
            <label className="block text-sm text-zinc-400 mb-2 flex items-center gap-2">
              <Package size={16} />
              Average Daily Sales (units)
            </label>
            <input
              type="number"
              value={inputs.avgDailySales}
              onChange={(e) => handleInputChange('avgDailySales', e.target.value)}
              className="w-full bg-zinc-900/50 border border-white/10 rounded-lg px-4 py-3 text-white focus:outline-none focus:ring-2 focus:ring-cyan-400"
              data-testid="daily-sales-input"
            />
            <p className="text-xs text-zinc-500 mt-1">Average units sold per day</p>
          </div>
          
          <div>
            <label className="block text-sm text-zinc-400 mb-2 flex items-center gap-2">
              <DollarSign size={16} />
              Product Margin ($)
            </label>
            <input
              type="number"
              value={inputs.productMargin}
              onChange={(e) => handleInputChange('productMargin', e.target.value)}
              className="w-full bg-zinc-900/50 border border-white/10 rounded-lg px-4 py-3 text-white focus:outline-none focus:ring-2 focus:ring-cyan-400"
              data-testid="margin-input"
            />
            <p className="text-xs text-zinc-500 mt-1">Profit margin per unit</p>
          </div>
          
          <div>
            <label className="block text-sm text-zinc-400 mb-2 flex items-center gap-2">
              <TrendingUp size={16} />
              Storage Cost per Unit ($)
            </label>
            <input
              type="number"
              value={inputs.storageCostPerUnit}
              onChange={(e) => handleInputChange('storageCostPerUnit', e.target.value)}
              className="w-full bg-zinc-900/50 border border-white/10 rounded-lg px-4 py-3 text-white focus:outline-none focus:ring-2 focus:ring-cyan-400"
              data-testid="storage-cost-input"
            />
            <p className="text-xs text-zinc-500 mt-1">Monthly storage cost per unit</p>
          </div>
        </div>

        <button
          onClick={calculateImpact}
          disabled={loading}
          className="btn-primary w-full flex items-center justify-center gap-2 disabled:opacity-50 py-4"
          data-testid="calculate-button"
        >
          {loading ? (
            <>
              <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-black"></div>
              Calculating...
            </>
          ) : (
            <>
              <Calculator size={20} />
              Calculate Business Impact
            </>
          )}
        </button>
      </div>

      {/* Results */}
      {impact && (
        <>
          {/* Summary Cards */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
            <div className="glass-card rounded-xl p-6 border-2 border-green-400/30 glow-cyan" data-testid="total-savings-card">
              <div className="flex items-center gap-2 mb-3">
                <DollarSign className="text-green-400" size={24} />
                <h3 className="text-sm text-zinc-400">Total Annual Savings</h3>
              </div>
              <div className="data-value text-4xl text-green-400 mb-2">
                ${impact.total_annual_savings.toLocaleString()}
              </div>
              <div className="text-xs text-zinc-500">From better forecasting</div>
            </div>

            <div className="glass-card rounded-xl p-6" data-testid="stockout-reduction-card">
              <div className="flex items-center gap-2 mb-3">
                <TrendingUp className="text-cyan-400" size={24} />
                <h3 className="text-sm text-zinc-400">Stockout Reduction</h3>
              </div>
              <div className="data-value text-4xl text-cyan-400 mb-2">
                {impact.stockout_reduction.toFixed(1)}%
              </div>
              <div className="text-xs text-zinc-500">Fewer stockout days</div>
            </div>

            <div className="glass-card rounded-xl p-6" data-testid="roi-card">
              <div className="flex items-center gap-2 mb-3">
                <Package className="text-purple-400" size={24} />
                <h3 className="text-sm text-zinc-400">ROI Improvement</h3>
              </div>
              <div className="data-value text-4xl text-purple-400 mb-2">
                {impact.roi_percentage.toFixed(1)}%
              </div>
              <div className="text-xs text-zinc-500">Return on investment</div>
            </div>
          </div>

          {/* Detailed Breakdown */}
          <div className="glass-card rounded-xl p-6 mb-8" data-testid="detailed-breakdown">
            <h3 className="text-lg font-heading font-semibold text-white mb-4">Detailed Impact Breakdown</h3>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="space-y-4">
                <div className="bg-zinc-900/50 rounded-lg p-4 border border-white/10">
                  <div className="text-sm text-zinc-400 mb-2">Current Annual Stockouts</div>
                  <div className="data-value text-2xl text-red-400">
                    {impact.current_annual_stockouts.toLocaleString()} units
                  </div>
                </div>
                
                <div className="bg-zinc-900/50 rounded-lg p-4 border border-white/10">
                  <div className="text-sm text-zinc-400 mb-2">Improved Annual Stockouts</div>
                  <div className="data-value text-2xl text-green-400">
                    {impact.improved_annual_stockouts.toLocaleString()} units
                  </div>
                </div>
              </div>
              
              <div className="space-y-4">
                <div className="bg-zinc-900/50 rounded-lg p-4 border border-white/10">
                  <div className="text-sm text-zinc-400 mb-2">Revenue Gain from Reduced Stockouts</div>
                  <div className="data-value text-2xl text-cyan-400">
                    ${impact.revenue_gain.toLocaleString()}
                  </div>
                </div>
                
                <div className="bg-zinc-900/50 rounded-lg p-4 border border-white/10">
                  <div className="text-sm text-zinc-400 mb-2">Storage Cost Savings</div>
                  <div className="data-value text-2xl text-purple-400">
                    ${impact.storage_savings.toLocaleString()}
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Business Recommendations */}
          <div className="glass-card rounded-xl p-6 bg-gradient-to-br from-cyan-400/5 to-purple-400/5 border border-cyan-400/20">
            <h3 className="text-lg font-heading font-semibold text-white mb-4">Business Recommendations</h3>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <h4 className="text-sm font-semibold text-cyan-400 mb-2 flex items-center gap-2">
                  <Users size={16} />
                  Staffing Optimization
                </h4>
                <ul className="text-sm text-zinc-300 space-y-1">
                  <li>• Schedule staff based on predicted demand peaks</li>
                  <li>• Reduce overtime costs by 20-30%</li>
                  <li>• Improve customer service with better coverage</li>
                </ul>
              </div>
              
              <div>
                <h4 className="text-sm font-semibold text-purple-400 mb-2 flex items-center gap-2">
                  <Package size={16} />
                  Inventory Management
                </h4>
                <ul className="text-sm text-zinc-300 space-y-1">
                  <li>• Optimize reorder points using forecasts</li>
                  <li>• Reduce safety stock by 15-25%</li>
                  <li>• Free up capital tied in excess inventory</li>
                </ul>
              </div>
              
              <div>
                <h4 className="text-sm font-semibold text-green-400 mb-2 flex items-center gap-2">
                  <AlertCircle size={16} />
                  Stockout Prevention
                </h4>
                <ul className="text-sm text-zinc-300 space-y-1">
                  <li>• Set up automated alerts for low inventory</li>
                  <li>• Retrain models weekly for best accuracy</li>
                  <li>• Monitor forecast accuracy continuously</li>
                </ul>
              </div>
              
              <div>
                <h4 className="text-sm font-semibold text-yellow-400 mb-2 flex items-center gap-2">
                  <TrendingUp size={16} />
                  Continuous Improvement
                </h4>
                <ul className="text-sm text-zinc-300 space-y-1">
                  <li>• A/B test different forecasting models</li>
                  <li>• Incorporate promotional events in models</li>
                  <li>• Track and analyze forecast errors</li>
                </ul>
              </div>
            </div>
          </div>
        </>
      )}

      {!impact && !loading && (
        <div className="glass-card rounded-xl p-12 text-center">
          <Calculator className="text-zinc-600 mx-auto mb-4" size={48} />
          <h3 className="text-xl font-heading font-semibold text-zinc-400 mb-2">Ready to Calculate Impact</h3>
          <p className="text-sm text-zinc-500">Fill in the parameters above and click "Calculate Business Impact"</p>
        </div>
      )}
    </div>
  );
};

export default BusinessImpact;