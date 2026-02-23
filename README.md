#### Demand Forecasting


Predict future demand to optimize inventory and logistics.

**Industry:** Retail, Supply Chain
**What you build:**  Predict future product demand to reduce stockouts

**Why recruiters like it:** Direct business cost savings. 

**Skills:** Time-series forecasting, Regression , data trends

**Dataset:** https://www.kaggle.com/datasets/carrie1/ecommerce-data
UCI Energy Consumption: https://archive.ics.uci.edu/ml/datase…
M5 Forecasting (Walmart Sales): https://www.kaggle.com/competitions/m...

**Upgrade:** Seasonality + festival impact modelling

**💼 Company:** Walmart uses ML forecasting to reduce stock-outs

**Resume Bullets:**
• Built time-series forecasting pipeline using Prophet and LSTM
• Reduced stock-out events by 22% across simulated stores

Retail needs to know how much inventory to stock
Energy companies need to predict consumption
Airlines need to predict passenger demand
Tech companies need to predict server load

**Start with Exploration:**
Look for “trends”
Look for “seasonality”=> weekly, monthly, yearly
Look for “Anamolies” => Why they occur??

**Build multiple models**
ARIMA, SARIMA,  
Use model base line like prophet
Deep learning approach like LSTM

Goal is not complexity , but the goal is comparison and reasoning.

1.Can you explain why one model outperforms the other?

2.Can you articulate trade-offs between accuracy and interpretability?

3.Can you identify where each model fails?

4.Connect your forecast to the business impact?

5.How better decisions improve cost, improve staffing decisions or prevent outrages

Retail, Supply chain, Finance, transportation teams.

6.In real time scenarios:

7.How often do you retrain it?

8.How would you trigger an alert?

**Beyond jupiter notebook**

# ChurnGuard AI Dashboard - Product Requirements Document

## Original Problem Statement
Create a Customer Segmentation and Retention Analysis dashboard using Telco Customer Churn dataset. Predict which users are likely to cancel a subscription using usage + behavioral data that covers ML Classification, Feature Engineering, Modeling, Business metrics including Customer Lifetime Value (CLV), Monthly Recurring Revenue (MRR), and Churn Rate by segment. Full suite dashboard with interactive filters, export reports, real-time predictions for new customers, and AI-powered recommendations for retention strategies using OpenAI GPT-4.0

## User Personas
1. **Business Analysts** - Analyze customer segments, track KPIs, identify churn patterns
2. **Retention Teams** - Get AI recommendations, prioritize high-risk customers, execute retention campaigns
3. **Product Managers** - Track feature importance, monitor model performance, export reports

## Core Requirements (Static)
- XGBoost ML model for churn prediction
- Customer segmentation analysis
- Business metrics (CLV, MRR, Churn Rate)
- Interactive filters and search
- Real-time predictions for new customers
- AI-powered retention recommendations (GPT-5.2)
- CSV export functionality

### Backend (FastAPI)
- ✅ XGBoost model training on startup with 7,043 synthetic customers
- ✅ `/api/dashboard/stats` - KPIs and model metrics
- ✅ `/api/customers` - Paginated customer list with filters
- ✅ `/api/customers/{id}` - Individual customer details
- ✅ `/api/predict` - Real-time churn prediction
- ✅ `/api/segments` - Customer segmentation by Contract, Internet, Risk Level
- ✅ `/api/ai-recommendations` - GPT-5.2 powered retention strategies
- ✅ `/api/export/customers` - CSV export
- ✅ `/api/charts/*` - Chart data endpoints

### Frontend (React + Shadcn UI)
- ✅ Dashboard - KPIs, Risk Distribution Pie, Model Performance, Charts
- ✅ Customers - Searchable, filterable table with pagination & detail modal
- ✅ Predictions - Full form with risk gauge visualization
- ✅ AI Insights - GPT-5.2 recommendations for high-risk customers
- ✅ Reports - Segment analysis, feature importance, export buttons

## Prioritized Backlog

### P0 (Critical)
- All P0 features implemented ✅

### P1 (High Priority - Future)
- Upload custom CSV dataset
- Scheduled batch predictions
- Email alerts for high-risk customers
- Historical churn tracking over time

### P2 (Medium Priority - Future)
- A/B testing for retention campaigns
- Customer cohort analysis
- Advanced model tuning interface
- Multiple ML model comparison
- Automated re-training pipeline

## Next Tasks
1. Add user authentication (JWT or Google OAuth)
2. Implement real Telco dataset CSV upload
3. Add email notifications for high-risk alerts
4. Create retention campaign tracking
5. Add dashboard date range filters

**Dataset:** 
- https://www.kaggle.com/datasets/blastchar/telco-customer-churn








