## 📊 HR Analytics & Attrition Prediction Dashboard

A comprehensive HR analytics solution that combines exploratory data analysis with machine learning to predict employee attrition. This production-ready dashboard helps HR managers identify at-risk employees and understand key factors driving turnover.

## 🚀 Features
- 📈 Interactive Analytics: Explore workforce demographics, salary distributions, and department trends

- 🎯 Attrition Analysis: Visualize attrition patterns across different employee segments

- 🔮 Predictive Modeling: Machine learning model with 94% F1-score for attrition prediction

- 🧮 Risk Calculator: Interactive tool to predict attrition risk for new employees

- 📊 Professional Visualizations: Plotly-powered interactive charts and graphs


## 🛠️ Tech Stack
- Python 3.8+
- Streamlit - Web dashboard framework
- Scikit-learn - Machine learning (RandomForest)
- Pandas & NumPy - Data manipulation
- Plotly - Interactive visualizations
- Matplotlib & Seaborn - Static visualizations

## 📁 Project Structure
```
hr-analytics-dashboard/
├── data/
│   ├── raw/                   # Original HR data files
│   └── processed/            # Cleaned and processed datasets
├── notebooks/
│   ├── 01_eda.ipynb          # Exploratory Data Analysis
│   ├── 02_feature_engineering.ipynb
│   ├── 03_attrition_analysis.ipynb
│   └── 04_model_training.ipynb
├── src/
│   ├── data/                 # Data loading & cleaning
│   ├── features/             # Feature engineering
│   ├── models/               # ML model training
│   ├── visualization/        # Plotting utilities
│   └── app/                  # Streamlit dashboard
├── models/                   # Serialized trained models
├── tests/                    # Unit tests
├── requirements.txt          # Python dependencies
├── run_pipeline.py          # Automated pipeline runner
├── run_dashboard.py         # Dashboard launcher
└── README.md
```

## 🏃 Quick Start

```bash
# Prerequisites
pip install -r requirements.txt

#1. Run Complete Pipeline
python run_pipeline.py

# 2. Launch Dashboard
python run_dashboard.py
```

# 3. Access Dashboard
Open http://localhost:8501 in your browser

## 📊 Model Performance
Our optimized RandomForest model achieved outstanding results:

- F1 Score: 0.94
- ROC-AUC: 0.999
- Accuracy: 97.7%

## Key Predictors of Attrition:

1. Salary vs Role Average (Most important)
2. Salary vs Department Average
3. Total Compensation
4. Age
5. Salary to Bonus Ratio

## 🔍 Insights Discovered
- Sales department has highest attrition rate (21.8%)
- Operations department has lowest attrition (18.8%)
- Employees with below-average salaries for their role are 3x more likely to leave
- Younger employees (25-35) show higher attrition rates
- Bonus structure significantly impacts retention

## 🎯 How to Use

1. Overview Analytics: Check workforce demographics and distributions
2. Attrition Analysis: Explore patterns and correlations
3. Prediction Insights: View risk distributions and key factors
4. Risk Calculator: Predict attrition risk for specific employees

## 🤝 Contributing
1. Fork the repository
2. Create a feature branch (git checkout -b feature/amazing-feature)
3. Commit changes (git commit -m 'Add amazing feature')
4. Push to branch (git push origin feature/amazing-feature)
5. Open a Pull Request

## 🙏 Acknowledgments
* HR dataset provided by [ME 🧑]
* Built with Streamlit framework
* Machine learning powered by Scikit-learn