# Defense Agricultural Price Forecasting System

> AI-powered agricultural price prediction system for defense procurement optimization

An intelligent forecasting system designed to predict agricultural commodity prices for military food supply procurement, utilizing LSTM deep learning models and comprehensive data analysis to optimize cost-effective procurement strategies.

---

## Table of Contents
- [Overview](#overview)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [Data Sources](#data-sources)
- [Modeling Approach](#modeling-approach)
- [Installation & Setup](#installation--setup)
- [Usage](#usage)
- [Results & Performance](#results--performance)
- [Visualization](#visualization)
- [Contributors](#contributors)

---

## Overview

This project develops a comprehensive agricultural price forecasting system specifically designed for defense procurement operations. The system analyzes historical pricing data, economic indicators, weather patterns, and market supply volumes to predict future prices of key agricultural commodities used in military food services.

### Problem Statement
Military food service operations require stable and cost-effective procurement of agricultural products. Price volatility in agricultural markets can significantly impact procurement budgets and planning. This system addresses these challenges by providing accurate price predictions to enable optimal procurement timing.

### Solution
- **LSTM-based deep learning models** for time series price prediction
- **Multi-variable analysis** incorporating economic indicators, weather data, and market logistics
- **Interactive dashboard** for procurement decision support
- **Power BI visualizations** for comprehensive data insights

---

## Key Features

- **AI-Powered Price Forecasting**: LSTM and SARIMAX models for accurate price predictions
- **Multi-Factor Analysis**: Considers economic indicators (GDP, interest rates, exchange rates), weather patterns, and market supply
- **Real-time Dashboard**: Flask-based web application for monitoring and analysis
- **Comprehensive EDA**: Extensive exploratory data analysis with correlation studies
- **Seasonal Insights**: Identifies seasonal price patterns and optimal procurement periods
- **Data Visualization**: Power BI integration for interactive reports and dashboards

---

## System Architecture

### Work Breakdown Structure (WBS)
![WBS](https://github.com/user-attachments/assets/8018477a-1dcd-4190-8ac7-d3c3d4d02866)

### Entity-Relationship Diagram (ERD)
![ERD](https://github.com/user-attachments/assets/30a8a0e2-3757-428e-8d6a-1fc22d9ccb00)

### System Architecture Diagram
![System Architecture](https://github.com/user-attachments/assets/7ac28cd8-6c41-40f8-bae3-21073578d6dc)

### System Flow and Configuration
![System Flow](https://github.com/user-attachments/assets/ce0dd4e3-2d11-4d13-8d27-f6f440505b1c)

---

## Technology Stack

### Core Technologies
- **Python 3.x** - Primary programming language
- **TensorFlow/Keras** - Deep learning framework for LSTM models
- **Pandas & NumPy** - Data manipulation and analysis
- **Matplotlib & Seaborn** - Data visualization
- **Scikit-learn** - Machine learning utilities and preprocessing

### Web Framework
- **Flask** - Backend web application framework
- **HTML/CSS/JavaScript** - Frontend development

### Database & Analytics
- **Excel/CSV** - Data storage and interchange
- **Power BI** - Business intelligence and visualization

### Time Series Analysis
- **Statsmodels** - SARIMAX statistical modeling
- **LSTM Networks** - Long Short-Term Memory neural networks

---

## Project Structure

```
Defense-Agri-Price-Forecasting/
│
├── 가격예측 AI 모델링/          # AI Modeling
│   ├── latest_lstm_prediction.ipynb    # Main LSTM prediction model
│   ├── SARIMAX_이현동.ipynb           # SARIMAX statistical model
│   ├── lstm_predict.ipynb              # LSTM training scripts
│   └── *.png                           # Model architecture visualizations
│
├── 전처리 부분/                 # Data Preprocessing
│   ├── filled_date_price.ipynb         # Date filling and interpolation
│   ├── new_product_price.ipynb         # New product data processing
│   └── EDAdata_make.ipynb              # EDA dataset creation
│
├── EDA(탐색적 데이터 분석)/      # Exploratory Data Analysis
│   ├── *.ipynb                         # Various EDA notebooks
│   ├── *.png                           # Correlation and trend visualizations
│   └── 전처리/                         # EDA preprocessing scripts
│
├── acorn_web/                   # Web Application
│   ├── app.py                          # Flask application
│   └── main/                           # HTML templates and assets
│       ├── main_dashboard.html         # Main dashboard
│       ├── menu_management.html        # Menu management interface
│       ├── sub_chart_*.html           # Economic, logistics, weather charts
│       ├── scripts/                    # JavaScript files
│       └── styles/                     # CSS stylesheets
│
├── DB/                          # Database and Raw Data
│   ├── 식품예측 독립변수 데이터/        # Independent variables
│   │   ├── GDP_*.xlsx                  # GDP economic indicators
│   │   ├── 환율_*.xlsx                 # Exchange rate data
│   │   ├── 출하량_가락/                # Market shipment volumes
│   │   └── 경유_*.xls                  # Diesel price data
│   └── 재배지날씨/                     # Cultivation area weather data
│
├── PowerBI/                     # Power BI Reports
│   ├── Project_Power_BI_File/          # Power BI project files
│   └── Project_Figma_Image/            # Design mockups
│
├── Server/                      # Server Scripts
│   ├── Ubuntu_transfer_csv.sh          # Ubuntu data transfer script
│   └── window_transfer_csv.ps1         # Windows data transfer script
│
├── ERD Table/                   # Database Schema Documentation
├── PT/                          # Presentation Materials
├── 기획서와주간업무일지/          # Planning & Weekly Reports
└── README.md                    # This file
```

---

## Data Sources

### Agricultural Price Data
- **Korea Agro-Fisheries & Food Trade Corporation**: Daily retail and wholesale prices
- **Garak Market**: Shipment volume data for major agricultural products
- **Product Categories**: Vegetables (cabbage, radish, green onion, garlic), Fruits (apple, strawberry, tangerine), Potatoes, Rice

### Economic Indicators
- **GDP**: Quarterly and annual economic activity indicators
- **Exchange Rates**: Daily USD/KRW exchange rates from Bank of Korea
- **Interest Rates**: Base rates and market lending rates
- **Oil Prices**: Daily diesel prices affecting transportation costs

### Weather Data
- **Korea Meteorological Administration**: Temperature, precipitation, humidity data
- **Cultivation Areas**: Region-specific weather data for major growing regions
  - Spring cabbage: Gangwon Yeongwol, Gyeongbuk Mungyeong
  - Fall cabbage: Jeonnam Haenam, Jeonbuk Gochang

### Market Logistics
- **Shipment Volumes**: Daily market inflow data from Garak Market
- **Import Data**: Monthly import quantities for key products

---

## Modeling Approach

### Data Preprocessing
![Data Preprocessing](https://github.com/user-attachments/assets/723986a9-1767-4db1-96fc-d50155b10d75)

**Preprocessing Steps:**
1. **Data Cleaning**: Handling missing values, outlier detection and treatment
2. **Date Normalization**: Filling gaps in time series data
3. **Feature Engineering**: Creating lag features, rolling averages, seasonal indicators
4. **Scaling**: RobustScaler for handling outliers in economic variables
5. **Integration**: Merging multiple data sources on temporal keys

### Model Architecture
![Modeling Approach](https://github.com/user-attachments/assets/453e6163-0afa-4857-aa65-e90e5811ec0a)

**LSTM Neural Network:**
- **Input Layer**: Multi-variate time series data (prices, economic indicators, weather, logistics)
- **LSTM Layers**: Stacked LSTM layers with dropout for regularization
- **Dense Layers**: Fully connected layers for final prediction
- **Output**: Price forecast for target agricultural products
- **Early Stopping**: Prevents overfitting during training

**SARIMAX Statistical Model:**
- Seasonal AutoRegressive Integrated Moving Average with eXogenous factors
- Captures seasonal patterns and trends in agricultural prices
- Incorporates external economic and weather variables

### Feature Selection
Based on correlation analysis:
- Economic variables with >0.5 correlation to target prices
- Seasonal harvest period indicators
- Weather conditions during cultivation periods
- Previous price lags (1-day, 7-day, 30-day)

---

## Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Virtual environment (recommended)

### Installation Steps

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/Defense-Agri-Price-Forecasting.git
cd Defense-Agri-Price-Forecasting-main
```

2. **Create virtual environment** (recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

*Note: If requirements.txt is not available, install these core packages:*
```bash
pip install flask pandas numpy tensorflow scikit-learn matplotlib seaborn statsmodels openpyxl jupyter
```

4. **Verify installation**
```bash
python -c "import tensorflow; print(tensorflow.__version__)"
```

---

## Usage

### Running the Web Application

1. **Navigate to the web app directory**
```bash
cd acorn_web
```

2. **Start the Flask server**
```bash
python app.py
```

3. **Access the dashboard**
Open your browser and navigate to: `http://localhost:5000`

### Available Routes
- `/` - Main Dashboard
- `/sub_chart_economy` - Economic Indicators Analysis
- `/sub_chart_logistics` - Logistics and Supply Analysis
- `/sub_chart_weather` - Weather Impact Analysis
- `/sub_chart_oil` - Oil Price Trends
- `/menu_management` - Menu and Product Management

### Running Price Predictions

1. **Navigate to modeling directory**
```bash
cd "가격예측 AI 모델링"
```

2. **Open Jupyter Notebook**
```bash
jupyter notebook latest_lstm_prediction.ipynb
```

3. **Execute cells** to train model and generate predictions

### Exploratory Data Analysis

```bash
cd "EDA(탐색적 데이터 분석)"
jupyter notebook
```

Open any of the EDA notebooks to explore:
- Correlation analysis between prices and economic variables
- Seasonal price patterns
- Regional weather impact studies
- Price volatility analysis

---

## Results & Performance

### Exploratory Data Analysis Insights
![EDA Results](https://github.com/user-attachments/assets/f3c5e155-2dc5-49db-b604-83fac02da115)

**Key Findings:**
- **Seasonal Price Variations**: Identified products with >50% monthly price variance
- **Stable Products**: Found items with <10% annual price fluctuation suitable for long-term contracts
- **Economic Correlations**: GDP and exchange rates show strong correlation (>0.5) with certain product prices
- **Weather Impact**: Temperature and precipitation during cultivation significantly affect yields and prices

### Model Performance
- **LSTM Model**: Captures long-term dependencies and complex patterns in price movements
- **SARIMAX Model**: Effectively models seasonal variations and trend components
- **Ensemble Approach**: Combining both models improves prediction accuracy

### Sample Predictions
- Predicted optimal procurement timing for seasonal vegetables
- Forecasted price trends for 1-3 month horizons
- Identified cost-saving opportunities through strategic timing

---

## Visualization

### Power BI Dashboard
![Power BI Visualization](https://github.com/user-attachments/assets/24abdadf-933c-4035-aebf-d4ecabc5d5a6)

**Dashboard Features:**
- Real-time price monitoring across multiple products
- Economic indicator trends and correlations
- Market supply and demand dynamics
- Weather impact visualization
- Procurement recommendations based on predictions

**Interactive Elements:**
- Date range filtering
- Product category selection
- Regional comparisons
- Export capabilities for reports

---

## Contributors

This project was developed as part of a defense procurement optimization initiative.

**Team Members:**
- Data Engineering & Preprocessing
- Machine Learning & Model Development
- Web Development & Dashboard
- Data Analysis & Visualization

---

## License

This project is developed for educational and research purposes.

---

## Acknowledgments

- **Data Sources**: Korea Agro-Fisheries & Food Trade Corporation, Bank of Korea, Korea Meteorological Administration
- **Research References**: Various academic papers on agricultural price forecasting using LSTM and time series analysis
- **Framework Credits**: TensorFlow, Flask, Power BI

---

## Contact & Support

For questions, issues, or contributions, please open an issue in this repository.

---

**Last Updated**: December 2024

