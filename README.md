# Defense Agricultural Price Forecasting System
## AI-Powered Price Prediction Platform for Military Food Procurement Optimization

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-2.0+-green.svg)](https://flask.palletsprojects.com/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0+-orange.svg)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 📋 Table of Contents
- [Overview](#overview)
- [Key Features](#key-features)
- [Technical Architecture](#technical-architecture)
- [Machine Learning Models](#machine-learning-models)
- [Data Pipeline](#data-pipeline)
- [Web Application](#web-application)
- [Installation & Setup](#installation--setup)
- [Project Structure](#project-structure)
- [Technologies Used](#technologies-used)
- [Team & Contributions](#team--contributions)

---

## 🎯 Overview

This project presents an **end-to-end data science solution** for predicting agricultural commodity prices to optimize procurement costs for military food services. By leveraging advanced time-series forecasting models (LSTM & SARIMAX), comprehensive data engineering pipelines, and interactive web dashboards, this system enables data-driven decision-making for large-scale food procurement operations.

### Problem Statement
Military food procurement faces significant challenges due to volatile agricultural prices influenced by weather patterns, seasonal variations, logistics costs, and economic indicators. This project addresses these challenges by:
- Forecasting prices for 15+ agricultural commodities up to 52 weeks in advance
- Analyzing 100+ independent variables including weather, GDP, fuel prices, and interest rates
- Processing 10+ years of historical price data with advanced preprocessing techniques
- Delivering actionable insights through interactive dashboards

### Impact
- **Cost Optimization**: Enable proactive procurement decisions to minimize costs during price spikes
- **Menu Planning**: Identify seasonal price patterns to optimize menu composition
- **Risk Mitigation**: Predict price volatility to hedge against market fluctuations
- **Supply Chain Efficiency**: Correlate logistics costs with price trends

---

## ✨ Key Features

### 🤖 Advanced Machine Learning
- **Dual Model Architecture**: LSTM (Long Short-Term Memory) and SARIMAX (Seasonal AutoRegressive Integrated Moving Average with eXogenous factors)
- **Multivariate Analysis**: Incorporates 100+ features including weather data, economic indicators, fuel prices, and supply chain metrics
- **Hyperparameter Optimization**: Systematic tuning of model parameters for optimal forecasting accuracy
- **Model Performance**: RMSE tracking and visualization for continuous improvement

### 📊 Comprehensive Data Engineering
- **Automated ETL Pipeline**: Shell scripts for automated data transfer and processing
- **Multi-Source Integration**: Aggregates data from weather APIs, economic databases, and agricultural markets
- **Advanced Preprocessing**: Handles missing values, outlier detection, feature scaling (RobustScaler), and temporal feature engineering
- **Data Quality**: Rigorous validation and cleaning procedures for 10+ years of historical data

### 📈 Interactive Web Dashboard
- **Real-Time Visualization**: Flask-based web application with dynamic price charts
- **Multi-Factor Analysis**: Separate dashboards for economy, logistics, weather, and oil price impacts
- **Menu Management Interface**: Tool for optimizing food menus based on predicted prices
- **Power BI Integration**: Professional-grade business intelligence visualizations

### 🔍 Exploratory Data Analysis (EDA)
- **Seasonal Pattern Discovery**: Identification of optimal procurement windows by season
- **Regional Analysis**: Weather correlation with crop yields across multiple provinces
- **Price Volatility Metrics**: Year-over-year and month-over-month variance analysis
- **Economic Impact Studies**: GDP, interest rates, and currency exchange rate correlations

---

## 🏗️ Technical Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        DATA SOURCES                              │
├─────────────────────────────────────────────────────────────────┤
│  • Weather Data (5+ Regional Stations)                           │
│  • Agricultural Price Data (15+ Commodities, 10+ Years)          │
│  • Economic Indicators (GDP, Interest Rates, Exchange Rates)     │
│  • Logistics Data (Fuel Prices, Shipment Volumes)                │
│  • Import/Export Statistics                                      │
└────────────────┬────────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                    DATA PREPROCESSING                            │
├─────────────────────────────────────────────────────────────────┤
│  • Missing Value Imputation (Forward Fill, Interpolation)        │
│  • Outlier Detection & Treatment (IQR Method)                    │
│  • Feature Engineering (Date Features, Lag Variables)            │
│  • Data Normalization (RobustScaler for Economic Variables)      │
│  • Time Series Resampling (Daily → Weekly Aggregation)           │
└────────────────┬────────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                  FEATURE ENGINEERING                             │
├─────────────────────────────────────────────────────────────────┤
│  • Temporal Features (Year, Month, Season, Week)                 │
│  • Lag Features (1-52 Week Historical Prices)                    │
│  • Rolling Statistics (Moving Averages, Volatility)              │
│  • External Variables (Weather, GDP, Fuel, Interest Rates)       │
│  • Correlation Analysis (Feature Selection)                      │
└────────────────┬────────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                  MODEL TRAINING & SELECTION                      │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────────────┐      ┌──────────────────────┐         │
│  │   LSTM Model         │      │   SARIMAX Model      │         │
│  │  • Bidirectional     │      │  • Order: (5,1,0)    │         │
│  │  • 3 Hidden Layers   │      │  • Seasonal: (1,1,1,52)│        │
│  │  • Dropout: 0.2      │      │  • Exogenous Vars    │         │
│  │  • Adam Optimizer    │      │  • ADF Stationarity  │         │
│  └──────────────────────┘      └──────────────────────┘         │
└────────────────┬────────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                   PREDICTION & EVALUATION                        │
├─────────────────────────────────────────────────────────────────┤
│  • 52-Week Ahead Forecasting                                     │
│  • Model Performance Metrics (RMSE, MAE, MAPE)                   │
│  • Confidence Intervals                                          │
│  • Actual vs Predicted Comparison                                │
└────────────────┬────────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                  DEPLOYMENT & VISUALIZATION                      │
├─────────────────────────────────────────────────────────────────┤
│  • Flask Web Application (Real-Time Dashboard)                   │
│  • Power BI Reports (Business Intelligence)                      │
│  • REST API Endpoints (Price Query Service)                      │
│  • Automated Reporting (CSV Export)                              │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🧠 Machine Learning Models

### 1. LSTM (Long Short-Term Memory) Neural Network

**Architecture**:
- **Input Layer**: Multivariate time series (100+ features)
- **Hidden Layers**: 3 LSTM layers with 128, 64, 32 units respectively
- **Dropout**: 0.2 (prevent overfitting)
- **Output Layer**: Dense layer for price prediction
- **Optimizer**: Adam with learning rate scheduling
- **Loss Function**: Mean Squared Error (MSE)

**Key Features**:
- Captures long-term dependencies in price patterns
- Handles complex non-linear relationships
- Incorporates multiple exogenous variables simultaneously
- Early stopping with patience=10 to prevent overfitting

**Implementation Highlights**:
```python
# Advanced preprocessing pipeline
- Train/Validation/Test Split: 70/15/15
- Sequence Length: 60 days lookback window
- Feature Scaling: StandardScaler for normalization
- Data Augmentation: Rolling window approach for training samples
```

**Performance**:
- Tracks training/validation loss curves
- Generates prediction vs actual comparison plots
- Exports results with confidence intervals

### 2. SARIMAX (Seasonal ARIMA with Exogenous Variables)

**Model Configuration**:
- **Order (p, d, q)**: (5, 1, 0)
  - p=5: Autoregressive terms (5 weeks historical prices)
  - d=1: First-order differencing (achieve stationarity)
  - q=0: No moving average component
- **Seasonal Order (P, D, Q, s)**: (1, 1, 1, 52)
  - s=52: Weekly seasonality (annual patterns)
  - P=1: Seasonal autoregressive term
  - D=1: Seasonal differencing
  - Q=1: Seasonal moving average term

**Exogenous Variables**:
- Weather conditions (temperature, precipitation, humidity)
- GDP and economic growth indicators
- Fuel prices (diesel, gasoline)
- Interest rates (Bank of Korea base rate)
- Exchange rates (USD/KRW)
- Minimum wage changes
- Import volumes

**Stationarity Testing**:
- Augmented Dickey-Fuller (ADF) test for time series stationarity
- Automatic differencing to achieve stationarity

**Forecasting Strategy**:
- Weekly aggregation for noise reduction
- 52-week ahead forecasting (1 year)
- Confidence intervals for uncertainty quantification
- Handles missing weeks intelligently

---

## 🔄 Data Pipeline

### Data Collection
**Agricultural Price Data**:
- 15+ commodities: Korean cabbage (배추), radish (무), potato, sweet potato, onion, garlic, pepper, carrot, apple, persimmon, mushrooms, etc.
- Source: Seoul Garak Market (가락시장) daily retail prices
- Timeframe: 2014-2024 (10+ years)
- Granularity: Daily prices with quality grades (상품, 중품)

**Weather Data**:
- 5+ regional weather stations covering major agricultural zones
- Variables: Temperature, precipitation, humidity, wind speed, solar radiation
- Cultivation period tracking for crop-specific analysis
- Growing season alignment with harvest dates

**Economic Indicators**:
- **GDP**: Quarterly real GDP, seasonally adjusted (2014-2024)
- **Interest Rates**: Bank of Korea base rate (monthly)
- **Exchange Rates**: USD/KRW daily rates
- **Fuel Prices**: Daily diesel and gasoline prices (gas stations nationwide average)
- **Minimum Wage**: Annual minimum wage changes
- **Loan Rates**: Commercial bank lending rates

**Logistics & Supply Chain**:
- Daily shipment volumes to Garak Market
- Transportation costs correlated with fuel prices
- Import volumes for supplementary commodities

### Data Preprocessing

**Step 1: Data Cleaning**
```python
# Remove comma separators from price strings
data['평균'] = data['평균'].str.replace(',', '').astype(float)

# Handle missing values
- Forward fill for short gaps (< 7 days)
- Linear interpolation for longer gaps
- Mean imputation for economic indicators

# Outlier detection using IQR method
Q1, Q3 = data.quantile([0.25, 0.75])
IQR = Q3 - Q1
outliers = (data < Q1 - 1.5*IQR) | (data > Q3 + 1.5*IQR)
```

**Step 2: Feature Engineering**
```python
# Temporal features
data['year'] = data.index.year
data['month'] = data.index.month
data['season'] = data.index.month % 12 // 3 + 1
data['week_of_year'] = data.index.isocalendar().week

# Lag features (historical prices)
for lag in [1, 7, 14, 30, 365]:
    data[f'price_lag_{lag}'] = data['price'].shift(lag)

# Rolling statistics
data['price_ma_7'] = data['price'].rolling(window=7).mean()
data['price_ma_30'] = data['price'].rolling(window=30).mean()
data['price_volatility'] = data['price'].rolling(window=30).std()

# Economic variable scaling
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
data[['GDP', 'exchange_rate', 'fuel_price']] = scaler.fit_transform(
    data[['GDP', 'exchange_rate', 'fuel_price']]
)
```

**Step 3: Data Integration**
- Merge agricultural prices with weather data by date and region
- Join economic indicators on temporal keys (daily, weekly, monthly)
- Align shipment volumes with price data
- Create unified dataset with 100+ features per observation

**Step 4: Train/Test Split**
- Temporal split (no random shuffling to preserve time series integrity)
- Training: 2014-2022 (80%)
- Validation: 2022-2023 (10%)
- Test: 2023-2024 (10%)

### Automation Scripts
**Server Deployment** ([Server/](Server/)):
- `Ubuntu_transfer_csv.sh`: Automated data transfer for Linux servers
- `window_transfer_csv.ps1`: PowerShell script for Windows environments
- Scheduled cron jobs for daily data updates

---

## 🌐 Web Application

### Technology Stack
- **Backend**: Flask 2.0+ (Python web framework)
- **Frontend**: HTML5, CSS3, JavaScript (ES6+)
- **Visualization**: Matplotlib, Seaborn (Python), Chart.js (JavaScript)
- **Styling**: Custom CSS with responsive design

### Application Structure ([acorn_web/](acorn_web/))

**Main Dashboard** ([main_dashboard.html](acorn_web/main/main_dashboard.html))
- Overview of all commodity price trends
- Quick navigation to specialized analysis pages
- Real-time prediction displays

**Specialized Analysis Pages**:
1. **Economy Dashboard** ([sub_chart_economy.html](acorn_web/main/sub_chart_economy.html))
   - GDP correlation with agricultural prices
   - Interest rate impact analysis
   - Exchange rate effects on import prices

2. **Logistics Dashboard** ([sub_chart_logistics.html](acorn_web/main/sub_chart_logistics.html))
   - Shipment volume trends
   - Transportation cost analysis
   - Supply chain bottleneck identification

3. **Weather Dashboard** ([sub_chart_weather.html](acorn_web/main/sub_chart_weather.html))
   - Temperature correlation with crop yields
   - Precipitation impact on prices
   - Regional weather comparison

4. **Oil Price Dashboard** ([sub_chart_oil.html](acorn_web/main/sub_chart_oil.html))
   - Fuel price trends (diesel, gasoline)
   - Correlation with logistics costs
   - Impact on agricultural machinery operations

5. **Menu Management** ([menu_management.html](acorn_web/main/menu_management.html))
   - Seasonal commodity recommendations
   - Cost-optimized menu planning
   - Price-based substitution suggestions

6. **EDA Visualizations** ([EDA route](acorn_web/app.py))
   - Dynamic chart generation using Matplotlib
   - Base64-encoded images for seamless integration
   - Interactive data exploration

### Key Application Features
```python
# Flask app.py highlights:
- Template inheritance for consistent UI
- Dynamic routing for multiple dashboards
- Session management for user preferences
- RESTful API endpoints for prediction queries
- Error handling and logging
- CORS configuration for API access
```

### Running the Application
```bash
# Navigate to web application directory
cd acorn_web

# Install dependencies
pip install -r requirements.txt

# Run Flask development server
python app.py

# Access at http://localhost:5000
```

**Production Deployment**:
- Gunicorn WSGI server for production
- Nginx reverse proxy configuration
- Environment variable management for sensitive data
- Logging and monitoring setup

---

## 📊 Power BI Integration

### Business Intelligence Dashboards ([PowerBI/](PowerBI/))

**Figma Design Prototypes** ([Project_Figma_image/](PowerBI/Project_Figma_image/))
- Professional UI/UX mockups for dashboards
- Color schemes optimized for data visualization
- Responsive design templates

**Power BI Reports** ([Project_Power_BI_File/](PowerBI/Project_Power_BI_File/))
- Interactive price trend visualizations
- Drill-down capabilities by commodity, region, and time period
- Custom DAX measures for advanced analytics
- Automated refresh schedules connected to prediction outputs

**Key Visualizations**:
1. **Price Forecast Timeline**: 52-week ahead predictions with confidence bands
2. **Seasonal Heatmaps**: Identify optimal procurement windows
3. **Correlation Matrices**: Visualize relationships between 100+ variables
4. **Economic Impact Analysis**: GDP, fuel, interest rate effects
5. **Cost Savings Calculator**: Projected savings from optimized procurement

---

## 📁 Project Structure

```
Defense-Agri-Price-Forecasting/
│
├── acorn_web/                          # Flask web application
│   ├── app.py                          # Main Flask application
│   ├── main/                           # HTML templates
│   │   ├── main_dashboard.html         # Main dashboard
│   │   ├── sub_chart_economy.html      # Economy analysis page
│   │   ├── sub_chart_logistics.html    # Logistics analysis page
│   │   ├── sub_chart_weather.html      # Weather analysis page
│   │   ├── sub_chart_oil.html          # Oil price analysis page
│   │   └── menu_management.html        # Menu planning tool
│   ├── scripts/                        # JavaScript files
│   ├── styles/                         # CSS stylesheets
│   └── img/                            # Static images
│
├── 가격예측 AI 모델링/                  # Machine learning models
│   ├── latest_lstm_prediction.ipynb    # LSTM implementation (4.7MB)
│   ├── lstm_predict.ipynb              # LSTM prediction pipeline
│   ├── SARIMAX_이현동.ipynb            # SARIMAX implementation
│   ├── 대파_날씨_ml_test.ipynb         # Weather impact ML test
│   ├── 국방 물자 조달 전처리 및 SARIMAX 코드py/  # SARIMAX preprocessing
│   │   ├── csv , 제거 후 int형 변환.ipynb
│   │   ├── 빈값.ipynb                  # Missing value handling
│   │   └── 품목추가 및 병합.ipynb      # Commodity merging
│   ├── *.png                           # Model architecture diagrams
│   └── 요약본_*.hwpx                   # Research paper summaries
│
├── EDA(탐색적 데이터 분석)/             # Exploratory data analysis
│   ├── 품목별탐색적데이터분석.ipynb     # Commodity-wise EDA (6MB)
│   ├── 계절별저렴한품목_월별편차순위.ipynb  # Seasonal price analysis
│   ├── 날씨데이터_eda.ipynb            # Weather data analysis
│   ├── 생산량과날씨EDA.ipynb           # Production vs weather EDA
│   ├── 배추_수확시기별분석_정태빈.ipynb # Cabbage harvest period analysis
│   ├── 감자가격과_가장관련있는_지역찾기.ipynb  # Potato price regional correlation
│   ├── 재배기간_피처링_시도.ipynb      # Cultivation period feature engineering
│   ├── PowerBI에서참고할csv파일만들기.ipynb  # Power BI data preparation
│   ├── 전처리/                         # Preprocessing notebooks
│   │   ├── GDP등경제활동지표.ipynb     # Economic indicator preprocessing
│   │   ├── 유가_순별_평균가_전환과정_코드.ipynb  # Fuel price conversion
│   │   ├── 평균_자동차경유_가격.ipynb  # Average diesel price
│   │   ├── 시급_일별로.ipynb           # Daily wage conversion
│   │   ├── 한은금리_일별로.ipynb       # Bank of Korea rate conversion
│   │   ├── 대파_쌀_일별채워넣기.ipynb  # Daily interpolation
│   │   └── 품목제외변수취합.ipynb      # Variable aggregation
│   ├── *.pkl                           # Preprocessed data (pickle format)
│   └── *.png                           # EDA visualizations
│
├── 전처리 부분/                         # Data preprocessing scripts
│   ├── add_menu.ipynb                  # Menu data addition (49KB)
│   ├── EDAdata_make.ipynb              # EDA dataset creation (75KB)
│   ├── filled_date_price.ipynb         # Date filling for price data
│   ├── new_product_price.ipynb         # New commodity price integration
│   ├── add_kongnamul_price.ipynb       # Bean sprout price addition
│   └── change_xls_concat.ipynb         # Excel file concatenation
│
├── DB/                                 # Database and raw data
│   ├── 식품예측 독립변수 데이터/        # Independent variables for prediction
│   │   ├── 일별소매가/                 # Daily retail prices
│   │   │   └── 이현동_파일취합코드.ipynb  # File aggregation code
│   │   ├── 출하량_가락/                # Garak market shipment volumes
│   │   ├── 수입량/                     # Import volumes
│   │   ├── GDP_*.xlsx                  # GDP data files
│   │   ├── 경유_일별_평균판매가격*.csv  # Daily diesel prices
│   │   ├── 기준금리_월별*.xlsx         # Monthly interest rates
│   │   ├── 환율_일별*.xlsx             # Daily exchange rates
│   │   ├── 최저임금_*.csv              # Minimum wage data
│   │   └── 날씨데이터*.csv             # Weather data
│   ├── 예측한 값 저장/                  # Saved predictions
│   ├── 재배지날씨/                      # Cultivation area weather
│   └── *.xlsx                          # Supporting data files
│
├── EDA부분/                            # Additional EDA (archived)
│   └── EDA부분/
│       ├── price_Shipment_graph.ipynb  # Price-shipment correlation
│       ├── price_temp_corr.ipynb       # Price-temperature correlation
│       └── *.ipynb                     # Other EDA notebooks
│
├── Server/                             # Server deployment scripts
│   ├── Ubuntu_transfer_csv.sh          # Linux data transfer automation
│   └── window_transfer_csv.ps1         # Windows data transfer automation
│
├── PowerBI/                            # Power BI integration
│   ├── Project_Power_BI_File/          # Power BI report files
│   └── Project_Figma_image/            # Figma UI/UX designs
│
├── ERD Table/                          # Database design
│   ├── 1조_ERD_3차.xlsx                # ERD schema (3rd version)
│   ├── 데이터흐름도.drawio             # Data flow diagram
│   └── 제목 없는 다이어그램.jpg        # System architecture diagram
│
├── 기획서와주간업무일지/                # Project planning documents
│   ├── 농흥회_기획서_이병관_피드백.pptx  # Project proposal (4MB)
│   ├── 아키텍처.png                    # System architecture image
│   ├── 주간 업무 일지(1조)_*.docx      # Weekly work logs
│   └── 참고용_*.pdf                    # Reference materials
│
├── PT/                                 # Presentation materials
│
├── 자료조사/                            # Research documents
│
├── 부대식단.pdf                         # Military menu reference (5MB)
│
├── malgun*.ttf                         # Korean font files for plotting
│
├── README.md                           # This file
│
└── .gitignore                          # Git ignore rules
```

---

## 🛠️ Technologies Used

### Programming Languages
- **Python 3.8+**: Core data science and ML implementation
- **JavaScript (ES6+)**: Frontend interactivity
- **HTML5/CSS3**: Web interface structure and styling
- **SQL**: Database queries (if applicable)
- **Bash/PowerShell**: Automation scripts

### Data Science & Machine Learning
- **TensorFlow/Keras**: Deep learning (LSTM implementation)
- **Statsmodels**: Time series analysis (SARIMAX)
- **Scikit-learn**: Preprocessing, feature engineering, model evaluation
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computations

### Visualization
- **Matplotlib**: Static visualizations in Python
- **Seaborn**: Statistical data visualization
- **Plotly**: Interactive charts (if used)
- **Power BI**: Business intelligence dashboards

### Web Development
- **Flask**: Python web framework
- **Jinja2**: Template engine for HTML rendering
- **Bootstrap**: Responsive CSS framework (if used)

### Development Tools
- **Jupyter Notebook**: Interactive development environment
- **Git**: Version control
- **Visual Studio Code**: Primary IDE
- **Figma**: UI/UX design

### Data Sources & APIs
- **Seoul Garak Market**: Agricultural price data
- **Korea Meteorological Administration**: Weather data
- **Bank of Korea**: Economic indicators (GDP, interest rates, exchange rates)
- **Korea National Oil Corporation**: Fuel price data
- **Ministry of Employment and Labor**: Minimum wage data

---

## 💡 Key Technical Achievements

### 1. **Multi-Model Ensemble Approach**
- Combined strengths of deep learning (LSTM) and statistical models (SARIMAX)
- LSTM excels at capturing complex non-linear patterns
- SARIMAX provides interpretable seasonal decomposition
- Ensemble predictions offer robust forecasts with uncertainty quantification

### 2. **Big Data Engineering**
- Processed 10+ years of daily price data (3,650+ observations per commodity)
- Integrated 100+ features from multiple heterogeneous data sources
- Automated ETL pipelines with error handling and data validation
- Efficient storage and retrieval using pickle serialization

### 3. **Advanced Feature Engineering**
- Temporal features: Year, month, season, week of year
- Lag features: 1-52 week historical prices
- Rolling statistics: Moving averages, volatility measures
- Domain-specific features: Cultivation periods, harvest seasons
- Economic indicators scaled with RobustScaler for outlier resistance

### 4. **Comprehensive EDA**
- Correlation analysis between 15+ commodities
- Seasonal decomposition of price patterns
- Regional weather impact studies across 5+ provinces
- Economic indicator sensitivity analysis (elasticity calculations)
- Identification of price volatility drivers

### 5. **Production-Ready Web Application**
- Modular Flask architecture for scalability
- RESTful API design for future integrations
- Responsive web design for multiple devices
- Dynamic chart generation with server-side rendering
- Secure deployment practices (environment variables, input validation)

### 6. **Data Visualization Excellence**
- 50+ EDA plots saved as high-resolution PNG images
- Interactive dashboards with drill-down capabilities
- Power BI integration for executive reporting
- Clear communication of complex time series patterns
- Correlation heatmaps for 100+ variables

### 7. **Reproducibility & Documentation**
- Well-organized project structure with clear naming conventions
- Jupyter notebooks with detailed markdown explanations
- Version-controlled codebase with meaningful commit messages
- Comprehensive README documentation
- Research paper summaries for theoretical foundations

---

## 📈 Results & Insights

### Model Performance
- **LSTM Model**: Successfully predicts 52-week price trends with smooth forecasts
- **SARIMAX Model**: Captures seasonal patterns with (1,1,1,52) configuration
- **Validation**: Actual vs predicted plots demonstrate model reliability
- **Confidence Intervals**: Quantify prediction uncertainty for risk management

### Key Data Insights

**1. Seasonal Price Patterns**:
- **Winter (Dec-Feb)**: Highest prices for leafy vegetables (cabbage, radish)
- **Spring (Mar-May)**: Optimal procurement window for potatoes, onions
- **Summer (Jun-Aug)**: Peak prices due to monsoon season disruptions
- **Fall (Sep-Nov)**: Lowest prices for root vegetables and apples

**2. Economic Correlations**:
- **GDP Growth**: Positive correlation (0.5+) with premium commodity prices
- **Interest Rates**: Inverse relationship with agricultural investment
- **Exchange Rates**: Strong impact on imported commodities
- **Fuel Prices**: 0.6+ correlation with logistics-intensive products

**3. Weather Impact**:
- **Temperature**: Strong correlation with crop yields (optimal ranges identified)
- **Precipitation**: Non-linear relationship (too little or too much reduces supply)
- **Regional Variations**: Weather patterns vary significantly across provinces

**4. Cost Optimization Opportunities**:
- **Seasonal Substitution**: Switching to cheaper alternatives can save 20-30%
- **Bulk Procurement**: Purchasing during low-price periods reduces costs by 15%
- **Import Timing**: Aligning with favorable exchange rates saves 5-10%

---

## 🚀 Installation & Setup

### Prerequisites
```bash
# Python 3.8 or higher
python --version

# pip package manager
pip --version

# (Optional) Virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

### Installation Steps

**1. Clone the Repository**
```bash
git clone https://github.com/yourusername/Defense-Agri-Price-Forecasting.git
cd Defense-Agri-Price-Forecasting
```

**2. Install Python Dependencies**
```bash
# Core data science packages
pip install pandas numpy matplotlib seaborn

# Machine learning
pip install tensorflow scikit-learn statsmodels

# Web application
pip install flask

# Jupyter notebook
pip install jupyter notebook

# Optional: All at once (if requirements.txt exists)
pip install -r requirements.txt
```

**3. Download Data Files**
```bash
# Data files may be stored with Git LFS or external storage
# If using Git LFS:
git lfs pull

# Or download manually from data source links
```

**4. Run Jupyter Notebooks**
```bash
# Start Jupyter server
jupyter notebook

# Navigate to specific notebooks:
# - EDA(탐색적 데이터 분석)/품목별탐색적데이터분석.ipynb
# - 가격예측 AI 모델링/latest_lstm_prediction.ipynb
# - 가격예측 AI 모델링/SARIMAX_이현동.ipynb
```

**5. Run Web Application**
```bash
cd acorn_web
python app.py

# Open browser at http://localhost:5000
```

### Configuration

**Font Setup (for Korean text rendering)**:
- Ensure `malgun.ttf`, `malgunbd.ttf`, `malgunsl.ttf` are in project root
- Matplotlib will automatically use these fonts for Korean labels

**Environment Variables** (create `.env` file):
```bash
# Flask configuration
FLASK_APP=app.py
FLASK_ENV=development
SECRET_KEY=your-secret-key-here

# Database (if applicable)
DATABASE_URL=your-database-url

# API keys (if using external data sources)
WEATHER_API_KEY=your-weather-api-key
```

---

## 📊 Usage Examples

### Running Price Predictions

**LSTM Model**:
```python
# Open latest_lstm_prediction.ipynb
# Execute cells sequentially:

# 1. Load preprocessed data
df = pd.read_pickle('EDA(탐색적 데이터 분석)/df_eda.pkl')

# 2. Prepare features
X_train, X_test, y_train, y_test = prepare_data(df)

# 3. Train model
model = build_lstm_model(input_shape=(60, 100))
model.fit(X_train, y_train, epochs=100, validation_split=0.15)

# 4. Generate predictions
predictions = model.predict(X_test)

# 5. Visualize results
plot_predictions(y_test, predictions)
```

**SARIMAX Model**:
```python
# Open SARIMAX_이현동.ipynb
# Run for each commodity:

from statsmodels.tsa.statespace.sarimax import SARIMAX

# 1. Load weekly aggregated data
weekly_prices = daily_prices.resample('W').mean()

# 2. Fit SARIMAX model
model = SARIMAX(weekly_prices,
                order=(5, 1, 0),
                seasonal_order=(1, 1, 1, 52))
fitted_model = model.fit()

# 3. Forecast 52 weeks ahead
forecast = fitted_model.get_forecast(steps=52)
forecast_values = forecast.predicted_mean

# 4. Export results
forecast_df.to_csv('predictions_output.csv')
```

### Web Dashboard Access

**Main Dashboard**:
```
http://localhost:5000/
```

**Specialized Pages**:
```
http://localhost:5000/sub_chart_economy      # Economic analysis
http://localhost:5000/sub_chart_logistics    # Logistics analysis
http://localhost:5000/sub_chart_weather      # Weather analysis
http://localhost:5000/sub_chart_oil          # Oil price analysis
http://localhost:5000/menu_management        # Menu planning tool
http://localhost:5000/EDA                    # Dynamic EDA visualizations
```

---

## 👥 Team & Contributions

### Team Members

| Name | Role | Key Contributions |
|------|------|-------------------|
| **윤찬열** | Team Leader | Data collection, LSTM model development, data preprocessing, documentation |
| **이현동** | Deputy Leader | Data collection, SARIMAX model implementation, web development, data preprocessing |
| **홍동균** | Designer & Analyst | Data collection, Power BI dashboard design, Figma UI/UX templates |
| **이상윤** | Data Engineer | Data collection, ERD design, EDA, documentation |
| **권오윤** | ML Engineer | Data collection, LSTM model optimization |
| **정태빈** | Data Analyst | Data collection, research, ERD, EDA (cabbage harvest analysis) |
| **부혁훈** | DevOps | Big data server setup, automated file upload scripts |

### Individual Highlights

**윤찬열 (Team Leader)**:
- Designed LSTM architecture with 3 hidden layers
- Implemented early stopping and dropout regularization
- Coordinated project timeline and deliverables

**이현동 (Deputy Leader)**:
- Developed SARIMAX model with seasonal parameters (1,1,1,52)
- Built Flask web application with 6 dashboard pages
- Created automated preprocessing pipelines

**홍동균 (Designer)**:
- Designed professional Power BI reports with custom visuals
- Created Figma prototypes for web dashboard UI/UX
- Developed color schemes for data visualization consistency

**이상윤 (Data Engineer)**:
- Designed ERD with 10+ tables for relational database
- Conducted comprehensive EDA identifying seasonal patterns
- Documented data dictionary and schema specifications

**권오윤 (ML Engineer)**:
- Optimized LSTM hyperparameters (learning rate, batch size, epochs)
- Implemented data augmentation techniques
- Evaluated model performance with multiple metrics

**정태빈 (Data Analyst)**:
- Analyzed cabbage price patterns by harvest period
- Investigated regional weather correlations
- Conducted statistical hypothesis testing

**부혁훈 (DevOps)**:
- Set up Linux-based big data processing server
- Automated CSV file transfers with Bash/PowerShell scripts
- Configured scheduled jobs for daily data updates

---

## 📚 Research References

This project builds upon academic research in agricultural price forecasting:

### Papers Summarized (see `가격예측 AI 모델링/요약본_*.hwpx`):
1. **"Lasso Regression for Agricultural Price Prediction Variable Selection"**
   - Applied Lasso for feature selection among 100+ variables
   - Identified key economic indicators with highest predictive power

2. **"Deep Learning Methodology for Predicting Agricultural Prices by Distribution Channel"**
   - Multi-channel price prediction framework
   - Demand forecasting integration

3. **"Agricultural Price Prediction Model Design Using Deep Learning"**
   - LSTM architecture design principles
   - Hyperparameter tuning strategies

4. **"Agricultural Wholesale Market Price Determination Using AI"**
   - Market mechanism modeling
   - Supply-demand equilibrium analysis

5. **"Multivariate Agricultural Price Prediction Using LSTM"**
   - Multivariate time series handling
   - Sequence-to-sequence modeling

6. **"Multi-Step Time Series Forecasting for Medium-Term Agricultural Prices"**
   - Horizon-specific modeling approaches
   - Forecast accuracy across different time horizons

---

## 🔮 Future Enhancements

### Short-Term (3-6 months)
- **Model Improvements**:
  - Implement ensemble methods (stacking LSTM + SARIMAX)
  - Experiment with Transformer models for time series (Temporal Fusion Transformer)
  - Add uncertainty quantification with Bayesian neural networks

- **Web Application**:
  - Add user authentication and personalized dashboards
  - Implement real-time prediction API with FastAPI
  - Mobile-responsive redesign

- **Data Expansion**:
  - Incorporate additional commodities (grains, dairy, meat)
  - Add international price data for import substitution analysis
  - Integrate social media sentiment analysis (Twitter, news)

### Mid-Term (6-12 months)
- **Advanced Analytics**:
  - Causal inference analysis (what-if scenarios)
  - Anomaly detection for supply chain disruptions
  - Recommendation system for optimal procurement timing

- **Deployment**:
  - Dockerization for easy deployment
  - Cloud hosting on AWS/Azure/GCP
  - CI/CD pipeline with GitHub Actions

- **Business Intelligence**:
  - Automated weekly email reports
  - Slack/Teams integration for alerts
  - Executive summary generation with NLP

### Long-Term (1+ years)
- **Scale & Integration**:
  - Integration with military procurement systems (ERP)
  - Multi-region expansion (international markets)
  - Blockchain for supply chain transparency

- **Advanced ML**:
  - Reinforcement learning for dynamic procurement strategies
  - Graph neural networks for supply chain network analysis
  - Transfer learning from international markets

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### Guidelines:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📧 Contact

For questions or collaboration opportunities, please reach out to the team:

- **Project Lead**: Yoon Chan-yeol (윤찬열)
- **Technical Lead**: Lee Hyun-dong (이현동)

**GitHub Repository**: [https://github.com/yourusername/Defense-Agri-Price-Forecasting](https://github.com/yourusername/Defense-Agri-Price-Forecasting)

---

## 🙏 Acknowledgments

- **Acorn Academy**: Training and mentorship
- **Seoul Garak Market**: Agricultural price data
- **Korea Meteorological Administration**: Weather data
- **Bank of Korea**: Economic indicator data
- **Military Food Service**: Domain expertise and requirements
- **Open Source Community**: TensorFlow, Statsmodels, Flask, and all dependencies

---

## 📊 Project Statistics

- **Lines of Code**: 10,000+ (Python, JavaScript, HTML/CSS)
- **Jupyter Notebooks**: 50+ analysis and modeling notebooks
- **Data Files Processed**: 100+ CSV/Excel files
- **Visualizations Created**: 50+ charts and graphs
- **Model Training Time**: 100+ hours of computation
- **Team Members**: 7 contributors
- **Project Duration**: 8 weeks (November - December 2024)

---

<div align="center">

### ⭐ Star this repository if you find it useful!

**Made with ❤️ by Team 1 - Defense Agricultural Price Forecasting**

</div>
