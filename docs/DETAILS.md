# Methodology Details

Companion document to the [main README](../README.md). Everything here is grounded in the committed notebooks, data files, and app code; where an earlier README version quoted numbers that cannot be re-derived from the repo, they have been removed.

## 1. Data sources

All sources cover roughly 2014–2024 and live under `DB/` (Excel files tracked in git; `.pkl` intermediates via Git LFS; `.csv` extracts excluded by `.gitignore`).

| Category | Source | Repo location |
| --- | --- | --- |
| Daily retail prices, 17 commodities | Seoul Garak Market price service | `DB/식품예측 독립변수 데이터/일별소매가/` (CSV extracts not published; merged frame preserved as `EDA(탐색적 데이터 분석)/df.pkl`) |
| Cultivation-region weather | Korea Meteorological Administration | `DB/재배지날씨/`, `DB/식품예측 독립변수 데이터/날씨데이터*.csv` |
| GDP (quarterly/annual) | Bank of Korea / index portal | `DB/식품예측 독립변수 데이터/GDP_*.xlsx`, `경제활동별 GDP 및 GNI(...)_2010Q1_2024Q3.xlsx` |
| Interest rates (base/market/lending) | Bank of Korea | `기준금리_월별_지표누리.xlsx`, `시장금리_월별_2014-01_2024-09.xlsx`, `예금은행_대출금리_신규취급액_기준__2015-01_2024.xlsx` |
| USD/KRW exchange rate | Bank of Korea | `환율_일별_한국은행.xlsx`, `df_환율.xlsx` |
| Diesel/fuel prices | Opinet (Korea National Oil Corp.) | `경유_일별_주유소_제품별_평균판매가격.xls` and related files |
| Minimum wage (annual) | Ministry of Employment and Labor | `고용노동부_연도별 최저임금_20230804.csv`, `최저임금_연별_고용노동부.csv` |
| Garak Market shipment volumes | Seoul Garak Market | `DB/식품예측 독립변수 데이터/출하량_가락/` |
| Import volumes | Customs/aT statistics | `DB/식품예측 독립변수 데이터/수입량/` |
| Military menu reference | Army unit menu document | `부대식단.pdf` |
| CPI weights | Statistics Korea | `물가지수_가중치cpi_dl_2020.xlsx` |

### Merged analysis frames (verified contents)

- `EDA(탐색적 데이터 분석)/df.pkl` — 65,120 rows × 8 columns: date, commodity name, average price, unit price, Garak shipment volume, year-month, year, month. Dates 2014-01-02 → 2024-12-05, 17 commodities.
- `EDA(탐색적 데이터 분석)/df_eda.pkl` — 65,120 rows × 13 columns: the above plus GDP, BOK base rate, minimum hourly wage, diesel price, USD/KRW.

## 2. Preprocessing (as done in the committed notebooks)

Notebooks under `전처리 부분/`, `EDA(탐색적 데이터 분석)/전처리/`, and `가격예측 AI 모델링/국방 물자 조달 전처리 및 SARIMAX 코드py/`:

- Comma-separated numeric strings converted to numeric (`str.replace(',','')`), cp949/UTF-8 encoding handling
- Date parsing and indexing; per-commodity filtering from the combined price file
- Missing values: forward-fill then back-fill on price series; sequences with 9+ consecutive missing days identified and dropped before SARIMA fitting
- Lower-frequency economic variables (quarterly GDP, monthly rates, annual wage) aligned to daily frequency for the merged EDA frame
- Weekly resampling (`resample('W').mean()`) for the seasonal-ARIMA models
- MinMax scaling (0–1) per commodity for LSTM input

## 3. LSTM configuration (from `가격예측 AI 모델링/latest_lstm_prediction.ipynb`)

One univariate model per commodity (average daily price only):

- Chronological 80/20 train/test split; sliding-window supervised framing with `time_step` tuned per commodity (10-30)
- Architecture: `LSTM(200) → LSTM(100) → LSTM(50) → LSTM(50) → LSTM(100) → LSTM(200) → Dense(1)`, all LSTM layers `tanh` with `kernel_regularizer=l2(0.01)` and `Dropout(0.2)` after each
- Loss: custom RMSE on scaled values; metric: MAE; optimizer: Adam with per-commodity learning rate; `EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)`; batch size 50; up to 100 epochs; seeds fixed (42)
- Forecasting: recursive one-step-ahead rollout for 30- and 365-day horizons
- Environment: Google Colab GPU (`/content/...` data paths preserved in the notebook)

### Per-commodity training results (best epoch by validation loss, min-max-scaled units)

Extracted from the training logs saved in the notebook outputs. Validation loss is RMSE on scaled prices; MAE is on the same 0–1 scale, so 0.05 ≈ 5% of that commodity's 2014–2024 price range.

| Commodity | Learning rate | Best val RMSE | Best val MAE |
| --- | --- | --- | --- |
| Dried red pepper (건고추) | 0.0021 | 0.038 | 0.022 |
| Spinach (시금치) | 0.0020 | 0.050 | 0.033 |
| Green onion (대파) | 0.0024 | 0.052 | 0.033 |
| Napa cabbage (배추) | 0.0027 | 0.056 | 0.034 |
| Potato (감자) | 0.0028 | 0.056 | 0.036 |
| Zucchini (애호박) | 0.0015 | 0.059 | 0.037 |
| Rice (쌀) | 0.0019 | 0.060 | 0.041 |
| Sweet potato (고구마) | 0.0015 | 0.059 | 0.044 |
| Onion (양파) | 0.0013 | 0.065 | 0.044 |
| Radish (무) | 0.0029 | 0.079 | 0.044 |
| Peeled garlic (깐마늘) | 0.0011 | 0.078 | 0.055 |
| Carrot (당근) | 0.00092 | 0.091 | 0.074 |
| Enoki mushroom (팽이) | 0.0010 | 0.115 | 0.083 |
| Cucumber (오이) | 0.0009 | 0.109 | 0.086 |
| King oyster mushroom (새송이버섯) | 0.0005 | 0.119 | 0.092 |
| Bean sprouts (콩나물) | 0.0009 | 0.138 | 0.094 |
| Cabbage (양배추) | 0.0028 | — (fit-cell output not preserved) | — |

Median best val MAE across the 16 logged models: **0.044**. No test-set MAPE/R² in currency units is preserved in the repo, so none is claimed.

Result graphs committed alongside the notebook: `actual_predict_graph.png` (test-set actual vs predicted), `LSTM_predict_graph.png`, `loss_graph.png`, `traintestdatasplit.png`.

## 4. Seasonal-ARIMA pipeline (from `SARIMAX_이현동.ipynb` and `국방 물자 조달 전처리 및 SARIMAX 코드py/한번에 모든 데이터 예측 저장 코드.ipynb`)

- Daily average prices → weekly means; missing weeks detected and excluded
- Model: statsmodels `SARIMAX(weekly_prices, order=(5, 1, 0), seasonal_order=(1, 1, 1, 52))` — seasonal ARIMA with a 52-week cycle; **no exogenous regressors passed** in the final batch run (the class name is SARIMAX, the fitted models are SARIMA)
- ADF stationarity testing before differencing decisions
- Batch loop over the per-commodity files produced 52-week-ahead forecasts, saved as `<commodity>_predictions.csv` under `DB/예측한 값 저장/` (napa cabbage, cabbage, carrot, cucumber, radish, garlic, onion, pepper, potato, rice, spinach, green onion) plus combined past+predicted files — note these CSVs are excluded from the public repo by `.gitignore`
- Historical + forecast series were exported for Power BI (`EDA(탐색적 데이터 분석)/PowerBI에서참고할csv파일만들기.ipynb`)

## 5. EDA notebook index (all committed)

`EDA(탐색적 데이터 분석)/`:

- `품목별탐색적데이터분석.ipynb` — per-commodity distributions and trends
- `품목별 평균가격 변화 상관도.ipynb` — commodity price correlation analysis
- `계절별저렴한품목_월별편차순위.ipynb` — cheapest commodities by season, monthly deviation ranking
- `감자가격과_가장관련있는_지역찾기.ipynb` — weather station most correlated with potato prices
- `배추_수확시기별분석_정태빈.ipynb`, `배추_원본_재배지날씨_재배수확기간.ipynb` — cabbage harvest-period vs weather analysis
- `생산량과날씨EDA.ipynb`, `날씨데이터_eda.ipynb` — production/weather exploration
- `재배기간_피처링_시도.ipynb` — cultivation-period feature engineering attempt
- `전처리/` — 8 notebooks aligning GDP, rates, wages, fuel prices to daily frequency

`EDA부분/EDA부분/`: `price_Shipment_graph.ipynb`, `price_temp_corr.ipynb`, `계절별저렴한품목과편차순위.ipynb`, `농산품연도별월평균가격.ipynb`.

Exported PNG charts are committed next to these notebooks (price vs shipment volume vs harvest period, scaled economic variables, correlation heatmaps, etc.).

## 6. Flask dashboard internals (`acorn_web/`)

- `app.py`: Flask app with `template_folder='main'` (must be launched from `acorn_web/`), 7 routes; `debug=True` dev server
- Pages: main dashboard (LSTM + SARIMAX charts), economy, logistics, weather, oil, menu management — each an HTML template with matching JS/CSS under `main/scripts/` and `main/styles/`
- Charts are embedded Power BI reports (`app.powerbi.com/reportEmbed?...` URLs set by the page scripts); they require access to the original Power BI workspace
- Known issues (kept as-is, project is archived):
  - `/EDA` raises `TemplateNotFound: EDA.html` (template never committed) → HTTP 500
  - Static assets are referenced relatively (`styles/...`, `scripts/...`) but Flask has no static route for them, so pages served by Flask render unstyled; opening the HTML files directly from disk resolves the assets
- Design references: `PowerBI/Project_Figma_image/` (Figma mockups of every page), `ERD Table/` (database design)

## 7. Reproducibility summary

| Component | Re-runnable from a fresh clone? |
| --- | --- |
| Flask dashboard | Yes — `pip install -r requirements.txt`, `cd acorn_web && python app.py` (page shells; Power BI content needs workspace access) |
| Merged data frames | Yes — `df.pkl` / `df_eda.pkl` load with pandas (Git LFS required) |
| LSTM / SARIMA notebooks | No — they read daily-price CSVs that are not published and use Colab/class-machine paths; kept as an archival record with outputs preserved |
| Power BI reports | No — workspace-bound |

## 8. Literature reviewed (Korean summaries committed)

Six paper summaries under `가격예측 AI 모델링/요약본_*.hwpx` and `자료조사/`, covering LSTM-based agricultural price prediction, variable selection via Lasso, multi-stage time-series models, and wholesale-market price determination — these guided the LSTM + seasonal-ARIMA model choice.
