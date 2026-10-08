# Defense Agricultural Price Forecasting

![Python](https://img.shields.io/badge/Python-3.x-blue.svg) ![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-orange.svg) ![statsmodels](https://img.shields.io/badge/statsmodels-SARIMAX-green.svg) ![Flask](https://img.shields.io/badge/Flask-dashboard-lightgrey.svg)

Forecasting Korean agricultural retail prices for military food procurement: per-commodity LSTM models and 52-week seasonal-ARIMA forecasts trained on 11 years of Seoul Garak Market data, served through a Flask dashboard with embedded Power BI reports. An 8-week team project (Nov–Dec 2024) in which I did most of the work.

## Verified results

| What | Number | Verified where |
| --- | --- | --- |
| Dataset | **65,120 daily retail-price rows, 17 commodities, 2014-01-02 → 2024-12-05** | `EDA(탐색적 데이터 분석)/df.pkl` (tracked via Git LFS) |
| LSTM models | **16 per-commodity Keras models with saved training logs** (a 17th block is labelled cabbage but loads the potato series; see below); best-epoch MAE **0.022–0.094 on min-max-scaled prices** (median 0.044) — roughly **2–9% of each commodity's 11-year price range**, measured on the last 20% of each series, which also chose the stopping epoch | training logs preserved in `가격예측 AI 모델링/latest_lstm_prediction.ipynb` |
| Baseline | **None computed.** No last-value or seasonal-naive forecast was scored on the same split, so these MAEs do not show the LSTM beats a simple rule | — |
| Long-horizon forecast | **52-week-ahead weekly forecasts per commodity**, seasonal ARIMA (statsmodels `SARIMAX`, order (5,1,0), seasonal (1,1,1,52)); fitted on the full series and **not scored against held-out data** | `가격예측 AI 모델링/SARIMAX_이현동.ipynb` and the batch pipeline in `국방 물자 조달 전처리 및 SARIMAX 코드py/` |
| Dashboard | **6 Flask pages, all serving HTTP 200** (verified 2026-07) | `acorn_web/app.py` |
| Scope | 44 Jupyter notebooks, 52 Excel source files, 66 committed PNGs — **41 analysis charts & data infographics** plus 25 non-chart images (Figma UI mockups, code screenshots, nav buttons, diagrams) | `git ls-files`; PNG breakdown by visual inspection (2026-07) |

> **Metrics honesty note**: an earlier version of this README quoted test MAPE, R², and accuracy percentages whose evaluation runs are not preserved in this repository. This README reports only numbers that can be re-derived from the committed notebook outputs and data files.

![LSTM test-set predictions vs actual prices (from latest_lstm_prediction.ipynb)](가격예측%20AI%20모델링/actual_predict_graph.png)

## Quick start

```bash
git clone https://github.com/CY-HYUN/Defense-Agri-Price-Forecasting-main.git
cd Defense-Agri-Price-Forecasting-main

pip install -r requirements.txt   # flask, pandas, matplotlib

cd acorn_web
python app.py
# open http://127.0.0.1:5000
```

Verified routes: `/` (main dashboard), `/sub_chart_economy`, `/sub_chart_logistics`, `/sub_chart_weather`, `/sub_chart_oil`, `/menu_management` — all return HTTP 200.

Known limitations of the local run (honest notes):

- The charts inside each page are **Power BI embeds** that require access to the original workspace; without it you see the page shells only.
- CSS/JS are referenced with relative paths that Flask does not serve as static files, so pages render unstyled via Flask. The committed Figma design below shows the intended UI.
- The `/EDA` route returns 500 (`EDA.html` template was never committed) — known dead code.
- **Data note**: all `*.csv` files are excluded by `.gitignore` (the daily retail-price extracts are not published). Excel sources under `DB/` and `.pkl` intermediates (Git LFS) are in the repo. The modeling notebooks were run on Colab and a class machine and contain hardcoded paths (`/content/...`, `C:/Users/acorn/...`), so they are an archival record of the analysis — browsable on GitHub, not re-runnable end-to-end from a fresh clone.

### Browsing the analysis without running it

The notebooks keep their cell outputs (training logs, plots), so the full analysis can be read directly on GitHub without any setup:

- `가격예측 AI 모델링/latest_lstm_prediction.ipynb` — 297 cells, one training block per commodity with preserved loss/MAE logs
- `가격예측 AI 모델링/국방 물자 조달 전처리 및 SARIMAX 코드py/한번에 모든 데이터 예측 저장 코드.ipynb` — batch 52-week forecasting loop
- `EDA(탐색적 데이터 분석)/` — analysis notebooks plus their exported PNG charts committed side by side

To open them locally instead: `pip install jupyter` and launch `jupyter notebook` from the repo root (the notebooks read data from paths that no longer exist, so re-execution requires repointing them at `DB/` and supplying the private CSVs).

![Dashboard design (Figma)](PowerBI/Project_Figma_image/Main%20Dashboard.png)

## Architecture

```text
DB/ (raw sources)                 전처리 부분/ + EDA(탐색적 데이터 분석)/        가격예측 AI 모델링/
  daily retail prices (Garak)  →    cleaning, gap handling, merging,      →    LSTM per commodity (Keras)
  weather (KMA stations)            weekly resampling, EDA charts              seasonal ARIMA, 52-week forecast
  GDP / BOK rate / FX / diesel                                                       ↓
  minimum wage / shipments                                              saved per-commodity prediction CSVs
                                                                                     ↓
                                                          acorn_web/ Flask dashboard + PowerBI/ reports
```

| Folder | Contents |
| --- | --- |
| `DB/` | Source data: daily price extracts, cultivation-region weather, GDP, interest/FX rates, diesel prices, minimum wage, Garak Market shipment volumes |
| `전처리 부분/`, `EDA(탐색적 데이터 분석)/`, `EDA부분/` | Preprocessing and EDA notebooks + exported charts |
| `가격예측 AI 모델링/` | LSTM and seasonal-ARIMA modeling notebooks, result graphs, paper summaries |
| `acorn_web/` | Flask app (`app.py`) + HTML/CSS/JS for 6 dashboard pages |
| `PowerBI/`, `ERD Table/` | BI report assets, Figma design images, ERD |
| `Server/` | CSV transfer scripts (bash + PowerShell) |
| `docs/DETAILS.md` | Full methodology detail (data catalog, model configs, per-commodity results) |

## Models (as implemented in the committed notebooks)

**LSTM (Keras/TensorFlow)** — one model per commodity, univariate price series:

- MinMax scaling fitted on each full series, then a chronological 80/20 train/test split; input window tuned per commodity (10-30 days)
- 6 stacked LSTM layers (200-100-50-50-100-200 units, tanh), dropout 0.2 and L2(0.01) on every layer, Dense(1) output
- Adam with a per-commodity tuned learning rate (0.0005–0.0029), custom RMSE loss, EarlyStopping (patience 10, best-weights restore) monitored on the 20% test split (`validation_data=(X_test, y_test)`), batch 50, ≤100 epochs, seed 42
- Recursive multi-step forecasting over a 30-day horizon (`future_days = 30` in every block; some plot titles say 365 days)

![Recursive LSTM forecast beyond the test window (red) after actual prices (blue)](가격예측%20AI%20모델링/LSTM_predict_graph.png)

**Seasonal ARIMA (statsmodels `SARIMAX` class)** — one model per commodity:

- Daily prices resampled to weekly means; runs with ≥9-day data gaps filtered
- Order (5,1,0), seasonal order (1,1,1,52), with one differencing step in each part; `adfuller` (ADF stationarity test) is imported but not called in the committed notebooks
- 52-week-ahead forecasts saved per commodity (napa cabbage, cabbage, carrot, cucumber, radish, garlic, onion, pepper, potato, rice, spinach, green onion + combined file)
- No exogenous regressors in the final batch run — the economic variables below were used in EDA and the dashboards, not as model inputs

### Per-commodity LSTM validation results (preserved training logs)

Best epoch by validation loss, where the "validation" set is the 20% test split; because that split also picked the epoch, these are optimistic estimates, not untouched test scores. Units are min-max-scaled prices, so an MAE of 0.05 ≈ 5% of that commodity's 2014–2024 price range. Full configuration table (learning rates, RMSE) in [docs/DETAILS.md](docs/DETAILS.md).

| Commodity | Best val MAE (scaled) | Commodity | Best val MAE (scaled) |
| --- | --- | --- | --- |
| Dried red pepper (건고추) | 0.022 | Onion (양파) | 0.044 |
| Spinach (시금치) | 0.033 | Radish (무) | 0.044 |
| Green onion (대파) | 0.033 | Peeled garlic (깐마늘) | 0.055 |
| Napa cabbage (배추) | 0.034 | Carrot (당근) | 0.074 |
| Potato (감자) | 0.036 | Enoki mushroom (팽이) | 0.083 |
| Zucchini (애호박) | 0.037 | Cucumber (오이) | 0.086 |
| Rice (쌀) | 0.041 | King oyster mushroom (새송이버섯) | 0.092 |
| Sweet potato (고구마) | 0.044 | Bean sprouts (콩나물) | 0.094 |

(The notebook's 17th block is headed cabbage/양배추, but it filters the price file on 감자 (potato) so its model is fitted on the potato series; its fit-cell output was not preserved. Cabbage has no LSTM result in this repository.)

## Data

11 years (2014–2024) of Korean market and macro data, integrated in `DB/` and merged into a single analysis frame:

| Category | Source |
| --- | --- |
| Daily retail prices (17 commodities) | Seoul Garak Market price service |
| Cultivation-region weather | Korea Meteorological Administration |
| GDP, interest rates, USD/KRW | Bank of Korea / national statistics portals |
| Diesel prices | Opinet (Korea National Oil Corp.) |
| Minimum wage | Ministry of Employment and Labor |
| Market shipment volumes, imports | Garak Market / customs statistics |
| Military menu reference | Army unit menu document (`부대식단.pdf`) |

## Key findings (from preserved outputs)

- Staple field crops (dried red pepper, spinach, green onion, napa cabbage, potato: 0.022–0.036) had lower scaled MAE than short-cycle/sprout items (cucumber, enoki, king oyster mushroom, bean sprouts: 0.083–0.094). Without a naive baseline per commodity, this may reflect how smooth each price series is rather than how well the model learned it.
- The EDA notebooks map seasonal price patterns, price–shipment-volume relationships, commodity price correlations, and cultivation-region weather vs price (e.g. `감자가격과_가장관련있는_지역찾기.ipynb` — finding the weather station most correlated with potato prices); 41 exported analysis charts are committed.
- The merged analysis frame (`df_eda.pkl`) aligns daily prices with GDP, Bank of Korea base rate, minimum wage, diesel price, USD/KRW, and Garak Market shipment volume.

## Limitations

- **No baseline.** Neither model was compared with a last-value or seasonal-naive forecast.
- **Test split used for early stopping.** The LSTM's stopping epoch was chosen on the same 20% split its MAE is reported on, so the MAEs are optimistic; there is no separate validation split.
- **Scaling before the split.** The MinMax scaler sees each full series, including the test period, before the 80/20 split — a small leak of the test range into training inputs.
- **Seasonal ARIMA is unscored.** The 52-week forecasts were fitted on all data and never compared with held-out weeks.
- **One split, one seed.** Each LSTM was trained once (seed 42) on one chronological split; run-to-run variation is not measured.

## Tech stack

Python (pandas, NumPy, scikit-learn, statsmodels, TensorFlow/Keras, matplotlib) · Flask + HTML/CSS/JS · Power BI · Git LFS for data intermediates.

## Project context

- **My role.** An 8-week team capstone (November–December 2024) in which I did most of the work end to end: the data pipeline from the Garak Market source, preprocessing and EDA, the per-commodity LSTM forecasting models, and the Flask dashboard that serves the forecasts.
- This repository is the project archive. Some notebooks keep the file names they had in the team's shared drive, and the code is kept as it ran in December 2024 rather than retrofitted.
- Model choices (LSTM + seasonal ARIMA) followed a review of six Korean papers on agricultural price forecasting; the summaries are committed alongside the modeling notebooks.

## More detail

- [docs/DETAILS.md](docs/DETAILS.md) — data source catalog, preprocessing steps, full model configuration, per-commodity training results, dashboard internals
- `자료조사/`, `가격예측 AI 모델링/요약본_*.hwpx` — Korean-language literature summaries (6 papers on agricultural price forecasting) that guided model selection
