# Defense Agricultural Price Forecasting

![Python](https://img.shields.io/badge/Python-3.x-blue.svg) ![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-orange.svg) ![statsmodels](https://img.shields.io/badge/statsmodels-SARIMAX-green.svg) ![Flask](https://img.shields.io/badge/Flask-dashboard-lightgrey.svg)

Forecasting Korean agricultural retail prices for military food procurement: per-commodity LSTM models and 52-week seasonal-ARIMA forecasts trained on 11 years of Seoul Garak Market data, served through a Flask dashboard with embedded Power BI reports. 8-week, 7-person team project (Nov–Dec 2024).

## Verified results

| What | Number | Verified where |
| --- | --- | --- |
| Dataset | **65,120 daily retail-price rows, 17 commodities, 2014-01-02 → 2024-12-05** | `EDA(탐색적 데이터 분석)/df.pkl` (tracked via Git LFS) |
| LSTM models | **17 per-commodity Keras models**; best-epoch validation MAE **0.022–0.094 on min-max-scaled prices** (median 0.044) — roughly **2–9% of each commodity's 11-year price range** | training logs preserved in `가격예측 AI 모델링/latest_lstm_prediction.ipynb` (16 of 17 blocks have saved outputs) |
| Long-horizon forecast | **52-week-ahead weekly forecasts per commodity**, seasonal ARIMA (statsmodels `SARIMAX`, order (5,1,0), seasonal (1,1,1,52)) | `가격예측 AI 모델링/SARIMAX_이현동.ipynb` and the batch pipeline in `국방 물자 조달 전처리 및 SARIMAX 코드py/` |
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

- MinMax scaling, chronological 80/20 train/test split, input window tuned per commodity (10-30 days)
- 6 stacked LSTM layers (200-100-50-50-100-200 units, tanh), dropout 0.2 and L2(0.01) on every layer, Dense(1) output
- Adam with a per-commodity tuned learning rate (0.0005–0.0029), custom RMSE loss, EarlyStopping (patience 10, best-weights restore), batch 50, ≤100 epochs, seed 42
- Recursive multi-step forecasting (30- and 365-day horizons)

![Recursive LSTM forecast beyond the test window (red) after actual prices (blue)](가격예측%20AI%20모델링/LSTM_predict_graph.png)

**Seasonal ARIMA (statsmodels `SARIMAX` class)** — one model per commodity:

- Daily prices resampled to weekly means; runs with ≥9-day data gaps filtered
- Order (5,1,0), seasonal order (1,1,1,52); ADF stationarity checks
- 52-week-ahead forecasts saved per commodity (napa cabbage, cabbage, carrot, cucumber, radish, garlic, onion, pepper, potato, rice, spinach, green onion + combined file)
- No exogenous regressors in the final batch run — the economic variables below were used in EDA and the dashboards, not as model inputs

### Per-commodity LSTM validation results (preserved training logs)

Best epoch by validation loss; units are min-max-scaled prices, so an MAE of 0.05 ≈ 5% of that commodity's 2014–2024 price range. Full configuration table (learning rates, RMSE) in [docs/DETAILS.md](docs/DETAILS.md).

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

(17th commodity, cabbage/양배추, was modeled but its fit-cell output was not preserved in the notebook.)

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

- Staple field crops (dried red pepper, spinach, green onion, napa cabbage, potato: 0.022–0.036) validated noticeably better than short-cycle/sprout items (cucumber, enoki, king oyster mushroom, bean sprouts: 0.083–0.094).
- The EDA notebooks map seasonal price patterns, price–shipment-volume relationships, commodity price correlations, and cultivation-region weather vs price (e.g. `감자가격과_가장관련있는_지역찾기.ipynb` — finding the weather station most correlated with potato prices); 41 exported analysis charts are committed.
- The merged analysis frame (`df_eda.pkl`) aligns daily prices with GDP, Bank of Korea base rate, minimum wage, diesel price, USD/KRW, and Garak Market shipment volume.

## Tech stack

Python (pandas, NumPy, scikit-learn, statsmodels, TensorFlow/Keras, matplotlib) · Flask + HTML/CSS/JS · Power BI · Git LFS for data intermediates.

## Project context

- 8-week team capstone (November–December 2024), 7 members covering data collection, preprocessing, EDA, modeling, dashboard, and BI reporting; weekly plans and work logs are preserved in `기획서와주간업무일지/` (Korean).
- This repository is the project archive — teammate-authored notebooks are preserved under their original names, and the code is kept as it ran in December 2024 rather than retrofitted.
- Model choices (LSTM + seasonal ARIMA) followed a review of six Korean papers on agricultural price forecasting; the summaries are committed alongside the modeling notebooks.

## More detail

- [docs/DETAILS.md](docs/DETAILS.md) — data source catalog, preprocessing steps, full model configuration, per-commodity training results, dashboard internals
- `자료조사/`, `가격예측 AI 모델링/요약본_*.hwpx` — Korean-language literature summaries (6 papers on agricultural price forecasting) that guided model selection
