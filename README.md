# 🇱🇰 Sri Lanka Inflation Prediction System

**Event-Driven Inflation Forecasting with SARIMAX + XGBoost + LLM Event Analysis**

## Overview

This system predicts Sri Lanka inflation using:
1. **SARIMAX** - Seasonal autoregressive forecasting baseline
2. **XGBoost** - Residual correction using economic indicators
3. **LLM Classification** - Event categorization from uploaded documents (OpenAI gpt-5.4-mini-2026-03-17)
4. **Impact Adjustment** - 4 weighting methods (OLS, MVO, PCA, Entropy) + Hybrid ensemble

## Quick Start

### 1. Install Dependencies

```bash
cd product
pip install -r requirements.txt
```

### 2. Setup OpenAI API Key (Optional, for LLM Features)

```bash
# Option A: Set environment variable
export OPENAI_API_KEY="your-api-key-here"

# Option B: Create .env file in parent directory
echo "OPENAI_API_KEY=your-api-key-here" > ../.env
```

### 3. Prepare Data Files

Place these files in the parent directory (`/home/suchitra/InflationPredNew/`):
- `SLData.csv` - Historical inflation & economic indicators
- `event_categorization_summary_combined.csv` - Pre-classified event summaries

### 4. Run the App

```bash
streamlit run app.py
```

App opens at: `http://localhost:8501`

---

## Module Overview

### `data_loader.py` (145 lines)
**Purpose:** Load and preprocess data

**Key Functions:**
- `load_inflation_data()` - Load SLData.csv
- `load_event_summary()` - Load event categorization CSV
- `engineer_features()` - Create 40+ engineered features:
  - Log transforms (Imports, Exports, Broad Money)
  - Differencing (Exchange Rate, Trade Balance)
  - Broad Money Growth (monthly %)
  - Lag features (1-18 months for pass-through effects)
  - PCA (2 components for collinear vars)
  - Inflation Spike Indicator (detect crisis periods)
- `load_and_preprocess()` - All-in-one pipeline

**Usage:**
```python
from data_loader import load_and_preprocess
df_fe, events_df = load_and_preprocess("SLData.csv", "event_summary.csv")
```

---

### `prediction_pipeline.py` (270+ lines)
**Purpose:** Generate rolling-window forecasts

**Key Functions:**
- `forecast_inflation()` - SARIMAX(1,0,0) + XGBoost residual booster
  - Rolling window: 12 months (configurable)
  - SARIMAX seasonal order: (0,0,1,6)
  - XGBoost: 200 estimators, max_depth=3
  - Returns: Date | Forecast_SARIMAX | Forecast_Boosted | Lower_CI | Upper_CI | XGB_corr
- `get_forecast_diagnostics()` - MAE, RMSE, MAPE, Bias metrics

**Returns DataFrame:**
```
Date                 2022-01-01
Forecast_SARIMAX     3.45
Forecast_Boosted     3.72
Lower_CI             2.15
Upper_CI             4.89
XGB_corr             0.27
```

---

### `llm_adjustment.py` (340+ lines)
**Purpose:** Adjust forecasts using LLM-classified events

**Key Functions:**
- `prepare_impact_factors()` - Extract event categories, create rolling effects
- `calculate_ols_weights()` - OLS fit to residuals
- `calculate_mvo_weights()` - Minimum Variance Optimization
- `calculate_pca_weights()` - Principal Component Analysis weighting
- `calculate_entropy_weights()` - Information-theoretic weighting
- `adjust_forecast_with_llm_events()` - Apply all methods + Hybrid (average of 4)
- `evaluate_adjustments()` - Calculate MAE, RMSE, MAPE, Directional Accuracy

**Returns DataFrame with new columns:**
```
Forecast_Boosted               3.72
Impact_Adjustment_ols          -0.15
Adjusted_Inflation_OLS         3.57
Adjusted_Inflation_MVO         3.62
Adjusted_Inflation_PCA         3.59
Adjusted_Inflation_Entropy     3.61
Adjusted_Inflation_Hybrid      3.60  ← Recommended (average of 4 methods)
```

---

### `document_processor.py` (430+ lines)
**Purpose:** Process uploaded documents and classify events

**Key Functions:**
- `parse_filename_to_date()` - Extract YYYY-MM from filename (e.g., "2025-10_article.pdf")
- `extract_text_from_upload()` - Read PDF/TXT/MD files
- `categorize_text_with_llm()` - Call OpenAI to classify events
- `process_uploaded_file()` - Single file processing pipeline
- `process_batch_uploads()` - Batch upload with progress tracking
- `EventAggregation` - Accumulate and aggregate events by month/category

**Supported Event Categories:**
1. Monetary Policy
2. Fiscal Policy
3. External and Global Shocks
4. Supply and Demand Shocks

**Intensity Scale:** -1.0 (deflationary) to +1.0 (inflationary)

---

### `app.py` (500+ lines)
**Purpose:** Main Streamlit UI with 4 tabs

**Sidebar:**
- 📅 Date range picker (start/end forecast dates)
- 🔧 Rolling window size (6-120 months)
- 📊 Data status & load button
- 🤖 LLM toggle (enable/disable OpenAI calls)

**Tab 1: Upload & Classify**
- Batch file uploader (PDF, TXT, MD)
- Automatic date parsing from filename
- LLM event classification (optional)
- Tables: Parsed dates + Extracted events (with intensity scores)

**Tab 2: Prediction Results**
- 📊 Generate forecasts button (SARIMAX + XGBoost + LLM adjustment)
- 📋 Results table (Date | Actual | SARIMAX | SARIMAX+XGB | All LLM variants | Hybrid)
- 📈 Charts:
  - Forecast comparison (Actual vs SARIMAX vs Hybrid)
  - Confidence intervals (95% CI)
  - Residuals (forecast errors)
- 📊 Performance metrics (MAE, RMSE, MAPE, Directional Accuracy)

**Tab 3: Event Summary**
- 📊 Bar charts:
  - Average intensity by event category
  - Event occurrence count
- 📋 Summary table (Category | Mean Intensity | Count)

**Tab 4: Export**
- 📥 Download forecast CSV
- 📥 Download events CSV

---

## Workflow Example

### 1. User Upload & Classification

```
User Action:
├─ Select date range: 2023-01-01 to 2025-09-01
├─ Upload 3 documents (auto-parsed):
│  ├─ 2024-06_report.pdf → Year: 2024, Month: 6 ✓
│  ├─ 2024-12_article.txt → Year: 2024, Month: 12 ✓
│  └─ invalid_file.md → Year: None, Status: ⚠️
├─ Enable LLM classification
└─ Click "Analyze Documents"

Result:
├─ Event Extraction (via LLM or uploaded summaries)
│  ├─ 2024-06: Monetary Policy (intensity: +0.45), External Shock (intensity: -0.20)
│  └─ 2024-12: Fiscal Policy (intensity: +0.30)
├─ Aggregation by category
└─ Save to session_state.custom_events_df
```

### 2. Forecast Generation

```
User Action: Click "Generate Forecasts"

Pipeline:
├─ Load historical data (SLData.csv)
├─ Engineer 40+ features (log, diff, lag, PCA, spike indicators)
├─ SARIMAX(1,0,0) rolling forecast
│  └─ 12-month window, seasonal order (0,0,1,6)
├─ XGBoost residual correction
├─ LLM Impact Adjustment
│  ├─ OLS weights (fit to residuals)
│  ├─ MVO weights (covariance-based)
│  ├─ PCA weights (first principal component)
│  ├─ Entropy weights (information-theoretic)
│  └─ Hybrid = average of 4 methods
└─ Save to session_state.result_df

Output: 45 rows × 14 columns (one row per month)
```

### 3. Results Display

```
Charts Generated:
├─ Forecast Comparison
│  └─ Actual (black line) vs SARIMAX (red dashed) vs Hybrid (blue solid)
│     with 95% confidence intervals (pink shaded)
├─ Residuals
│  └─ Bar chart of forecast errors colored by sign
└─ Event Summary
   └─ Bar charts of category intensity & occurrence counts

Tables Displayed:
├─ Full forecast data (exportable CSV)
├─ Performance metrics (MAE: 2.15, RMSE: 2.89, etc.)
└─ Event details (category, intensity, description, reasoning)
```

---

## File Structure

```
/home/suchitra/InflationPredNew/product/
├── app.py                          # Main Streamlit UI (500+ lines)
├── data_loader.py                  # Data loading & preprocessing (145 lines)
├── prediction_pipeline.py          # SARIMAX + XGBoost (270+ lines)
├── llm_adjustment.py               # LLM event impact (340+ lines)
├── document_processor.py           # Document upload & LLM classification (430+ lines)
├── requirements.txt                # Dependencies
├── README.md                        # This file
└── .streamlit/
    └── config.toml                 # Streamlit configuration (optional)
```

---

## Key Parameters

### Forecast Pipeline
- **Rolling window:** 12 months (configurable in Streamlit)
- **Minimum training:** 6 months
- **SARIMAX order:** (1, 0, 0) - AR(1) model
- **Seasonal order:** (0, 0, 1, 6) - Seasonal MA(1) with 6-month cycle
- **XGBoost:** 200 trees, max_depth=3, learning_rate=0.05

### LLM Event Categorization
- **Model:** gpt-5.4-mini-2026-03-17 (cost-efficient)
- **Temperature:** 0.2 (deterministic)
- **Max tokens:** Automatic (from LangChain)
- **Timeout:** 30 seconds per document

### Impact Adjustment
- **Factor window:** 2-month rolling average (shifted 1 month back)
- **Methods:** OLS, MVO, PCA, Entropy, Hybrid (equal average)
- **Weighting:** Can be customized per session

---

## Troubleshooting

### "Data file not found"
- Ensure `SLData.csv` and `event_categorization_summary_combined.csv` are in parent directory
- Update file paths in `data_loader.py` if needed

### "Could not initialize LLM"
- Check `OPENAI_API_KEY` environment variable
- Create `.env` file in parent directory with key
- Try running without LLM (uncheck "Use LLM" in sidebar)

### "Text extraction failed"
- Ensure PDF is not corrupted (try opening in Adobe)
- Use TXT format as fallback
- Check file permissions

### "Forecast generation timeout"
- Reduce rolling window size
- Reduce forecast period (fewer months to forecast)
- Check system resources (RAM, CPU)

---

## Future Enhancements

1. **Multi-step Forecasting** - Predict 6-12 months ahead (currently 1-month)
2. **Real-time Updates** - Auto-fetch new articles daily
3. **Database Integration** - Persist forecasts and results
4. **API Dashboard** - REST API for programmatic access
5. **Mobile App** - Streamlit Mobile version
6. **Advanced Diagnostics** - ARIMA/GARCH comparison, Bayesian methods
7. **Feedback Loop** - Retrain models with new data quarterly

---

## Citation & References

**Papers/Methods:**
- SARIMAX: Box-Jenkins time series methodology
- XGBoost: Chen & Guestrin (2016), "XGBoost: A Scalable Tree Boosting System"
- PCA Weighting: Jolliffe (2002), "Principal Component Analysis"
- Entropy Weighting: Shannon (1948), Information Theory

**Data Sources:**
- Central Bank of Sri Lanka (CBSL)
- World Bank economic indicators
- News archives

---

## License & Contact

**Project:** Sri Lanka Inflation Prediction System  
**Academic:** B.Sc. Computer Science & Artificial Intelligence (Group Project)  
**Date:** 2026

For questions or issues, contact the development team.

---

**Last Updated:** 2026-03-24
