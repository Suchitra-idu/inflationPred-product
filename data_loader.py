"""
Data Loading and Preprocessing Module
Handles loading and preparing inflation and event data for forecasting
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from typing import Tuple, Optional


def load_inflation_data(csv_path: str) -> pd.DataFrame:
    """
    Load inflation time series data from CSV.
    
    Args:
        csv_path: Path to SLData.csv or similar file
        
    Returns:
        DataFrame with datetime index and inflation/economic indicators
    """
    df = pd.read_csv(csv_path)
    
    # Drop Country column if exists
    if "Country" in df.columns:
        df = df.drop(columns=["Country"])
    
    # Parse Time as datetime and set as index
    df["Time"] = pd.to_datetime(df["Time"], errors="coerce")
    df = df.set_index("Time").sort_index()
    
    # Convert all other columns to numeric
    df = df.apply(pd.to_numeric, errors="coerce")
    
    # Drop rows with missing Time or completely empty rows
    df = df.dropna(how="all")
    
    return df


def load_event_summary(csv_path: str) -> pd.DataFrame:
    """
    Load event categorization summary data.

    Args:
        csv_path: Path to event_categorization_summary_combined.csv

    Returns:
        DataFrame with datetime index and event category intensities
    """
    impact_df = pd.read_csv(csv_path)
    impact_df.columns = impact_df.columns.str.strip().str.replace(" ", "_")

    # If year/month columns exist, use them
    if "year" in impact_df.columns and "month" in impact_df.columns:
        impact_df["year"] = pd.to_numeric(impact_df["year"], errors="coerce")
        impact_df["month"] = pd.to_numeric(impact_df["month"], errors="coerce")

        if impact_df["year"].isna().any() or impact_df["month"].isna().any():
            raise ValueError("Event data has invalid year/month values")

        impact_df["Date"] = pd.to_datetime(
            impact_df["year"].astype(int).astype(str)
            + "-"
            + impact_df["month"].astype(int).astype(str).str.zfill(2)
            + "-01"
        )
        impact_df = impact_df.set_index("Date").sort_index()
        return impact_df.drop(columns=["year", "month"], errors="ignore")

    # Fallback: Date field in case file already has it
    if "Date" in impact_df.columns:
        impact_df["Date"] = pd.to_datetime(impact_df["Date"], errors="coerce")
        impact_df = impact_df.dropna(subset=["Date"]).set_index("Date").sort_index()
        return impact_df

    # Fallback: old 'wide category row' format where first row is year and second is month.
    raw = pd.read_csv(csv_path, header=None)
    if raw.shape[0] > 2 and str(raw.iloc[0, 0]).strip().lower() == "year" and str(raw.iloc[1, 0]).strip().lower() == "month":
        years = raw.iloc[0, 1:].astype(str).tolist()
        months = raw.iloc[1, 1:].astype(str).tolist()
        category_rows = raw.iloc[2:, :].copy()
        category_rows.columns = ["category"] + [f"{y}-{m}" for y, m in zip(years, months)]
        category_rows = category_rows.set_index("category")

        # Build time-indexed DataFrame
        out = []
        for col in category_rows.columns:
            y, m = col.split("-")
            try:
                dt = pd.to_datetime(f"{int(y)}-{int(m):02d}-01")
            except Exception:
                continue
            values = category_rows[col]
            entry = values.to_frame().T
            entry.index = [dt]
            out.append(entry)

        if out:
            result = pd.concat(out).fillna(0)
            return result.sort_index()

    raise KeyError("Unknown event summary format: missing year/month/date")


def engineer_features(df: pd.DataFrame, rolling_months: int = 12, pca_fit_end: str = None) -> pd.DataFrame:
    """
    Engineer features for SARIMAX + XGBoost modeling.
    Creates log transforms, differences, lags, PCA components, and spike indicators.
    
    Args:
        df: Raw inflation DataFrame with economic indicators
        rolling_months: Window for rolling forecasts
        pca_fit_end: ISO date string (e.g. "2022-01-01"). PCA/scaler are fit only
            on data strictly before this date to avoid leakage into the forecast
            period. If None, uses all available data (legacy behaviour).

    Returns:
        DataFrame with engineered features
    """
    df_fe = df.copy()
    
    # === Log transforms for skewed variables ===
    df_fe["log_Imports"] = np.log(
        df_fe["Imports Merchandise"].replace(0, np.nan)
    ).ffill()
    df_fe["log_Exports"] = np.log(
        df_fe["Exports Merchandise"].replace(0, np.nan)
    ).ffill()
    df_fe["log_BroadMoney"] = np.log(
        df_fe["Broad Money"].replace(0, np.nan)
    ).ffill()
    
    # === Differencing for non-stationary variables ===
    df_fe["diff_ExchangeRate"] = df_fe["Official exchange rate"].diff()
    df_fe["diff_TradeBalance"] = df_fe["Trade Balance"].diff()
    df_fe["diff_log_BroadMoney"] = df_fe["log_BroadMoney"].diff()
    
    df_fe = df_fe.dropna()
    
    # === Broad Money Growth (monthly %) ===
    df_fe["BroadMoney_growth"] = df_fe["log_BroadMoney"].diff() * 100
    
    # === Lag features for Exchange Rate (1-18 months) ===
    for lag in range(1, 19):
        df_fe[f"lag{lag}_ExchangeRate"] = df_fe["diff_ExchangeRate"].shift(lag)
    
    df_fe = df_fe.dropna()
    
    # === PCA on collinear variables ===
    # LEAKAGE FIX: fit scaler and PCA only on pre-forecast history so the
    # principal components are not informed by test-period variance.
    collinear_vars = ["log_Imports", "log_Exports", "BroadMoney_growth"]
    if pca_fit_end is not None:
        pca_fit_data = df_fe.loc[df_fe.index < pd.Timestamp(pca_fit_end), collinear_vars].dropna()
    else:
        pca_fit_data = df_fe[collinear_vars].dropna()
    scaler = StandardScaler().fit(pca_fit_data)
    pca = PCA(n_components=2).fit(scaler.transform(pca_fit_data))
    df_fe[["PC1", "PC2"]] = pca.transform(scaler.transform(df_fe[collinear_vars]))
    
    # === Inflation Spike Indicator ===
    # Flag periods with large inflation movements (potential crisis/shock)
    threshold = 3  # 3% absolute change
    df_fe["Inflation_spike"] = (df_fe["Inflation"].diff().abs() > threshold).astype(int)
    
    # Lagged spike features (capture persistence of shocks)
    df_fe["Inflation_spike_lag1"] = df_fe["Inflation_spike"].shift(1).fillna(0)
    df_fe["Inflation_spike_lag2"] = df_fe["Inflation_spike"].shift(2).fillna(0)
    df_fe["Inflation_spike_lag3"] = df_fe["Inflation_spike"].shift(3).fillna(0)
    
    # Weighted spike (emphasize spike events in XGBoost)
    df_fe["Inflation_spike_weighted"] = df_fe["Inflation_spike"] * 3
    
    return df_fe


def get_feature_columns() -> list:
    """
    Features used by XGBoost residual corrector.

    Design choices:
    - 6 ER lags (not 18): with ~24 training samples, 18 lags = severe overfitting
    - No PC1/PC2: PCA of 3 variables on a small sample adds noise, not signal
    - No Inflation_spike_weighted: requires knowing Inflation[t] → data leakage
    - Spike lags (lag1-3) are safe: they use Inflation[t-1/2/3], known at forecast time
    """
    feats = ["diff_ExchangeRate", "BroadMoney_growth"]
    feats.extend([f"lag{lag}_ExchangeRate" for lag in range(1, 7)])   # 6 lags
    feats.extend([
        "Inflation_spike_lag1",
        "Inflation_spike_lag2",
        "Inflation_spike_lag3",
    ])
    return feats


def load_and_preprocess(
    data_path: str,
    rolling_months: int = 12,
    pca_fit_end: str = None
) -> pd.DataFrame:
    """
    Load and preprocess inflation data. Event data is no longer loaded here —
    all event categorization comes from documents uploaded by the user at runtime.

    Args:
        data_path: Path to inflation data CSV
        rolling_months: Window for rolling forecasts
        pca_fit_end: ISO date string; PCA/scaler fit only on data before this date.

    Returns:
        Engineered inflation DataFrame
    """
    df = load_inflation_data(data_path)
    return engineer_features(df, rolling_months, pca_fit_end=pca_fit_end)


if __name__ == "__main__":
    # Example usage
    import sys
    
    # Adjust paths for local testing
    base_dir = Path(__file__).parent.parent
    data_path = base_dir / "SLData.csv"
    events_path = base_dir / "event_categorization_summary_combined.csv"
    
    if data_path.exists():
        df_fe, events_df = load_and_preprocess(str(data_path), str(events_path))
        print("Data loaded and preprocessed successfully!")
        print(f"Engineered data shape: {df_fe.shape}")
        print(f"Features: {get_feature_columns()}")
    else:
        print(f"Data file not found: {data_path}")
