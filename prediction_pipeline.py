"""
Inflation Prediction Pipeline Module
Implements rolling window SARIMAX forecast
"""

import pandas as pd
import numpy as np
import warnings
from typing import Tuple, List
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox

warnings.filterwarnings("ignore")


def forecast_inflation(
    df_fe: pd.DataFrame,
    start_date: str = "2022-01-01",
    end_date: str = "2025-09-01",
    rolling_months: int = 24,
    min_train_months: int = 6,
    base_order: Tuple[int, int, int] = (1, 0, 0),
    base_seasonal: Tuple[int, int, int, int] = (0, 0, 1, 12),
    feature_columns: List[str] = None,   # unused — kept for backward compat
    xgb_params: dict = None,             # unused — XGBoost removed
    verbose: bool = True
) -> pd.DataFrame:
    """
    Generate rolling-window SARIMAX inflation forecasts.

    Args:
        df_fe: Engineered features DataFrame with datetime index
        start_date: Start of forecast period (ISO format)
        end_date: End of forecast period (ISO format)
        rolling_months: Rolling window size for training
        min_train_months: Minimum training data required
        base_order: SARIMAX order (p, d, q)
        base_seasonal: SARIMAX seasonal order (P, D, Q, s)
        feature_columns: Unused (kept for backward compatibility)
        xgb_params: Unused (XGBoost removed — evaluation showed it hurt performance)
        verbose: Print progress messages

    Returns:
        DataFrame with columns:
        - Forecast_SARIMAX / Forecast_Boosted: SARIMAX forecast (identical)
        - Lower_CI, Upper_CI: Confidence intervals
        - XGB_corr: Always 0.0 (kept for backward compat)
    """
    # Ensure datetime index
    df_fe.index = pd.to_datetime(df_fe.index)

    test_index    = pd.date_range(start=start_date, end=end_date, freq="MS")
    last_data_date = df_fe.index.max()

    # Historical seasonal factors: per-month deviation from grand mean.
    # Applied to future multi-step forecasts so they show realistic seasonal
    # variation instead of converging to a flat line.
    historical_inflation = df_fe["Inflation"].dropna()
    overall_mean    = historical_inflation.mean()
    seasonal_factors = (
        historical_inflation.groupby(historical_inflation.index.month).mean()
        - overall_mean
    )

    sarimax_preds    = []
    sarimax_ci_lower = []
    sarimax_ci_upper = []
    forecast_dates   = []

    for t in test_index:
        # Training window — strictly excludes the month being forecast
        if t > last_data_date:
            train_end = last_data_date
        else:
            train_end = t - pd.DateOffset(months=1)

        train_start = train_end - pd.DateOffset(months=rolling_months - 1)
        if train_start < df_fe.index.min():
            train_start = df_fe.index.min()

        train_slice = df_fe.loc[train_start:train_end]
        if len(train_slice) < min_train_months:
            if verbose:
                print(f"Skipping {t.date()}: only {len(train_slice)} months of training data")
            continue

        y_train = train_slice["Inflation"]

        # Fit SARIMAX
        try:
            res = SARIMAX(
                y_train,
                order=base_order,
                seasonal_order=base_seasonal,
                enforce_stationarity=True,   # prevents explosive AR extrapolation
                enforce_invertibility=True,
            ).fit(disp=False)
        except Exception as e:
            if verbose:
                print(f"SARIMAX failed at {t.date()}: {str(e)[:80]}")
            continue

        # Forecast (multi-step for future months)
        try:
            months_ahead = (
                (t.year - last_data_date.year) * 12
                + (t.month - last_data_date.month)
            )
            steps = max(1, months_ahead)
            pred_res = res.get_forecast(steps=steps)
            pred_mean = float(pred_res.predicted_mean.iloc[-1])
            ci        = pred_res.conf_int()
            lower     = float(ci.iloc[-1, 0])
            upper     = float(ci.iloc[-1, 1])

            # Overlay seasonal pattern for future months
            if t > last_data_date:
                seasonal_adj = float(seasonal_factors.get(t.month, 0.0)) * 0.7
                seasonal_adj = np.clip(seasonal_adj, -0.5, 0.5)
                pred_mean += seasonal_adj
                lower     += seasonal_adj
                upper     += seasonal_adj

            # Cap CI width at ±1.5 pp
            lower = max(lower, pred_mean - 1.5)
            upper = min(upper, pred_mean + 1.5)

        except Exception:
            pred_mean = float(y_train.iloc[-1])
            lower = upper = pred_mean

        sarimax_preds.append(pred_mean)
        sarimax_ci_lower.append(lower)
        sarimax_ci_upper.append(upper)
        forecast_dates.append(t)

        if verbose:
            print(f"Forecast {t.date()}: {pred_mean:.3f}  CI=[{lower:.2f}, {upper:.2f}]")

    forecast_df = pd.DataFrame(
        {
            "Forecast_SARIMAX": sarimax_preds,
            "Forecast_Boosted": sarimax_preds,   # backward compat alias
            "Lower_CI":         sarimax_ci_lower,
            "Upper_CI":         sarimax_ci_upper,
            "XGB_corr":         [0.0] * len(sarimax_preds),  # backward compat
        },
        index=pd.to_datetime(forecast_dates),
    )
    forecast_df["Date"] = forecast_df.index
    forecast_df = forecast_df.reset_index(drop=True)

    if verbose:
        print(f"\nForecasts generated: {len(forecast_df)} months")

    return forecast_df


def get_forecast_diagnostics(
    forecast_df: pd.DataFrame,
    actual_inflation: pd.Series = None,
) -> dict:
    """
    Calculate diagnostic statistics for forecast quality.

    Args:
        forecast_df: Output from forecast_inflation()
        actual_inflation: Actual inflation values (optional, for error metrics)

    Returns:
        Dictionary of diagnostic metrics
    """
    diagnostics = {
        "forecast_count": len(forecast_df),
        "date_range":     f"{forecast_df['Date'].min()} to {forecast_df['Date'].max()}",
        "sarimax_mean":   forecast_df["Forecast_SARIMAX"].mean(),
        "sarimax_std":    forecast_df["Forecast_SARIMAX"].std(),
        "ci_width_mean":  (forecast_df["Upper_CI"] - forecast_df["Lower_CI"]).mean(),
    }

    if actual_inflation is not None:
        from sklearn.metrics import mean_absolute_error, mean_squared_error

        forecast_dates   = pd.to_datetime(forecast_df["Date"])
        alignment_index  = forecast_dates.isin(actual_inflation.index)

        if alignment_index.any():
            y_true = actual_inflation.loc[forecast_dates[alignment_index]].values
            y_pred = forecast_df.loc[alignment_index, "Forecast_SARIMAX"].values
            diagnostics.update({
                "MAE":             mean_absolute_error(y_true, y_pred),
                "RMSE":            float(np.sqrt(mean_squared_error(y_true, y_pred))),
                "MAPE":            float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-8)))),
                "Bias":            float(np.mean(y_pred - y_true)),
                "test_obs_count":  len(y_true),
            })

    return diagnostics
