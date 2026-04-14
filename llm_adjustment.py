"""
LLM Event Impact Adjustment Module
Integrates event categorization with inflation forecasts using multiple weighting methods
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, List
from sklearn.decomposition import PCA
from numpy.linalg import inv


def _normalize_events_df(events_df: pd.DataFrame) -> pd.DataFrame:
    """Normalize event dataframe for consistent category and datetime handling."""
    df = events_df.copy()

    # If year/month columns exist, build a Date index
    if "year" in df.columns and "month" in df.columns:
        df["year"] = pd.to_numeric(df["year"], errors="coerce")
        df["month"] = pd.to_numeric(df["month"], errors="coerce")

        if df["year"].isna().any() or df["month"].isna().any():
            raise ValueError("Event data has invalid year/month values")

        df["Date"] = pd.to_datetime(
            df["year"].astype(int).astype(str)
            + "-"
            + df["month"].astype(int).astype(str).str.zfill(2)
            + "-01"
        )
        df = df.set_index("Date").sort_index()
        df = df.drop(columns=["year", "month"], errors="ignore")

    # If Date column exists, set it as index
    elif "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"]).set_index("Date").sort_index()

    # Ensure we have datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Event dataframe index must be datetime-like")

    # Normalize category names to underscores so they can be matched easily
    df.columns = (
        df.columns
          .str.strip()
          .str.replace(" ", "_", regex=False)
          .str.replace("(Domestic)", "", regex=False)
          .str.replace("__", "_", regex=False)
          .str.strip("_")
    )

    return df


def prepare_impact_factors(
    events_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    signal_decay: float = 0.80
) -> pd.DataFrame:
    """
    Prepare event impact factors for adjustment.
    Aggregate event category intensities and create rolling effects.

    Args:
        events_df: Event categorization summary with Date index
        forecast_df: Forecast DataFrame from prediction_pipeline

    Returns:
        DataFrame with impact factors aligned to forecast dates
    """
    events_df = _normalize_events_df(events_df)

    # Define canonical event categories that we can support
    canonical_categories = [
        "External_and_Global_Shocks",
        "Fiscal_Policy",
        "Monetary_Policy",
        "Supply_and_Demand_Shocks"
    ]

    # attempt to map categories either exact or by prefix
    selected_categories = []
    for cat in canonical_categories:
        if cat in events_df.columns:
            selected_categories.append(cat)
        else:
            # try partial match from original with/without parentheses
            alt = cat.replace("_", " ")
            matches = [c for c in events_df.columns if c.lower().startswith(alt.lower())]
            if matches:
                selected_categories.append(matches[0])

    if not selected_categories:
        raise ValueError("No recognized event categories found in events_df")

    # Convert forecast dates to datetime
    forecast_dates = pd.to_datetime(forecast_df["Date"])

    # Collapse any duplicate columns (from merging historical + uploaded data) by averaging
    events_df = events_df.groupby(level=0).mean() if events_df.index.duplicated().any() else events_df
    events_df = events_df.T.groupby(level=0).mean().T  # deduplicate columns

    # Prepare per-factor rolling effects with exponential decay into future months.
    # Instead of filling missing future months with 0 (which would make the model
    # "forget" all LLM history the moment events_df runs out), we carry the last
    # known signal forward with a monthly decay of 0.80 (~half-life 3 months).
    # Economically: a rate hike or supply shock has a lingering effect that fades.
    DECAY_PER_MONTH = signal_decay

    factor_effects = {}
    for cat in selected_categories:
        col = events_df[cat]
        if isinstance(col, pd.DataFrame):
            col = col.mean(axis=1)
        rolled = col.rolling(2, min_periods=1).mean()

        # Align to forecast dates; NaN = no event data for that month
        reindexed = rolled.reindex(forecast_dates)

        # Find last date with actual LLM data and build decay forward from it.
        # Use fillna with a pre-built Series to avoid pandas in-place assignment issues.
        last_known_idx = reindexed.last_valid_index()
        if last_known_idx is not None:
            last_val = float(reindexed.loc[last_known_idx])
            future_dates = forecast_dates[forecast_dates > last_known_idx]
            if len(future_dates) > 0:
                months_out = np.array([
                    (d.year - last_known_idx.year) * 12 + (d.month - last_known_idx.month)
                    for d in future_dates
                ])
                decay_series = pd.Series(
                    last_val * (DECAY_PER_MONTH ** months_out),
                    index=future_dates
                )
                reindexed = reindexed.fillna(decay_series)
        else:
            reindexed = reindexed.fillna(0)

        # Fill any remaining NaN (gaps within the known period) with 0
        reindexed = reindexed.fillna(0)
        factor_effects[cat] = reindexed

    X_factors = pd.DataFrame(factor_effects, index=forecast_dates)

    return X_factors


def calculate_ols_weights(
    X_factors: pd.DataFrame,
    residuals: np.ndarray
) -> pd.Series:
    """
    Calculate OLS (Ordinary Least Squares) weights for event impacts.
    Fitted to historical forecast residuals.
    
    Args:
        X_factors: DataFrame of factor intensities
        residuals: Array of forecast residuals (actual - predicted)
        
    Returns:
        Series of OLS weights by factor
    """
    X = X_factors.values
    y = residuals
    
    # Least squares solution
    weights_ls, *_ = np.linalg.lstsq(X, y, rcond=None)
    weights_ols = pd.Series(weights_ls, index=X_factors.columns)
    
    return weights_ols


def calculate_mvo_weights(X_factors: pd.DataFrame) -> pd.Series:
    """
    Calculate Minimum Variance Optimization (MVO) weights.
    Inverse variance weighting based on covariance matrix.
    
    Args:
        X_factors: DataFrame of factor intensities
        
    Returns:
        Series of MVO weights by factor
    """
    S = X_factors.cov().values
    k = S.shape[0]
    ones = np.ones(k)
    
    # Add regularization to avoid singular matrix
    invS = inv(S + 1e-8 * np.eye(k))
    w_mv = invS.dot(ones) / (ones.dot(invS).dot(ones))
    
    weights_mvo = pd.Series(w_mv, index=X_factors.columns)
    
    return weights_mvo


def calculate_pca_weights(X_factors: pd.DataFrame) -> pd.Series:
    """
    Calculate PCA-based weights using first principal component.
    Captures dominant variance direction in factors.
    
    Args:
        X_factors: DataFrame of factor intensities
        
    Returns:
        Series of normalized PCA weights by factor
    """
    pca = PCA(n_components=1)
    pca.fit(X_factors.fillna(0).values)
    
    weights_pca = pd.Series(
        np.abs(pca.components_[0]),
        index=X_factors.columns
    )
    weights_pca /= weights_pca.sum()
    
    return weights_pca


def calculate_entropy_weights(X_factors: pd.DataFrame) -> pd.Series:
    """
    Calculate entropy-based weights using information theory approach.
    Higher entropy factors get lower weights.
    
    Args:
        X_factors: DataFrame of factor intensities
        
    Returns:
        Series of entropy weights by factor
    """
    A = X_factors.copy()
    A = A - A.min() + 1e-8  # Shift to positive
    
    # Normalize to probability distribution
    P = A.div(A.sum(axis=0), axis=1)
    
    # Calculate entropy for each factor (column)
    entropy = -(P * np.log(P + 1e-12)).sum(axis=0)
    
    # Weight inversely to entropy: high entropy → low weight
    d = 1 - entropy / np.log(len(P.columns))
    weights_entropy = d / d.sum()
    
    return weights_entropy


def adjust_forecast_with_llm_events(
    forecast_df: pd.DataFrame,
    events_df: Optional[pd.DataFrame],
    actual_inflation: Optional[pd.Series] = None,
    methods: List[str] = None,
    llm_hybrid_weight: float = 0.5,
    mean_reversion: bool = True,
    signal_decay: float = 0.80,
) -> pd.DataFrame:
    """
    Adjust inflation forecasts using LLM-classified event impacts.
    Applies multiple weighting methods: OLS, MVO, PCA, Entropy, and Hybrid.

    Args:
        forecast_df: Output from prediction_pipeline.forecast_inflation()
        events_df: Event categorization summary DataFrame or None
        actual_inflation: Actual inflation values for residual-based weighting
        methods: List of methods to apply. Default: ['OLS', 'MVO', 'PCA', 'Entropy', 'Hybrid']

    Returns:
        DataFrame with original forecasts + all adjusted variants
    """

    if methods is None:
        methods = ['OLS', 'MVO', 'PCA', 'Entropy', 'Hybrid']

    # Make a copy to avoid modifying input
    result_df = forecast_df.copy()

    if events_df is None or (isinstance(events_df, pd.DataFrame) and events_df.empty):
        raise ValueError("Event dataframe is required and cannot be empty for LLM adjustment.")

    
    # === Prepare factor impacts ===
    X_factors = prepare_impact_factors(events_df, forecast_df, signal_decay=signal_decay)

    if X_factors.empty or X_factors.shape[1] == 0:
        raise ValueError("Prepared event factor matrix is empty; check event data categories.")

    # === Calculate weight vectors for each method ===
    weights = {}
    
    if 'OLS' in methods:
        # OLS: fit to residuals if actual inflation provided.
        # LEAKAGE FIX: use only the first 70% of overlapping dates to fit weights
        # so that the remaining dates remain a genuine out-of-sample holdout.
        if actual_inflation is not None:
            forecast_dates = pd.to_datetime(forecast_df["Date"])
            alignment = forecast_dates.isin(actual_inflation.index)

            if alignment.any():
                residuals_series = (
                    actual_inflation.loc[forecast_dates[alignment]] -
                    result_df.loc[alignment, "Forecast_Boosted"]
                )
                X_aligned = X_factors.reindex(forecast_dates[alignment]).dropna()
                residuals_series = residuals_series.reindex(X_aligned.index).dropna()
                X_aligned = X_aligned.loc[residuals_series.index]

                if len(X_aligned) >= 2:
                    # Fit on the earlier 70% only
                    n_fit = max(1, int(len(X_aligned) * 0.7))
                    weights['OLS'] = calculate_ols_weights(
                        X_aligned.iloc[:n_fit],
                        residuals_series.values[:n_fit]
                    )
                else:
                    weights['OLS'] = pd.Series(1.0 / len(X_factors.columns), index=X_factors.columns)
            else:
                weights['OLS'] = pd.Series(1.0 / len(X_factors.columns), index=X_factors.columns)
        else:
            weights['OLS'] = pd.Series(1.0 / len(X_factors.columns), index=X_factors.columns)
    
    # LEAKAGE FIX: for unsupervised weight methods (MVO, PCA, Entropy) fit on the
    # historical portion only (dates where actual inflation is known). Fall back to
    # full X_factors if no historical overlap exists.
    forecast_dates = pd.to_datetime(forecast_df["Date"])
    if actual_inflation is not None:
        hist_mask = forecast_dates.isin(actual_inflation.index).values
        X_factors_hist = X_factors.iloc[hist_mask] if hist_mask.any() else X_factors
    else:
        X_factors_hist = X_factors

    if 'MVO' in methods:
        weights['MVO'] = calculate_mvo_weights(X_factors_hist)

    if 'PCA' in methods:
        weights['PCA'] = calculate_pca_weights(X_factors_hist)

    if 'Entropy' in methods:
        weights['Entropy'] = calculate_entropy_weights(X_factors_hist)
    
    # === Apply adjustments ===
    for method, w in weights.items():
        # Align weights with factor columns (safely handle accidental mismatch)
        w_aligned = w.reindex(X_factors.columns, fill_value=0.0)

        if len(w_aligned) != X_factors.shape[1]:
            raise ValueError(
                f"Incompatible dimensions in weight vector for method {method}: "
                f"X_factors cols={X_factors.shape[1]}, weights={len(w_aligned)}"
            )

        try:
            adjustment = X_factors.values @ w_aligned.values
        except ValueError as ex:
            raise ValueError(
                f"Incompatible dimensions during adjustment multiply for method {method}: "
                f"X_factors {X_factors.shape}, w {w_aligned.shape} - {ex}"
            )

        result_df[f"Impact_Adjustment_{method.lower()}"] = adjustment
        result_df[f"Adjusted_Inflation_{method}"] = (
            result_df["Forecast_Boosted"] + adjustment
        )
    
    # === LLM-Only forecast (fully independent of SARIMAX/XGBoost) ===
    # Starts from the last known actual inflation value and builds autoregressively:
    #   llm_only[t] = llm_only[t-1] + mean_intensity[t] * hist_std
    # This is purely event-driven — no statistical model involved.
    if not X_factors.empty:
        raw_signal = X_factors.mean(axis=1)

        if actual_inflation is not None and len(actual_inflation.dropna()) > 0:
            last_actual = float(actual_inflation.dropna().iloc[-1])
            hist_std = max(float(actual_inflation.dropna().std()), 1e-6)
        else:
            last_actual = 0.0
            hist_std = 0.3

        # AR(1) with mean reversion + LLM event shock each step:
        #   value[t] = alpha * value[t-1] + (1-alpha) * hist_mean + signal[t] * hist_std
        # alpha=0.75: moderate pull-back to mean when signals are weak,
        # but strong events (signal ±1) still produce meaningful moves.
        alpha = 0.75
        hist_mean = float(actual_inflation.dropna().mean()) if actual_inflation is not None else last_actual

        llm_values = []
        prev = last_actual
        for sig in raw_signal.values:
            if mean_reversion:
                prev = alpha * prev + (1 - alpha) * hist_mean + sig * hist_std
            else:
                prev = prev + sig * hist_std
            llm_values.append(prev)

        result_df["LLM_Only_Forecast"] = llm_values

    # === Hybrid: blend SARIMAX+XGBoost and LLM-Only ===
    if 'Hybrid' in methods and "LLM_Only_Forecast" in result_df.columns:
        result_df["Adjusted_Inflation_Hybrid"] = (
            (1 - llm_hybrid_weight) * result_df["Forecast_Boosted"] +
            llm_hybrid_weight       * result_df["LLM_Only_Forecast"]
        )

    return result_df


def evaluate_adjustments(
    result_df: pd.DataFrame,
    actual_inflation: pd.Series,
    baseline_col: str = "Forecast_Boosted"
) -> pd.DataFrame:
    """
    Evaluate forecast accuracy across all adjustment methods.
    Calculate MAE, RMSE, MAPE, Bias, and directional accuracy.
    
    Args:
        result_df: Output from adjust_forecast_with_llm_events()
        actual_inflation: Actual inflation values
        baseline_col: Column to use as baseline for comparison
        
    Returns:
        DataFrame with evaluation metrics for each forecast variant
    """
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    
    forecast_dates = pd.to_datetime(result_df["Date"])
    alignment_index = forecast_dates.isin(actual_inflation.index)
    
    if not alignment_index.any():
        raise ValueError("No date alignment between forecasts and actual inflation")
    
    y_true = actual_inflation.loc[forecast_dates[alignment_index]].values
    
    # Find all forecast columns
    forecast_cols = [c for c in result_df.columns if c.startswith("Adjusted_Inflation_")]
    forecast_cols.insert(0, baseline_col)
    forecast_cols = list(dict.fromkeys(forecast_cols))  # Remove duplicates, preserve order
    
    results = []
    
    for col in forecast_cols:
        if col not in result_df.columns:
            continue
        
        y_pred = result_df.loc[alignment_index, col].values
        
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mape = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-8)))
        bias = np.mean(y_pred - y_true)
        
        # Directional accuracy: do predictions match actual direction?
        dir_match = np.sign(np.diff(y_pred)) == np.sign(np.diff(y_true))
        dir_acc = np.mean(dir_match)
        
        results.append({
            "Model": col,
            "MAE": mae,
            "RMSE": rmse,
            "MAPE": mape,
            "Bias": bias,
            "Dir_Accuracy": dir_acc
        })
    
    eval_df = pd.DataFrame(results).set_index("Model")
    
    return eval_df


if __name__ == "__main__":
    # Example usage
    from data_loader import load_and_preprocess
    from prediction_pipeline import forecast_inflation
    from pathlib import Path
    
    base_dir = Path(__file__).parent.parent
    data_path = base_dir / "SLData.csv"
    events_path = base_dir / "event_categorization_summary_combined.csv"
    
    if data_path.exists() and events_path.exists():
        # Load data
        df_fe, events_df = load_and_preprocess(str(data_path), str(events_path))
        
        # Generate base forecasts
        forecast_df = forecast_inflation(
            df_fe,
            start_date="2022-01-01",
            end_date="2025-09-01",
            verbose=False
        )
        
        # Adjust with LLM events
        result_df = adjust_forecast_with_llm_events(
            forecast_df,
            events_df,
            actual_inflation=df_fe["Inflation"],
            methods=['OLS', 'MVO', 'PCA', 'Entropy', 'Hybrid']
        )
        
        print("Adjusted forecasts (first 5 rows):")
        adjusted_cols = [c for c in result_df.columns if "Adjusted_Inflation" in c]
        print(result_df[["Date", "Forecast_Boosted"] + adjusted_cols].head())
        
        # Evaluate
        eval_df = evaluate_adjustments(result_df, df_fe["Inflation"])
        print("\nEvaluation metrics:")
        print(eval_df.round(4))
