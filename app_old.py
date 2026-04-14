"""
Streamlit App: Sri Lanka Inflation Prediction with Event Analysis
Main UI for the inflation prediction system with document upload and LLM categorization
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import io
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from langchain_openai import ChatOpenAI

# Import project modules
from data_loader import load_and_preprocess, get_feature_columns
from prediction_pipeline import forecast_inflation, get_forecast_diagnostics
from llm_adjustment import adjust_forecast_with_llm_events, evaluate_adjustments
from document_processor import (
    process_batch_uploads,
    parse_filename_to_date,
    EventAggregation
)

# Try to import LLM (optional, for document classification)
try:
    from langchain_openai import ChatOpenAI as _ChatOpenAI
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    _ChatOpenAI = None  # type: ignore

import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


# === PAGE CONFIG ===
st.set_page_config(
    page_title="Inflation Prediction System | Sri Lanka",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# === CUSTOM STYLING ===
st.markdown("""
    <style>
        /* Main container */
        .main { padding: 0.5rem; }
        
        /* Header styling */
        h1 { 
            color: #0f3460;
            font-size: 2.5rem;
            margin-bottom: 0.5rem;
            font-weight: 700;
            letter-spacing: -0.5px;
        }
        
        h2 {
            color: #0f3460;
            font-size: 1.8rem;
            margin-top: 1.5rem;
            margin-bottom: 1rem;
            font-weight: 600;
            border-bottom: 3px solid #16a34a;
            padding-bottom: 0.5rem;
        }
        
        h3 {
            color: #1e3a5f;
            font-size: 1.2rem;
            margin-top: 1rem;
            font-weight: 600;
        }
        
        /* Button styling */
        .stButton > button {
            background-color: #16a34a;
            color: white;
            font-weight: 600;
            padding: 0.75rem 1.5rem;
            border: none;
            border-radius: 6px;
            transition: all 0.3s ease;
        }
        
        .stButton > button:hover {
            background-color: #15803d;
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(22, 163, 74, 0.3);
        }
        
        /* Sidebar styling */
        [data-testid="stSidebar"] {
            background-color: #f8fafc;
            border-right: 1px solid #e2e8f0;
        }
        
        [data-testid="stSidebar"] h2 {
            color: #0f3460;
            border-color: #16a34a;
        }
        
        /* Tab styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 1rem;
        }
        
        .stTabs [data-baseweb="tab"] {
            background-color: #f1f5f9;
            font-weight: 600;
            color: #64748b;
            padding: 0.75rem 1rem;
            border-radius: 6px;
        }
        
        .stTabs [aria-selected="true"] {
            background-color: #16a34a !important;
            color: white !important;
        }
        
        /* Cards and containers */
        .stMetric {
            background-color: #f8fafc;
            padding: 1.5rem;
            border-radius: 8px;
            border-left: 4px solid #16a34a;
        }
        
        /* Spinners and progress */
        .stSpinner > div > div {
            border-top-color: #16a34a;
        }
        
        /* Success/error/info messages */
        .stAlert {
            border-radius: 6px;
            font-weight: 500;
        }
        
        /* Dataframe styling */
        [data-testid="stDataFrame"] {
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
        }
    </style>
    """, unsafe_allow_html=True)

st.markdown("<h1>Inflation Prediction System</h1>", unsafe_allow_html=True)
st.markdown("<p style='font-size: 1.1rem; color: #64748b; margin-bottom: 2rem;'>Advanced forecasting with SARIMAX, XGBoost, and LLM-based event analysis</p>", unsafe_allow_html=True)


# === CACHING ===

@st.cache_data
def load_historical_data():
    """Load and preprocess historical inflation and event data"""
    base_dir = Path(__file__).parent.parent
    data_path = base_dir / "SLData.csv"
    events_path = base_dir / "event_categorization_summary_combined.csv"
    
    if not data_path.exists():
        st.error(f"Data file not found: {data_path}")
        return None, None
    
    if not events_path.exists():
        st.error(f"Events file not found: {events_path}")
        return None, None
    
    try:
        df_fe, events_df = load_and_preprocess(str(data_path), str(events_path))
        return df_fe, events_df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, None


@st.cache_resource
def get_llm_client() -> "ChatOpenAI | None":
    """Initialize OpenAI LLM client"""
    if not LLM_AVAILABLE:
        return None
    
    try:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return None
        
        llm = _ChatOpenAI(model="gpt-5.4-mini-2026-03-17", temperature=0.2)
        return llm
    except Exception as e:
        st.warning(f"Could not initialize LLM: {str(e)}")
        return None


# === SIDEBAR CONFIGURATION ===

with st.sidebar:
    st.markdown("<h2 style='margin-top: 0;'>Configuration</h2>", unsafe_allow_html=True)
    
    with st.container():
        st.markdown("<h3>Forecast Period</h3>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "Start Date",
                value=datetime(2022, 1, 1),
                min_value=datetime(1996, 1, 1),
                max_value=datetime(2026, 12, 31),
                label_visibility="collapsed"
            )
        with col2:
            end_date = st.date_input(
                "End Date",
                value=datetime(2025, 9, 1),
                min_value=datetime(1996, 1, 1),
                max_value=datetime(2026, 12, 31),
                label_visibility="collapsed"
            )
        
        if start_date > end_date:
            st.error("Start date must be before end date")
    
    st.divider()
    
    with st.container():
        st.markdown("<h3>Forecast Settings</h3>", unsafe_allow_html=True)
        rolling_months = st.slider(
            "Rolling Window (months)",
            min_value=6,
            max_value=120,
            value=12,
            step=6,
            label_visibility="collapsed"
        )
        
        verbose = st.checkbox("Show detailed progress", value=False)
    
    st.divider()
    
    with st.container():
        st.markdown("<h3>Data Management</h3>", unsafe_allow_html=True)
        if st.button("Reload Historical Data", use_container_width=True):
            st.session_state.data_loaded = False
            st.rerun()
    
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False


# === MAIN TABS ===

tab1, tab2, tab3, tab4 = st.tabs([
    "Documents",
    "Forecasts",
    "Events",
    "Export"
])


# ============================================================================
# TAB 1: DOCUMENTS
# ============================================================================

with tab1:
    st.markdown("<h2>Upload & Analyze Documents</h2>", unsafe_allow_html=True)
    st.markdown("Upload economic reports or articles with dates in filenames (e.g., `2025-10_article.pdf`)")
    
    st.divider()
    
    # File uploader
    col_upload, col_settings = st.columns([2, 1])
    
    with col_upload:
        uploaded_files = st.file_uploader(
            "Select files (PDF, TXT, Markdown)",
            type=["pdf", "txt", "md"],
            accept_multiple_files=True,
            key="file_uploader"
        )
    
    with col_settings:
        st.markdown("<p style='margin-top: 15px;'></p>", unsafe_allow_html=True)
        use_llm = st.checkbox(
            "Use AI Classification",
            value=True,
            help="Classify events using OpenAI LLM (requires API key)"
        )
        skip_existing = st.checkbox(
            "Skip Existing",
            value=True,
            help="Skip months already in event summary"
        )
    
    if uploaded_files:
        st.divider()
        st.markdown("<h3>Upload Summary</h3>", unsafe_allow_html=True)
        
        # Display file list with parsed dates
        file_info = []
        for file in uploaded_files:
            year, month, success = parse_filename_to_date(file.name)
            status = "Valid" if success else "Invalid"
            file_info.append({
                "File Name": file.name,
                "Parsed Date": f"{year:04d}-{month:02d}" if success else "Not found",
                "Status": status
            })
        
        st.dataframe(pd.DataFrame(file_info), use_container_width=True, hide_index=True)
        
        # Process button
        if st.button("Analyze Documents", key="analyze_btn", use_container_width=False):
            with st.spinner("Processing documents..."):
                # Prepare files
                file_bytes_list = [
                    (file.getvalue(), file.name)
                    for file in uploaded_files
                ]
                
                # Get LLM if needed
                llm = None
                if use_llm and LLM_AVAILABLE:
                    llm = get_llm_client()
                    if not llm:
                        st.warning("AI not available. Using text extraction only.")
                
                try:
                    # Process files
                    aggregation, results = process_batch_uploads(
                        file_bytes_list,
                        llm=llm,
                        skip_llm=(not use_llm)
                    )
                    
                    # Store in session state
                    st.session_state.aggregation = aggregation
                    st.session_state.processing_results = results
                    st.session_state.events_processed = True
                    
                    st.success(f"Successfully processed {len(results)} document(s)")
                    
                except Exception as e:
                    st.error(f"Error processing files: {str(e)}")
        
        # Display results if available
        if st.session_state.get('events_processed', False):
            st.divider()
            st.markdown("<h3>Extracted Events</h3>", unsafe_allow_html=True)
            
            aggregation = st.session_state.aggregation
            
            # Summary by month
            summary_df = aggregation.get_summary_df()
            if not summary_df.empty:
                st.markdown("**Summary by Month & Category**")
                st.dataframe(summary_df, use_container_width=True, hide_index=True)
            
            # Detailed view
            detail_df = aggregation.get_detail_df()
            if not detail_df.empty:
                st.markdown("**Detailed Events**")
                st.dataframe(detail_df, use_container_width=True, hide_index=True)
            
            # Save to session for use in other tabs
            st.session_state.custom_events_df = summary_df


# ============================================================================
# TAB 2: FORECASTS
# ============================================================================

with tab2:
    st.markdown("<h2>Inflation Forecast Results</h2>", unsafe_allow_html=True)
    
    # Load data
    if not st.session_state.get('data_loaded', False):
        with st.spinner("Loading historical data..."):
            df_fe, events_df = load_historical_data()
            if df_fe is not None:
                st.session_state.df_fe = df_fe
                st.session_state.events_df = events_df
                st.session_state.data_loaded = True
            else:
                st.error("Could not load data. Check data files.")
    
    if st.session_state.get('data_loaded', False):
        df_fe = st.session_state.df_fe
        events_df = st.session_state.events_df
        
        # Generate forecasts
        col1, col2 = st.columns([2, 1])
        
        with col1:
            if st.button("Generate Forecasts", key="forecast_btn", use_container_width=True):
                with st.spinner("Running SARIMAX + XGBoost forecasts... (this may take 1-2 minutes)"):
                    try:
                        # Base forecast
                        forecast_df = forecast_inflation(
                            df_fe,
                            start_date=start_date.strftime("%Y-%m-%d"),
                            end_date=end_date.strftime("%Y-%m-%d"),
                            rolling_months=rolling_months,
                            feature_columns=get_feature_columns(),
                            verbose=verbose
                        )
                        
                        # LLM adjustment
                        events_df_to_use = df_fe  # Default: use pre-classified events
                        if st.session_state.get('custom_events_df') is not None:
                            events_df_to_use = st.session_state.custom_events_df
                        
                        result_df = adjust_forecast_with_llm_events(
                            forecast_df,
                            events_df_to_use,
                            actual_inflation=df_fe["Inflation"],
                            methods=['OLS', 'MVO', 'PCA', 'Entropy', 'Hybrid']
                        )
                        
                        st.session_state.forecast_df = forecast_df
                        st.session_state.result_df = result_df
                        
                        st.success("Forecasts generated successfully!")
                    
                    except Exception as e:
                        st.error(f"Error generating forecasts: {str(e)}")
                        import traceback
                        st.write(traceback.format_exc())
        
        # Display results
        if st.session_state.get('result_df') is not None:
            result_df = st.session_state.result_df
            forecast_df = st.session_state.forecast_df
            
            st.divider()
            st.markdown("<h3>Forecast Data</h3>", unsafe_allow_html=True)
            
            # Select columns to display
            display_cols = st.multiselect(
                "Select columns to display",
                options=[c for c in result_df.columns if c not in ["Date"]],
                default=[
                    "Date", "Forecast_Boosted", "Adjusted_Inflation_Hybrid",
                    "Lower_CI", "Upper_CI"
                ],
                key="col_select"
            )
            
            if "Date" not in display_cols:
                display_cols = ["Date"] + display_cols
            
            st.dataframe(result_df[display_cols], use_container_width=True, hide_index=True)
            
            st.divider()
            st.markdown("<h3>Forecast Charts</h3>", unsafe_allow_html=True)
            
            # Chart 1: Forecast Comparison
            col_chart1 = st.container()
            with col_chart1:
                fig, ax = plt.subplots(figsize=(14, 5))
                
                forecast_dates = pd.to_datetime(result_df["Date"])
                
                # Plot actual inflation
                actual_dates = df_fe.index
                actual_values = df_fe["Inflation"]
                ax.plot(actual_dates, actual_values, 'o-', label="Actual Inflation", color="#0f3460", linewidth=2, markersize=4)
                
                # Plot forecasts
                ax.plot(forecast_dates, result_df["Forecast_Boosted"], '--', label="SARIMAX+XGB", color="#ea580c", linewidth=1.5)
                ax.plot(forecast_dates, result_df["Adjusted_Inflation_Hybrid"], '-', label="Hybrid (AI-Adjusted)", color="#16a34a", linewidth=2)
                
                # Confidence intervals
                ax.fill_between(
                    forecast_dates,
                    result_df["Lower_CI"],
                    result_df["Upper_CI"],
                    alpha=0.15,
                    color="#16a34a",
                    label="95% Confidence Interval"
                )
                
                ax.set_xlabel("Date", fontsize=11, fontweight="500")
                ax.set_ylabel("Inflation (%)", fontsize=11, fontweight="500")
                ax.set_title("Inflation Forecast: SARIMAX+XGB vs AI-Adjusted Hybrid", fontsize=13, fontweight="bold", pad=20)
                ax.legend(loc="best", fontsize=10)
                ax.grid(True, alpha=0.2, linestyle="--")
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)
            
            # Chart 2: Residuals
            st.markdown("<h3>Forecast Residuals</h3>", unsafe_allow_html=True)
            
            fig, ax = plt.subplots(figsize=(14, 4))
            
            # Align actual with forecast
            actual_aligned = df_fe.loc[forecast_dates, "Inflation"]
            residuals = actual_aligned.values - result_df["Forecast_Boosted"].values
            
            colors = ['#dc2626' if x < 0 else '#16a34a' for x in residuals]
            ax.bar(forecast_dates, residuals, color=colors, alpha=0.7, edgecolor='none')
            ax.axhline(0, color='#0f3460', linestyle='-', linewidth=1.5)
            ax.set_xlabel("Date", fontsize=11, fontweight="500")
            ax.set_ylabel("Residual (%)", fontsize=11, fontweight="500")
            ax.set_title("Forecast Errors by Month", fontsize=13, fontweight="bold", pad=20)
            ax.grid(True, alpha=0.2, axis='y', linestyle="--")
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
            
            # Evaluation metrics
            st.divider()
            st.markdown("<h3>Model Performance Metrics</h3>", unsafe_allow_html=True)
            
            try:
                eval_df = evaluate_adjustments(
                    result_df,
                    df_fe["Inflation"],
                    baseline_col="Forecast_Boosted"
                )
                st.dataframe(eval_df, use_container_width=True, hide_index=True)
            except Exception as e:
                st.info(f"Could not evaluate metrics: {str(e)}")


# ============================================================================
# TAB 3: EVENTS
# ============================================================================

with tab3:
    st.markdown("<h2>Event Category Analysis</h2>", unsafe_allow_html=True)
    
    if st.session_state.get('custom_events_df') is not None:
        events_summary = st.session_state.custom_events_df
        
        st.markdown("<h3>Event Summary by Category</h3>", unsafe_allow_html=True)
        
        # Aggregate by category
        category_cols = [
            "Monetary Policy",
            "Fiscal Policy",
            "External and Global Shocks",
            "Supply and Demand Shocks"
        ]
        
        category_cols = [c for c in category_cols if c in events_summary.columns]
        
        if category_cols:
            # Chart 1: Average intensity by category
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            
            # Mean intensity
            mean_intensity = events_summary[category_cols].mean()
            colors = ['#dc2626' if x < 0 else '#16a34a' for x in mean_intensity]
            ax1.barh(mean_intensity.index, mean_intensity.values, color=colors, alpha=0.75, edgecolor='none')
            ax1.set_xlabel("Average Intensity", fontsize=11, fontweight="500")
            ax1.set_title("Average Event Intensity by Category", fontsize=12, fontweight="bold", pad=15)
            ax1.axvline(0, color='#0f3460', linestyle='-', linewidth=1.5)
            ax1.grid(True, alpha=0.2, axis='x', linestyle="--")
            
            # Event count (non-zero)
            event_counts = (events_summary[category_cols] != 0).sum()
            bars = ax2.bar(range(len(event_counts)), event_counts.values, color='#16a34a', alpha=0.75, edgecolor='none')
            ax2.set_ylabel("Occurrence Count", fontsize=11, fontweight="500")
            ax2.set_title("Event Occurrences by Category", fontsize=12, fontweight="bold", pad=15)
            ax2.set_xticks(range(len(event_counts)))
            ax2.set_xticklabels(event_counts.index, rotation=45, ha='right')
            ax2.grid(True, alpha=0.2, axis='y', linestyle="--")
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}',
                        ha='center', va='bottom', fontsize=10, fontweight="500")
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Summary table
            st.divider()
            st.markdown("<h3>Category Statistics</h3>", unsafe_allow_html=True)
            
            summary_table = pd.DataFrame({
                "Category": category_cols,
                "Mean Intensity": [f"{events_summary[c].mean():.3f}" for c in category_cols],
                "Occurrences": [(events_summary[c] != 0).sum() for c in category_cols],
                "Total Events": [events_summary[c].notna().sum() for c in category_cols]
            })
            st.dataframe(summary_table, use_container_width=True, hide_index=True)
    else:
        st.info("Upload and analyze documents in the 'Documents' tab to see event summaries.")


# ============================================================================
# TAB 4: EXPORT
# ============================================================================

with tab4:
    st.markdown("<h2>Export Results</h2>", unsafe_allow_html=True)
    st.markdown("Download your analysis results as CSV files for reporting and further analysis.")
    
    st.divider()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<h3>Forecast Data</h3>", unsafe_allow_html=True)
        
        if st.session_state.get('result_df') is not None:
            result_df = st.session_state.result_df
            
            # CSV download
            csv_data = result_df.to_csv(index=False)
            st.download_button(
                label="Download Forecast CSV",
                data=csv_data,
                file_name=f"inflation_forecast_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                key="download_forecast_csv"
            )
            
            st.caption(f"Contains {len(result_df)} rows × {len(result_df.columns)} columns")
        else:
            st.info("Generate forecasts in the 'Forecasts' tab first.")
    
    with col2:
        st.markdown("<h3>Event Data</h3>", unsafe_allow_html=True)
        
        if st.session_state.get('custom_events_df') is not None:
            events_df = st.session_state.custom_events_df
            
            # CSV download
            csv_data = events_df.to_csv(index=False)
            st.download_button(
                label="Download Events CSV",
                data=csv_data,
                file_name=f"events_summary_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                key="download_events_csv"
            )
            
            st.caption(f"Contains {len(events_df)} rows × {len(events_df.columns)} columns")
        else:
            st.info("Process documents in the 'Documents' tab first.")


# === FOOTER ===

st.markdown("---")
st.markdown(
    """
    ### 📚 About This System
    
    This system combines:
    - **SARIMAX**: Time series forecasting with seasonal patterns
    - **XGBoost**: Residual correction using economic indicators
    - **LLM Analysis**: Event categorization via ChatGPT (optional)
    - **Impact Adjustment**: Multiple weighting methods (OLS, MVO, PCA, Entropy)
    
    **Data sources:**
    - Sri Lanka Central Bank economic indicators
    - News articles & reports (user-uploaded)
    - Pre-classified event summaries
    """
)
