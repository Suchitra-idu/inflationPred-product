"""
Streamlit App: Sri Lanka Inflation Prediction with Event Analysis
Clean tab-based UI for the inflation prediction system
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from pathlib import Path
import io
import os

# Load .env so OPENROUTER_API_KEY is available
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env")
except ImportError:
    pass

# Import project modules
from data_loader import load_and_preprocess, get_feature_columns
from prediction_pipeline import forecast_inflation, get_forecast_diagnostics
from llm_adjustment import adjust_forecast_with_llm_events, evaluate_adjustments
from document_processor import (
    process_batch_uploads,
    parse_filename_to_date,
    EventAggregation,
    scrape_url_text,
    extract_text_from_upload,
)

# Try to import OpenAI (optional)
try:
    from openai import OpenAI as _OpenAI
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    _OpenAI = None

import matplotlib.pyplot as plt
import warnings
import traceback
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
        /* Main colors: Professional blue, green accent, warm orange for data */
        :root {
            --primary: #1e3a8a;
            --secondary: #059669;
            --accent: #f97316;
            --light-bg: #f8fafc;
            --border: #e2e8f0;
        }
        
        .main { padding: 1rem; background-color: white; }
        
        h1 { 
            color: #1e3a8a;
            font-size: 2.2rem;
            margin-bottom: 0.3rem;
            font-weight: 700;
            letter-spacing: -0.5px;
        }
        
        h2 {
            color: #1e3a8a;
            font-size: 1.6rem;
            margin-top: 1.2rem;
            margin-bottom: 1rem;
            font-weight: 600;
            border-bottom: 2px solid #059669;
            padding-bottom: 0.5rem;
        }
        
        h3 {
            color: #334155;
            font-size: 1.1rem;
            font-weight: 600;
            margin-top: 0.8rem;
        }

        p { color: #475569; line-height: 1.6; }
        
        /* Button styling */
        .stButton > button,
        .stDownloadButton > button {
            background-color: #1e3a8a !important;
            color: #ffffff !important;
            font-weight: 600;
            padding: 0.6rem 1.2rem;
            border: none !important;
            border-radius: 6px;
            transition: all 0.2s ease;
        }

        .stButton > button:hover,
        .stDownloadButton > button:hover {
            background-color: #1e40af !important;
            color: #ffffff !important;
            box-shadow: 0 2px 8px rgba(30, 58, 138, 0.25);
        }

        .stButton > button *,
        .stDownloadButton > button * {
            color: #ffffff !important;
        }

        /* Sidebar */
        [data-testid="stSidebar"] {
            background-color: #f8fafc;
            border-right: 1px solid #e2e8f0;
        }

        [data-testid="stSidebar"] h2 {
            font-size: 1.3rem;
            color: #1e3a8a;
            margin-top: 0;
        }

        /* Tabs */
        [data-baseweb="tab-list"] {
            border-bottom: 2px solid #e2e8f0;
        }

        [aria-selected="true"] {
            color: #1e3a8a;
            border-bottom: 3px solid #059669;
        }

        /* Cards and containers */
        .stMetric {
            background-color: #f8fafc;
            padding: 1rem;
            border-radius: 8px;
            border-left: 4px solid #059669;
        }

        div[data-testid="stExpander"] {
            background-color: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 6px;
        }

        /* Tables */
        .stDataFrame {
            margin: 1rem 0;
        }

        /* Divider */
        hr { border-color: #e2e8f0; }

        /* Status messages */
        .stSuccess { 
            background-color: #ecfdf5;
            border-left: 4px solid #059669;
        }

        .stError {
            background-color: #fef2f2;
            border-left: 4px solid #dc2626;
        }

        .stWarning {
            background-color: #fffbeb;
            border-left: 4px solid #d97706;
        }

        .stInfo {
            background-color: #eff6ff;
            border-left: 4px solid #3b82f6;
        }

        /* Subheader for sections */
        .section-header {
            color: #334155;
            font-size: 0.95rem;
            font-weight: 500;
            margin-top: 1rem;
            margin-bottom: 0.5rem;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
    </style>
    """, unsafe_allow_html=True)

st.markdown("<h1>Inflation Prediction System</h1>", unsafe_allow_html=True)
st.markdown("<p style='font-size: 1rem; color: #64748b; margin-bottom: 1.5rem;'><strong>Sri Lanka Economic Forecasting</strong><br/>Time-series SARIMAX time-series + AI-powered event analysis</p>", unsafe_allow_html=True)


# === SESSION STATE ===
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'events_processed' not in st.session_state:
    st.session_state.events_processed = False
if 'forecast_generated' not in st.session_state:
    st.session_state.forecast_generated = False
if 'scraped_docs' not in st.session_state:
    st.session_state.scraped_docs = []
if 'doc_texts' not in st.session_state:
    st.session_state.doc_texts = {}


# === DOCUMENT PREVIEW DIALOG ===
@st.dialog("Document Preview", width="large")
def preview_dialog(title: str, text: str):
    """Modal popup to preview extracted document text."""
    st.markdown(f"**{title}**")
    preview = text[:5000] if len(text) > 5000 else text
    st.text_area("Content", preview, height=400, disabled=True, label_visibility="collapsed")
    if len(text) > 5000:
        st.caption(f"Showing first 5,000 of {len(text):,} characters")


# === CACHING ===
@st.cache_data
def load_historical_data(pca_fit_end: str = None):
    """Load and preprocess historical inflation data. Events come from user uploads only."""
    base_dir = Path(__file__).parent.parent
    data_path = base_dir / "SLData.csv"

    if not data_path.exists():
        st.error(f"Data file not found: {data_path}")
        return None

    try:
        return load_and_preprocess(str(data_path), pca_fit_end=pca_fit_end)
    except Exception as e:
        tb = traceback.format_exc()
        st.error(f"Error loading data:\n{tb}")
        return None


@st.cache_resource
def get_llm_client() -> "_OpenAI | None":
    """Initialize OpenRouter client (OpenAI-compatible)"""
    if not LLM_AVAILABLE:
        return None
    try:
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            return None
        return _OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
    except Exception:
        return None


# === SIDEBAR CONFIGURATION ===
with st.sidebar:
    st.markdown("<h2 style='margin-top: 0;'>Settings</h2>", unsafe_allow_html=True)
    
    st.markdown("<div class='section-header'>Forecast Period</div>", unsafe_allow_html=True)
    start_date = st.date_input(
        "Start Month",
        value=datetime(2025, 9, 1),
        min_value=datetime(1996, 1, 1),
        max_value=datetime(2026, 12, 1),
        help="Select the first month of the forecast window (day is set to 1).",
        label_visibility="collapsed"
    )

    forecast_months = st.slider(
        "Forecast Horizon (months)",
        min_value=3,
        max_value=24,
        value=9,
        step=1,
        help="Number of months to forecast from the selected start month."
    )

    # Normalize start to first day of month
    start_date = start_date.replace(day=1)
    end_date = start_date + relativedelta(months=forecast_months - 1)

    # Convert to datetime.date for consistent comparisons (Streamlit returns date objects)
    if isinstance(end_date, datetime):
        end_date = end_date.date()

    if end_date > datetime(2026, 12, 31).date():
        st.warning("Forecast end exceeds 2026-12-31; please reduce horizon or shift start month.")
    
    st.markdown("<div class='section-header' style='margin-top: 1.5rem;'>Model Settings</div>", unsafe_allow_html=True)
    rolling_months = st.slider(
        "Rolling Window (months)",
        min_value=6, max_value=120, value=24, step=6,
        help="Number of months of historical data used to train each rolling forecast step."
    )

    st.markdown("<div class='section-header' style='margin-top: 1rem;'>LLM Settings</div>", unsafe_allow_html=True)

    use_llm_adjustment = st.checkbox("Enable LLM Adjustment", value=True,
        help="When disabled, only SARIMAX+XGBoost forecast is shown. Useful for comparing with/without event data.")

    if use_llm_adjustment:
        st.caption("Hybrid Weight — share of LLM-Only in the Hybrid forecast")
        llm_hybrid_weight = st.slider(
            "Hybrid LLM Weight",
            min_value=0.0, max_value=1.0, value=0.7, step=0.05,
            help="0.0 = pure SARIMAX+XGBoost, 1.0 = pure LLM-Only, 0.5 = equal blend."
        )
        st.caption("LLM Multiplier — scales all event intensities")
        llm_multiplier = st.slider(
            "LLM Multiplier",
            min_value=0.5, max_value=3.0, value=1.7, step=0.1,
            help="Multiplies all +/- event intensities. 1.0 = raw LLM output, 1.5 = 50% stronger signals."
        )
        mean_reversion = st.checkbox("Mean reversion in LLM-Only", value=True,
            help="When enabled, LLM-Only pulls back toward historical mean between events. Disable for pure cumulative drift.")
        st.caption("Signal Decay — how fast past LLM events fade into future months")
        signal_decay = st.slider(
            "Signal Decay (per month)",
            min_value=0.5, max_value=1.0, value=0.8, step=0.05,
            help="0.8 = signal halves every ~3 months. 1.0 = last known signal held constant forever. 0.5 = rapid fade."
        )
    else:
        llm_hybrid_weight = 0.7
        llm_multiplier = 1.7
        mean_reversion = True
        signal_decay = 0.8

    verbose = st.checkbox("Verbose logging", value=False)
    
    st.markdown("<div class='section-header' style='margin-top: 1.5rem;'>Data</div>", unsafe_allow_html=True)
    if st.button("Refresh Data", width="stretch"):
        st.cache_data.clear()
        st.session_state.data_loaded = False
        st.success("Cache cleared")


# === MAIN TABS ===
tab_upload, tab_analyze, tab_forecast, tab_results, tab_events, tab_export = st.tabs([
    "Step 1: Upload",
    "Step 2: Analyze",
    "Step 3: Forecast",
    "Step 4: Results",
    "Events",
    "Export"
])


# ============================================================================
# TAB 1: UPLOAD DOCUMENTS
# ============================================================================
with tab_upload:
    st.markdown("<h2>Upload Economic Documents</h2>", unsafe_allow_html=True)
    st.markdown("Upload reports/articles or paste a URL to scrape. Filenames should contain dates (e.g., `2024-03_article.pdf`)")

    st.divider()

    # --- File Upload ---
    uploaded_files = st.file_uploader(
        "Select files",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
        key="file_uploader"
    )

    if uploaded_files:
        st.session_state.uploaded_files = uploaded_files
        # Pre-extract text for preview (cached in session state)
        for f in uploaded_files:
            if f.name not in st.session_state.doc_texts:
                try:
                    st.session_state.doc_texts[f.name] = extract_text_from_upload(
                        f.getvalue(), f.name
                    )
                except Exception as e:
                    st.session_state.doc_texts[f.name] = f"[Error extracting text: {e}]"

    # --- URL Scraping ---
    st.markdown("<h3>Or Add from URL</h3>", unsafe_allow_html=True)
    url_col1, url_col2, url_col3 = st.columns([3, 1, 1])
    with url_col1:
        url_input = st.text_input(
            "URL",
            placeholder="https://example.com/article",
        )
    with url_col2:
        url_year = st.number_input("Year", min_value=1996, max_value=2026, value=2025, key="url_year")
    with url_col3:
        url_month = st.number_input("Month", min_value=1, max_value=12, value=1, key="url_month")

    if st.button("Scrape & Add URL"):
        if not url_input or not url_input.startswith("http"):
            st.error("Please enter a valid URL starting with http:// or https://")
        else:
            with st.spinner("Scraping URL..."):
                try:
                    text = scrape_url_text(url_input)
                    short_url = url_input[:60] + ("..." if len(url_input) > 60 else "")
                    st.session_state.scraped_docs.append({
                        'url': url_input,
                        'text': text,
                        'year': url_year,
                        'month': url_month,
                        'label': short_url,
                    })
                    st.session_state.doc_texts[f"url_{short_url}"] = text
                    st.success(f"Scraped {len(text):,} characters from URL")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to scrape URL: {e}")

    st.divider()

    # --- All Sources (files + URLs) with preview ---
    all_sources = []

    for f in st.session_state.get('uploaded_files', []):
        year, month, success = parse_filename_to_date(f.name)
        all_sources.append({
            'type': 'file',
            'name': f.name,
            'date': f"{year:04d}-{month:02d}" if success else "Invalid",
            'size': f"{len(f.getvalue()) / 1024:.1f} KB",
            'status': "Valid" if success else "Invalid",
            'text_key': f.name,
        })

    for idx, doc in enumerate(st.session_state.get('scraped_docs', [])):
        all_sources.append({
            'type': 'url',
            'name': doc['label'],
            'date': f"{doc['year']:04d}-{doc['month']:02d}",
            'size': f"{len(doc['text']) / 1024:.1f} KB",
            'status': "Valid",
            'text_key': f"url_{doc['label']}",
            'url_index': idx,
        })

    if all_sources:
        st.markdown(f"<h3>All Sources ({len(all_sources)})</h3>", unsafe_allow_html=True)

        for i, src in enumerate(all_sources):
            icon = "📄" if src['type'] == 'file' else "🌐"
            border = "#059669" if src['status'] == "Valid" else "#dc2626"

            col_info, col_preview, col_remove = st.columns([5, 1, 0.5])

            with col_info:
                st.markdown(
                    f"<div style='border-left:4px solid {border};padding:6px 12px;"
                    f"background:#f9fafb;border-radius:0 6px 6px 0;margin-bottom:2px'>"
                    f"{icon} <b>{src['name']}</b> &nbsp;·&nbsp; {src['date']} &nbsp;·&nbsp; "
                    f"{src['size']} &nbsp;·&nbsp; "
                    f"<span style='color:{border}'>{src['status']}</span></div>",
                    unsafe_allow_html=True,
                )

            with col_preview:
                text = st.session_state.doc_texts.get(src['text_key'], '')
                if text and st.button("👁 Preview", key=f"preview_{i}"):
                    preview_dialog(src['name'], text)

            with col_remove:
                if src['type'] == 'url':
                    if st.button("✕", key=f"remove_{i}"):
                        st.session_state.scraped_docs.pop(src['url_index'])
                        if src['text_key'] in st.session_state.doc_texts:
                            del st.session_state.doc_texts[src['text_key']]
                        st.rerun()

        st.info(f"Ready to process {len(all_sources)} source(s). Proceed to 'Step 2: Analyze'")
    else:
        st.info("Upload files or paste a URL to begin. Filenames should contain dates like YYYY-MM")


# ============================================================================
# TAB 2: ANALYZE DOCUMENTS
# ============================================================================
with tab_analyze:
    st.markdown("<h2>Analyze Documents</h2>", unsafe_allow_html=True)
    
    has_uploads = 'uploaded_files' in st.session_state and st.session_state.uploaded_files
    has_scraped = bool(st.session_state.get('scraped_docs', []))
    if not has_uploads and not has_scraped:
        st.warning("No sources added. Go to 'Step 1: Upload' to add documents or URLs.")
    else:
        st.markdown("<h3>Analysis Options</h3>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)

        with col1:
            use_llm = st.checkbox("Use AI Classification", value=True, help="OpenAI LLM event extraction")
        with col2:
            skip_existing = st.checkbox("Skip existing events", value=True)

        if st.button("Start Analysis", width="stretch", type="primary"):
            with st.spinner("Analyzing documents..."):
                file_bytes_list = []
                if has_uploads:
                    file_bytes_list = [(f.getvalue(), f.name) for f in st.session_state.uploaded_files]
                for doc in st.session_state.get('scraped_docs', []):
                    synthetic_name = f"{doc['year']}-{doc['month']:02d}_url.txt"
                    file_bytes_list.append((doc['text'].encode('utf-8'), synthetic_name))
                
                llm = None
                if use_llm and LLM_AVAILABLE:
                    llm = get_llm_client()
                    if not llm:
                        st.warning("AI not available. Extracting text only.")
                
                try:
                    aggregation, results = process_batch_uploads(
                        file_bytes_list,
                        llm=llm,
                        skip_llm=(not use_llm)
                    )
                    
                    st.session_state.aggregation = aggregation
                    st.session_state.processing_results = results
                    st.session_state.events_processed = True
                    
                    st.success(f"Analysis complete: {len(results)} document(s) processed")
                    
                except Exception as e:
                    tb = traceback.format_exc()
                    st.error(f"Error: {str(e)}\n{tb}")
        
        if st.session_state.get('events_processed', False):
            st.divider()
            st.markdown("<h3>Analysis Results</h3>", unsafe_allow_html=True)

            aggregation = st.session_state.aggregation
            results = st.session_state.get('processing_results', [])

            # --- Category colours ---
            CAT_COLORS = {
                "External_and_Global_Shocks":         ("#1e3a8a", "#dbeafe"),
                "Fiscal_Policy":                       ("#7c2d12", "#ffedd5"),
                "Monetary_Policy":                     ("#065f46", "#d1fae5"),
                "Supply_and_Demand_Shocks_(Domestic)": ("#581c87", "#f3e8ff"),
                "Supply_and_Demand_Shocks":            ("#581c87", "#f3e8ff"),
            }

            def cat_badge(cat: str) -> str:
                fg, bg = CAT_COLORS.get(cat, ("#374151", "#f3f4f6"))
                label = cat.replace("_", " ").replace("(Domestic)", "").strip()
                return (
                    f"<span style='background:{bg};color:{fg};padding:2px 8px;"
                    f"border-radius:12px;font-size:0.78rem;font-weight:600'>{label}</span>"
                )

            def intensity_bar(val: float) -> str:
                clamped = max(min(val, 1.0), -1.0)
                pct = abs(clamped) * 50  # half the bar width (50% = full side)
                if clamped < 0:
                    color = "#2563eb"
                    # bar grows left from centre
                    bar_style = f"position:absolute;right:50%;width:{pct:.0f}%;background:{color};height:8px;border-radius:4px"
                else:
                    color = "#dc2626"
                    # bar grows right from centre
                    bar_style = f"position:absolute;left:50%;width:{pct:.0f}%;background:{color};height:8px;border-radius:4px"
                return (
                    f"<div style='display:flex;align-items:center;gap:6px'>"
                    f"<span style='font-size:0.75rem;color:#6b7280;width:32px;text-align:right'>-1</span>"
                    f"<div style='flex:1;position:relative;background:#e5e7eb;border-radius:4px;height:8px'>"
                    f"<div style='{bar_style}'></div>"
                    f"<div style='position:absolute;left:50%;top:-2px;width:1px;height:12px;background:#374151'></div>"
                    f"</div>"
                    f"<span style='font-size:0.75rem;color:#6b7280;width:24px'>+1</span>"
                    f"<span style='font-size:0.8rem;color:#6b7280;width:40px;text-align:right'>{val:+.2f}</span></div>"
                )

            import calendar

            # --- Per-document cards ---
            st.markdown("**Documents Processed**")
            for res in results:
                meta = res.get('metadata')
                ok = res.get('success', False) and (meta and meta.parse_success)
                nevents = len(res.get('events', []))
                fname = meta.filename if meta else "unknown"
                date_str = (
                    f"{calendar.month_name[meta.parsed_month]} {meta.parsed_year}"
                    if meta and meta.parse_success else "date not parsed"
                )
                border = "#059669" if ok else "#dc2626"
                icon = "✓" if ok else "✗"
                st.markdown(
                    f"<div style='border-left:4px solid {border};padding:8px 12px;"
                    f"margin-bottom:6px;background:#f9fafb;border-radius:0 6px 6px 0'>"
                    f"<b>{icon} {fname}</b> &nbsp;·&nbsp; {date_str} &nbsp;·&nbsp; "
                    f"<span style='color:#6b7280'>{nevents} event(s) extracted</span>"
                    + (f"<br><span style='color:#dc2626;font-size:0.82rem'>{res.get('error','')}</span>" if not ok else "")
                    + "</div>",
                    unsafe_allow_html=True
                )

            # --- Detailed events grouped by month ---
            detail_df = aggregation.get_detail_df()
            if not detail_df.empty:
                st.markdown("---")
                st.markdown("**Extracted Events**")
                for (yr, mo), grp in detail_df.groupby(["year", "month"]):
                    month_label = f"{calendar.month_name[int(mo)]} {int(yr)}"
                    with st.expander(f"📅 {month_label}  —  {len(grp)} event(s)", expanded=True):
                        for _, row in grp.iterrows():
                            st.markdown(cat_badge(row['category']), unsafe_allow_html=True)
                            st.markdown(
                                intensity_bar(float(row['intensity'])),
                                unsafe_allow_html=True
                            )
                            st.markdown(
                                f"<p style='margin:4px 0 2px;font-size:0.92rem'>{row['description']}</p>",
                                unsafe_allow_html=True
                            )
                            if row.get('reasoning'):
                                st.markdown(
                                    f"<p style='margin:0 0 10px;font-size:0.82rem;color:#6b7280;"
                                    f"font-style:italic'>{row['reasoning']}</p>",
                                    unsafe_allow_html=True
                                )
                            st.markdown("<hr style='margin:6px 0;border:none;border-top:1px solid #e5e7eb'>",
                                        unsafe_allow_html=True)

            # Accumulate uploaded events across multiple analysis runs
            summary_df = aggregation.get_summary_df()
            if not summary_df.empty:
                if 'custom_events_df' in st.session_state and not st.session_state.custom_events_df.empty:
                    st.session_state.custom_events_df = pd.concat(
                        [st.session_state.custom_events_df, summary_df], ignore_index=True
                    ).drop_duplicates(subset=["year", "month"], keep="last")
                else:
                    st.session_state.custom_events_df = summary_df

            st.success("Ready to generate forecasts — go to Step 3.")


# ============================================================================
# TAB 3: GENERATE FORECASTS
# ============================================================================
with tab_forecast:
    st.markdown("<h2>Generate Forecasts</h2>", unsafe_allow_html=True)
    
    # Load historical data
    if not st.session_state.data_loaded:
        with st.spinner("Loading historical data..."):
            df_fe = load_historical_data(
                pca_fit_end=start_date.strftime("%Y-%m-%d")
            )
            if df_fe is not None:
                st.session_state.df_fe = df_fe
                st.session_state.data_loaded = True
                st.success("Data loaded")
            else:
                st.error("Could not load data")
    
    if st.session_state.data_loaded:
        st.markdown("<h3>Configuration</h3>", unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Start Date", start_date.strftime("%Y-%m-%d"))
        with col2:
            st.metric("End Date", end_date.strftime("%Y-%m-%d"))
        with col3:
            st.metric("Rolling Window", f"{rolling_months} months")
        
        st.divider()
        
        if st.button("Generate Forecasts", width="stretch", type="primary"):
            with st.spinner("Running SARIMAX forecasts... This may take 1-2 minutes"):
                try:
                    df_fe = st.session_state.df_fe

                    # Generate base forecast
                    forecast_df = forecast_inflation(
                        df_fe,
                        start_date=start_date.strftime("%Y-%m-%d"),
                        end_date=end_date.strftime("%Y-%m-%d"),
                        rolling_months=rolling_months,
                        feature_columns=get_feature_columns(),
                        verbose=verbose
                    )
                    
                    # LLM adjustment — only from user-uploaded documents
                    custom_events = st.session_state.get('custom_events_df', None)
                    has_events = (
                        use_llm_adjustment
                        and custom_events is not None
                        and isinstance(custom_events, pd.DataFrame)
                        and not custom_events.empty
                    )

                    if has_events:
                        events_df_to_use = custom_events.copy()
                        # Convert year/month to Date index if needed
                        if "year" in events_df_to_use.columns and "month" in events_df_to_use.columns:
                            events_df_to_use["Date"] = pd.to_datetime(
                                events_df_to_use["year"].astype(int).astype(str) + "-" +
                                events_df_to_use["month"].astype(int).astype(str).str.zfill(2) + "-01"
                            )
                            events_df_to_use = events_df_to_use.drop(columns=["year", "month"]).set_index("Date")
                        events_df_to_use.columns = (
                            events_df_to_use.columns.str.strip()
                            .str.replace(" ", "_", regex=False)
                            .str.replace("(Domestic)", "", regex=False)
                            .str.replace("__", "_", regex=False)
                            .str.strip("_")
                        )
                        # Apply LLM multiplier to all intensity columns
                        numeric_cols = events_df_to_use.select_dtypes(include="number").columns
                        events_df_to_use[numeric_cols] = (events_df_to_use[numeric_cols] * llm_multiplier).clip(-1, 1)
                        result_df = adjust_forecast_with_llm_events(
                            forecast_df,
                            events_df_to_use,
                            actual_inflation=df_fe["Inflation"],
                            methods=['OLS', 'MVO', 'PCA', 'Entropy', 'Hybrid'],
                            llm_hybrid_weight=llm_hybrid_weight,
                            mean_reversion=mean_reversion,
                            signal_decay=signal_decay,
                        )
                    else:
                        # No uploaded events — use base forecast only
                        result_df = forecast_df.copy()
                        st.info("No event documents uploaded. Showing SARIMAX forecast only. Upload documents in Step 1 to enable LLM adjustment.")

                    st.session_state.forecast_df = forecast_df
                    st.session_state.result_df = result_df
                    st.session_state.forecast_generated = True

                    st.success("Forecasts generated successfully! See results in Step 4.")
                
                except Exception as e:
                    tb = traceback.format_exc()
                    st.error(f"Error: {str(e)}\n{tb}")


# ============================================================================
# TAB 4: VIEW RESULTS
# ============================================================================
with tab_results:
    st.markdown("<h2>Forecast Results</h2>", unsafe_allow_html=True)
    
    if not st.session_state.get('forecast_generated', False):
        st.info("No forecasts generated yet. Complete Step 3 first.")
    else:
        result_df = st.session_state.result_df
        forecast_df = st.session_state.forecast_df
        df_fe = st.session_state.get('df_fe')

        if df_fe is None or 'Inflation' not in df_fe.columns:
            st.warning("Historical inflation data unavailable for accuracy metrics.")
            df_fe = None

        # Charts
        st.markdown("<h3>Forecast Comparison</h3>", unsafe_allow_html=True)

        fig, ax = plt.subplots(figsize=(14, 5))

        forecast_dates = pd.to_datetime(result_df["Date"])
        chart_start = forecast_dates.min() - pd.DateOffset(years=1)

        # Actual — only show 1 year before forecast start onwards
        hist_slice = df_fe[df_fe.index >= chart_start]
        ax.plot(hist_slice.index, hist_slice["Inflation"], 'o-', label="Actual",
                color="#1e3a8a", linewidth=2.5, markersize=5, alpha=0.8)

        # Forecasts
        ax.plot(forecast_dates, result_df["Forecast_Boosted"], '--',
                label="SARIMAX", color="#f97316", linewidth=1.8, alpha=0.7)
        if use_llm_adjustment and "Adjusted_Inflation_Hybrid" in result_df.columns:
            ax.plot(forecast_dates, result_df["Adjusted_Inflation_Hybrid"],
                    '-', label="Hybrid (AI-Adjusted)", color="#059669", linewidth=2.5)
            if "LLM_Only_Forecast" in result_df.columns:
                ax.plot(forecast_dates, result_df["LLM_Only_Forecast"],
                        ':', label="LLM-Only Signal", color="#7c3aed", linewidth=2.0, alpha=0.85)

        # CI
        ax.fill_between(forecast_dates, result_df["Lower_CI"], result_df["Upper_CI"],
                        alpha=0.12, color="#059669", label="95% CI")

        # Clamp y-axis to sensible range based on visible data
        all_visible = pd.concat([
            hist_slice["Inflation"],
            result_df["Forecast_Boosted"],
            result_df.get("Adjusted_Inflation_Hybrid", pd.Series(dtype=float)),
            result_df.get("LLM_Only_Forecast", pd.Series(dtype=float))
        ]).dropna()
        y_pad = 0.5
        ax.set_ylim(all_visible.min() - y_pad, all_visible.max() + y_pad)

        ax.set_xlim(left=chart_start)
        ax.set_xlabel("Date", fontsize=11, fontweight="500")
        ax.set_ylabel("Inflation (%)", fontsize=11, fontweight="500")
        ax.set_title("Inflation Actual vs Forecasts", fontsize=13, fontweight="bold", pad=15)
        ax.legend(loc="best", fontsize=10, framealpha=0.95)
        ax.grid(True, alpha=0.25, linestyle="--")
        plt.xticks(rotation=45, fontsize=9)
        plt.tight_layout()
        st.pyplot(fig)
        
        # Data table
        st.divider()
        st.markdown("<h3>Forecast Data</h3>", unsafe_allow_html=True)
        
        all_cols = list(result_df.columns)
        default_cols = ["Date", "Actual_Inflation", "Forecast_Boosted", "Adjusted_Inflation_Hybrid", "Lower_CI", "Upper_CI"]
        default_cols = [c for c in default_cols if c in all_cols]

        display_cols = st.multiselect(
            "Select columns",
            options=all_cols,
            default=default_cols
        )

        if "Date" not in display_cols and "Date" in all_cols:
            display_cols = ["Date"] + display_cols
        
        st.dataframe(result_df[display_cols], width="stretch", hide_index=True)


# ============================================================================
# TAB 5: EVENTS SUMMARY
# ============================================================================
with tab_events:
    st.markdown("<h2>Event Summary</h2>", unsafe_allow_html=True)

    if st.session_state.data_loaded:
        events_df = st.session_state.get('custom_events_df', None)
        
        if events_df is None or not isinstance(events_df, pd.DataFrame) or events_df.empty:
            st.info("No event data available yet. Run Step 2 (Analyze) to extract events.")
        else:
            st.markdown("<h3>Average Intensity by Category</h3>", unsafe_allow_html=True)

            # custom_events_df has one column per category with intensity values
            ev_display = events_df.copy()
            if "year" in ev_display.columns and "month" in ev_display.columns:
                ev_display["Date"] = pd.to_datetime(
                    ev_display["year"].astype(int).astype(str) + "-" +
                    ev_display["month"].astype(int).astype(str).str.zfill(2) + "-01"
                )
                ev_display = ev_display.drop(columns=["year", "month"])

            cat_cols = [c for c in ev_display.columns if c not in ("Date",)]
            if cat_cols:
                mean_intensities = ev_display[cat_cols].mean()
                colors = ['#2563eb' if v < 0 else '#dc2626' for v in mean_intensities]

                col1, col2 = st.columns([1, 2])
                with col1:
                    st.markdown("**Mean intensity (across uploaded months)**")
                    for cat, val in mean_intensities.items():
                        st.write(f"- {cat.replace('_', ' ')}: {val:+.3f}")

                with col2:
                    fig, ax = plt.subplots(figsize=(10, 4))
                    mean_intensities.plot(kind='barh', ax=ax, color=colors)
                    ax.set_xlabel("Mean Intensity")
                    ax.set_title("Average LLM Event Intensity by Category", fontweight="bold")
                    ax.axvline(0, color='black', linewidth=0.8)
                    ax.set_xlim(-1, 1)
                    plt.tight_layout()
                    st.pyplot(fig)

            st.divider()
            st.markdown("<h3>Full Event Data</h3>", unsafe_allow_html=True)
            if "Date" in ev_display.columns:
                ev_display.insert(0, "Year", pd.to_datetime(ev_display["Date"]).dt.year)
                ev_display.insert(1, "Month", pd.to_datetime(ev_display["Date"]).dt.strftime("%B"))
                ev_display = ev_display.drop(columns=["Date"])
            st.dataframe(ev_display, width="stretch", hide_index=True)


# ============================================================================
# TAB 6: EXPORT
# ============================================================================
with tab_export:
    st.markdown("<h2>Export Results</h2>", unsafe_allow_html=True)
    
    if not st.session_state.get('forecast_generated', False):
        st.info("Generate forecasts first to export results.")
    else:
        result_df = st.session_state.result_df
        
        # CSV Export
        st.markdown("<h3>Forecast Results</h3>", unsafe_allow_html=True)
        csv = result_df.to_csv(index=False)
        st.download_button(
            "Download Forecast CSV",
            data=csv,
            file_name=f"inflation_forecast_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            width="stretch"
        )
        
        # Events Export
        if st.session_state.get('events_processed', False):
            st.markdown("<h3>Extracted Events</h3>", unsafe_allow_html=True)
            events_csv = st.session_state.aggregation.get_summary_df().to_csv(index=False)
            st.download_button(
                "Download Events CSV",
                data=events_csv,
                file_name=f"extracted_events_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                width="stretch"
            )
        
        # Preview
        st.divider()
        st.markdown("<h3>Preview</h3>", unsafe_allow_html=True)
        st.dataframe(result_df, width="stretch", hide_index=True)


# === FOOTER ===
st.divider()
st.markdown("""
    <div style='text-align: center; color: #94a3b8; font-size: 0.9rem; padding: 1rem;'>
        <p>Sri Lanka Inflation Prediction System | Built with SARIMAX and LLM event analysis</p>
        <p>© 2026 - Research Project</p>
    </div>
""", unsafe_allow_html=True)
