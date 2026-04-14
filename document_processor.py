"""
Document Processing Module for Streamlit
Handles file uploads, text extraction, LLM-based event categorization, and aggregation
"""

import re
import json
import io
import sys
from pathlib import Path
from typing import Tuple, List, Dict, Optional, TYPE_CHECKING
from dataclasses import dataclass
import pandas as pd

try:
    import pypdf
except Exception:
    pypdf = None

try:
    from openai import OpenAI as _OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    _OpenAI = None  # type: ignore


# === Data Structures ===

@dataclass
class Event:
    """Represents a single economic event"""
    category: str
    description: str
    intensity: float
    reasoning: str


@dataclass
class DocumentMetadata:
    """Metadata extracted from uploaded document"""
    filename: str
    parsed_year: Optional[int]
    parsed_month: Optional[int]
    parse_success: bool
    error_message: str = ""


class EventAggregation:
    """Aggregates events by category and month"""
    
    def __init__(self):
        self.events_by_month: Dict[Tuple[int, int], List[Event]] = {}
        self.categories = [
            "Monetary Policy",
            "Fiscal Policy",
            "External and Global Shocks",
            "Supply and Demand Shocks"
        ]
    
    def add_event(self, year: int, month: int, event: Event) -> None:
        """Add event to aggregation"""
        key = (year, month)
        if key not in self.events_by_month:
            self.events_by_month[key] = []
        self.events_by_month[key].append(event)
    
    def get_summary_df(self) -> pd.DataFrame:
        """
        Get aggregated summary as DataFrame.
        Returns rows with: year, month, and intensity by category
        """
        rows = []
        
        for (year, month), events in sorted(self.events_by_month.items()):
            row = {"year": year, "month": month}
            
            # Aggregate by category
            for cat in self.categories:
                cat_events = [e for e in events if e.category == cat]
                if cat_events:
                    avg_intensity = sum(e.intensity for e in cat_events) / len(cat_events)
                    row[cat] = avg_intensity
                else:
                    row[cat] = 0.0
            
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    def get_detail_df(self) -> pd.DataFrame:
        """
        Get detailed view of all extracted events.
        Returns: year, month, category, intensity, description, reasoning
        """
        rows = []
        
        for (year, month), events in sorted(self.events_by_month.items()):
            for event in events:
                rows.append({
                    "year": year,
                    "month": month,
                    "category": event.category,
                    "intensity": event.intensity,
                    "description": event.description,
                    "reasoning": event.reasoning
                })
        
        return pd.DataFrame(rows)


# === Filename Parsing ===

def parse_filename_to_date(filename: str) -> Tuple[Optional[int], Optional[int], bool]:
    """
    Extract year and month from filename using regex pattern YYYY-MM.
    
    Args:
        filename: Name of uploaded file (e.g., "2025-10_article.pdf")
        
    Returns:
        Tuple of (year, month, success)
        Example: (2025, 10, True) or (None, None, False)
    """
    # Pattern: YYYY-MM (e.g., 2025-10, 2021-05)
    pattern = r"(20\d{2})[-_/ ](0[1-9]|1[0-2])"
    
    match = re.search(pattern, filename)
    if match:
        year = int(match.group(1))
        month = int(match.group(2))
        return year, month, True
    
    return None, None, False


def validate_date(year: Optional[int], month: Optional[int]) -> bool:
    """Check if extracted date is valid"""
    if year is None or month is None:
        return False
    return 1996 <= year <= 2026 and 1 <= month <= 12


# === Text Extraction ===

def read_text_from_bytes(file_bytes: bytes, encoding: str = "utf-8") -> str:
    """Extract text from text file bytes"""
    try:
        return file_bytes.decode(encoding, errors="ignore")
    except Exception as e:
        raise ValueError(f"Failed to read text file: {str(e)}")


def read_pdf_from_bytes(file_bytes: bytes) -> str:
    """Extract text from PDF file bytes"""
    if pypdf is None:
        raise RuntimeError("pypdf not installed. Install with: pip install pypdf")
    
    try:
        pdf_file = io.BytesIO(file_bytes)
        reader = pypdf.PdfReader(pdf_file)
        texts = []
        
        for page in reader.pages:
            try:
                text = page.extract_text()
                if text:
                    texts.append(text)
            except Exception:
                continue
        
        if not texts:
            raise ValueError("No text could be extracted from PDF")
        
        return "\n".join(texts)
    except Exception as e:
        raise ValueError(f"Failed to read PDF: {str(e)}")


def extract_text_from_upload(file_bytes: bytes, filename: str) -> str:
    """
    Extract text from uploaded file (PDF, TXT, or MD).
    
    Args:
        file_bytes: Raw file bytes
        filename: Filename (used to determine file type)
        
    Returns:
        Extracted text
        
    Raises:
        ValueError: If file type not supported or extraction fails
    """
    suffix = Path(filename).suffix.lower()
    
    if suffix in [".txt", ".md"]:
        return read_text_from_bytes(file_bytes)
    elif suffix == ".pdf":
        return read_pdf_from_bytes(file_bytes)
    else:
        raise ValueError(f"Unsupported file type: {suffix}. Supported: .pdf, .txt, .md")


# === URL Scraping ===

def scrape_url_text(url: str) -> str:
    """
    Scrape text content from a web URL.

    Args:
        url: The URL to scrape

    Returns:
        Extracted text content

    Raises:
        ValueError: If scraping fails
        RuntimeError: If required packages not installed
    """
    try:
        import requests
        from bs4 import BeautifulSoup
    except ImportError:
        raise RuntimeError(
            "requests and beautifulsoup4 are required for URL scraping. "
            "Install with: pip install requests beautifulsoup4"
        )

    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (compatible; InflationPredictor/1.0)'
        }
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')

        # Remove non-content elements
        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
            tag.decompose()

        text = soup.get_text(separator='\n', strip=True)
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        cleaned = '\n'.join(lines)

        if not cleaned:
            raise ValueError("No text content found on the page")

        return cleaned

    except Exception as e:
        if "requests" in type(e).__module__:
            raise ValueError(f"Failed to fetch URL: {str(e)}")
        raise


# === LLM Event Categorization ===

SYSTEM_PROMPT = (
    "You are a quantitative economic analyst specializing in Sri Lankan inflation dynamics.\n"
    "Your job is to read central bank reports or news excerpts and extract events that drive CORE INFLATION "
    "(underlying price trends, excluding volatile food and energy).\n\n"
    "INTENSITY SCORING — use the FULL [-1, +1] range. Be bold and specific:\n"
    "  +0.9 to +1.0 : Emergency rate cut, massive fiscal stimulus, severe currency collapse, hyperinflationary shock, A war\n"
    "  +0.6 to +0.8 : Significant rate cut (≥100 bps), large government deficit spending, major currency depreciation\n"
    "  +0.3 to +0.5 : Moderate rate cut (25–75 bps), moderate fiscal expansion, supply bottleneck\n"
    "  +0.1 to +0.2 : Minor policy easing, small subsidy increase, mild demand pickup\n"
    "   0.0         : Neutral/no material impact on core inflation\n"
    "  -0.1 to -0.2 : Minor tightening, small subsidy cut, mild demand softening\n"
    "  -0.3 to -0.5 : Moderate rate hike (25–75 bps), fiscal consolidation, import surge easing prices\n"
    "  -0.6 to -0.8 : Significant rate hike (≥100 bps), large fiscal austerity, strong currency appreciation\n"
    "  -0.9 to -1.0 : Emergency rate hike, severe demand destruction, deflationary collapse\n\n"
    "RULES:\n"
    "- Do NOT cluster scores around 0.3–0.5. Differentiate strongly between weak and strong events.\n"
    "- Assign NEGATIVE scores for deflationary/tightening events — do not default to positive.\n"
    "- Extract ALL distinct events; a document may have 1 or many.\n"
    "- Respond ONLY with the required JSON schema."
)

USER_PROMPT_TEMPLATE = (
    "Analyze the document below and extract every economic event affecting CORE INFLATION in Sri Lanka.\n\n"
    "Return a JSON object with this exact schema:\n"
    "{{\n"
    "  \"events\": [\n"
    "    {{\n"
    "      \"category\": \"<one of: Monetary Policy | Fiscal Policy | External and Global Shocks | Supply and Demand Shocks>\",\n"
    "      \"description\": \"<concise event description, max 20 words>\",\n"
    "      \"intensity\": <float from -1.0 to +1.0 — use the full range, calibrated to the scoring guide>,\n"
    "      \"reasoning\": \"<why this intensity? cite specific numbers or facts from the document>\"\n"
    "    }}\n"
    "  ]\n"
    "}}\n\n"
    "IMPORTANT: Use the FULL intensity range. A rate hike of 200 bps should be around -0.8, not -0.3. "
    "A major currency depreciation should be +0.7 or higher. Do not bunch scores near zero.\n\n"
    "Document:\n{doc_text}\n"
)


def clamp_intensity(value: float) -> float:
    """Clamp intensity to [-1, 1] range"""
    return max(-1.0, min(1.0, float(value)))


def parse_llm_output(llm_text: str) -> List[Event]:
    """
    Parse LLM JSON output to extract events.
    Includes JSON and regex fallback parsing.
    
    Args:
        llm_text: Raw text from LLM response
        
    Returns:
        List of extracted Event objects
    """
    events = []
    
    try:
        # Try to find JSON in text
        json_start = llm_text.find('{')
        json_end = llm_text.rfind('}') + 1
        
        if json_start != -1 and json_end > json_start:
            json_text = llm_text[json_start:json_end]
            data = json.loads(json_text)
        else:
            data = json.loads(llm_text)
        
        if isinstance(data, dict) and "events" in data:
            for event_data in data["events"]:
                if isinstance(event_data, dict):
                    event = Event(
                        category=event_data.get("category", ""),
                        description=event_data.get("description", ""),
                        intensity=clamp_intensity(event_data.get("intensity", 0.0)),
                        reasoning=event_data.get("reasoning", "")
                    )
                    events.append(event)
    
    except (json.JSONDecodeError, Exception):
        # Fallback: regex extraction
        category_matches = re.findall(r'"category":\s*"([^"]*)"', llm_text)
        description_matches = re.findall(r'"description":\s*"([^"]*)"', llm_text)
        intensity_matches = re.findall(r'"intensity":\s*([+-]?\d*\.?\d+)', llm_text)
        
        for i in range(min(len(category_matches), len(description_matches), len(intensity_matches))):
            event = Event(
                category=category_matches[i],
                description=description_matches[i],
                intensity=clamp_intensity(float(intensity_matches[i])),
                reasoning="Extracted via regex fallback"
            )
            events.append(event)
    
    return events


def categorize_batch_with_llm(
    llm: "_OpenAI | None",
    docs: List[Tuple[str, str]]   # list of (label, text)
) -> Dict[str, Tuple[List[Event], str]]:
    """
    Send all documents in a single OpenAI call and return events per label.

    Args:
        llm: openai.OpenAI client instance
        docs: list of (label, text) — label is e.g. "2026-02"

    Returns:
        dict mapping label -> (List[Event], raw_json_string)
    """
    if llm is None or not docs:
        return {}

    # Build one user message containing all docs
    parts = []
    for label, text in docs:
        parts.append(f"=== DOCUMENT: {label} ===\n{text[:3000]}")
    combined = "\n\n".join(parts)

    batch_prompt = (
        "You will receive multiple documents, each labeled '=== DOCUMENT: <id> ==='.\n"
        "For EACH document, identify economic events affecting CORE INFLATION.\n"
        "Return a single JSON object where each key is the document id and the value "
        "follows this schema:\n"
        "{\n"
        "  \"<doc_id>\": {\n"
        "    \"events\": [\n"
        "      {\n"
        "        \"category\": string,  // one of: Monetary Policy, Fiscal Policy, "
        "External and Global Shocks, Supply and Demand Shocks\n"
        "        \"description\": string,\n"
        "        \"intensity\": number,  // -1 to +1\n"
        "        \"reasoning\": string\n"
        "      }\n"
        "    ]\n"
        "  }\n"
        "}\n\n"
        f"Documents:\n{combined}"
    )

    response = llm.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": batch_prompt},
        ],
        temperature=0,
    )
    raw = response.choices[0].message.content or ""

    # Parse top-level JSON
    results: Dict[str, Tuple[List[Event], str]] = {}
    try:
        start = raw.find("{")
        end = raw.rfind("}") + 1
        data = json.loads(raw[start:end])
        for label, payload in data.items():
            events = parse_llm_output(json.dumps(payload))
            results[label] = (events, json.dumps(payload))
    except Exception:
        # Fallback: try parsing as if it were a single-doc response for each label
        for label, _ in docs:
            results[label] = ([], raw)

    return results


# === Main Processing Pipeline ===

def process_uploaded_file(
    file_bytes: bytes,
    filename: str,
    llm: "_OpenAI | None" = None,
    skip_llm: bool = False
) -> Dict:
    """
    Complete processing pipeline for a single uploaded file.
    
    Args:
        file_bytes: Raw file bytes
        filename: Uploaded filename
        llm: ChatOpenAI instance (required if skip_llm=False)
        skip_llm: If True, skip LLM classification (testing mode)
        
    Returns:
        Dictionary with results:
        {
            'metadata': DocumentMetadata,
            'text': extracted text,
            'events': List[Event],
            'llm_output': raw LLM response,
            'success': bool,
            'error': error message if any
        }
    """
    result = {
        'metadata': None,
        'text': '',
        'events': [],
        'llm_output': '',
        'success': False,
        'error': ''
    }
    
    try:
        # === Parse filename ===
        year, month, parse_success = parse_filename_to_date(filename)
        
        metadata = DocumentMetadata(
            filename=filename,
            parsed_year=year,
            parsed_month=month,
            parse_success=parse_success and validate_date(year, month),
            error_message="" if (parse_success and validate_date(year, month)) 
                         else f"Could not parse YYYY-MM date from filename"
        )
        result['metadata'] = metadata
        
        if not metadata.parse_success:
            result['error'] = metadata.error_message
            # Continue anyway; use for LLM classification
        
        # === Extract text ===
        try:
            text = extract_text_from_upload(file_bytes, filename)
            result['text'] = text
        except Exception as e:
            result['error'] = f"Text extraction failed: {str(e)}"
            return result
        
        # === Call LLM (if enabled) ===
        if not skip_llm:
            if llm is None:
                result['error'] = "LLM instance not provided"
                return result
            
            try:
                events, llm_output = categorize_text_with_llm(llm, text)
                result['events'] = events
                result['llm_output'] = llm_output
            except Exception as e:
                result['error'] = f"LLM categorization failed: {str(e)}"
                return result
        
        result['success'] = True
        return result
    
    except Exception as e:
        result['error'] = f"Unexpected error: {str(e)}"
        return result


def process_batch_uploads(
    uploaded_files: List[Tuple[bytes, str]],
    llm: "_OpenAI | None" = None,
    skip_llm: bool = False,
    progress_callback = None
) -> Tuple[EventAggregation, List[Dict]]:
    """
    Process multiple uploaded files and aggregate results.
    
    Args:
        uploaded_files: List of (file_bytes, filename) tuples
        llm: ChatOpenAI instance
        skip_llm: Skip LLM classification
        progress_callback: Optional callback for progress updates
        
    Returns:
        Tuple of (EventAggregation, List of processing results)
    """
    aggregation = EventAggregation()
    n = len(uploaded_files)
    results = []

    # Step 1 — extract text from all files (fast, local)
    if progress_callback:
        progress_callback(0, n, "Extracting text from documents...")

    for file_bytes, filename in uploaded_files:
        year, month, parse_success = parse_filename_to_date(filename)
        metadata = DocumentMetadata(
            filename=filename,
            parsed_year=year,
            parsed_month=month,
            parse_success=parse_success and validate_date(year, month),
            error_message="" if (parse_success and validate_date(year, month))
                         else "Could not parse YYYY-MM date from filename"
        )
        result = {'metadata': metadata, 'text': '', 'events': [], 'llm_output': '', 'success': False, 'error': ''}
        try:
            result['text'] = extract_text_from_upload(file_bytes, filename)
        except Exception as e:
            result['error'] = f"Text extraction failed: {str(e)}"
        results.append(result)

    # Step 2 — one LLM call for all documents
    if not skip_llm and llm is not None:
        if progress_callback:
            progress_callback(0, n, f"Sending {n} document(s) to AI in one request...")

        docs_for_llm = []
        for r in results:
            if r['text'] and r['metadata'].parse_success:
                label = f"{r['metadata'].parsed_year}-{str(r['metadata'].parsed_month).zfill(2)}"
                docs_for_llm.append((label, r['text']))

        if docs_for_llm:
            batch_results = categorize_batch_with_llm(llm, docs_for_llm)
            for r in results:
                if r['metadata'].parse_success:
                    label = f"{r['metadata'].parsed_year}-{str(r['metadata'].parsed_month).zfill(2)}"
                    if label in batch_results:
                        r['events'], r['llm_output'] = batch_results[label]
                        r['success'] = True
                    elif r['text']:
                        r['success'] = True  # text extracted, just no events found
                elif r['text']:
                    r['success'] = True
    elif not skip_llm and llm is None:
        for r in results:
            r['error'] = "LLM instance not provided"
    else:
        for r in results:
            if r['text']:
                r['success'] = True

    # Step 3 — aggregate events
    for r in results:
        if r['success'] and r['metadata'].parse_success and r['events']:
            for event in r['events']:
                aggregation.add_event(r['metadata'].parsed_year, r['metadata'].parsed_month, event)

    if progress_callback:
        progress_callback(n, n, "Done")

    return aggregation, results


if __name__ == "__main__":
    # Example usage / testing
    import sys
    
    # Test filename parsing
    test_filenames = [
        "2025-10_article.pdf",
        "2021-05_report.txt",
        "article_2024-03.md",
        "invalid_file.pdf"
    ]
    
    print("=== Filename Parsing Test ===")
    for fname in test_filenames:
        year, month, success = parse_filename_to_date(fname)
        print(f"{fname:30} → Year: {year}, Month: {month}, Success: {success}")
