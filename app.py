import os
import io
import re
import json
import time
import ssl
import httpx
import certifi
import socket
import random
import logging
import requests
from pathlib import Path
from typing import Optional, Dict, List
from datetime import datetime, timedelta
from contextlib import asynccontextmanager
from urllib.parse import urlparse

from fastapi import FastAPI, Form, UploadFile, File, HTTPException, Request, Depends
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

# Anthropic SDK
from anthropic import Anthropic, APIStatusError, APITimeoutError, RateLimitError, APIConnectionError

# PDF extraction imports
try:
    import pypdf
    PDF_EXTRACTION_AVAILABLE = True
except ImportError:
    try:
        from PyPDF2 import PdfReader as PyPDF2Reader
        PDF_EXTRACTION_AVAILABLE = True
    except ImportError:
        PDF_EXTRACTION_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("pipewrench")

# === URL WHITELIST CONFIGURATION ===
WHITELIST_URL = "https://raw.githubusercontent.com/rmkenv/pipewrench_mvp/main/custom_whitelist.json"
URL_REGEX = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')

EMBEDDED_WHITELIST = [
    {"url": "https://www.epa.gov", "description": "EPA Regulations"},
    {"url": "https://www.osha.gov", "description": "OSHA Standards"},
    {"url": "https://www.fhwa.dot.gov", "description": "FHWA Standards"},
    {"url": "https://www.awwa.org", "description": "Water Standards"},
    {"url": "https://www.apwa.net", "description": "APWA Resources"},
    {"url": "https://www.asce.org", "description": "ASCE Standards"},
]
whitelist_urls: List[str] = []

def fetch_whitelist():
    global whitelist_urls
    try:
        logger.info(f"Fetching whitelist from {WHITELIST_URL}...")
        response = requests.get(WHITELIST_URL, timeout=15)
        response.raise_for_status()
        data = response.json()
        whitelist_urls = [entry["url"] for entry in data if "url" in entry]
        logger.info(f"✅ Loaded {len(whitelist_urls)} URLs from external whitelist")
    except Exception as e:
        logger.warning(f"⚠️ Failed to fetch external whitelist: {e}")
        whitelist_urls = [entry["url"] for entry in EMBEDDED_WHITELIST]
        logger.info(f"✅ Using embedded whitelist with {len(whitelist_urls)} URLs")

def get_whitelisted_domains():
    domains = set()
    for url in whitelist_urls:
        parsed = urlparse(url)
        if parsed.netloc:
            domains.add(parsed.netloc)
    return domains

def get_total_whitelisted_urls():
    return len(whitelist_urls)

def is_url_whitelisted(url: str) -> bool:
    try:
        parsed = urlparse(url)
        for whitelisted_url in whitelist_urls:
            whitelisted_parsed = urlparse(whitelisted_url)
            if (parsed.netloc == whitelisted_parsed.netloc and
                parsed.path.startswith(whitelisted_parsed.path)):
                return True
    except Exception:
        return False
    return False

# === ENV CONFIG ===
DRAWING_PROCESSING_API_URL = os.getenv("DRAWING_PROCESSING_API_URL", "http://localhost:8001/parse")
SESSION_EXPIRY_HOURS = int(os.getenv("SESSION_EXPIRY_HOURS", "24"))
DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower() == "true"

ENVIRONMENT = os.getenv("ENVIRONMENT", "production")
IS_RENDER = bool(os.getenv("RENDER", ""))  # kept for compatibility
# Anthropic model configuration
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-latest")

# === APPLICATION STATE ===
class AppState:
    def __init__(self):
        self.anthropic_client: Optional[Anthropic] = None
        self.session_manager: Optional['SessionManager'] = None
        self.http_client: Optional[httpx.Client] = None
app_state = AppState()

# === LIFESPAN ===
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=" * 70)
    logger.info("PipeWrench AI - Municipal DPW Knowledge Capture System (Anthropic/Claude)")
    logger.info(f"Environment: {ENVIRONMENT}")
    logger.info(f"Debug Mode: {DEBUG_MODE}")
    logger.info("=" * 70)

    # Session manager
    app_state.session_manager = SessionManager()
    logger.info("✅ Session manager initialized")

    # PDF extraction availability
    if not PDF_EXTRACTION_AVAILABLE:
        logger.warning("⚠️ PDF extraction library not available")
    else:
        logger.info("✅ PDF extraction library available")

    # Fetch whitelist
    fetch_whitelist()
    logger.info(f"✅ Whitelisted URLs: {get_total_whitelisted_urls()}")

    # Configuration info
    logger.info(f"✅ Departments: {len(DEPARTMENT_PROMPTS)}")
    logger.info(f"✅ Job Roles: {len(JOB_ROLES)}")
    logger.info(f"✅ Session Expiry: {SESSION_EXPIRY_HOURS} hours")

    # DNS basic sanity (optional)
    try:
        ip = socket.gethostbyname("anthropic.com")
        logger.info(f"✅ DNS Resolution: anthropic.com -> {ip}")
    except Exception as e:
        logger.warning(f"DNS resolution check failed: {e}")

    # HTTP client
    try:
        timeout_config = httpx.Timeout(connect=90.0, read=240.0, write=90.0, pool=60.0)
        limits_config = httpx.Limits(max_connections=50, max_keepalive_connections=10, keepalive_expiry=30.0)
        try:
            app_state.http_client = httpx.Client(
                timeout=timeout_config,
                limits=limits_config,
                verify=certifi.where(),
                http2=False,
                follow_redirects=True,
                transport=httpx.HTTPTransport(retries=5, verify=certifi.where())
            )
        except Exception as ssl_error:
            logger.warning(f"⚠️ SSL verification failed for httpx: {ssl_error}")
            app_state.http_client = httpx.Client(
                timeout=timeout_config,
                limits=limits_config,
                verify=False,
                http2=False,
                follow_redirects=True
            )
            logger.warning("⚠️ HTTP client (httpx) running WITHOUT SSL verification")
    except Exception as e:
        logger.warning(f"Failed to init http client: {e}")

    # Anthropic client
    if ANTHROPIC_API_KEY:
        try:
            app_state.anthropic_client = Anthropic(api_key=ANTHROPIC_API_KEY)
            logger.info("✅ Anthropic client initialized")
            logger.info(f"    Model: {ANTHROPIC_MODEL}")
            if DEBUG_MODE:
                try:
                    logger.info("Testing Anthropic API connection...")
                    test = app_state.anthropic_client.messages.create(
                        model=ANTHROPIC_MODEL,
                        max_tokens=32,
                        messages=[{"role":"user","content":"test"}],
                        timeout=30
                    )
                    if test and getattr(test, "id", None):
                        logger.info("✅ Anthropic API connection verified!")
                except Exception as test_error:
                    logger.warning(f"⚠️ Startup API test failed: {type(test_error).__name__}: {test_error}")
        except Exception as e:
            logger.error(f"❌ Anthropic client initialization failed: {e}", exc_info=True)
            app_state.anthropic_client = None
    else:
        logger.warning("⚠️ ANTHROPIC_API_KEY not found - running in DEMO MODE")
        app_state.anthropic_client = None

    logger.info("=" * 70)
    logger.info("🚀 Application startup complete")
    logger.info("=" * 70)
    yield
    logger.info("Application shutting down...")
    if app_state.http_client:
        app_state.http_client.close()
        logger.info("✅ HTTP client closed")

# === CREATE APP ===
app = FastAPI(
    title="PipeWrench AI",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs" if DEBUG_MODE else None,
    redoc_url="/redoc" if DEBUG_MODE else None
)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
)

# Static & templates
static_dir = Path("static")
templates_dir = Path("templates")
if static_dir.exists():
    app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates") if templates_dir.exists() else None

# === JOB ROLES ===
JOB_ROLES = {
    "general": {
        "name": "General DPW Staff",
        "context": "You are assisting general Department of Public Works staff with municipal infrastructure questions."
    },
    "director": {
        "name": "DPW Director",
        "context": "You are assisting a DPW Director with strategic planning, policy decisions, and departmental oversight."
    },
    "engineer": {
        "name": "Civil Engineer",
        "context": "You are assisting a licensed civil engineer with technical engineering standards, design specifications, and compliance requirements."
    },
    "project_manager": {
        "name": "Project Manager",
        "context": "You are assisting a project manager with construction management, scheduling, budgeting, and contractor coordination."
    },
    "inspector": {
        "name": "Construction Inspector",
        "context": "You are assisting a construction inspector with field inspection procedures, quality control, and compliance verification."
    },
    "maintenance": {
        "name": "Maintenance Supervisor",
        "context": "You are assisting a maintenance supervisor with asset management, preventive maintenance, and repair operations."
    },
    "environmental": {
        "name": "Environmental Compliance Officer",
        "context": "You are assisting an environmental compliance officer with EPA regulations, stormwater management, and environmental permits."
    },
    "safety": {
        "name": "Safety Officer",
        "context": "You are assisting a safety officer with OSHA compliance, workplace safety, and accident prevention."
    }
}

# === DEPARTMENTS ===
DEPARTMENT_PROMPTS = {
    "general_public_works": {
        "name": "General Public Works",
        "prompt": """You are a specialized AI assistant for Municipal Public Works departments. 
You help preserve institutional knowledge and provide accurate, cited information from approved sources."""
    },
    "water_sewer": {
        "name": "Water & Sewer",
        "prompt": """You are a specialized AI assistant for Water & Sewer departments. 
You provide expert guidance on water distribution, wastewater treatment, and utility infrastructure."""
    },
    "streets_highways": {
        "name": "Streets & Highways",
        "prompt": """You are a specialized AI assistant for Streets & Highways departments.
You provide guidance on road maintenance, traffic management, and transportation infrastructure."""
    },
    "environmental": {
        "name": "Environmental Compliance",
        "prompt": """You are a specialized AI assistant for Environmental Compliance.
You help with EPA regulations, stormwater management, and environmental permitting."""
    },
    "safety": {
        "name": "Safety & Training",
        "prompt": """You are a specialized AI assistant for Safety & Training.
You provide guidance on OSHA compliance, workplace safety, and training programs."""
    },
    "administration": {
        "name": "Administration & Planning",
        "prompt": """You are a specialized AI assistant for DPW Administration & Planning.
You help with budgeting, project planning, and departmental management."""
    }
}

# === DEPENDENCIES ===
def get_anthropic_client() -> Optional[Anthropic]:
    return app_state.anthropic_client

def get_session_manager() -> 'SessionManager':
    if app_state.session_manager is None:
        raise HTTPException(status_code=500, detail="Session manager not initialized")
    return app_state.session_manager

# === HELPERS ===
def get_role_info(role_key: str):
    role = JOB_ROLES.get(role_key)
    if role:
        return {"title": role["name"], "focus_areas": ["General DPW operations"]}
    return None

def build_system_prompt(department_key: str, role_key: Optional[str]) -> str:
    base = DEPARTMENT_PROMPTS.get(department_key, DEPARTMENT_PROMPTS["general_public_works"]).get("prompt", "")
    role_txt = ""
    if role_key:
        role = get_role_info(role_key)
        if role:
            areas = role.get("focus_areas", [])
            role_txt = (
                f"\n\nROLE CONTEXT:\n- Title: {role.get('title', role_key)}\n- Focus Areas:\n"
                + "\n".join(f"  - {a}" for a in areas)
            )
    whitelist_notice = (
        f"\n\nURL RESTRICTIONS:\n"
        f"- Only cite and reference sources from approved whitelist\n"
        f"- Include the specific URL for each citation\n"
        f"- If info is not in whitelist, clearly state that it cannot be verified from approved sources\n"
        f"- All child pages of whitelisted URLs are permitted\n"
        f"- Total Whitelisted URLs: {get_total_whitelisted_urls()}\n"
        f"- Approved Domains: {', '.join(sorted(list(get_whitelisted_domains()))[:25])}"
        + ("..." if len(get_whitelisted_domains()) > 25 else "")
    )
    return base + role_txt + whitelist_notice

def extract_text_from_pdf(content: bytes) -> str:
    if not PDF_EXTRACTION_AVAILABLE:
        return "[ERROR: PDF extraction library not installed. Install pypdf or PyPDF2.]"
    try:
        try:
            import pypdf
            pdf_reader = pypdf.PdfReader(io.BytesIO(content))
            text = ""
            for page_num, page in enumerate(pdf_reader.pages):
                page_text = page.extract_text()
                if page_text:
                    text += f"\n--- Page {page_num + 1} ---\n{page_text}"
            if text.strip():
                logger.info(f"Extracted {len(text)} characters from PDF using pypdf")
                return text
            else:
                return "[PDF appears to be empty or contains only images]"
        except Exception:
            from PyPDF2 import PdfReader as PyPDF2Reader
            pdf_reader = PyPDF2Reader(io.BytesIO(content))
            text = ""
            for page_num, page in enumerate(pdf_reader.pages):
                page_text = page.extract_text()
                if page_text:
                    text += f"\n--- Page {page_num + 1} ---\n{page_text}"
            if text.strip():
                logger.info(f"Extracted {len(text)} characters from PDF using PyPDF2")
                return text
            else:
                return "[PDF appears to be empty or contains only images]"
    except Exception as e:
        logger.error(f"Error extracting PDF text: {e}", exc_info=True)
        return f"[Error extracting PDF text: {str(e)}]"

def generate_mock_response(query: str, context: str, system_prompt: str, has_document: bool) -> str:
    department_match = re.search(r'You are a specialized AI assistant for ([\w\s&]+)', system_prompt)
    department_name = department_match.group(1).strip() if department_match else 'N/A'
    mock_response = """[DEMO MODE - Anthropic API key not configured]

Your question: {query}

This is a demonstration response. To get real AI-powered answers:
1. Set the ANTHROPIC_API_KEY environment variable.
2. Restart the application.

Configuration:
- Department: {department_name}
- Document uploaded: {has_document_status} (Preview: {context_preview}...)

*All functionality is ready; needs API key.*"""
    return mock_response.format(
        query=query,
        department_name=department_name,
        has_document_status='Yes' if has_document else 'No',
        context_preview=context[:50]
    )

def enforce_whitelist_on_text(text: str) -> str:
    if not text:
        return text
    bad_urls = []
    for url in set(URL_REGEX.findall(text)):
        url_clean = url.rstrip('.,);]')
        if not is_url_whitelisted(url_clean):
            bad_urls.append(url_clean)
    if not bad_urls:
        return text
    note = "\n\n[COMPLIANCE NOTICE]\n" \
           "The following URLs are not in the approved whitelist and must not be cited:\n" + \
           "\n".join(f"- {u}" for u in sorted(bad_urls)) + \
           "\n\nPlease revise citations to use only approved sources."
    return text + note

def sanitize_html(text: str) -> str:
    if not text:
        return ""
    return (text
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&#x27;")
            .replace("/", "&#x2F;"))

# === SESSION MANAGEMENT ===
class SessionManager:
    def __init__(self):
        self.sessions: Dict[str, Dict] = {}

    def cleanup_expired_sessions(self):
        now = datetime.now()
        expired = []
        for session_id, session_data in list(self.sessions.items()):
            created_at = datetime.fromisoformat(session_data.get("created_at", now.isoformat()))
            if now - created_at > timedelta(hours=SESSION_EXPIRY_HOURS):
                expired.append(session_id)
        for session_id in expired:
            del self.sessions[session_id]

    def get_session(self, session_id: str) -> Optional[Dict]:
        self.cleanup_expired_sessions()
        return self.sessions.get(session_id)

    def create_session(self, session_id: str, data: Dict) -> None:
        self.sessions[session_id] = {
            **data,
            "created_at": datetime.now().isoformat(),
            "document_context": "",
            "documents": [],
            "questions": []
        }
        logger.info(f"Created session: {session_id}")

    def update_session(self, session_id: str, updates: Dict) -> None:
        if session_id in self.sessions:
            self.sessions[session_id].update(updates)

    def get_session_count(self) -> int:
        self.cleanup_expired_sessions
        return len(self.sessions)

# === MODELS ===
class QueryRequest(BaseModel):
    session_id: Optional[str] = None
    query: str
    role: Optional[str] = None
    department: Optional[str] = "general_public_works"

class UploadResponse(BaseModel):
    session_id: str
    filename: str
    pages: int
    message: str
    is_asbuilt: bool = False

class SystemInfoResponse(BaseModel):
    total_whitelisted_urls: int
    whitelisted_domains: List[str]
    roles: List[str]
    departments: List[str]
    config: Dict

# === DIAGNOSTICS ===
@app.get("/api/test-connection")
async def test_connection():
    results = {}
    # Basic Anthropic check
    try:
        if ANTHROPIC_API_KEY:
            test_client = Anthropic(api_key=ANTHROPIC_API_KEY)
            results["anthropic_init"] = "✅ Client created"
            try:
                r = test_client.messages.create(
                    model=ANTHROPIC_MODEL,
                    max_tokens=16,
                    messages=[{"role":"user","content":"hi"}],
                    timeout=30
                )
                results["anthropic_api_call"] = "✅ API call successful" if getattr(r, "id", None) else "❌ Empty response"
            except Exception as e:
                results["anthropic_api_call"] = f"❌ Failed: {str(e)[:200]}"
        else:
            results["anthropic_init"] = "❌ No API key"
    except Exception as e:
        results["anthropic_init"] = f"❌ Failed: {str(e)[:200]}"

    return {
        "timestamp": datetime.now().isoformat(),
        "environment": ENVIRONMENT,
        "debug_mode": DEBUG_MODE,
        "diagnostics": results
    }

# === CORE: LLM RESPONSE (Anthropic) ===
def generate_llm_response(
    query: str,
    context: str,
    system_prompt: str,
    has_document: bool,
    anthropic_client: Optional[Anthropic]
) -> str:
    if not anthropic_client:
        return generate_mock_response(query, context, system_prompt, has_document)

    max_retries = 5
    base_delay = 5.0

    user_text_parts = [f"User query: {query}"]
    if context:
        user_text_parts.append(f"\n\nDOCUMENT CONTEXT (for RAG/citation only): {context[:8000]}")
    else:
        user_text_parts.append("\n\nNo document uploaded")
    user_text = "\n".join(user_text_parts)

    for attempt in range(max_retries):
        try:
            logger.info(f"🔄 Anthropic API call attempt {attempt + 1}/{max_retries}")
            resp = anthropic_client.messages.create(
                model=ANTHROPIC_MODEL,
                max_tokens=8192,
                temperature=0.7,
                system=system_prompt,
                messages=[{"role": "user", "content": [{"type": "text", "text": user_text}]}],
                timeout=240
            )
            if resp and resp.content:
                text = "".join(getattr(part, "text", "") for part in resp.content)
                if text.strip():
                    logger.info(f"✅ Anthropic API call successful on attempt {attempt + 1}")
                    return text
            raise HTTPException(status_code=500, detail="Empty response from LLM.")
        except (APITimeoutError, APIConnectionError) as e:
            retryable = True
            detail = "Connection/timeout"
        except RateLimitError as e:
            retryable = True
            detail = "Rate limit"
        except APIStatusError as e:
            status = getattr(e, "status_code", None)
            retryable = status in (500, 502, 503, 504)
            detail = f"API status {status}"
        except Exception as e:
            retryable = attempt == 0
            detail = f"Unexpected: {type(e).__name__}"

        if not retryable or attempt >= max_retries - 1:
            raise HTTPException(status_code=503, detail=f"LLM error: {detail}")

        delay = base_delay * (2 ** attempt)
        total_delay = delay + random.uniform(delay * 0.1, delay * 0.3)
        logger.info(f"    ⏳ Retrying in {total_delay:.1f}s... ({detail})")
        time.sleep(total_delay)

    raise HTTPException(status_code=503, detail="Service temporarily unavailable after retries.")

# === API: Query ===
@app.post("/api/query")
async def query_endpoint(
    request_data: QueryRequest,
    session_manager: 'SessionManager' = Depends(get_session_manager),
    anthropic_client: Optional[Anthropic] = Depends(get_anthropic_client)
):
    session_id = request_data.session_id
    query = request_data.query
    role = request_data.role or "general"
    department = request_data.department or "general_public_works"

    if not session_id:
        session_id = f"temp-{random.randint(1000, 9999)}"
        session_manager.create_session(session_id, {"role": role, "department": department})
        logger.info(f"No session ID provided, created temporary session: {session_id}")

    session = session_manager.get_session(session_id)
    if not session:
        session_manager.create_session(session_id, {"role": role, "department": department})
        session = session_manager.get_session(session_id)

    document_context = session.get("document_context", "")
    has_document = bool(document_context)
    system_prompt = build_system_prompt(department, role)

    try:
        llm_response = generate_llm_response(
            query=query,
            context=document_context,
            system_prompt=system_prompt,
            has_document=has_document,
            anthropic_client=anthropic_client
        )
    except HTTPException as e:
        logger.error(f"Error during LLM generation: {e.detail}")
        raise e

    final_response = enforce_whitelist_on_text(llm_response)

    session.get("questions", []).append(query)
    session_manager.update_session(session_id, {"questions": session.get("questions")})

    return {"response": final_response, "session_id": session_id, "model_used": ANTHROPIC_MODEL}

# === API: Upload PDF ===
@app.post("/api/upload", response_model=UploadResponse)
async def upload_file(
    session_id: str = Form(...),
    department: str = Form(...),
    role: str = Form(...),
    file: UploadFile = File(...),
    session_manager: 'SessionManager' = Depends(get_session_manager)
):
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Unsupported file type. Only PDF files are supported.")

    session = session_manager.get_session(session_id)
    if not session:
        session_manager.create_session(session_id, {"role": role, "department": department})
        session = session_manager.get_session(session_id)

    try:
        contents = await file.read()
    except Exception as e:
        logger.error(f"Error reading file: {e}")
        raise HTTPException(status_code=500, detail="Failed to read uploaded file.")

    extracted_text = extract_text_from_pdf(contents)
    if "[Error" in extracted_text or "[ERROR" in extracted_text:
        raise HTTPException(status_code=500, detail=f"Failed to process PDF: {extracted_text}")

    page_count = 0
    if extracted_text.startswith("--- Page 1 ---"):
        page_count = extracted_text.count("--- Page")

    session_manager.update_session(
        session_id,
        {"document_context": extracted_text, "documents": [file.filename]}
    )
    logger.info(f"Stored {len(extracted_text)} chars from '{file.filename}' in session {session_id}")

    return UploadResponse(
        session_id=session_id,
        filename=file.filename,
        pages=page_count,
        message=f"Successfully extracted {page_count} pages and stored context for RAG.",
        is_asbuilt=False
    )

# === Root (UI) ===
@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    if templates is None:
        raise HTTPException(status_code=500, detail="Jinja2Templates directory 'templates' not found.")
    departments = [{"value": k, "name": v["name"]} for k, v in DEPARTMENT_PROMPTS.items()]
    roles = [{"value": k, "name": v["name"]} for k, v in JOB_ROLES.items()]
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "departments": departments,
            "roles": roles,
            "is_demo_mode": ANTHROPIC_API_KEY is None,
            "model_name": ANTHROPIC_MODEL,
            "is_render": IS_RENDER,
        }
    )
