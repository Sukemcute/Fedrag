"""
Streamlit Chatbot UI for RAG with Privacy Protection
Multi-Topic Q&A Chatbot Interface with Federated Learning
"""
import streamlit as st
import requests
import json
from datetime import datetime
from typing import Optional, Dict, Any, List
import time
import html
import re
from translations import get_text, get_topic_examples


# Configuration
API_BASE_URL = "http://localhost:8000"
DEFAULT_SESSION_ID = "streamlit_session"

# Topic configurations
TOPIC_CONFIG = {
    "finance": {
        "icon": "💰",
        "color": "#0ea5e9",
        "retrievers": ["financial", "general"],
        "example_icon": "📊"
    },
    "healthcare": {
        "icon": "🏥",
        "color": "#10b981",
        "retrievers": ["general", "technical"],
        "example_icon": "🩺"
    },
    "legal": {
        "icon": "⚖️",
        "color": "#8b5cf6",
        "retrievers": ["legal", "general"],
        "example_icon": "📜"
    },
    "education": {
        "icon": "🎓",
        "color": "#f59e0b",
        "retrievers": ["general", "technical"],
        "example_icon": "📚"
    },
    "technology": {
        "icon": "💻",
        "color": "#3b82f6",
        "retrievers": ["technical", "general"],
        "example_icon": "⚡"
    },
    "general": {
        "icon": "🌍",
        "color": "#6b7280",
        "retrievers": ["general"],
        "example_icon": "💡"
    }
}


def _clean_html_text(text: str) -> str:
    """Remove HTML tags/artifacts and return cleaned plain text (may be empty)."""
    if not text:
        return ""
    
    raw = str(text)
    
    # Step 1: Remove ALL HTML tags (multiple passes to catch nested)
    for _ in range(3):  # Multiple passes for nested tags
        raw = re.sub(r'<[^>]*>', '', raw)
    
    # Step 2: Remove specific problematic patterns
    patterns_to_remove = [
        "</div>", "<div>", "<div ", "</div ", 
        "<div class=\"message-time\">", "<div class='message-time'>",
        "<div class=\"message-time\"", "<div class='message-time'",
        "<span>", "</span>", "<p>", "</p>",
        "<br>", "<br/>", "<br />",
    ]
    for pattern in patterns_to_remove:
        raw = raw.replace(pattern, "")
    
    # Step 3: Decode HTML entities
    raw = raw.replace("&lt;", "").replace("&gt;", "")
    raw = raw.replace("&amp;", "&").replace("&nbsp;", " ")
    raw = raw.replace("&#96;", "`")
    
    # Step 4: Try html.unescape to catch any remaining entities
    try:
        raw = html.unescape(raw)
    except:
        pass
    
    # Step 5: Clean whitespace
    raw = " ".join(raw.split())  # Normalize whitespace
    raw = raw.strip()
    
    return raw


def init_session_state():
    """Initialize session state variables"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "session_id" not in st.session_state:
        st.session_state.session_id = f"{DEFAULT_SESSION_ID}_{int(time.time())}"
    
    if "language" not in st.session_state:
        st.session_state.language = "vi"  # Default to Vietnamese
    
    if "topic" not in st.session_state:
        st.session_state.topic = "finance"
    
    if "thinking_state" not in st.session_state:
        st.session_state.thinking_state = None
    
    # AUTO-CLEAN: Force clean all existing messages on init
    if "messages" in st.session_state and st.session_state.messages:
        cleaned = []
        for m in st.session_state.messages:
            content = _clean_html_text(m.get("content", ""))
            if content and content not in ["</div>", "<div>", "div", "/div", "...", "."]:
                m["content"] = content
                cleaned.append(m)
        st.session_state.messages = cleaned


# Page configuration
st.set_page_config(
    page_title=get_text("page_title", st.session_state.get("language", "vi")),
    page_icon=get_text("page_icon", st.session_state.get("language", "vi")),
    layout="wide",
    initial_sidebar_state="expanded"
)


# CSS Styles (keeping the original beautiful design)
st.markdown("""
<style>
    /* Nền chung */
    html, body, [data-testid="stAppViewContainer"] {
        background-color: #f8fafc !important;
    }

    /* Container chính */
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 8rem !important;
        max-width: 900px !important;
        margin: 0 auto !important;
        background: #ffffff !important;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Sidebar background */
    [data-testid="stSidebar"] {
        background-color: #ffffff !important;
        border-right: 1px solid #e5e7eb !important;
    }

    /* Sidebar title */
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3 {
        color: #1f2937 !important;
        font-weight: 600 !important;
    }

    /* Normal text in sidebar */
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] span {
        color: #374151 !important;
        font-size: 0.95rem !important;
    }

    /* Status block */
    .sidebar-status-block {
        background: #e8fdf2 !important;
        color: #0f5132 !important;
        border: 1px solid #a7e3c3 !important;
        padding: 1rem !important;
        border-radius: 12px !important;
        margin-bottom: 0.7rem !important;
        font-weight: 600;
    }

    /* Info block */
    .sidebar-info-block {
        background: #e0f0ff !important;
        color: #1e3a8a !important;
        border: 1px solid #93c5fd !important;
        padding: 1rem !important;
        border-radius: 12px !important;
        margin-top: 0.8rem !important;
        font-weight: 600;
    }

    /* FL Community Card */
    .fl-community-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        padding: 1.2rem !important;
        border-radius: 12px !important;
        margin: 1rem 0 !important;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3) !important;
    }
    
    .fl-stat {
        display: flex;
        justify-content: space-between;
        margin: 0.5rem 0;
        font-size: 0.9rem;
    }
    
    .fl-stat-label {
        opacity: 0.9;
    }
    
    .fl-stat-value {
        font-weight: 700;
    }

    /* Thinking indicator */
    .thinking-container {
        background: linear-gradient(135deg, #fef3c7 0%, #fed7aa 100%);
        border: 2px solid #f59e0b;
        border-radius: 12px;
        padding: 1rem 1.5rem;
        margin: 1rem 0;
        animation: pulse 2s ease-in-out infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.8; }
    }
    
    .thinking-title {
        font-weight: 700;
        color: #92400e;
        margin-bottom: 0.5rem;
    }
    
    .thinking-state {
        color: #78350f;
        font-size: 0.95rem;
    }

    /* Privacy masked info */
    .masked-info-container {
        background: #fef3c7;
        border: 2px solid #fde047;
        border-radius: 12px;
        padding: 1rem;
        margin: 0.75rem 0;
    }
    
    .masked-comparison {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 1rem;
        margin-top: 0.75rem;
    }
    
    .masked-box {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 0.75rem;
    }
    
    .masked-label {
        font-weight: 600;
        font-size: 0.85rem;
        color: #6b7280;
        margin-bottom: 0.5rem;
    }
    
    .masked-text {
        font-size: 0.9rem;
        color: #1f2937;
        line-height: 1.5;
    }
    
    .masked-highlight {
        background: #fef3c7;
        padding: 0.1rem 0.3rem;
        border-radius: 4px;
        font-weight: 600;
    }

    /* Radio button container */
    .stRadio > div > label {
        background: #f9fafb !important;
        border: 2px solid #e5e7eb !important;
        border-radius: 12px !important;
        padding: 0.75rem 1rem !important;
        color: #1f2937 !important;
        transition: all 0.25s ease;
    }

    .stRadio > div > label:hover {
        border-color: #3b82f6 !important;
        background: #eff6ff !important;
    }

    input:checked + div {
        border: 2px solid #3b82f6 !important;
        background: #dbeafe !important;
    }

    /* Checkbox text */
    .stCheckbox > label {
        color: #1f2937 !important;
    }

    /* Section divider spacing */
    [data-testid="stSidebar"] hr {
        margin-top: 1.2rem !important;
        margin-bottom: 1.2rem !important;
        border-color: #e5e7eb !important;
    }

    /* Theme colors */
    :root {
        --primary: #0ea5e9;
        --primary-light: #e0f2fe;
        --primary-dark: #0284c7;
        --secondary: #06b6d4;
        --success: #3b82f6;
        --warning: #f59e0b;
        --danger: #ef4444;
        --neutral-50: #f9fafb;
        --neutral-100: #f3f4f6;
        --neutral-200: #e5e7eb;
        --neutral-300: #d1d5db;
        --neutral-600: #4b5563;
        --neutral-700: #374151;
        --neutral-800: #1f2937;
    }

    .main {
        background: #f8fafc !important;
    }
    
    /* Chat container */
    .chat-container {
        max-width: 900px;
        margin: 0 auto;
        padding: 0 1rem;
    }
    
    /* Message bubbles */
    .message {
        display: flex;
        gap: 1rem;
        margin-bottom: 1.5rem;
        animation: fadeIn 0.3s ease-in;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .message.user {
        flex-direction: row-reverse;
    }
    
    .message.user .message-content {
        align-items: flex-end !important;
    }
    
    .avatar {
        width: 40px;
        height: 40px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.3rem;
        flex-shrink: 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }
    
    .avatar.user {
        background: linear-gradient(135deg, #0ea5e9 0%, #06b6d4 100%);
        color: white;
    }
    
    .avatar.bot {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        color: white;
    }
    
    .message-content {
        flex: 1;
        max-width: 85%;
        display: flex;
        flex-direction: column;
    }
    
    .message-bubble {
        padding: 1rem 1.25rem;
        border-radius: 1.25rem;
        line-height: 1.6;
        word-wrap: break-word;
        font-size: 0.95rem;
        color: #374151;
        background: #ffffff !important;
        border: 1px solid #e5e7eb;
    }
    
    .message.user .message-bubble {
        background: #ffffff !important;
        border: 1px solid #d1d5db;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
        color: #374151;
    }
    
    .message.bot .message-bubble {
        background: #ffffff !important;
        border: 1px solid #d1d5db;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
        color: #374151;
    }
    
    /* Privacy badge */
    .privacy-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.25rem;
        padding: 0.4rem 0.75rem;
        border-radius: 0.75rem;
        font-size: 0.8rem;
        font-weight: 600;
        margin-top: 0.75rem;
        width: fit-content;
    }
    
    .privacy-badge.protected {
        background: #dbeafe;
        color: #075985;
        border: 1px solid #93c5fd;
    }
    
    .privacy-badge.warning {
        background: #fef3c7;
        color: #92400e;
        border: 1px solid #fde047;
    }
    
    /* Text area styling */
    .stTextArea textarea {
        border-radius: 1.25rem !important;
        border: 2px solid var(--neutral-200) !important;
        padding: 1rem 1.5rem !important;
        font-size: 0.95rem !important;
        resize: none !important;
        transition: all 0.3s !important;
        background: white !important;
        color: #374151 !important;
    }
    
    .stTextArea textarea::placeholder {
        color: #9ca3af !important;
    }
    
    .stTextArea textarea:focus {
        border-color: var(--primary) !important;
        box-shadow: 0 0 0 4px rgba(14, 165, 233, 0.1) !important;
        outline: none !important;
    }
    
    /* Buttons */
    .stButton button {
        border-radius: 1.25rem !important;
        padding: 0.75rem 2rem !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        border: none !important;
        transition: all 0.3s !important;
        cursor: pointer !important;
    }
    
    .stButton button[kind="primary"] {
        background: linear-gradient(135deg, var(--primary) 0%, var(--primary-dark) 100%) !important;
        color: white !important;
    }
    
    .stButton button[kind="primary"]:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(14, 165, 233, 0.3) !important;
    }
    
    /* Welcome screen */
    .welcome-screen {
        max-width: 700px;
        margin: 4rem auto;
        text-align: center;
    }
    
    .welcome-title {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }
    
    .welcome-subtitle {
        font-size: 1.1rem;
        color: var(--neutral-600);
        margin-bottom: 2rem;
    }
    
    .example-card {
        background: white;
        border: 2px solid var(--neutral-200);
        border-radius: 1rem;
        padding: 1.5rem;
        margin-bottom: 1rem;
        cursor: pointer;
        transition: all 0.3s;
        text-align: left;
    }
    
    .example-card:hover {
        border-color: var(--primary);
        background: var(--primary-light);
        box-shadow: 0 8px 24px rgba(14, 165, 233, 0.15);
        transform: translateY(-4px);
    }
    
    .example-icon {
        font-size: 1.75rem;
        margin-bottom: 0.75rem;
    }
    
    .example-text {
        color: var(--neutral-700);
        font-size: 0.95rem;
        font-weight: 500;
    }
    
    /* Typing indicator */
    .typing-indicator {
        display: inline-flex;
        gap: 0.3rem;
        padding: 1rem;
    }
    
    .typing-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: var(--primary);
        animation: typing 1.4s infinite;
    }
    
    .typing-dot:nth-child(2) {
        animation-delay: 0.2s;
    }
    
    .typing-dot:nth-child(3) {
        animation-delay: 0.4s;
    }
    
    @keyframes typing {
        0%, 60%, 100% { transform: translateY(0); }
        30% { transform: translateY(-10px); }
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: var(--neutral-100);
    }
    
    ::-webkit-scrollbar-thumb {
        background: var(--primary) !important;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: var(--primary-dark) !important;
    }
    
    /* Expander styling */
    [data-testid="stExpander"] * {
        color: #111827 !important;
        opacity: 1 !important;
    }

    [data-testid="stExpander"] pre,
    [data-testid="stExpander"] code {
        color: #111827 !important;
        background: #f9fafb !important;
        border: 1px solid #e5e7eb !important;
        border-radius: 0.75rem !important;
        padding: 0.75rem 1rem !important;
        line-height: 1.5 !important;
    }
</style>
""", unsafe_allow_html=True)


def check_api_health() -> Dict[str, Any]:
    """Check if API is healthy"""
    try:
        response = requests.get(f"{API_BASE_URL}/api/health", timeout=5)
        if response.status_code == 200:
            return {"status": "healthy", "data": response.json()}
        return {"status": "unhealthy", "error": f"HTTP {response.status_code}"}
    except requests.exceptions.ConnectionError:
        return {"status": "offline", "error": "Cannot connect to API"}
    except Exception as e:
        return {"status": "error", "error": str(e)}


def check_federated_health() -> Dict[str, Any]:
    """Check if Federated Server is ready"""
    try:
        response = requests.get(f"{API_BASE_URL}/api/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            if data.get("federated_ready"):
                return {
                    "status": "ready",
                    "num_clients": data.get("num_clients", 0),
                    "grid_status": data.get("flower_grid_status", "unknown"),
                    "data_sources": data.get("data_sources", [])
                }
            else:
                return {
                    "status": "waiting_clients",
                    "num_clients": data.get("num_clients", 0),
                    "grid_status": data.get("flower_grid_status", "unknown")
                }
        return {"status": "unhealthy"}
    except Exception:
        return {"status": "offline"}


def send_query(
    question: str,
    apply_privacy: bool = True,
    retriever_type: Optional[str] = None,
    use_federated: bool = False,
    topic: str = "finance"
) -> Dict[str, Any]:
    """Send query to API"""
    try:
        payload = {
            "question": question,
            "session_id": st.session_state.session_id,
            "apply_privacy": apply_privacy,
            "retriever_type": retriever_type,
            "use_federated": use_federated,
            "topic": topic
        }
        
        endpoint = "/api/query/federated" if use_federated else "/api/query"
        
        response = requests.post(
            f"{API_BASE_URL}{endpoint}",
            json=payload,
            timeout=60
        )
        
        if response.status_code == 200:
            return {"success": True, "data": response.json()}
        else:
            return {"success": False, "error": f"HTTP {response.status_code}: {response.text}"}
            
    except requests.exceptions.Timeout:
        return {"success": False, "error": "Request timeout (>60s)"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def display_thinking_state(state: str, num_clients: int = 0):
    """Display AI thinking state"""
    lang = st.session_state.language
    
    if state == "processing":
        text = get_text("thinking_processing", lang)
    elif state == "searching":
        text = get_text("thinking_searching", lang)
    elif state == "federated":
        text = get_text("thinking_federated", lang).format(num_clients)
    elif state == "privacy":
        text = get_text("thinking_privacy", lang)
    elif state == "generating":
        text = get_text("thinking_generating", lang)
    elif state == "finalizing":
        text = get_text("thinking_finalizing", lang)
    else:
        text = state
    
    st.markdown(f"""
    <div class="thinking-container">
        <div class="thinking-title">{get_text("thinking_title", lang)}</div>
        <div class="thinking-state">{text}</div>
    </div>
    """, unsafe_allow_html=True)


def display_masked_info(original: str, masked: str, pii_items: List[Dict]):
    """Display comparison of original and masked text"""
    lang = st.session_state.language
    
    if not pii_items:
        st.info(get_text("privacy_no_pii", lang))
        return
    
    # Highlight masked items in both texts
    masked_display = masked
    for item in pii_items:
        entity_type = item.get("entity_type", "")
        masked_display = masked_display.replace(
            item.get("masked_value", ""),
            f'<span class="masked-highlight">{item.get("masked_value", "")}</span>'
        )
    
    st.markdown(f"""
    <div class="masked-info-container">
        <div style="font-weight: 700; margin-bottom: 0.75rem; color: #92400e;">
            {get_text("privacy_masked_info", lang)}
        </div>
        <div class="masked-comparison">
            <div class="masked-box">
                <div class="masked-label">{get_text("privacy_original_text", lang)}</div>
                <div class="masked-text">{html.escape(original[:200])}...</div>
            </div>
            <div class="masked-box">
                <div class="masked-label">{get_text("privacy_masked_text", lang)}</div>
                <div class="masked-text">{masked_display[:200]}...</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def display_fl_community(fed_health: Dict[str, Any]):
    """Display Federated Learning community information"""
    lang = st.session_state.language
    
    if fed_health["status"] != "ready":
        return
    
    num_clients = fed_health.get("num_clients", 0)
    data_sources = fed_health.get("data_sources", [])
    
    st.markdown(f"""
    <div class="fl-community-card">
        <div style="font-weight: 700; font-size: 1.1rem; margin-bottom: 0.75rem;">
            {get_text("fl_community", lang)}
        </div>
        <div class="fl-stat">
            <span class="fl-stat-label">{get_text("fl_participants", lang)}:</span>
            <span class="fl-stat-value">{num_clients} {get_text("fl_active_nodes", lang)}</span>
        </div>
        <div class="fl-stat">
            <span class="fl-stat-label">{get_text("fl_status", lang)}:</span>
            <span class="fl-stat-value">✓ {get_text("fl_ready", lang)}</span>
        </div>
        <div class="fl-stat">
            <span class="fl-stat-label">{get_text("fl_data_sources", lang)}:</span>
            <span class="fl-stat-value">{len(data_sources)} sources</span>
        </div>
        <div class="fl-stat">
            <span class="fl-stat-label">{get_text("fl_last_sync", lang)}:</span>
            <span class="fl-stat-value">{datetime.now().strftime("%H:%M:%S")}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


def display_message(role: str, content: str, metadata: Optional[Dict] = None, timestamp: str = None):
    """Display a chat message"""
    lang = st.session_state.language
    
    # Avatar
    avatar_emoji = "👤" if role == "user" else "🤖"
    avatar_class = "user" if role == "user" else "bot"
    message_class = "user" if role == "user" else "bot"
    
    # Privacy badge
    privacy_html = ""
    if metadata and metadata.get("privacy_applied"):
        privacy_html = f'<div class="privacy-badge protected">{get_text("privacy_badge_protected", lang)}</div>'
    elif metadata and metadata.get("privacy_applied") is False and role == "assistant":
        privacy_html = f'<div class="privacy-badge warning">{get_text("privacy_badge_none", lang)}</div>'

    # Clean content - Remove ALL HTML tags
    raw = _clean_html_text(content or "")
    
    # If content is empty or only contains HTML artifacts, skip
    if not raw or raw in ["</div>", "<div>", "...", ""]:
        return

    # Escape content properly
    safe_content = html.escape(raw)
    safe_content = safe_content.replace("`", "&#96;")
    safe_content = safe_content.replace("\n", "<br>")

    st.markdown(f"""
    <div class="message {message_class}">
        <div class="avatar {avatar_class}">{avatar_emoji}</div>
        <div class="message-content">
            <div class="message-bubble">{safe_content}</div>
            {privacy_html}
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Privacy stats
    if metadata and metadata.get("privacy_stats") and role == "assistant":
        with st.expander(get_text("privacy_details", lang)):
            stats = metadata["privacy_stats"]
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric(get_text("privacy_pii_detected", lang), stats.get("pii_detected", 0))
            with col2:
                st.metric(get_text("privacy_removed", lang), stats.get("sentences_removed", 0))
            with col3:
                st.metric(get_text("privacy_density", lang), f"{stats.get('pii_density', 0):.1%}")
            with col4:
                st.metric(get_text("privacy_risk", lang), f"{stats.get('average_risk', 0):.2f}")
            
            # Show masked information if available
            if stats.get("pii_items"):
                st.divider()
                display_masked_info(
                    original=metadata.get("original_text", content),
                    masked=content,
                    pii_items=stats.get("pii_items", [])
                )
    
    # Sources
    if metadata and metadata.get("source_nodes") and role == "assistant":
        with st.expander(get_text("sources", lang)):
            for i, node in enumerate(metadata["source_nodes"][:3], 1):
                st.markdown(f"**{get_text('source_number', lang).format(i, node.get('score', 0))}**")
                src_text = html.escape(node.get("text", "")[:200]) + "..."
                st.markdown(f"<pre>{src_text}</pre>", unsafe_allow_html=True)
                if i < len(metadata["source_nodes"][:3]):
                    st.divider()


def sidebar_config():
    """Sidebar configuration"""
    lang = st.session_state.language
    
    with st.sidebar:
        st.title(get_text("settings", lang))
        
        # Language Selection
        st.subheader(get_text("language", lang))
        language = st.selectbox(
            get_text("select_language", lang),
            options=["vi", "en"],
            format_func=lambda x: "🇻🇳 Tiếng Việt" if x == "vi" else "🇬🇧 English",
            index=0 if st.session_state.language == "vi" else 1,
            key="language_selector"
        )
        if language != st.session_state.language:
            st.session_state.language = language
            st.rerun()
        
        st.divider()
        
        # Topic Selection
        st.subheader(get_text("topic", lang))
        topic_options = list(TOPIC_CONFIG.keys())
        topic_labels = [f"{TOPIC_CONFIG[t]['icon']} {get_text(f'topic_{t}', lang)}" for t in topic_options]
        
        selected_topic_idx = topic_options.index(st.session_state.topic) if st.session_state.topic in topic_options else 0
        selected_topic = st.selectbox(
            get_text("select_topic", lang),
            options=topic_options,
            format_func=lambda x: f"{TOPIC_CONFIG[x]['icon']} {get_text(f'topic_{x}', lang)}",
            index=selected_topic_idx,
            key="topic_selector"
        )
        if selected_topic != st.session_state.topic:
            st.session_state.topic = selected_topic
            st.rerun()
        
        st.divider()
        
        # API Status
        st.subheader(get_text("api_status", lang))
        health = check_api_health()
        
        if health["status"] == "healthy":
            st.success(get_text("api_online", lang))
            data = health.get("data", {})
            if data.get("privacy_enabled"):
                st.info(get_text("privacy_enabled", lang))
        elif health["status"] == "offline":
            st.error(get_text("api_offline", lang))
            st.warning(f"{get_text('api_start_command', lang)}\n```bash\npython -m uvicorn api.main:app --port 8000\n```")
        else:
            st.error(f"❌ {health.get('error')}")
        
        st.divider()
        
        # RAG Mode
        st.subheader(get_text("rag_mode", lang))
        rag_mode = st.radio(
            get_text("rag_mode_select", lang),
            [get_text("mode_single", lang), get_text("mode_federated", lang)],
            help=get_text("federated_help", lang),
            key="rag_mode_radio",
        )
        use_federated = get_text("mode_federated", lang) in rag_mode

        if use_federated:
            fed_health = check_federated_health()
            if fed_health["status"] == "ready":
                st.success(get_text("mode_federated_info", lang).format(fed_health['num_clients']))
                # Display FL Community Card
                display_fl_community(fed_health)
            elif fed_health["status"] == "waiting_clients":
                st.warning(get_text("mode_federated_waiting", lang).format(fed_health['num_clients']))
            else:
                st.error(get_text("mode_federated_offline", lang))
        else:
            st.info(get_text("mode_single_info", lang))
        
        st.divider()
        
        # Privacy Settings
        st.subheader(get_text("privacy_settings", lang))
        apply_privacy = st.checkbox(
            get_text("privacy_enable", lang),
            value=True,
            help=get_text("privacy_help", lang)
        )
        
        st.divider()
        
        # Retriever
        st.subheader(get_text("retriever", lang))
        topic_config = TOPIC_CONFIG.get(st.session_state.topic, TOPIC_CONFIG["general"])
        retriever_options = topic_config["retrievers"]
        
        retriever_type = st.selectbox(
            get_text("retriever_type", lang),
            ["default"] + retriever_options
        )
        
        st.divider()
        
        # Chat History
        st.subheader(get_text("chat_history", lang))
        message_count = len([m for m in st.session_state.messages if m["role"] == "user"])
        st.write(get_text("messages_count", lang).format(message_count))
        
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("🗑️ Clear", use_container_width=True, help="Clear all chat history"):
                st.session_state.messages = []
                st.rerun()
        with col2:
            if st.button("🧹 Clean", use_container_width=True, help="Remove HTML tags from messages"):
                # Force clean all messages
                cleaned = []
                for m in st.session_state.messages:
                    cleaned_content = _clean_html_text(m.get("content", ""))
                    if cleaned_content and cleaned_content not in ["</div>", "<div>", "div", "/div"]:
                        m["content"] = cleaned_content
                        cleaned.append(m)
                st.session_state.messages = cleaned
                st.success("✓ HTML cleaned!")
                st.rerun()
        with col3:
            if st.button("🔄 Reload", use_container_width=True, help="Force reload app"):
                st.cache_data.clear()
                st.cache_resource.clear()
                st.rerun()
    
    return {
        "apply_privacy": apply_privacy,
        "retriever_type": retriever_type if retriever_type != "default" else None,
        "api_healthy": health["status"] == "healthy",
        "use_federated": use_federated,
        "fed_health": check_federated_health() if use_federated else None
    }


def show_welcome_screen():
    """Show welcome screen with example questions"""
    lang = st.session_state.language
    topic = st.session_state.topic
    topic_config = TOPIC_CONFIG.get(topic, TOPIC_CONFIG["general"])
    
    welcome_msg_key = f"welcome_{topic}"
    welcome_msg = get_text(welcome_msg_key, lang)
    
    st.markdown(f"""
    <div class="welcome-screen">
        <div class="welcome-title">{topic_config['icon']} {get_text("welcome_title", lang)}</div>
        <div class="welcome-subtitle">
            {get_text("welcome_subtitle", lang).format(welcome_msg)}
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"### {get_text('try_asking', lang)}")
    
    # Get topic-specific examples
    examples = get_topic_examples(topic, lang)
    
    cols = st.columns(2)
    for idx, example in enumerate(examples):
        with cols[idx % 2]:
            st.markdown(f"""
            <div class="example-card">
                <div class="example-icon">{example['icon']}</div>
                <div class="example-text">{example['text']}</div>
            </div>
            """, unsafe_allow_html=True)


def main():
    """Main application"""
    # Initialize session state
    init_session_state()
    lang = st.session_state.language
    
    # AGGRESSIVE Clean messages - Remove ALL HTML artifacts
    cleaned_messages = []
    for m in st.session_state.messages:
        original_content = m.get("content", "")
        
        # Clean content aggressively
        cleaned_content = _clean_html_text(original_content)
        
        # Skip if content is empty, only whitespace, or HTML artifact
        if not cleaned_content:
            continue
        if cleaned_content in ["</div>", "<div>", "div", "/div", "...", ".", ""]:
            continue
        if len(cleaned_content) < 2:  # Too short, likely artifact
            continue
        
        # Update message with cleaned content
        m["content"] = cleaned_content
        cleaned_messages.append(m)
    
    st.session_state.messages = cleaned_messages

    # Sidebar configuration
    config = sidebar_config()
    
    # Welcome screen
    if len(st.session_state.messages) == 0:
        show_welcome_screen()
    
    # Check for HTML artifacts WARNING + DEBUG
    html_messages = []
    for idx, m in enumerate(st.session_state.messages):
        content = m.get("content", "")
        if "</div>" in content or "<div" in content.lower():
            html_messages.append((idx, m["role"], content[:80]))
    
    if html_messages:
        st.error("🐛 **DEBUG: HTML tags detected in messages!**", icon="🚨")
        with st.expander("🔍 View problematic messages (Debug)"):
            for idx, role, preview in html_messages:
                st.code(f"Message #{idx} ({role}): {preview}...")
        st.warning("👉 Click **🧹 Clean** button in sidebar → Chat History to fix", icon="⚠️")
    
    # Display message history
    for message in st.session_state.messages:
        display_message(
            role=message["role"],
            content=_clean_html_text(message["content"]),
            metadata=message.get("metadata"),
            timestamp=message.get("timestamp")
        )
    
    # Form input
    input_container = st.container()
    with input_container:
        with st.form(key="chat_form", clear_on_submit=True):
            user_input = st.text_area(
                "Message",
                placeholder=get_text("message_placeholder", lang),
                label_visibility="collapsed",
                height=80,
                key="chat_input"
            )
            
            col1, col2 = st.columns([1, 5])
            with col1:
                submit = st.form_submit_button(get_text("send_button", lang), type="primary", use_container_width=True)
    
    # Handle user submission
    if submit and user_input:
        if not config["api_healthy"]:
            st.error(get_text("error_api_unavailable", lang))
            return
        
        clean_user = _clean_html_text(user_input)
        if clean_user:
            st.session_state.messages.append({
                "role": "user",
                "content": clean_user,
                "timestamp": datetime.now().isoformat()
            })
        
        st.rerun()
    
    # Call backend if last message is user
    if len(st.session_state.messages) > 0 and st.session_state.messages[-1]["role"] == "user":
        # Show thinking states
        thinking_container = st.empty()
        
        with thinking_container:
            # Stage 1: Processing
            display_thinking_state("processing")
            time.sleep(0.5)
            
            # Stage 2: Searching
            display_thinking_state("searching")
            time.sleep(0.5)
            
            # Stage 3: Federated (if applicable)
            if config.get("use_federated") and config.get("fed_health"):
                num_clients = config["fed_health"].get("num_clients", 0)
                display_thinking_state("federated", num_clients)
                time.sleep(0.5)
            
            # Stage 4: Privacy protection
            if config["apply_privacy"]:
                display_thinking_state("privacy")
                time.sleep(0.5)
            
            # Stage 5: Generating
            display_thinking_state("generating")
            
            result = send_query(
                question=st.session_state.messages[-1]["content"],
                apply_privacy=config["apply_privacy"],
                retriever_type=config["retriever_type"],
                use_federated=config.get("use_federated", False),
                topic=st.session_state.topic
            )
            
            # Stage 6: Finalizing
            display_thinking_state("finalizing")
            time.sleep(0.3)
        
        # Clear thinking states
        thinking_container.empty()
        
        if result["success"]:
            data = result["data"]
            answer_raw = data.get("answer", "").strip()
            
            # DEBUG: Log raw answer if contains HTML
            if "</div>" in answer_raw or "<div" in answer_raw.lower():
                st.warning(f"⚠️ DEBUG: Backend returned HTML in answer!", icon="🐛")
                with st.expander("🔍 View raw response"):
                    st.code(f"Raw answer (first 300 chars):\n{answer_raw[:300]}")
            
            # AGGRESSIVE Clean HTML tags from answer
            answer = _clean_html_text(answer_raw)
            
            # Additional safety check - decode any entities
            if answer:
                try:
                    answer = html.unescape(answer)
                except:
                    pass
                answer = answer.strip()
            
            # Final verification
            if "</div>" in answer or "<div" in answer.lower():
                st.error("❌ HTML still present after cleaning! This is a bug.", icon="🚨")
                st.code(f"After cleaning: {answer[:200]}")
            
            # Skip if answer is empty or only contains HTML artifacts  
            if not answer or answer in ["</div>", "<div>", "div", "/div", "...", ".", ""]:
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": get_text("error_no_answer", lang).format(data.get('routed_to', 'unknown')),
                    "metadata": {
                        "privacy_applied": False,
                        "privacy_stats": None,
                        "source_nodes": [],
                        "response_time": data.get("response_time", 0)
                    },
                    "timestamp": datetime.now().isoformat()
                })
            else:
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "metadata": {
                        "privacy_applied": data.get("privacy_applied", False),
                        "privacy_stats": data.get("privacy_stats"),
                        "source_nodes": data.get("source_nodes", []),
                        "response_time": data.get("response_time", 0),
                        "original_text": data.get("original_answer", answer)
                    },
                    "timestamp": datetime.now().isoformat()
                })
        else:
            st.session_state.messages.append({
                "role": "assistant",
                "content": get_text("error_prefix", lang).format(result['error']),
                "timestamp": datetime.now().isoformat()
            })
        
        st.rerun()


if __name__ == "__main__":
    main()
