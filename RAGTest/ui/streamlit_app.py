"""
Streamlit Chatbot UI for RAG with Privacy Protection
Financial Q&A Chatbot Interface - ChatGPT-style Design
"""
import streamlit as st
import requests
import json
from datetime import datetime
from typing import Optional, Dict, Any
import time


# Configuration
API_BASE_URL = "http://localhost:8000"
DEFAULT_SESSION_ID = "streamlit_session"


# Page configuration
st.set_page_config(
    page_title="Financial Q&A Chatbot",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)


st.markdown("""
<style>
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Main container */
    .main {
        background: #f7f7f8;
    }
    
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 8rem !important;
        max-width: 100% !important;
    }
    
    /* Chat container */
    .chat-container {
        max-width: 800px;
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
    
    .avatar {
        width: 36px;
        height: 36px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.2rem;
        flex-shrink: 0;
    }
    
    .avatar.user {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    .avatar.bot {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
    }
    
    .message-content {
        flex: 1;
        max-width: 85%;
    }
    
    .message-bubble {
        padding: 1rem 1.25rem;
        border-radius: 1rem;
        line-height: 1.6;
        word-wrap: break-word;
    }
    
    .message.user .message-bubble {
        background: white;
        border: 1px solid #e5e7eb;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
    }
    
    .message.bot .message-bubble {
        background: white;
        border: 1px solid #e5e7eb;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
    }
    
    .message-time {
        font-size: 0.75rem;
        color: #9ca3af;
        margin-top: 0.5rem;
    }
    
    /* Privacy badge */
    .privacy-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.25rem;
        padding: 0.25rem 0.5rem;
        border-radius: 0.5rem;
        font-size: 0.75rem;
        font-weight: 600;
        margin-top: 0.5rem;
    }
    
    .privacy-badge.protected {
        background: #dcfce7;
        color: #166534;
    }
    
    .privacy-badge.warning {
        background: #fef3c7;
        color: #92400e;
    }
    
    /* Input container - fixed at bottom */
    .input-container {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background: white;
        border-top: 1px solid #e5e7eb;
        padding: 1.5rem;
        z-index: 1000;
        box-shadow: 0 -4px 6px -1px rgba(0,0,0,0.1);
    }
    
    /* Adjust for sidebar */
    @media (min-width: 768px) {
        .input-container {
            left: 21rem;
            width: calc(100% - 21rem);
        }
    }
    
    .input-wrapper {
        max-width: 800px;
        margin: 0 auto;
    }
    
    /* Text area styling */
    .stTextArea textarea {
        border-radius: 1.5rem !important;
        border: 2px solid #e5e7eb !important;
        padding: 1rem 1.5rem !important;
        font-size: 1rem !important;
        resize: none !important;
        transition: border-color 0.2s !important;
    }
    
    .stTextArea textarea:focus {
        border-color: #667eea !important;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1) !important;
        outline: none !important;
    }
    
    /* Buttons */
    .stButton button {
        border-radius: 1.5rem !important;
        padding: 0.75rem 2rem !important;
        font-weight: 600 !important;
        border: none !important;
        transition: all 0.2s !important;
    }
    
    .stButton button[kind="primary"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
    }
    
    .stButton button[kind="primary"]:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4) !important;
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background: white;
    }
    
    /* Welcome screen */
    .welcome-screen {
        max-width: 600px;
        margin: 4rem auto;
        text-align: center;
    }
    
    .welcome-title {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }
    
    .welcome-subtitle {
        font-size: 1.1rem;
        color: #6b7280;
        margin-bottom: 2rem;
    }
    
    .example-card {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 1rem;
        padding: 1rem 1.5rem;
        margin-bottom: 0.75rem;
        cursor: pointer;
        transition: all 0.2s;
        text-align: left;
    }
    
    .example-card:hover {
        border-color: #667eea;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.15);
        transform: translateY(-2px);
    }
    
    .example-icon {
        font-size: 1.5rem;
        margin-bottom: 0.5rem;
    }
    
    .example-text {
        color: #374151;
        font-size: 0.95rem;
    }
    
    /* Loading animation */
    .typing-indicator {
        display: inline-flex;
        gap: 0.25rem;
        padding: 1rem;
    }
    
    .typing-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #9ca3af;
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
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: #f9fafb !important;
        border-radius: 0.5rem !important;
        border: 1px solid #e5e7eb !important;
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f1f1;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #cbd5e1;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #94a3b8;
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


def send_query(question: str, apply_privacy: bool = True, retriever_type: Optional[str] = None) -> Dict[str, Any]:
    """Send query to API"""
    try:
        payload = {
            "question": question,
            "session_id": st.session_state.session_id,
            "apply_privacy": apply_privacy,
            "retriever_type": retriever_type
        }
        
        response = requests.post(
            f"{API_BASE_URL}/api/query",
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


def display_message(role: str, content: str, metadata: Optional[Dict] = None, timestamp: str = None):
    """Display a chat message with ChatGPT-style design"""
    
    # Avatar emoji
    avatar_emoji = "👤" if role == "user" else "🤖"
    avatar_class = "user" if role == "user" else "bot"
    message_class = "user" if role == "user" else "bot"
    
    # Privacy badge
    privacy_html = ""
    if metadata and metadata.get("privacy_applied"):
        privacy_html = '<div class="privacy-badge protected">🔒 Privacy Protected</div>'
    elif metadata and metadata.get("privacy_applied") == False and role == "assistant":
        privacy_html = '<div class="privacy-badge warning">⚠️ No Privacy</div>'
    
    # Time display
    time_str = ""
    if timestamp:
        try:
            dt = datetime.fromisoformat(timestamp)
            time_str = dt.strftime("%H:%M")
        except:
            time_str = ""
    
    # Message HTML
    st.markdown(f"""
    <div class="message {message_class}">
        <div class="avatar {avatar_class}">{avatar_emoji}</div>
        <div class="message-content">
            <div class="message-bubble">{content}</div>
            {privacy_html}
            <div class="message-time">{time_str}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Show privacy details in expander if available
    if metadata and metadata.get("privacy_stats") and role == "assistant":
        with st.expander("🔍 Privacy Details"):
            stats = metadata["privacy_stats"]
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("PII Detected", stats.get("pii_detected", 0))
            with col2:
                st.metric("Removed", stats.get("sentences_removed", 0))
            with col3:
                st.metric("PII Density", f"{stats.get('pii_density', 0):.1%}")
            with col4:
                st.metric("Risk", f"{stats.get('average_risk', 0):.2f}")
    
    # Show source documents if available
    if metadata and metadata.get("source_nodes") and role == "assistant":
        with st.expander("📄 Sources"):
            for i, node in enumerate(metadata["source_nodes"][:3], 1):
                st.markdown(f"**Source {i}** (Score: {node.get('score', 0):.3f})")
                st.text(node.get("text", "")[:200] + "...")
                if i < len(metadata["source_nodes"][:3]):
                    st.divider()


def sidebar_config():
    """Sidebar configuration"""
    with st.sidebar:
        st.title("⚙️ Settings")
        
        # API Status
        st.subheader("API Status")
        health = check_api_health()
        
        if health["status"] == "healthy":
            st.success("✅ Online")
            data = health.get("data", {})
            if data.get("privacy_enabled"):
                st.info("🔒 Privacy Enabled")
        elif health["status"] == "offline":
            st.error("❌ Offline")
            st.warning("Start API:\n```bash\ncd RAGTest/api\npython main.py\n```")
        else:
            st.error(f"❌ {health.get('error')}")
        
        st.divider()
        
        # Privacy Settings
        st.subheader("Privacy Settings")
        apply_privacy = st.checkbox(
            "Enable Privacy Protection",
            value=True,
            help="Protect PII in responses"
        )
        
        st.divider()
        
        # Retriever Selection
        st.subheader("Retriever")
        st.info("🚧 MOCK Mode: All queries → default")
        retriever_type = st.selectbox(
            "Type",
            ["default", "financial", "general", "technical", "legal"]
        )
        
        st.divider()
        
        # Chat History
        st.subheader("Chat History")
        message_count = len([m for m in st.session_state.messages if m["role"] == "user"])
        st.write(f"📝 {message_count} messages")
        
        if st.button("🗑️ Clear History", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    
    return {
        "apply_privacy": apply_privacy,
        "retriever_type": retriever_type if retriever_type != "default" else None,
        "api_healthy": health["status"] == "healthy"
    }


def show_welcome_screen():
    """Show welcome screen with example questions"""
    st.markdown("""
    <div class="welcome-screen">
        <div class="welcome-title">💰 Financial Q&A Chatbot</div>
        <div class="welcome-subtitle">
            Ask me anything about financial reports, earnings, and market data
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 💡 Try asking:")
    
    examples = [
        {"icon": "📊", "text": "What is 3M's revenue in 2019?"},
        {"icon": "👔", "text": "Who is the CEO of Apple?"},
        {"icon": "💹", "text": "Show me Tesla's profit margin"},
        {"icon": "📈", "text": "What are the main products of Microsoft?"},
    ]
    
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
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "session_id" not in st.session_state:
        st.session_state.session_id = f"{DEFAULT_SESSION_ID}_{int(time.time())}"
    
    # Sidebar configuration
    config = sidebar_config()
    
    # Show welcome screen if no messages
    if len(st.session_state.messages) == 0:
        show_welcome_screen()
    
    # Display chat history
    for message in st.session_state.messages:
        display_message(
            role=message["role"],
            content=message["content"],
            metadata=message.get("metadata"),
            timestamp=message.get("timestamp")
        )
    
    # Fixed input container at bottom
    input_container = st.container()
    with input_container:
        with st.form(key="chat_form", clear_on_submit=True):
            user_input = st.text_area(
                "Message",
                placeholder="Type your question here... (Shift+Enter for new line)",
                label_visibility="collapsed",
                height=80,
                key="chat_input"
            )
            
            col1, col2 = st.columns([1, 5])
            with col1:
                submit = st.form_submit_button("Send 🚀", type="primary", use_container_width=True)
    
    # Process input
    if submit and user_input:
        if not config["api_healthy"]:
            st.error("❌ API is not available. Please start the FastAPI server.")
            return
        
        # Add user message
        st.session_state.messages.append({
            "role": "user",
            "content": user_input,
            "timestamp": datetime.now().isoformat()
        })
        
        # Rerun to show user message
        st.rerun()
    
    # If last message is from user, get bot response
    if len(st.session_state.messages) > 0 and st.session_state.messages[-1]["role"] == "user":
        with st.spinner(""):
            # Show typing indicator
            st.markdown("""
            <div class="message bot">
                <div class="avatar bot">🤖</div>
                <div class="message-content">
                    <div class="message-bubble">
                        <div class="typing-indicator">
                            <div class="typing-dot"></div>
                            <div class="typing-dot"></div>
                            <div class="typing-dot"></div>
                        </div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Send query to API
            result = send_query(
                question=st.session_state.messages[-1]["content"],
                apply_privacy=config["apply_privacy"],
                retriever_type=config["retriever_type"]
            )
        
        if result["success"]:
            data = result["data"]
            
            # Add assistant message
            st.session_state.messages.append({
                "role": "assistant",
                "content": data["answer"],
                "metadata": {
                    "privacy_applied": data.get("privacy_applied", False),
                    "privacy_stats": data.get("privacy_stats"),
                    "source_nodes": data.get("source_nodes", []),
                    "response_time": data.get("response_time", 0)
                },
                "timestamp": datetime.now().isoformat()
            })
        else:
            # Add error message
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"❌ Error: {result['error']}",
                "timestamp": datetime.now().isoformat()
            })
        
        # Rerun to show bot response
        st.rerun()


if __name__ == "__main__":
    main()
