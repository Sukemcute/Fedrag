"""
Internationalization (i18n) translations for the chatbot UI
"""

TRANSLATIONS = {
    "vi": {
        # Page title
        "page_title": "Chatbot RAG với Bảo Vệ Quyền Riêng Tư",
        "page_icon": "🤖",
        
        # Sidebar
        "settings": "⚙️ Cài Đặt",
        "language": "🌐 Ngôn Ngữ",
        "select_language": "Chọn ngôn ngữ",
        
        # Topic selection
        "topic": "📚 Chủ Đề",
        "select_topic": "Chọn chủ đề",
        "topic_finance": "💰 Tài chính",
        "topic_healthcare": "🏥 Y tế",
        "topic_legal": "⚖️ Pháp lý",
        "topic_education": "🎓 Giáo dục",
        "topic_technology": "💻 Công nghệ",
        "topic_general": "🌍 Tổng hợp",
        
        # API Status
        "api_status": "Trạng Thái API",
        "api_online": "✅ API Đang Hoạt Động",
        "api_offline": "❌ API Không Hoạt Động",
        "api_start_command": "Khởi động API:",
        "privacy_enabled": "🔒 Đã Bật Bảo Mật",
        
        # Federated Community
        "fl_community": "🌐 Cộng Đồng Liên Minh",
        "fl_participants": "Số bên tham gia",
        "fl_status": "Trạng thái",
        "fl_ready": "Sẵn sàng",
        "fl_waiting": "Đang chờ...",
        "fl_offline": "Không kết nối",
        "fl_total_nodes": "Tổng số nút",
        "fl_active_nodes": "Nút hoạt động",
        "fl_data_sources": "Nguồn dữ liệu",
        "fl_last_sync": "Đồng bộ lần cuối",
        
        # RAG Mode
        "rag_mode": "Chế Độ RAG",
        "rag_mode_select": "Chọn chế độ RAG:",
        "mode_single": "Máy Đơn",
        "mode_federated": "Liên Minh (Nhiều Bên)",
        "mode_single_info": "🖥️ Chế độ đơn: Truy vấn RAG cục bộ nhanh",
        "mode_federated_info": "🌐 Đã sẵn sàng: {} bên tham gia ✓",
        "mode_federated_waiting": "🌐 Server đang chờ các bên tham gia... ({} đã kết nối)",
        "mode_federated_offline": "🌐 Server liên minh ngoại tuyến",
        "federated_help": "Đơn: Nhanh, cục bộ. Liên minh: Chậm hơn nhưng câu trả lời tổng hợp từ nhiều bên",
        
        # Privacy Settings
        "privacy_settings": "Cài Đặt Bảo Mật",
        "privacy_enable": "Bật Bảo Vệ Quyền Riêng Tư",
        "privacy_help": "Bảo vệ thông tin cá nhân trong câu trả lời",
        "privacy_badge_protected": "🔒 Đã Bảo Vệ Quyền Riêng Tư",
        "privacy_badge_none": "⚠️ Không Bảo Mật",
        
        # Privacy Details
        "privacy_details": "🔍 Chi Tiết Bảo Mật",
        "privacy_pii_detected": "PII Phát Hiện",
        "privacy_removed": "Đã Xóa",
        "privacy_density": "Mật Độ PII",
        "privacy_risk": "Rủi Ro",
        "privacy_masked_info": "📋 Thông Tin Đã Che",
        "privacy_original_text": "Văn bản gốc",
        "privacy_masked_text": "Văn bản đã che",
        "privacy_no_pii": "Không phát hiện thông tin cá nhân",
        
        # Retriever
        "retriever": "Bộ Truy Xuất",
        "retriever_type": "Loại",
        "retriever_default": "Mặc định",
        "retriever_financial": "Tài chính",
        "retriever_general": "Tổng quát",
        "retriever_technical": "Kỹ thuật",
        "retriever_legal": "Pháp lý",
        
        # Chat History
        "chat_history": "Lịch Sử Chat",
        "messages_count": "📝 {} tin nhắn",
        "clear_history": "🗑️ Xóa Lịch Sử",
        
        # Welcome Screen
        "welcome_title": "Chatbot Hỏi Đáp Thông Minh",
        "welcome_subtitle": "Hỏi tôi bất cứ điều gì về {}",
        "try_asking": "💡 Thử hỏi:",
        
        # Topic-specific welcome messages
        "welcome_finance": "báo cáo tài chính, thu nhập và dữ liệu thị trường",
        "welcome_healthcare": "thông tin y tế, bệnh tật và phương pháp điều trị",
        "welcome_legal": "luật pháp, quy định và tư vấn pháp lý",
        "welcome_education": "giáo dục, khóa học và tài liệu học tập",
        "welcome_technology": "công nghệ, phần mềm và đổi mới",
        "welcome_general": "bất kỳ chủ đề nào",
        
        # Example questions by topic
        "example_finance_1": "Doanh thu của 3M năm 2019 là bao nhiêu?",
        "example_finance_2": "CEO của Apple là ai?",
        "example_finance_3": "Biên lợi nhuận của Tesla",
        "example_finance_4": "Sản phẩm chính của Microsoft là gì?",
        
        "example_healthcare_1": "Triệu chứng của bệnh COVID-19 là gì?",
        "example_healthcare_2": "Cách phòng ngừa bệnh tiểu đường?",
        "example_healthcare_3": "Tác dụng phụ của thuốc kháng sinh",
        "example_healthcare_4": "Chế độ ăn cho người huyết áp cao",
        
        "example_legal_1": "Quy định về bảo vệ dữ liệu cá nhân?",
        "example_legal_2": "Quyền của người lao động là gì?",
        "example_legal_3": "Thủ tục ly hôn tại Việt Nam",
        "example_legal_4": "Luật sở hữu trí tuệ",
        
        # AI Thinking States
        "thinking_title": "🧠 AI Đang Suy Nghĩ",
        "thinking_processing": "🔄 Đang xử lý câu hỏi...",
        "thinking_searching": "🔍 Đang tìm kiếm thông tin...",
        "thinking_federated": "🌐 Đang truy vấn {} bên tham gia...",
        "thinking_privacy": "🔒 Đang bảo vệ thông tin cá nhân...",
        "thinking_generating": "✍️ Đang tạo câu trả lời...",
        "thinking_finalizing": "✅ Đang hoàn thiện...",
        
        # Messages
        "message_placeholder": "Nhập câu hỏi của bạn... (Shift+Enter để xuống dòng)",
        "send_button": "Gửi 🚀",
        "error_api_unavailable": "❌ API không khả dụng. Vui lòng khởi động server FastAPI.",
        "error_no_answer": "❌ Không tạo được câu trả lời. Chế độ: {}",
        "error_prefix": "❌ Lỗi: {}",
        
        # Sources
        "sources": "📄 Nguồn Tham Khảo",
        "source_number": "Nguồn {} (Điểm: {:.3f})",
        
        # Response info
        "response_time": "Thời gian phản hồi",
        "routed_to": "Định tuyến đến",
    },
    
    "en": {
        # Page title
        "page_title": "RAG Chatbot with Privacy Protection",
        "page_icon": "🤖",
        
        # Sidebar
        "settings": "⚙️ Settings",
        "language": "🌐 Language",
        "select_language": "Select language",
        
        # Topic selection
        "topic": "📚 Topic",
        "select_topic": "Select topic",
        "topic_finance": "💰 Finance",
        "topic_healthcare": "🏥 Healthcare",
        "topic_legal": "⚖️ Legal",
        "topic_education": "🎓 Education",
        "topic_technology": "💻 Technology",
        "topic_general": "🌍 General",
        
        # API Status
        "api_status": "API Status",
        "api_online": "✅ API Online",
        "api_offline": "❌ API Offline",
        "api_start_command": "Start API:",
        "privacy_enabled": "🔒 Privacy Enabled",
        
        # Federated Community
        "fl_community": "🌐 Federated Community",
        "fl_participants": "Participants",
        "fl_status": "Status",
        "fl_ready": "Ready",
        "fl_waiting": "Waiting...",
        "fl_offline": "Offline",
        "fl_total_nodes": "Total nodes",
        "fl_active_nodes": "Active nodes",
        "fl_data_sources": "Data sources",
        "fl_last_sync": "Last sync",
        
        # RAG Mode
        "rag_mode": "RAG Mode",
        "rag_mode_select": "Select RAG mode:",
        "mode_single": "Single Machine",
        "mode_federated": "Federated (Multi-Client)",
        "mode_single_info": "🖥️ Single Mode: Fast local RAG queries",
        "mode_federated_info": "🌐 Federated Ready: {} clients connected ✓",
        "mode_federated_waiting": "🌐 Server waiting for clients... ({} connected)",
        "mode_federated_offline": "🌐 Federated Server offline",
        "federated_help": "Single: Fast, local. Federated: Slower but ensemble answer from multiple clients",
        
        # Privacy Settings
        "privacy_settings": "Privacy Settings",
        "privacy_enable": "Enable Privacy Protection",
        "privacy_help": "Protect PII in responses",
        "privacy_badge_protected": "🔒 Privacy Protected",
        "privacy_badge_none": "⚠️ No Privacy",
        
        # Privacy Details
        "privacy_details": "🔍 Privacy Details",
        "privacy_pii_detected": "PII Detected",
        "privacy_removed": "Removed",
        "privacy_density": "PII Density",
        "privacy_risk": "Risk",
        "privacy_masked_info": "📋 Masked Information",
        "privacy_original_text": "Original text",
        "privacy_masked_text": "Masked text",
        "privacy_no_pii": "No personal information detected",
        
        # Retriever
        "retriever": "Retriever",
        "retriever_type": "Type",
        "retriever_default": "Default",
        "retriever_financial": "Financial",
        "retriever_general": "General",
        "retriever_technical": "Technical",
        "retriever_legal": "Legal",
        
        # Chat History
        "chat_history": "Chat History",
        "messages_count": "📝 {} messages",
        "clear_history": "🗑️ Clear History",
        
        # Welcome Screen
        "welcome_title": "Intelligent Q&A Chatbot",
        "welcome_subtitle": "Ask me anything about {}",
        "try_asking": "💡 Try asking:",
        
        # Topic-specific welcome messages
        "welcome_finance": "financial reports, earnings, and market data",
        "welcome_healthcare": "healthcare information, diseases and treatments",
        "welcome_legal": "laws, regulations and legal advice",
        "welcome_education": "education, courses and learning materials",
        "welcome_technology": "technology, software and innovation",
        "welcome_general": "any topic",
        
        # Example questions by topic
        "example_finance_1": "What is 3M's revenue in 2019?",
        "example_finance_2": "Who is the CEO of Apple?",
        "example_finance_3": "Show me Tesla's profit margin",
        "example_finance_4": "What are the main products of Microsoft?",
        
        "example_healthcare_1": "What are the symptoms of COVID-19?",
        "example_healthcare_2": "How to prevent diabetes?",
        "example_healthcare_3": "Side effects of antibiotics",
        "example_healthcare_4": "Diet for high blood pressure",
        
        "example_legal_1": "Regulations on personal data protection?",
        "example_legal_2": "What are workers' rights?",
        "example_legal_3": "Divorce procedures in Vietnam",
        "example_legal_4": "Intellectual property law",
        
        # AI Thinking States
        "thinking_title": "🧠 AI Thinking",
        "thinking_processing": "🔄 Processing question...",
        "thinking_searching": "🔍 Searching information...",
        "thinking_federated": "🌐 Querying {} participants...",
        "thinking_privacy": "🔒 Protecting personal information...",
        "thinking_generating": "✍️ Generating answer...",
        "thinking_finalizing": "✅ Finalizing...",
        
        # Messages
        "message_placeholder": "Type your question here... (Shift+Enter for new line)",
        "send_button": "Send 🚀",
        "error_api_unavailable": "❌ API is not available. Please start the FastAPI server.",
        "error_no_answer": "❌ No answer generated. Mode: {}",
        "error_prefix": "❌ Error: {}",
        
        # Sources
        "sources": "📄 Sources",
        "source_number": "Source {} (Score: {:.3f})",
        
        # Response info
        "response_time": "Response time",
        "routed_to": "Routed to",
    }
}


def get_text(key: str, lang: str = "vi", **kwargs) -> str:
    """
    Get translated text by key
    
    Args:
        key: Translation key
        lang: Language code (vi or en)
        **kwargs: Format arguments for string formatting
    
    Returns:
        Translated text
    """
    translations = TRANSLATIONS.get(lang, TRANSLATIONS["vi"])
    text = translations.get(key, key)
    
    if kwargs:
        try:
            return text.format(**kwargs)
        except:
            return text
    
    return text


def get_topic_examples(topic: str, lang: str = "vi") -> list:
    """
    Get example questions for a topic
    
    Args:
        topic: Topic name (finance, healthcare, legal, etc.)
        lang: Language code
    
    Returns:
        List of example questions
    """
    examples_map = {
        "finance": [
            ("📊", f"example_{topic}_1"),
            ("👔", f"example_{topic}_2"),
            ("💹", f"example_{topic}_3"),
            ("📈", f"example_{topic}_4"),
        ],
        "healthcare": [
            ("🩺", f"example_{topic}_1"),
            ("💊", f"example_{topic}_2"),
            ("⚕️", f"example_{topic}_3"),
            ("🏥", f"example_{topic}_4"),
        ],
        "legal": [
            ("⚖️", f"example_{topic}_1"),
            ("📜", f"example_{topic}_2"),
            ("🔨", f"example_{topic}_3"),
            ("📋", f"example_{topic}_4"),
        ],
    }
    
    if topic not in examples_map:
        topic = "finance"
    
    examples = []
    for icon, key in examples_map[topic]:
        examples.append({
            "icon": icon,
            "text": get_text(key, lang)
        })
    
    return examples
