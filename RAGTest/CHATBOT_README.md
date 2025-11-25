# 💬 RAG Chatbot with Privacy Protection

## ⚡ Quick Start

### 1️⃣ Install Dependencies

```bash
cd RAGTest
pip install -r requirements.txt
```

### 2️⃣ Start Chatbot

**Linux/Mac:**
```bash
./start_chatbot.sh
```

**Windows:**
```cmd
start_chatbot.bat
```

**Manual:**
```bash
# Terminal 1: API
cd RAGTest/api
python main.py

# Terminal 2: UI
cd RAGTest/ui
streamlit run streamlit_app.py
```

### 3️⃣ Open Browser

- **Streamlit UI**: http://localhost:8501
- **API Docs**: http://localhost:8000/api/docs
- **API Health**: http://localhost:8000/api/health

---

## 🎯 Usage

### Streamlit UI

1. **Check API Status** (sidebar):
   - ✅ Green = API Online
   - ❌ Red = API Offline

2. **Configure Privacy** (sidebar):
   - Toggle privacy protection
   - Select retriever type (MOCK)

3. **Ask Questions**:
   ```
   What is 3M's revenue in 2019?
   Who is the CEO of Apple?
   Show me Tesla's profit margin
   ```

4. **View Privacy Stats**:
   - Click "🔍 Privacy Details"
   - See PII detected, risk scores, etc.

5. **View Sources**:
   - Click "📄 Source Documents"
   - See relevant document chunks

### API Endpoints

#### 1. Query

```bash
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is 3M'\''s revenue?",
    "apply_privacy": true
  }'
```

#### 2. Health Check

```bash
curl http://localhost:8000/api/health
```

#### 3. Stats

```bash
curl http://localhost:8000/api/stats
```

---

## 🏗️ Architecture

```
┌─────────────┐
│   User      │
└─────┬───────┘
      │
      ▼
┌─────────────────┐
│  Streamlit UI   │ (Port 8501)
└─────┬───────────┘
      │ HTTP
      ▼
┌─────────────────┐
│  FastAPI        │ (Port 8000)
│  Backend        │
└─────┬───────────┘
      │
      ▼
┌─────────────────┐
│  Router Module  │ (MOCK)
│  (Future: ML)   │
└─────┬───────────┘
      │
      ▼
┌─────────────────────────────────────┐
│          RAG Pipeline               │
│  ┌──────────┐  ┌─────────────┐     │
│  │ Retrieve │→ │  Rerank     │     │
│  └──────────┘  └─────────────┘     │
│          ↓                          │
│  ┌──────────────────────────┐      │
│  │   LLM Generation         │      │
│  └──────────────────────────┘      │
└─────┬───────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────┐
│      Privacy Module                 │
│  ┌──────────┐  ┌────────────────┐  │
│  │ Presidio │→ │ Eraser4RAG     │  │
│  └──────────┘  └────────────────┘  │
│  ┌──────────┐  ┌────────────────┐  │
│  │ TenSEAL  │  │ Flower         │  │
│  └──────────┘  └────────────────┘  │
└─────┬───────────────────────────────┘
      │
      ▼
┌─────────────────┐
│   Protected     │
│   Response      │
└─────────────────┘
```
---

## 🚧 Router Module (MOCK)

### Current State

Router hiện tại là **MOCK implementation**:
- Phân tích keywords để detect domain
- Luôn route về "default" retriever
- Log reasoning để prepare cho future

### Future Implementation

```python
# 1. Train ML classifier
classifier = train_query_classifier(training_data)

# 2. Create multiple retrievers
retrievers = {
    "financial": FinancialRetriever(...),
    "technical": TechnicalRetriever(...),
    "legal": LegalRetriever(...),
}

# 3. Route queries
domain = classifier.predict(query)
retriever = retrievers[domain]
response = retriever.query(query)
```

### Training Data Format

```json
{
  "queries": [
    {"text": "What is 3M's revenue?", "domain": "financial"},
    {"text": "How does OAuth work?", "domain": "technical"},
    {"text": "Show me the contract terms", "domain": "legal"}
  ]
}
```

---

## 📊 Privacy Module Details

### Flow

```
Input Text
    ↓
[Presidio]
    ↓ (detect PII)
Anonymized Text
    ↓
[Eraser4RAG]
    ↓ (remove risky sentences)
Sanitized Text
    ↓
[TenSEAL]
    ↓ (encrypt if needed)
Encrypted Payload
    ↓
[Flower]
    ↓ (federated aggregation)
Final Response
```

### Privacy Stats

```json
{
  "pii_detected": 2,
  "pii_density": 0.08,
  "sentences_removed": 1,
  "average_risk": 0.45,
  "encrypted": false,
  "entities": ["PERSON", "MONEY"]
}
```

### Example

**Original:**
```
John Smith earned $50,000 last year at Apple Inc.
```

**Protected (with `use_entity_type_tags=true`):**
```
<PERSON> earned <MONEY> last year at <ORG>.
```

---

## 🐛 Troubleshooting

### API không start

```bash
# Check port
lsof -i :8000  # Mac/Linux
netstat -ano | findstr :8000  # Windows

# Kill process
kill -9 <PID>
```

### UI không connect

1. Check API health: http://localhost:8000/api/health
2. Check sidebar: "API Status" = "✅ API Online"
3. Check CORS in `api/main.py`

### Embeddings error

```toml
# config.toml
[settings]
embeddings = "BAAI/bge-base-en"  # Use default
```

### Privacy không hoạt động

```toml
# config.toml
[privacy]
enable_privacy_summary = true
```

```python
# Request
{
  "question": "...",
  "apply_privacy": true  # Must be true
}
```

---

## 📈 Performance

### Benchmarks

```bash
python benchmark_privacy.py
```

So sánh:
- Response time: with/without privacy
- Response quality: NLG metrics
- Privacy stats: PII detected, sentences removed

### Optimization

1. **Reduce chunk_size**: Faster retrieval
2. **Use GPU**: Faster embeddings & LLM
3. **Cache**: Redis for repeated queries
4. **Batch**: Process multiple queries

---

## 🔐 Security (Production)

### 1. CORS

```python
# api/main.py
allow_origins=["https://yourdomain.com"]
```

### 2. API Keys

```python
from fastapi.security import APIKeyHeader

@app.post("/api/query")
async def query(api_key: str = Depends(verify_api_key)):
    ...
```

### 3. Rate Limiting

```python
from slowapi import Limiter

limiter = Limiter(key_func=get_remote_address)

@app.post("/api/query")
@limiter.limit("10/minute")
async def query():
    ...
```

### 4. HTTPS

Deploy with SSL certificate (Let's Encrypt, Cloudflare)

---

## 📚 Documentation

- **QUICK_START.md**: Hướng dẫn nhanh 5 phút
- **CHATBOT_SETUP.md**: Setup chi tiết, troubleshooting
- **RUN_GUIDE.md**: Chạy các scripts khác
- **PRIVACY_MODULE.md**: Chi tiết privacy module

---

## 🚀 Deployment

### Docker

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

# Start script
CMD ["bash", "start_chatbot.sh"]
```

```bash
docker build -t rag-chatbot .
docker run -p 8000:8000 -p 8501:8501 rag-chatbot
```

### Docker Compose

```yaml
version: '3.8'

services:
  api:
    build: .
    command: python api/main.py
    ports:
      - "8000:8000"
    volumes:
      - ./config.toml:/app/config.toml
  
  ui:
    build: .
    command: streamlit run ui/streamlit_app.py
    ports:
      - "8501:8501"
    depends_on:
      - api
```

---

## 🧪 Testing

```bash
# Unit tests
pytest tests/

# Integration test
python test_end_to_end.py

# Benchmark
python benchmark_privacy.py
```

---

## 🎨 Customization

### UI Theme

Sửa CSS trong `ui/streamlit_app.py`

### API Response

Sửa models trong `api/models.py`

### Privacy Rules

Sửa `config.toml` privacy section

---

## 📝 API Documentation

### Auto-generated Docs

- **Swagger UI**: http://localhost:8000/api/docs
- **ReDoc**: http://localhost:8000/api/redoc

### Example Response

```json
{
  "question": "What is 3M's revenue?",
  "answer": "3M's revenue in 2019 was approximately <MONEY>.",
  "original_answer": "3M's revenue in 2019 was $32.1 billion.",
  "privacy_applied": true,
  "privacy_stats": {
    "pii_detected": 1,
    "pii_density": 0.05,
    "sentences_removed": 0,
    "average_risk": 0.3,
    "encrypted": false,
    "entities": ["MONEY"]
  },
  "source_nodes": [...],
  "response_time": 2.5,
  "routed_to": "default"
}
```

---

## 🤝 Contributing

1. Fork repo
2. Create feature branch
3. Implement changes
4. Add tests
5. Submit PR

---

## 📄 License

MIT License

---

## 👥 Team

RAG Team - Financial AI Research

---

## 🔗 Links

- **API Docs**: http://localhost:8000/api/docs
- **Streamlit**: http://localhost:8501
- **GitHub**: (your repo)

---

## 📞 Support

- Issues: GitHub Issues
- Email: (your email)
- Docs: See CHATBOT_SETUP.md

---

**Built with ❤️ using FastAPI + Streamlit + LlamaIndex + Privacy Tools**

