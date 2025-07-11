# 🔍 TrustAI – Real-Time Misinformation Detection Engine

**TrustAI** is an AI-powered web application that fact-checks user-submitted claims using a Retrieval-Augmented Generation (RAG) pipeline. It combines OpenAI GPT-4, Weaviate (vector database), and LangChain to deliver a trust score, confidence rating, and detailed explanation — all in real time.

> 🧠 Powered by GPT-4, LangChain, and Weaviate  
> 🌐 Deployed via FastAPI (API) + Streamlit (UI)

---

## 🌐 Live Demo

👉 [Launch TrustAI](https://your-streamlit-url.streamlit.app)  
No sign-in required. Enter a claim and get instant analysis.

---

## 🧠 How It Works

1. User submits a claim or article snippet.
2. The system embeds the claim and queries Weaviate for similar fact-checked statements.
3. GPT-4 compares results and generates:
   - ✅ A **trust score** (0–100)
   - ✅ A **confidence score**
   - ✅ A natural language **explanation**
   - ✅ Links to **similar claims**

---

## 💡 Key Features

- 🔎 Real-time misinformation detection
- 📊 Trust and confidence scores with color-coded feedback
- 🧠 GPT-4 explanations and verdicts
- 🔗 Vector similarity search with Weaviate
- ⚙️ RESTful API (FastAPI) + optional frontend (Streamlit or React)
- 🚀 Hosted and ready to demo

---

## 🧰 Tech Stack

| Layer        | Technology |
|--------------|------------|
| LLM          | OpenAI GPT-4 |
| RAG Pipeline | LangChain |
| Vector DB    | Weaviate |
| API Server   | FastAPI |
| Frontend     | Streamlit (React optional) |
| Deployment   | Railway (API) + Streamlit Cloud (UI) |
| Data Sources | PolitiFact, Snopes, Wikipedia |

---

## 🧪 Local Development (for contributors)

### 1. Clone this repo

```bash
git clone https://github.com/imakkar/TrustAI.git
cd TrustAI
```

### 2. Set up environment variables

Create a `.env` file in the root:

```env
OPENAI_API_KEY=your-openai-key
SERPAPI_API_KEY=your-serpapi-key
```

> ⚠️ These keys are **not required** to use the public demo — only for local development.

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Start Weaviate locally

```bash
docker run -d --name weaviate -p 8080:8080   -e AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED=true   semitechnologies/weaviate:1.23.7
```

### 5. Ingest fact-check dataset

```bash
cd backend
python ingest_data.py
```

### 6. Start the API

```bash
uvicorn main:app --reload
```

### 7. Run the frontend (Streamlit)

```bash
cd ../frontend
streamlit run app.py
```

Visit: [http://localhost:8501](http://localhost:8501)

---

## 📂 Project Structure

```
trustai/
├── backend/           # FastAPI app + RAG pipeline
├── frontend/          # Streamlit UI
├── data/              # factbase.json
├── weaviate_setup/    # Vector schema
├── requirements.txt   # Dependencies
├── .env               # API keys (excluded)
└── README.md
```

---

## 📡 API Endpoints

| Method | Endpoint         | Description            |
|--------|------------------|------------------------|
| POST   | `/check_claim`   | Submit a claim for analysis |
| GET    | `/health`        | API health check       |
| GET    | `/docs`          | Swagger API docs       |

Example:
```bash
curl -X POST http://localhost:8000/check_claim \
  -H "Content-Type: application/json" \
  -d '{"claim": "Vaccines cause autism"}'
```

---

## 👨‍💻 Author

Built by [Ishan Makkar](https://github.com/imakkar)  
ML/AI + Product + Data | CS + DS @ Rutgers University

---

## 🪪 License

MIT License – Free for personal and commercial use.

---

> ⭐️ If you find this project useful, please consider starring the repo on GitHub!
