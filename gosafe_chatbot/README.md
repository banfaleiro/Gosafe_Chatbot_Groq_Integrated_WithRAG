# Gosafe Fire, Security and Safety — RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot for **Gosafe**, a fire safety company in
Fatorda, Madgaon, Goa. Built following the same 10-stage RAG flow as a standard
"Building a RAG Assistant" notebook — but using the **Groq API** for generation instead
of OpenAI, and a **Streamlit** UI themed around Gosafe's red/white branding.

## The 10-stage flow

| # | Stage | Where it happens |
|---|-------|-------------------|
| 1 | Create a small knowledge base | `knowledge_base/*.txt` |
| 2 | Split documents into chunks | `chunk_documents()` in `rag_engine.py` |
| 3 | Convert chunks into embeddings | `KnowledgeBase.build_index()` (TF-IDF) |
| 4 | Convert a question into an embedding | `KnowledgeBase.embed_query()` |
| 5 | Retrieve the most relevant chunks | `KnowledgeBase.retrieve()` |
| 6 | Place retrieved chunks inside a prompt | `build_prompt()` |
| 7 | Ask an LLM to answer only from those chunks | `call_groq()` |
| 8 | Show the sources used | source "pills" under each answer in the UI |
| 9 | Compare RAG with ordinary generation | "Compare RAG vs. plain LLM" toggle |
| 10 | Simple Agentic RAG workflow | `agentic_answer()` — checks retrieval confidence before answering |

No LangChain, LlamaIndex, FAISS, Chroma, MCP server, or external vector database is used.
Embeddings are computed locally with a TF-IDF vectorizer (scikit-learn), so the only network
calls in the whole pipeline are the requests to the Groq API (guardrail Layer 3, and the final
answer generation).

## Guardrail system (`guardrails.py`)

Every query passes through a 3-layer pipeline **before** it reaches the RAG system:

| Layer | Type | Catches | Cost |
|-------|------|---------|------|
| 1 | Rule-based (regex) | Restricted business topics (e.g. compensation, internal financials, credentials), obvious prompt-injection phrasing, obvious false-authority claims | Free, instant |
| 2 | Constitutional moderation (rule-based) | Content modeled on Article 19(2) reasonable-restriction grounds: public order, security of the state, decency/morality, defamation, incitement to an offence, etc. | Free, instant |
| 3 | LLM-based final check (Groq) | Subtle/indirect injection or authority-claim attempts that regex would miss | 1 extra Groq call, only runs if Layers 1-2 pass |

The pipeline **short-circuits**: as soon as a layer blocks a query, later layers (and the RAG/
generation step) are skipped entirely. Each blocked query gets a classification shown in the UI
as a colored badge:

- 🟢 **Safe** - proceeds to the RAG pipeline as normal
- 🔴 **Injection Attempt** - tries to override/extract system instructions
- 🟣 **False Authority Attempt** - tries to impersonate an admin/owner/developer
- 🟠 **Restricted Topic** - asks for confidential business info (e.g. compensation)
- 🟡 **Constitutional Concern** - flagged by the Layer 2 moderation categories

Toggle each layer on/off from the sidebar (useful for testing/demoing), and enable "Show guardrail
trace" to see exactly which layer and pattern fired for any given answer.

**Disclaimer**: Layer 2 is a simplified, keyword-based proxy for Article 19(2) of the Constitution
of India, built for demonstration purposes within this chatbot. It is not a substitute for legal
review, a certified content-moderation API, or advice from a qualified lawyer.

## Setup

```bash
cd gosafe_chatbot
pip install -r requirements.txt
export GROQ_API_KEY="your-groq-api-key"   # get one free at console.groq.com
streamlit run app.py
```

If you don't want to use an environment variable, you can also paste your Groq API key
directly into the sidebar once the app is running.

## Project structure

```
gosafe_chatbot/
├── app.py                     # Streamlit UI (Gosafe-themed)
├── rag_engine.py               # RAG pipeline: chunking, TF-IDF retrieval, Groq calls
├── guardrails.py                # 3-layer guardrail pipeline (rule-based, constitutional, LLM)
├── requirements.txt
├── README.md
└── knowledge_base/
    ├── 01_company_overview.txt
    ├── 02_services.txt
    ├── 03_products.txt
    ├── 04_team_and_experts.txt
    ├── 05_amc_and_compliance.txt
    └── 06_contact_and_faq.txt
```

## Customizing

- **Add more knowledge**: drop additional `.txt` files into `knowledge_base/` and click
  "Rebuild knowledge base index" in the sidebar (or restart the app).
- **Change the model**: edit `GROQ_MODEL` in `rag_engine.py` (any Groq-hosted chat model works).
- **Adjust retrieval**: tune `CHUNK_SIZE`, `CHUNK_OVERLAP`, `TOP_K`, and `RELEVANCE_THRESHOLD`
  at the top of `rag_engine.py`.
