"""
Gosafe RAG Engine
==================
Implements a Retrieval-Augmented Generation pipeline following the same
10-stage flow as a typical "Building a RAG Assistant" notebook, but using
the Groq API for the final answer generation instead of OpenAI.

Stages implemented in this file:
 1. Create a small knowledge base       -> load_documents()
 2. Split documents into chunks         -> chunk_documents()
 3. Convert chunks into embeddings      -> KnowledgeBase.build_index()
 4. Convert a question into an embedding-> KnowledgeBase.embed_query()
 5. Retrieve the most relevant chunks   -> KnowledgeBase.retrieve()
 6. Place retrieved chunks in a prompt  -> build_prompt()
 7. Ask an LLM to answer only from those chunks -> call_groq()
 8. Show the sources used               -> handled in answer_question() return value
 9. Compare RAG vs ordinary generation  -> generate_plain_answer()
10. Simple Agentic RAG workflow         -> agentic_answer()

No LangChain, LlamaIndex, FAISS, Chroma, MCP server, or external vector
database is used. Embeddings are built with a lightweight local TF-IDF
vectorizer (scikit-learn) so the whole pipeline runs offline except for
the single call to the Groq API for generation.
"""

import os
import glob
from dataclasses import dataclass
from typing import List, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from groq import Groq

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

KB_DIR = os.path.join(os.path.dirname(__file__), "knowledge_base")
CHUNK_SIZE = 700          # characters per chunk
CHUNK_OVERLAP = 120       # characters of overlap between chunks
TOP_K = 4                 # number of chunks to retrieve per question
RELEVANCE_THRESHOLD = 0.08  # minimum similarity score to trust retrieval (stage 10, agentic check)

GROQ_MODEL = "openai/gpt-oss-120b"

SYSTEM_PROMPT = (
    "You are the official Gosafe Fire, Security and Safety virtual assistant. "
    "Gosafe is a fire safety company based in Fatorda, Madgaon, Goa, established in 2011. "
    "Answer the customer's question using ONLY the information given in the CONTEXT below. "
    "If the answer is not contained in the context, politely say you don't have that specific "
    "detail and suggest the customer contact the Gosafe office directly for a precise answer. "
    "Never invent prices, phone numbers, or facts that are not in the context. "
    "Keep answers friendly, professional, and concise, in the voice of a helpful fire-safety "
    "company representative."
)


@dataclass
class Chunk:
    text: str
    source: str


# ---------------------------------------------------------------------------
# Stage 1: Create / load the small knowledge base
# ---------------------------------------------------------------------------

def load_documents(kb_dir: str = KB_DIR) -> List[Tuple[str, str]]:
    """Load every .txt document in the knowledge base directory.

    Returns a list of (filename, full_text) tuples.
    """
    docs = []
    for path in sorted(glob.glob(os.path.join(kb_dir, "*.txt"))):
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        docs.append((os.path.basename(path), text))
    return docs


# ---------------------------------------------------------------------------
# Stage 2: Split documents into chunks
# ---------------------------------------------------------------------------

def chunk_documents(docs: List[Tuple[str, str]],
                     chunk_size: int = CHUNK_SIZE,
                     overlap: int = CHUNK_OVERLAP) -> List[Chunk]:
    """Simple sliding-window character chunker with overlap.

    Splits on paragraph boundaries first, then packs paragraphs into
    chunks up to chunk_size, carrying a small overlap forward so context
    isn't lost at chunk boundaries.
    """
    all_chunks: List[Chunk] = []

    for filename, text in docs:
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        current = ""
        for para in paragraphs:
            if len(current) + len(para) + 1 <= chunk_size:
                current = f"{current}\n{para}".strip()
            else:
                if current:
                    all_chunks.append(Chunk(text=current, source=filename))
                # start new chunk, carrying overlap from the end of the previous chunk
                tail = current[-overlap:] if current else ""
                current = f"{tail}\n{para}".strip()
        if current:
            all_chunks.append(Chunk(text=current, source=filename))

    return all_chunks


# ---------------------------------------------------------------------------
# Stages 3-5: Embeddings + retrieval, wrapped in a KnowledgeBase class
# ---------------------------------------------------------------------------

class KnowledgeBase:
    def __init__(self, kb_dir: str = KB_DIR):
        self.kb_dir = kb_dir
        self.chunks: List[Chunk] = []
        self.vectorizer: TfidfVectorizer | None = None
        self.matrix = None
        self.build_index()

    def build_index(self):
        """Stages 1-3: load docs, chunk them, convert chunks into embeddings."""
        docs = load_documents(self.kb_dir)
        self.chunks = chunk_documents(docs)
        texts = [c.text for c in self.chunks]

        # TF-IDF acts as our local, offline embedding model (stage 3).
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.matrix = self.vectorizer.fit_transform(texts)

    def embed_query(self, question: str):
        """Stage 4: convert a question into an embedding using the same vectorizer."""
        return self.vectorizer.transform([question])

    def retrieve(self, question: str, top_k: int = TOP_K) -> List[Tuple[Chunk, float]]:
        """Stage 5: retrieve the most relevant chunks for a question."""
        query_vec = self.embed_query(question)
        scores = cosine_similarity(query_vec, self.matrix)[0]
        ranked_idx = scores.argsort()[::-1][:top_k]
        return [(self.chunks[i], float(scores[i])) for i in ranked_idx]


# ---------------------------------------------------------------------------
# Stage 6: Place the retrieved chunks inside a prompt
# ---------------------------------------------------------------------------

def build_prompt(question: str, retrieved: List[Tuple[Chunk, float]]) -> str:
    context_blocks = []
    for i, (chunk, score) in enumerate(retrieved, start=1):
        context_blocks.append(f"[Source {i}: {chunk.source}]\n{chunk.text}")
    context = "\n\n".join(context_blocks)

    return (
        f"CONTEXT:\n{context}\n\n"
        f"CUSTOMER QUESTION:\n{question}\n\n"
        "Answer the question using only the context above."
    )


# ---------------------------------------------------------------------------
# Stage 7: Ask Groq to answer only from those chunks
# ---------------------------------------------------------------------------

def get_groq_client(api_key: str | None = None) -> Groq:
    key = api_key or os.environ.get("GROQ_API_KEY")
    if not key:
        raise ValueError(
            "No Groq API key found. Set the GROQ_API_KEY environment variable "
            "or enter your key in the sidebar of the app."
        )
    return Groq(api_key=key)


def call_groq(prompt: str, api_key: str | None = None,
              system_prompt: str = SYSTEM_PROMPT,
              model: str = GROQ_MODEL,
              temperature: float = 0.3) -> str:
    client = get_groq_client(api_key)
    completion = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
    )
    return completion.choices[0].message.content


# ---------------------------------------------------------------------------
# Stage 9: Compare RAG with ordinary (non-retrieval) generation
# ---------------------------------------------------------------------------

def generate_plain_answer(question: str, api_key: str | None = None,
                           model: str = GROQ_MODEL) -> str:
    """Answer the question with NO retrieved context, for side-by-side comparison."""
    plain_system_prompt = (
        "You are a general-purpose AI assistant. Answer the user's question about a "
        "fire safety company called Gosafe as best you can from general knowledge. "
        "You have NOT been given any specific documents about this company, so be honest "
        "if you are unsure or would be guessing."
    )
    return call_groq(question, api_key=api_key, system_prompt=plain_system_prompt, model=model)


# ---------------------------------------------------------------------------
# Stage 10: Simple Agentic RAG workflow
# ---------------------------------------------------------------------------
#
# The "agent" makes one small decision before answering: it checks whether the
# retrieved chunks are actually relevant enough to the question (using the
# similarity scores from stage 5). If they are, it answers normally from the
# context (the standard RAG path). If none of the retrieved chunks clear the
# relevance threshold, the agent recognises this is likely outside the
# knowledge base's scope and routes to a safe fallback response instead of
# forcing the LLM to answer from weak/irrelevant context.

def agentic_answer(kb: KnowledgeBase, question: str, api_key: str | None = None,
                    top_k: int = TOP_K, model: str = GROQ_MODEL) -> dict:
    retrieved = kb.retrieve(question, top_k=top_k)
    best_score = retrieved[0][1] if retrieved else 0.0

    if best_score < RELEVANCE_THRESHOLD:
        # Agent decision: low-confidence retrieval -> don't guess, route to contact info.
        fallback = (
            "I don't have specific information about that in Gosafe's knowledge base. "
            "For an accurate answer, please contact the Gosafe office at Shop No. 5, "
            "Stadium Road, Near KTC, Fatorda, Madgaon, Goa 403601, and our team will help you directly."
        )
        return {
            "answer": fallback,
            "sources": [],
            "route": "fallback (low retrieval confidence)",
            "retrieved": retrieved,
        }

    prompt = build_prompt(question, retrieved)
    answer = call_groq(prompt, api_key=api_key, model=model)
    sources = sorted({chunk.source for chunk, score in retrieved})
    return {
        "answer": answer,
        "sources": sources,
        "route": "RAG (context found)",
        "retrieved": retrieved,
    }


# ---------------------------------------------------------------------------
# Convenience: a single "answer_question" call combining stages 4-8
# ---------------------------------------------------------------------------

def answer_question(kb: KnowledgeBase, question: str, api_key: str | None = None,
                     top_k: int = TOP_K, model: str = GROQ_MODEL) -> dict:
    retrieved = kb.retrieve(question, top_k=top_k)
    prompt = build_prompt(question, retrieved)
    answer = call_groq(prompt, api_key=api_key, model=model)
    sources = sorted({chunk.source for chunk, score in retrieved})
    return {"answer": answer, "sources": sources, "retrieved": retrieved}
