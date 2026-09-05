"""
Gosafe Fire, Security and Safety - RAG Chatbot UI
===================================================
Run with:  streamlit run app.py

"""


import os
import streamlit as st

from rag_engine import KnowledgeBase, answer_question, generate_plain_answer, agentic_answer, GROQ_MODEL
from guardrails import screen_query, CLASSIFICATION_LABELS, REFUSAL_MESSAGES

# ---------------------------------------------------------------------------
# Page config + Gosafe-themed styling (red / white / charcoal, matches signage)
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Gosafe | Fire, Security and Safety Assistant",
    page_icon="🧯",
    layout="centered",
)

GOSAFE_RED = "#C8102E"
GOSAFE_DARK = "#1A1A1A"

st.markdown(
    """
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <style>
        html, body, [class*="css"] {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
            color: #000000 !important;
        }}
        .stApp {{
            background-color: #000000;
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            color: #f5f5f5; 
        }}
        section[data-testid="stSidebar"] {{
            background-color: {GOSAFE_DARK};
        }}
        section[data-testid="stSidebar"] * {{
            color: #f7f5f2 !important;
        }}
        .gosafe-header {{
            background: linear-gradient(90deg, {GOSAFE_RED} 0%, #8f0c1f 100%);
            padding: 22px 28px;
            border-radius: 10px;
            color: white;
            margin-bottom: 18px;
        }}
        .gosafe-header h1 {{
            margin: 0;
            font-size: 30px;
            font-weight: 800;
            letter-spacing: 0.5px;
        }}
        .gosafe-header p {{
            margin: 4px 0 0 0;
            font-size: 15px;
            opacity: 0.95;
        }}
        .gosafe-badge {{
            display: inline-block;
            background-color: {GOSAFE_RED};
            color: white;
            padding: 2px 10px;
            border-radius: 999px;
            font-size: 12px;
            font-weight: 600;
            margin-right: 6px;
        }}
        
        div[data-testid="stChatMessage"] {{
            border-radius: 12px;
            background-color: #1a1a1a;
            color: #f5f5f5;
        }}
        
        .source-pill {{
            display: inline-block;
            background-color: #fdeaea;
            color: {GOSAFE_RED};
            border: 1px solid {GOSAFE_RED};
            border-radius: 999px;
            padding: 2px 10px;
            font-size: 12px;
            margin: 2px 4px 2px 0;
        }}
        .stButton>button {{
            background-color: {GOSAFE_RED};
            color: white;
            border-radius: 8px;
            border: none;
        }}
        .stButton>button:hover {{
            background-color: #8f0c1f;
            color: white;
        }}
        .guardrail-badge {{
            display: inline-block;
            padding: 3px 12px;
            border-radius: 999px;
            font-size: 12px;
            font-weight: 700;
            margin-bottom: 8px;
        }}
        .badge-safe {{ background-color: #e7f7ea; color: #1e7d34; border: 1px solid #1e7d34; }}
        .badge-prompt_injection {{ background-color: #fdeaea; color: #c8102e; border: 1px solid #c8102e; }}
        .badge-false_authority {{ background-color: #f1e7fb; color: #6a1baa; border: 1px solid #6a1baa; }}
        .badge-restricted_topic {{ background-color: #fdf1e2; color: #b8600a; border: 1px solid #b8600a; }}
        .badge-constitutional_concern {{ background-color: #fdf9e2; color: #a68a00; border: 1px solid #a68a00; }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="gosafe-header">
        <h1>🧯 GOSAFE</h1>
        <p>Fire, Security and Safety &nbsp;|&nbsp; Shop No. 5, Stadium Road, Near KTC, Fatorda, Madgaon, Goa 403601</p>
        <p>Established 2011 &nbsp;•&nbsp; Suppression Systems &nbsp;•&nbsp; Hose Reels &nbsp;•&nbsp; Extinguishers &nbsp;•&nbsp; Fire Panels &nbsp;•&nbsp; CCTV &nbsp;•&nbsp; AMC</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar: API key, settings, agentic mode, comparison mode
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("### ⚙️ Settings")

    default_key = os.environ.get("GROQ_API_KEY", "")
    api_key = st.text_input(
        "Groq API Key",
        value="",
        type="password",
        placeholder="gsk_..." if not default_key else "Using GROQ_API_KEY from environment",
        help="Get a free key at console.groq.com. Leave blank to use the GROQ_API_KEY environment variable.",
    )
    effective_key = api_key.strip() or default_key or None

    st.markdown("---")
    agentic_mode = st.toggle(
        "🤖 Agentic RAG mode",
        value=True,
        help="When on, the assistant first checks whether retrieved knowledge is actually "
             "relevant before answering. If nothing relevant is found, it routes to a safe "
             "'contact us' fallback instead of guessing (Step 10 of the RAG flow).",
    )
    compare_mode = st.toggle(
        "⚖️ Compare RAG vs. plain LLM",
        value=False,
        help="Shows the RAG (grounded) answer side-by-side with an answer generated without "
             "any Gosafe documents, so you can see what retrieval adds (Step 9 of the RAG flow).",
    )
    top_k = st.slider("Chunks to retrieve (Top-K)", min_value=1, max_value=8, value=4)

    st.markdown("---")
    st.markdown("### 🛡️ Guardrails")
    st.caption("Every query passes through this 3-layer pipeline before reaching the RAG system.")
    layer1_on = st.toggle("Layer 1 - Rule-based filter", value=True,
                           help="Regex/keyword checks for restricted topics, prompt injection, and false authority claims.")
    layer2_on = st.toggle("Layer 2 - Constitutional moderation", value=True,
                           help="Rule-based moderation modeled on Article 19(2) reasonable-restriction grounds.")
    layer3_on = st.toggle("Layer 3 - LLM final check", value=True,
                           help="Only runs if Layers 1-2 pass. Uses an LLM to catch subtle/indirect attempts.")
    show_trace = st.checkbox("Show guardrail trace for each answer", value=False)

    st.markdown("---")
    st.markdown(f"**Model:** `{GROQ_MODEL}` via Groq API")
    st.markdown("**Knowledge base:** 6 local documents (company overview, services, products, "
                "team, AMC/compliance, contact & FAQ)")

    if st.button("🔄 Rebuild knowledge base index"):
        st.session_state.pop("kb", None)
        st.rerun()

# ---------------------------------------------------------------------------
# Build / cache the knowledge base (Stages 1-3 run once and are cached)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Indexing Gosafe knowledge base...")
def load_kb():
    return KnowledgeBase()

kb = load_kb()

# ---------------------------------------------------------------------------
# Chat state
# ---------------------------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Hello! 👋 I'm the Gosafe virtual assistant. I can help with questions about our "
                "fire extinguishers, suppression systems, hose reels, fire alarm panels, AMC plans, "
                "and more. How can I help you today?"
            ),
            "sources": [],
            "classification": None,
        }
    ]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🧯" if msg["role"] == "assistant" else "🧑"):
        if msg["role"] == "user" and msg.get("classification"):
            icon, label = CLASSIFICATION_LABELS.get(msg["classification"], ("⚪", "Unknown"))
            st.markdown(
                f'<span class="guardrail-badge badge-{msg["classification"]}">{icon} {label}</span>',
                unsafe_allow_html=True,
            )
        st.markdown(msg["content"])
        if msg.get("sources"):
            pills = "".join(f'<span class="source-pill">📄 {s}</span>' for s in msg["sources"])
            st.markdown(pills, unsafe_allow_html=True)
        if msg.get("trace"):
            with st.expander("🔍 Guardrail trace"):
                for line in msg["trace"]:
                    st.text(line)

# Suggested quick questions
st.markdown("**Try asking:**")
cols = st.columns(3)
suggestions = [
    "What fire suppression systems do you install?",
    "Who are your fire safety experts?",
    "What does your AMC plan cover?",
]
clicked_suggestion = None
for col, s in zip(cols, suggestions):
    if col.button(s, use_container_width=True):
        clicked_suggestion = s

# ---------------------------------------------------------------------------
# Handle input
# ---------------------------------------------------------------------------

user_input = st.chat_input("Ask about our fire safety products, services, or AMC plans...")
question = clicked_suggestion or user_input

if question:
    # ---- Guardrail screening happens BEFORE anything reaches the RAG pipeline ----
    guardrail_result = screen_query(
        question,
        api_key=effective_key,
        enable_layer1=layer1_on,
        enable_layer2=layer2_on,
        enable_layer3=layer3_on,
    )
    icon, label = CLASSIFICATION_LABELS.get(guardrail_result.classification, ("⚪", "Unknown"))

    st.session_state.messages.append({
        "role": "user",
        "content": question,
        "sources": [],
        "classification": guardrail_result.classification,
    })
    with st.chat_message("user", avatar="🧑"):
        st.markdown(
            f'<span class="guardrail-badge badge-{guardrail_result.classification}">{icon} {label}</span>',
            unsafe_allow_html=True,
        )
        st.markdown(question)

    with st.chat_message("assistant", avatar="🧯"):
        if not guardrail_result.passed:
            # Blocked - never reaches the RAG pipeline or a second LLM call for generation.
            refusal = REFUSAL_MESSAGES.get(
                guardrail_result.classification,
                "I'm not able to help with that request.",
            )
            st.warning(f"🛡️ Blocked at Layer {guardrail_result.layer_triggered}: {label}")
            st.markdown(refusal)
            if show_trace:
                with st.expander("🔍 Guardrail trace"):
                    for line in guardrail_result.trace:
                        st.text(line)
            st.session_state.messages.append({
                "role": "assistant",
                "content": refusal,
                "sources": [],
                "classification": None,
                "trace": guardrail_result.trace if show_trace else [],
            })

        elif not effective_key:
            st.error(
                "⚠️ No Groq API key found. Please enter your key in the sidebar, or set the "
                "GROQ_API_KEY environment variable before running the app."
            )
        else:
            try:
                with st.spinner("Retrieving relevant Gosafe documents..."):
                    if agentic_mode:
                        result = agentic_answer(kb, question, api_key=effective_key, top_k=top_k)
                        rag_answer = result["answer"]
                        rag_sources = result["sources"]
                        route_note = result["route"]
                    else:
                        result = answer_question(kb, question, api_key=effective_key, top_k=top_k)
                        rag_answer = result["answer"]
                        rag_sources = result["sources"]
                        route_note = "RAG (agentic mode off)"

                st.markdown(rag_answer)
                if rag_sources:
                    pills = "".join(f'<span class="source-pill">📄 {s}</span>' for s in rag_sources)
                    st.markdown(pills, unsafe_allow_html=True)
                st.caption(f"Route: {route_note}")
                if show_trace:
                    with st.expander("🔍 Guardrail trace"):
                        for line in guardrail_result.trace:
                            st.text(line)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": rag_answer,
                    "sources": rag_sources,
                    "classification": None,
                    "trace": guardrail_result.trace if show_trace else [],
                })

                if compare_mode:
                    with st.spinner("Generating plain (non-RAG) answer for comparison..."):
                        plain_answer = generate_plain_answer(question, api_key=effective_key)
                    with st.expander("⚖️ Compare: answer WITHOUT Gosafe's knowledge base"):
                        st.markdown(plain_answer)
                        st.caption(
                            "This answer was generated with no access to Gosafe's documents - "
                            "notice it may be generic, vague, or incorrect about company-specific details."
                        )

            except Exception as e:
                st.error(f"Something went wrong: {e}")

st.markdown("---")
st.caption(
    "🧯 Gosafe Fire, Security and Safety • Shop No. 5, Stadium Road, Near KTC, Fatorda, Madgaon, Goa 403601 "
    "• This assistant answers from Gosafe's internal knowledge base using Retrieval-Augmented Generation (RAG) "
    "powered by the Groq API. For quotes or emergencies, please contact the office directly."
)
st.caption(
    "🛡️ Guardrail Layer 2 uses a simplified, keyword-based proxy for Article 19(2) of the Constitution of "
    "India and is for demonstration purposes only - it is not a substitute for legal review or a certified "
    "moderation service."
)
