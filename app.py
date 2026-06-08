"""
Patent Intelligence Dashboard
Streamlit app for semantic patent search and AI-powered analysis.
"""

import streamlit as st
import json
import os
import re
import numpy as np

# ---------------------------------------------------------------------------
# Page config — must be first Streamlit call
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Patent Intelligence Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS for premium look
# ---------------------------------------------------------------------------

st.markdown("""
<style>
    /* Import Inter font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* Global font */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Header gradient banner */
    .dashboard-header {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        border: 1px solid rgba(255,255,255,0.08);
    }
    .dashboard-header h1 {
        color: #ffffff;
        font-size: 2rem;
        font-weight: 800;
        margin: 0 0 0.4rem 0;
        letter-spacing: -0.5px;
    }
    .dashboard-header p {
        color: rgba(255,255,255,0.6);
        font-size: 0.95rem;
        margin: 0;
        font-weight: 400;
    }

    /* Risk score cards */
    .risk-card {
        padding: 1.25rem;
        border-radius: 12px;
        text-align: center;
        border: 1px solid rgba(255,255,255,0.06);
    }
    .risk-card.low { background: linear-gradient(135deg, #064e3b, #065f46); }
    .risk-card.medium { background: linear-gradient(135deg, #78350f, #92400e); }
    .risk-card.high { background: linear-gradient(135deg, #7f1d1d, #991b1b); }
    .risk-card .score { font-size: 2.5rem; font-weight: 800; color: #fff; }
    .risk-card .label { font-size: 0.8rem; color: rgba(255,255,255,0.7); text-transform: uppercase; letter-spacing: 1px; margin-top: 0.25rem; }

    /* Overlap badge */
    .overlap-badge {
        display: inline-block;
        padding: 0.3rem 0.75rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .overlap-low { background: #064e3b; color: #6ee7b7; }
    .overlap-medium { background: #78350f; color: #fcd34d; }
    .overlap-high { background: #7f1d1d; color: #fca5a5; }

    /* Section cards */
    .section-card {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }
    .section-card h3 {
        font-size: 1rem;
        font-weight: 700;
        color: #e2e8f0;
        margin: 0 0 0.75rem 0;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* Chunk preview */
    .chunk-preview {
        background: rgba(0,0,0,0.2);
        border: 1px solid rgba(255,255,255,0.05);
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 0.75rem;
        font-size: 0.85rem;
    }
    .chunk-preview .chunk-meta {
        color: #818cf8;
        font-weight: 600;
        font-size: 0.8rem;
        margin-bottom: 0.4rem;
    }

    /* Patent library sidebar */
    .patent-chip {
        background: rgba(99,102,241,0.15);
        border: 1px solid rgba(99,102,241,0.3);
        border-radius: 8px;
        padding: 0.5rem 0.75rem;
        font-size: 0.8rem;
        font-weight: 600;
        color: #a5b4fc;
        display: inline-block;
        margin: 0.15rem 0;
    }

    /* Hide default streamlit footer */
    footer { visibility: hidden; }

    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 0.5rem 1.25rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASE_DIR = os.path.expanduser("~/Desktop/patent-intelligence")
CHUNKS_DIR = os.path.join(BASE_DIR, "chunks")
EMBEDDINGS_DIR = os.path.join(BASE_DIR, "embeddings")
OLLAMA_LLM_MODEL = "gemma:7b"

OVERLAP_CATEGORIES = [
    "Communication Coaching",
    "Emotion Detection",
    "Conversation Guidance",
    "Workplace Communication",
    "Interview Automation",
    "Language Learning",
    "Cultural Fluency",
]

# ---------------------------------------------------------------------------
# Cached resource loaders
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading AI models...")
def load_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("BAAI/bge-small-en-v1.5")


@st.cache_resource(show_spinner="Loading patent index...")
def load_faiss_index():
    import faiss
    index_path = os.path.join(EMBEDDINGS_DIR, "patent_index.faiss")
    return faiss.read_index(index_path)


@st.cache_resource(show_spinner="Loading patent metadata...")
def load_metadata():
    meta_path = os.path.join(EMBEDDINGS_DIR, "embeddings.json")
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_patent_list():
    """List all patent folders in chunks/."""
    if not os.path.isdir(CHUNKS_DIR):
        return []
    patents = sorted([
        d for d in os.listdir(CHUNKS_DIR)
        if os.path.isdir(os.path.join(CHUNKS_DIR, d))
    ])
    return patents


def read_chunk_file(patent_id, filename):
    """Read a chunk text file for a given patent."""
    path = os.path.join(CHUNKS_DIR, patent_id, filename)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    return None


def get_chunk_files(patent_id):
    """List available chunk files for a patent."""
    patent_path = os.path.join(CHUNKS_DIR, patent_id)
    if not os.path.isdir(patent_path):
        return []
    return sorted([
        f for f in os.listdir(patent_path)
        if f.endswith(".txt")
    ])


# ---------------------------------------------------------------------------
# FAISS search
# ---------------------------------------------------------------------------

def search_patents(query, top_k=8):
    """Run semantic search and return (chunks, context_string)."""
    model = load_model()
    index = load_faiss_index()
    metadata = load_metadata()

    query_embedding = model.encode(query).astype("float32")
    D, I = index.search(np.array([query_embedding]), top_k)

    chunks = []
    context_parts = []

    for rank, idx in enumerate(I[0], start=1):
        if idx < 0 or idx >= len(metadata):
            continue
        chunk = metadata[idx]
        chunks.append({
            "rank": rank,
            "patent": chunk["patent"],
            "file": chunk["file"],
            "text": chunk["text"],
        })
        # Truncate text (max 2000 chars per chunk for deeper context locally)
        context_parts.append(
            f"PATENT: {chunk['patent']}\n"
            f"FILE: {chunk['file']}\n\n"
            f"{chunk['text'][:2000]}\n"
        )

    context = "\n---\n".join(context_parts)
    return chunks, context


# ---------------------------------------------------------------------------
# Ollama LLM call with structured JSON output
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an expert patent analyst working for a startup founder.

The startup is Verity — an AI platform helping users improve:
- Workplace communication
- Social confidence
- Professional conversations
- Cultural fluency
- Communication coaching

Your job is to analyze patent context retrieved via semantic search and return a structured JSON report.

CRITICAL INSTRUCTIONS:
{
  "executive_summary": "A concise 2-3 sentence summary of what the patent landscape contains, why the retrieved patents matter, and major conclusions.",
  "risk_score": <integer 1-10>,
  "risk_level": "<Low|Medium|High>",
  "key_findings": ["finding 1", "finding 2", ...],
  "relevant_patents": [
    {"patent_number": "xxxxxxxxx", "retrieval_rank": 1, "reason": "why this patent matters"}
  ],
  "overlap_analysis": {
    "Communication Coaching": "<Low|Medium|High>",
    "Emotion Detection": "<Low|Medium|High>",
    "Conversation Guidance": "<Low|Medium|High>",
    "Workplace Communication": "<Low|Medium|High>",
    "Interview Automation": "<Low|Medium|High>",
    "Language Learning": "<Low|Medium|High>",
    "Cultural Fluency": "<Low|Medium|High>"
  },
  "differentiation_opportunities": ["opportunity 1", "opportunity 2", ...],
  "recommended_actions": ["action 1", "action 2", ...]
}

3. risk_score: 1 = no risk, 10 = critical infringement risk.
4. risk_level: Low (1-3), Medium (4-6), High (7-10).
5. Be specific — cite patent numbers and claim details where relevant.
6. Focus on what matters to a founder deciding whether to pivot, differentiate, or proceed.
"""


def analyze_patents(query, context):
    """Call local Ollama LLM and return parsed JSON dict (or None on failure)."""
    import ollama

    user_message = (
        f"Question:\n{query}\n\n"
        f"Patent Context:\n{context}\n\n"
        f"Return your analysis as valid JSON only."
    )

    try:
        response = ollama.chat(
            model=OLLAMA_LLM_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            options={"temperature": 0.2}
        )
        raw = response['message']['content'].strip()
        parsed = parse_llm_json(raw)
        
        # Attach metrics
        parsed["_metrics"] = {
            "prompt_tokens": response.get("prompt_eval_count", 0),
            "completion_tokens": response.get("eval_count", 0),
            "total_time_s": response.get("total_duration", 0) / 1e9,
            "eval_time_s": response.get("eval_duration", 0) / 1e9,
        }
        return parsed
    except Exception as e:
        st.error(
            f"LLM call failed: {e}. Make sure Ollama is running and "
            f"`ollama pull {OLLAMA_LLM_MODEL}` has been run."
        )
        return None


def parse_llm_json(raw_text):
    """Attempt to parse JSON from LLM response, handling common issues."""
    # Strip markdown code fences if present
    cleaned = re.sub(r'^```(?:json)?\s*', '', raw_text, flags=re.MULTILINE)
    cleaned = re.sub(r'```\s*$', '', cleaned, flags=re.MULTILINE)
    cleaned = cleaned.strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # Try to find JSON object in the text
        match = re.search(r'\{.*\}', cleaned, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
        # Return raw text wrapped in a fallback structure
        return {
            "executive_summary": cleaned[:1000],
            "risk_score": 0,
            "risk_level": "Unknown",
            "key_findings": ["Could not parse structured output — raw response shown in executive summary."],
            "relevant_patents": [],
            "overlap_analysis": {},
            "differentiation_opportunities": [],
            "recommended_actions": ["Re-run the analysis or check the LLM response."],
            "_raw": cleaned,
        }


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

def render_risk_card(score, level):
    """Render the risk assessment card."""
    css_class = "low"
    if level == "High":
        css_class = "high"
    elif level == "Medium":
        css_class = "medium"

    priority = "Low"
    if score >= 7:
        priority = "High"
    elif score >= 4:
        priority = "Medium"

    cols = st.columns(3)
    with cols[0]:
        st.markdown(f"""
        <div class="risk-card {css_class}">
            <div class="score">{score}</div>
            <div class="label">Risk Score</div>
        </div>
        """, unsafe_allow_html=True)
    with cols[1]:
        st.markdown(f"""
        <div class="risk-card {css_class}">
            <div class="score" style="font-size:1.5rem">{level}</div>
            <div class="label">Risk Level</div>
        </div>
        """, unsafe_allow_html=True)
    with cols[2]:
        st.markdown(f"""
        <div class="risk-card {css_class}">
            <div class="score" style="font-size:1.5rem">{priority}</div>
            <div class="label">Read Priority</div>
        </div>
        """, unsafe_allow_html=True)


def render_overlap_grid(overlap):
    """Render overlap analysis as a colored category grid."""
    if not overlap:
        st.info("No overlap data available.")
        return

    cols = st.columns(4)
    for i, category in enumerate(OVERLAP_CATEGORIES):
        level = overlap.get(category, "N/A")
        css_class = "overlap-low"
        if level == "High":
            css_class = "overlap-high"
        elif level == "Medium":
            css_class = "overlap-medium"

        with cols[i % 4]:
            st.markdown(f"""
            <div style="margin-bottom:0.75rem">
                <div style="font-size:0.8rem; color:#94a3b8; margin-bottom:0.25rem; font-weight:500">{category}</div>
                <span class="overlap-badge {css_class}">{level}</span>
            </div>
            """, unsafe_allow_html=True)


def render_bullet_list(items, icon="•"):
    """Render a list of items as styled bullets."""
    if not items:
        st.caption("None identified.")
        return
    for item in items:
        st.markdown(f"&nbsp;&nbsp;{icon}&nbsp;&nbsp;{item}")


def render_patents_table(patents):
    """Render relevant patents as a table."""
    if not patents:
        st.caption("No patents identified.")
        return
    for p in patents:
        num = p.get("patent_number", "Unknown")
        rank = p.get("retrieval_rank", "—")
        reason = p.get("reason", "")
        st.markdown(f"""
        <div class="chunk-preview">
            <div class="chunk-meta">#{rank} — {num}</div>
            <div style="color:#cbd5e1; font-size:0.85rem">{reason}</div>
        </div>
        """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Sidebar — Patent Library Browser
# ---------------------------------------------------------------------------

def render_sidebar():
    """Patent library browser in the sidebar."""
    st.sidebar.markdown("## Patent Library")
    st.sidebar.caption("Browse your analyzed patent portfolio")

    patents = get_patent_list()
    if not patents:
        st.sidebar.warning("No patents found in chunks/ directory.")
        return

    selected = st.sidebar.selectbox(
        "Select Patent",
        ["— Select a Patent —"] + patents,
        key="patent_selector",
    )

    if selected == "— Select a Patent —":
        st.sidebar.markdown("---")
        st.sidebar.caption(f"{len(patents)} patents available")
        return

    st.sidebar.markdown("---")

    # Show available chunk files
    files = get_chunk_files(selected)
    st.sidebar.caption(f"{len(files)} chunk files")

    # Display key sections
    display_order = [
        ("abstract.txt", "Abstract"),
        ("claim_1.txt", "Claim 1"),
        ("independent_claims.txt", "Independent Claims"),
        ("key_claims.txt", "Key Claims"),
        ("invention_summary.txt", "Invention Summary"),
        ("metadata.txt", "Metadata"),
    ]

    for filename, label in display_order:
        content = read_chunk_file(selected, filename)
        if content:
            with st.sidebar.expander(label, expanded=(filename == "abstract.txt")):
                st.text(content[:3000])
                if len(content) > 3000:
                    st.caption(f"... truncated ({len(content):,} chars total)")


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------

def main():
    # Sidebar
    render_sidebar()

    # Header
    st.markdown("""
    <div class="dashboard-header">
        <h1>Patent Intelligence Dashboard</h1>
        <p>Search and analyze patent portfolios using semantic search and AI-powered patent intelligence.</p>
    </div>
    """, unsafe_allow_html=True)

    # Search interface
    col_search, col_slider = st.columns([3, 1])
    with col_search:
        query = st.text_input(
            "Patent Question",
            placeholder="e.g., Which patents overlap with AI communication coaching?",
            key="query_input",
        )
    with col_slider:
        top_k = st.slider(
            "Number of Chunks",
            min_value=3,
            max_value=8,
            value=8,
            key="top_k_slider",
        )

    analyze_clicked = st.button("Analyze", type="primary", use_container_width=True)

    st.markdown("---")

    # Run analysis
    if analyze_clicked and query.strip():
        # Step 1: FAISS retrieval
        with st.spinner("Searching patent database..."):
            chunks, context = search_patents(query, top_k)

        if not chunks:
            st.error("No results found. Try a different query.")
            return

        # Step 2: LLM analysis
        with st.spinner("Analyzing patents..."):
            result = analyze_patents(query, context)

        if not result:
            st.error("Analysis failed. Please try again.")
            return

        # Store in session state for tab rendering
        st.session_state["last_query"] = query
        st.session_state["last_chunks"] = chunks
        st.session_state["last_context"] = context
        st.session_state["last_result"] = result

    # Render results if available
    if "last_result" in st.session_state:
        render_results(
            st.session_state["last_result"],
            st.session_state["last_chunks"],
            st.session_state["last_context"],
            st.session_state.get("last_query", ""),
        )
    else:
        # Empty state
        st.markdown("""
        <div style="text-align:center; padding:3rem; color:#64748b">
            <div style="font-size:1.1rem; font-weight:600; margin-bottom:0.5rem">Ready to analyze</div>
            <div style="font-size:0.9rem">Enter a patent question above and click Analyze to get started.</div>
        </div>
        """, unsafe_allow_html=True)


def render_results(result, chunks, context, query):
    """Render the full analysis dashboard."""

    # Tabs
    tab_analysis, tab_claims, tab_chunks = st.tabs([
        "Analysis",
        "Claims",
        "Retrieved Chunks",
    ])

    # ── Tab 1: Analysis ──────────────────────────────────────────────
    with tab_analysis:

        # Performance Metrics (if available)
        metrics = result.get("_metrics")
        if metrics:
            with st.expander("LLM Performance Metrics"):
                m_cols = st.columns(4)
                m_cols[0].metric("Total Time", f"{metrics['total_time_s']:.2f}s")
                m_cols[1].metric("Context Tokens", f"{metrics['prompt_tokens']}")
                m_cols[2].metric("Generated Tokens", f"{metrics['completion_tokens']}")
                
                tps = 0
                if metrics['eval_time_s'] > 0:
                    tps = metrics['completion_tokens'] / metrics['eval_time_s']
                m_cols[3].metric("Tokens / Sec", f"{tps:.1f} t/s")

        # Executive Summary
        st.markdown('<div class="section-card"><h3>Executive Summary</h3>', unsafe_allow_html=True)
        st.markdown(result.get("executive_summary", "No summary available."))
        st.markdown('</div>', unsafe_allow_html=True)

        # Risk Assessment
        st.markdown("#### Risk Assessment")
        render_risk_card(
            result.get("risk_score", 0),
            result.get("risk_level", "Unknown"),
        )
        st.markdown("")

        # Key Findings
        col_left, col_right = st.columns(2)

        with col_left:
            st.markdown("#### Key Findings")
            render_bullet_list(result.get("key_findings", []))

            st.markdown("")
            st.markdown("#### Differentiation Opportunities")
            render_bullet_list(result.get("differentiation_opportunities", []))

        with col_right:
            st.markdown("#### Recommended Actions")
            render_bullet_list(result.get("recommended_actions", []))

            st.markdown("")
            st.markdown("#### Relevant Patents")
            render_patents_table(result.get("relevant_patents", []))

        # Overlap Analysis — full width
        st.markdown("---")
        st.markdown("#### Overlap With Verity")
        render_overlap_grid(result.get("overlap_analysis", {}))

    # ── Tab 2: Claims ────────────────────────────────────────────────
    with tab_claims:
        st.markdown("#### Patent Claims Viewer")
        st.caption("Review claims for patents retrieved in the analysis.")

        # Deduplicate patents from chunks
        seen_patents = []
        for chunk in chunks:
            if chunk["patent"] not in seen_patents:
                seen_patents.append(chunk["patent"])

        if not seen_patents:
            st.info("No patents retrieved.")
        else:
            for patent_id in seen_patents:
                with st.expander(patent_id, expanded=False):
                    claim_files = [
                        ("claim_1.txt", "Claim 1"),
                        ("independent_claims.txt", "Independent Claims"),
                        ("key_claims.txt", "Key Claims"),
                        ("full_claims.txt", "Full Claims"),
                    ]
                    for filename, label in claim_files:
                        content = read_chunk_file(patent_id, filename)
                        if content:
                            st.markdown(f"**{label}**")
                            st.text(content[:5000])
                            if len(content) > 5000:
                                st.caption(f"... truncated ({len(content):,} chars)")
                            st.markdown("---")
                        else:
                            st.caption(f"{label}: not available")

    # ── Tab 3: Retrieved Chunks ──────────────────────────────────────
    with tab_chunks:
        st.markdown("#### RAG Context — Retrieved Chunks")
        st.caption(f"These {len(chunks)} chunks were sent to the LLM for analysis.")

        for chunk in chunks:
            st.markdown(f"""
            <div class="chunk-preview">
                <div class="chunk-meta">#{chunk['rank']} — {chunk['patent']} / {chunk['file']}</div>
                <div style="color:#cbd5e1; font-size:0.83rem; white-space:pre-wrap">{chunk['text'][:800]}</div>
            </div>
            """, unsafe_allow_html=True)

        # Full context in collapsible
        with st.expander("Full context string sent to LLM"):
            st.text(context[:10000])
            if len(context) > 10000:
                st.caption(f"... truncated ({len(context):,} chars total)")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
