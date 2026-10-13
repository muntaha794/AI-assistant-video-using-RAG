import html
import re
import time

import streamlit as st

from vector_store import (
    build_vector_store,
    chunk_count,
    load_vector_store,
)

st.set_page_config(page_title="Meeting Recall", page_icon="🎙️", layout="wide")

SAMPLE = """Priya: Okay, let's start with the launch timeline. Engineering says the beta is on track for March 14th, but QA flagged two blockers in the payments flow.
Marcus: The payment blockers are mostly around refunds. If we cut partial refunds from the beta, we can still hit the date. I'd recommend that.
Priya: Agreed. Decision: partial refunds move to the post-beta release. Marcus owns the follow-up ticket.
Lena: On marketing, the landing page copy is approved. We still need final screenshots from design by next Wednesday.
Dev: I can get those to you by Tuesday. Also, budget-wise, we're 12 percent under on paid acquisition, so we have room to add a second creative test.
Priya: Good. Lena, go ahead and plan the second creative test. Let's budget up to four thousand dollars.
Marcus: One risk I want to raise: the support team has not been trained on the new onboarding flow. We should schedule a training session before the beta opens.
Lena: I'll set up the training for March 10th and invite the whole support team.
Priya: Last item: customer interviews. We have five scheduled this week. Dev, please share the notes in the product channel by Friday.
Dev: Will do. Summary: refunds deferred, screenshots by Tuesday, support training on the 10th, interview notes Friday."""

# ---------- styling ----------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,700&family=Manrope:wght@400;500;600&display=swap');

:root{
  --bg:#0E1319; --surface:#151C25; --surface2:#1B2430; --line:#263241;
  --text:#E7EBF0; --muted:#8C9AAB; --accent:#F2B84B; --accent-soft:rgba(242,184,75,.14);
}
html, body, [class*="css"], .stApp { font-family:'Manrope',sans-serif; color:var(--text); }
.stApp { background:
   radial-gradient(900px 400px at 85% -10%, rgba(242,184,75,.07), transparent 60%),
   var(--bg); }
header[data-testid="stHeader"]{ background:transparent; }
.block-container{ padding-top:2.2rem; max-width:1100px; }
section[data-testid="stSidebar"]{ background:var(--surface); border-right:1px solid var(--line); }

h1.hero{ font-family:'Fraunces',serif; font-weight:700; font-size:3rem; line-height:1.05;
  letter-spacing:-.02em; margin:0 0 .4rem 0; }
.hero-sub{ color:var(--muted); font-size:1.02rem; max-width:560px; margin-bottom:1.6rem; }

.stat-row{ display:flex; gap:12px; margin-bottom:1.6rem; flex-wrap:wrap; }
.stat{ background:var(--surface); border:1px solid var(--line); border-radius:12px;
  padding:12px 18px; min-width:130px; }
.stat b{ font-family:'Fraunces',serif; font-size:1.6rem; display:block; line-height:1.1; }
.stat span{ color:var(--muted); font-size:.82rem; }

/* inputs */
.stTextInput input{ background:var(--surface2)!important; border:1px solid var(--line)!important;
  border-radius:12px!important; color:var(--text)!important; padding:.85rem 1rem!important; font-size:1.05rem!important; }
.stTextInput input:focus{ border-color:var(--accent)!important; box-shadow:0 0 0 3px var(--accent-soft)!important; }
.stTextArea textarea{ background:var(--surface2)!important; border:1px solid var(--line)!important; color:var(--text)!important; border-radius:12px!important; }
.stButton>button{ border-radius:10px; border:1px solid var(--line); background:var(--surface2); color:var(--text); font-weight:600; }
.stButton>button:hover{ border-color:var(--accent); color:var(--accent); }
.stButton>button[kind="primary"]{ background:var(--accent); color:#1A1405; border-color:var(--accent); }
.stButton>button[kind="primary"]:hover{ filter:brightness(1.08); color:#1A1405; }
.stTabs [data-baseweb="tab-list"]{ gap:6px; border-bottom:1px solid var(--line); }
.stTabs [aria-selected="true"]{ color:var(--accent)!important; }

/* result cards: styled like transcript passages */
.hit{ background:var(--surface); border:1px solid var(--line); border-left:3px solid var(--accent);
  border-radius:4px 14px 14px 4px; padding:16px 20px; margin:0 0 14px 0; }
.hit-top{ display:flex; justify-content:space-between; align-items:center; margin-bottom:10px; gap:12px; }
.hit-title{ font-weight:600; font-size:.9rem; color:var(--text); }
.hit-title small{ color:var(--muted); font-weight:500; margin-left:6px; }
.meter{ display:flex; align-items:center; gap:8px; color:var(--muted); font-size:.8rem; white-space:nowrap; }
.meter i{ display:block; width:90px; height:6px; border-radius:99px; background:var(--surface2); overflow:hidden; }
.meter i u{ display:block; height:100%; background:var(--accent); border-radius:99px; }
.hit p{ margin:0; line-height:1.7; font-size:.98rem; color:#D4DAE2; white-space:pre-wrap; }
mark{ background:var(--accent-soft); color:var(--accent); padding:0 3px; border-radius:4px; }

.chunk{ background:var(--surface); border:1px solid var(--line); border-radius:10px;
  padding:12px 16px; margin-bottom:10px; font-size:.92rem; color:#C4CCD6; white-space:pre-wrap; line-height:1.6; }
.chunk b{ color:var(--accent); font-size:.8rem; display:block; margin-bottom:4px; }
.empty{ border:1px dashed var(--line); border-radius:14px; padding:36px; text-align:center; color:var(--muted); }
#MainMenu, footer{ visibility:hidden; }
</style>
""",
    unsafe_allow_html=True,
)


# ---------- helpers ----------
@st.cache_resource(show_spinner="Loading embedding model…")
def get_store():
    return load_vector_store()


def highlight(text: str, query: str) -> str:
    safe = html.escape(text)
    words = {w for w in re.findall(r"\w+", query) if len(w) > 2}
    if not words:
        return safe
    pattern = re.compile(r"\b(" + "|".join(re.escape(html.escape(w)) for w in words) + r")\b", re.I)
    return pattern.sub(r"<mark>\1</mark>", safe)


def match_pct(distance: float) -> int:
    # Chroma returns a distance (lower = closer). Map it onto a 0-100 scale for display.
    return int(round(100 / (1 + max(distance, 0))))


# ---------- sidebar ----------
with st.sidebar:
    st.markdown("### Transcript")
    uploaded = st.file_uploader("Upload a .txt file", type=["txt"], label_visibility="collapsed")
    if "draft" not in st.session_state:
        st.session_state.draft = ""
    if uploaded is not None:
        st.session_state.draft = uploaded.read().decode("utf-8", errors="ignore")
    if st.button("Use sample meeting", use_container_width=True):
        st.session_state.draft = SAMPLE
    st.text_area("Or paste text", key="draft", height=200, placeholder="Paste your meeting transcript here…")

    st.markdown("### Indexing")
    chunk_size = st.slider("Chunk size (characters)", 200, 1500, 500, 50)
    overlap = st.slider("Overlap", 0, 300, 50, 10)
    build = st.button("Index transcript", type="primary", use_container_width=True)

    st.markdown("### Search")
    k = st.slider("Passages to return", 1, 10, 4)

if build:
    text = st.session_state.draft.strip()
    if not text:
        st.sidebar.error("Add a transcript first, by upload or paste.")
    elif overlap >= chunk_size:
        st.sidebar.error("Overlap must be smaller than chunk size.")
    else:
        t0 = time.time()
        with st.spinner("Splitting and embedding…"):
            build_vector_store(text, chunk_size, overlap)
            get_store.clear()
        st.session_state.indexed_words = len(text.split())
        st.session_state.build_secs = time.time() - t0
        st.sidebar.success("Transcript indexed.")

# ---------- header ----------
n_chunks = chunk_count()
st.markdown('<h1 class="hero">Meeting Recall</h1>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-sub">Ask a question in plain words and jump straight to the part of the meeting where it came up.</div>',
    unsafe_allow_html=True,
)

words = st.session_state.get("indexed_words")
secs = st.session_state.get("build_secs")
st.markdown(
    f"""<div class="stat-row">
      <div class="stat"><b>{n_chunks}</b><span>passages indexed</span></div>
      <div class="stat"><b>{f"{words:,}" if words else "–"}</b><span>words this session</span></div>
      <div class="stat"><b>{f"{secs:.1f}s" if secs else "–"}</b><span>last index time</span></div>
    </div>""",
    unsafe_allow_html=True,
)

tab_search, tab_chunks = st.tabs(["Search", "All passages"])

# ---------- search ----------
with tab_search:
    if n_chunks == 0:
        st.markdown(
            '<div class="empty">Nothing to search yet.<br>Add a transcript in the sidebar and select <b>Index transcript</b>.</div>',
            unsafe_allow_html=True,
        )
    else:
        query = st.text_input(
            "Search",
            placeholder="What did we decide about refunds?",
            label_visibility="collapsed",
        )
        examples = ["What was decided?", "Who owns which action?", "Any risks mentioned?", "Budget"]
        cols = st.columns(len(examples))
        for col, ex in zip(cols, examples):
            if col.button(ex, key=f"ex_{ex}", use_container_width=True):
                query = ex

        if query:
            t0 = time.time()
            results = get_store().similarity_search_with_score(query, k=k)
            elapsed = (time.time() - t0) * 1000
            st.caption(f"{len(results)} passages in {elapsed:.0f} ms")

            export = [f"# Results for: {query}\n"]
            for doc, dist in results:
                idx = doc.metadata.get("chunk_index", "?")
                pct = match_pct(dist)
                st.markdown(
                    f"""<div class="hit">
                      <div class="hit-top">
                        <div class="hit-title">Passage {idx}<small>{len(doc.page_content)} characters</small></div>
                        <div class="meter">{pct}% match<i><u style="width:{pct}%"></u></i></div>
                      </div>
                      <p>{highlight(doc.page_content, query)}</p>
                    </div>""",
                    unsafe_allow_html=True,
                )
                export.append(f"## Passage {idx} ({pct}% match)\n\n{doc.page_content}\n")

            st.download_button(
                "Download results (.md)",
                "\n".join(export),
                file_name="search_results.md",
                mime="text/markdown",
            )

# ---------- all chunks ----------
with tab_chunks:
    if n_chunks == 0:
        st.markdown('<div class="empty">Index a transcript to see how it was split.</div>', unsafe_allow_html=True)
    else:
        data = get_store().get()
        rows = sorted(zip(data["metadatas"], data["documents"]), key=lambda r: r[0].get("chunk_index", 0))
        for meta, text in rows:
            st.markdown(
                f'<div class="chunk"><b>Passage {meta.get("chunk_index")}</b>{html.escape(text)}</div>',
                unsafe_allow_html=True,
            )