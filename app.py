"""
app.py — AI Meeting Assistant
Day 1: file upload + text input + summarization tab only.
Run with: streamlit run app.py
"""

import streamlit as st
import yaml

# ── page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Meeting Assistant",
    page_icon="🎙️",
    layout="wide",
)

# ── load config ────────────────────────────────────────────────────────────────
with open("config/config.yaml") as f:
    CONFIG = yaml.safe_load(f)

# ── header ─────────────────────────────────────────────────────────────────────
st.title("🎙️ AI Meeting Assistant")
st.caption("Summarize meetings, extract action items, and ask questions — all from a transcript.")
st.divider()

# ── sidebar — transcript input ──────────────────────────────────────────────────
with st.sidebar:
    st.header("📄 Transcript")

    mode = st.radio(
        "How do you want to input the transcript?",
        ["Upload .txt file", "Paste text"],
        label_visibility="collapsed",
    )

    transcript = ""

    if mode == "Upload .txt file":
        uploaded = st.file_uploader("Upload transcript", type=["txt"])
        if uploaded:
            transcript = uploaded.read().decode("utf-8", errors="replace")
            st.success(f"✅ {uploaded.name} loaded ({len(transcript.split())} words)")

    else:
        transcript = st.text_area(
            "Paste transcript here",
            height=280,
            placeholder="Sarah: Let's start...\nJohn: Sure...",
        )

    # load sample button
    if st.button("Load sample transcript", use_container_width=True):
        try:
            with open("data/sample_transcripts/example_meeting.txt",encoding="utf-8") as f:
                transcript = f.read()
            st.success("Sample transcript loaded!")
            # store in session so it persists after button click
            st.session_state["transcript"] = transcript
        except FileNotFoundError:
            st.error("Sample file not found.")

    # sync session state
    if transcript:
        st.session_state["transcript"] = transcript

    if "transcript" in st.session_state and st.session_state["transcript"]:
        with st.expander("Preview"):
            st.text(st.session_state["transcript"][:500] + "...")

    st.divider()
    st.caption(f"Model: `{CONFIG['models']['summarizer']}`")

# ── guard — no transcript yet ──────────────────────────────────────────────────
if "transcript" not in st.session_state or not st.session_state["transcript"].strip():
    st.info("👈 Upload or paste a meeting transcript in the sidebar to get started.")
    st.stop()

transcript = st.session_state["transcript"]

# ── tabs (Day 1: only Summary tab, others coming soon) ─────────────────────────
tab_summary, tab_actions, tab_sentiment, tab_qa = st.tabs([
    "📝 Summary",
    "✅ Action items — coming Day 2",
    "😊 Sentiment — coming Day 3",
    "💬 Q&A — coming Day 4",
])

# ── Tab 1: Summary ─────────────────────────────────────────────────────────────
with tab_summary:
    st.subheader("Meeting summary")
    st.caption("Generates a concise overview of what was discussed.")

    col1, col2 = st.columns([1, 3])
    with col1:
        run = st.button("Generate summary", type="primary", use_container_width=True)

    if run:
        with st.spinner("Summarizing... (first run downloads the model, ~1 min)"):
            from src.summarizer import summarize
            summary = summarize(transcript)
            st.session_state["summary"] = summary

    if "summary" in st.session_state:
        st.markdown("**Summary:**")
        st.info(st.session_state["summary"])

        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Words in transcript", len(transcript.split()))
        col_b.metric("Words in summary",    len(st.session_state["summary"].split()))
        col_c.metric("Compression ratio",
                     f"{len(transcript.split()) // max(len(st.session_state['summary'].split()), 1)}x")
    else:
        st.caption("Click the button above to generate a summary.")

# ── Tab 2-4: placeholders for now ──────────────────────────────────────────────
with tab_actions:
    st.info("🚧 Coming on Day 2 — action item extraction.")

with tab_sentiment:
    st.info("🚧 Coming on Day 3 — sentiment analysis.")

with tab_qa:
    st.info("🚧 Coming on Day 4 — RAG-powered Q&A.")