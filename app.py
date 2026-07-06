"""
app.py — AI Meeting Assistant
Day 1: file upload + text input + summarization tab only.
Run with: streamlit run app.py
"""

import streamlit as st
import yaml
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
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

# ── build RAG index whenever transcript changes ────────────────────────────────
if "transcript" in st.session_state and st.session_state["transcript"].strip():
    current = st.session_state["transcript"]
    if st.session_state.get("rag_transcript") != current:
        with st.spinner("Indexing transcript for Q&A..."):
            from src.rag import build_index
            chunks, index = build_index(current)
            st.session_state["rag_chunks"]     = chunks
            st.session_state["rag_index"]      = index
            st.session_state["rag_transcript"] = current

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

# ── Tab 2: Action items ────────────────────────────────────────────────────────
with tab_actions:
    st.subheader("Action items")
    st.caption("Extracts tasks and responsible people from the transcript.")

    col1, _ = st.columns([1, 3])
    with col1:
        run_actions = st.button("Extract action items", type="primary", use_container_width=True)

    if run_actions:
        with st.spinner("Extracting tasks... (first run downloads the model ~1 min)"):
            from src.action_extractor import extract_actions
            actions = extract_actions(st.session_state["transcript"])
            st.session_state["actions"] = actions

    if "actions" in st.session_state:
        actions = st.session_state["actions"]
        st.caption(f"{len(actions)} action item(s) found")
        st.divider()

        for i, item in enumerate(actions, 1):
            col_check, col_text = st.columns([0.05, 0.95])

            with col_check:
                st.checkbox("Done", key=f"action_{i}", value=False, label_visibility="hidden")

            with col_text:
                st.markdown(f"**{item['task']}**")
                if item["owner"] != "Unassigned":
                    st.caption(f"👤 {item['owner']}")
                else:
                    st.caption("👤 Owner not specified")

            st.divider()

        # download as text file
        action_text = "\n".join(
            [f"[ ] {a['task']} — {a['owner']}" for a in actions]
        )
        st.download_button(
            label="Download action items",
            data=action_text,
            file_name="action_items.txt",
            mime="text/plain",
            use_container_width=False,
        )

    else:
        st.caption("Click the button above to extract action items.")

# ── Tab 3: Sentiment ───────────────────────────────────────────────────────────
with tab_sentiment:
    st.subheader("Sentiment analysis")
    st.caption("Analyzes the emotional tone of the meeting per speaker.")

    col1, _ = st.columns([1, 3])
    with col1:
        run_sentiment = st.button(
            "Analyze sentiment", type="primary", use_container_width=True
        )

    if run_sentiment:
        with st.spinner("Analyzing sentiment..."):
            from src.sentiment import analyze
            result = analyze(st.session_state["transcript"])
            st.session_state["sentiment"] = result

    if "sentiment" in st.session_state:
        result   = st.session_state["sentiment"]
        overall  = result["overall"]
        speakers = result["by_speaker"]
        turns    = result["turns"]

        # ── overall metric ─────────────────────────────────────────────
        st.divider()
        emoji_map = {"POSITIVE": "😊", "NEGATIVE": "😟", "NEUTRAL": "😐"}
        emoji     = emoji_map.get(overall["label"], "😐")
        col_m, col_chart = st.columns([1, 2])
        with col_m:
            st.metric(
                label="Overall meeting tone",
                value=f"{emoji} {overall['label'].capitalize()}",
                delta=f"{overall['score']}% of turns",
            )

        # ── per-speaker bar chart ──────────────────────────────────────
        if speakers:
            with col_chart:
                matplotlib.use("Agg")
                names     = list(speakers.keys())
                pos_vals  = [speakers[n].get("POSITIVE", 0) for n in names]
                neu_vals  = [speakers[n].get("NEUTRAL",  0) for n in names]
                neg_vals  = [speakers[n].get("NEGATIVE", 0) for n in names]
                x         = np.arange(len(names))
                width     = 0.25

                fig, ax = plt.subplots(figsize=(5, 2.8))
                ax.bar(x - width,     pos_vals, width, label="Positive", color="#22c55e")
                ax.bar(x,             neu_vals, width, label="Neutral",  color="#94a3b8")
                ax.bar(x + width,     neg_vals, width, label="Negative", color="#ef4444")

                ax.set_xticks(x)
                ax.set_xticklabels(names, fontsize=10)
                ax.set_ylabel("Turns", fontsize=9)
                ax.set_title("Sentiment by speaker", fontsize=10)
                ax.legend(fontsize=9)
                ax.spines[["top", "right"]].set_visible(False)
                plt.tight_layout()

                st.pyplot(fig)
                plt.close(fig)
        # ── turn by turn breakdown ─────────────────────────────────────
        st.divider()
        st.caption("Turn-by-turn breakdown")

        for turn in turns:
            icon_map = {"POSITIVE": "🟢", "NEGATIVE": "🔴", "NEUTRAL": "🟡"}
            icon     = icon_map.get(turn["label"], "⚪")
            col_a, col_b = st.columns([0.05, 0.95])

            with col_a:
                st.write(icon)
            with col_b:
                st.markdown(
                    f"**{turn['speaker']}** — {turn['text'][:80]}"
                    f"{'...' if len(turn['text']) > 80 else ''}"
                )
                st.caption(
                    f"{turn['label'].capitalize()} · {turn['score']}% confidence"
                )

    else:
        st.caption("Click the button above to analyze sentiment.")


# ── Tab 4: Q&A ─────────────────────────────────────────────────────────────────
with tab_qa:
    st.subheader("Ask a question about this meeting")
    st.caption("Powered by RAG — retrieves relevant parts of the transcript before answering.")

    # suggested questions as quick-click buttons
    st.markdown("**Try asking:**")
    col_q1, col_q2, col_q3 = st.columns(3)

    suggested = [
        "Who is responsible for the API?",
        "When is the next sync meeting?",
        "What did the client complain about?",
    ]

    clicked_suggestion = None
    with col_q1:
        if st.button(suggested[0], use_container_width=True):
            clicked_suggestion = suggested[0]
    with col_q2:
        if st.button(suggested[1], use_container_width=True):
            clicked_suggestion = suggested[1]
    with col_q3:
        if st.button(suggested[2], use_container_width=True):
            clicked_suggestion = suggested[2]

    # text input — pre-filled with suggestion if clicked
    question = st.text_input(
        "Your question",
        value=clicked_suggestion or "",
        placeholder="e.g. Who will fix the response time bug?",
    )

    col_ask, _ = st.columns([1, 3])
    with col_ask:
        ask = st.button("Get answer", type="primary", use_container_width=True)

    if ask and question.strip():
        if "rag_chunks" not in st.session_state:
            st.warning("Transcript not indexed yet — please wait a moment and try again.")
        else:
            with st.spinner("Searching the transcript..."):
                from src.rag import answer as rag_answer
                result = rag_answer(
                    question,
                    st.session_state["rag_chunks"],
                    st.session_state["rag_index"],
                )
                st.session_state["qa_result"]   = result
                st.session_state["qa_question"] = question

    # display result
    if "qa_result" in st.session_state:
        st.divider()
        st.markdown(f"**Q: {st.session_state['qa_question']}**")

        ans = st.session_state["qa_result"]["answer"]
        if "not discussed" in ans.lower():
            st.warning(f"🤷 {ans}")
        else:
            st.success(f"💬 {ans}")

        with st.expander("View retrieved context"):
            for i, chunk in enumerate(
                st.session_state["qa_result"]["context"], 1
            ):
                st.markdown(f"**Chunk {i}:**")
                st.text(chunk)
                st.divider()

    elif ask and not question.strip():
        st.warning("Please type a question first.")
    else:
        st.caption("Type a question above and click Get answer.")