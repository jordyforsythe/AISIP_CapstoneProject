import streamlit as st
import numpy as np
import re
import string
import json
import os
import csv
from datetime import datetime

# ── Page config ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Nigerian Political Misinformation Detector",
    page_icon="🇳🇬",
    layout="centered"
)

# ── Styling ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .result-box {
        padding: 20px; border-radius: 10px;
        margin: 15px 0; text-align: center;
        font-size: 22px; font-weight: bold;
    }
    .human { background-color: #1a472a; color: #69db7c; border: 2px solid #69db7c; }
    .ai    { background-color: #6b2737; color: #ff6b6b; border: 2px solid #ff6b6b; }
    .info-box {
        background-color: #1e2130; border-radius: 8px;
        padding: 15px; margin: 10px 0;
        border-left: 4px solid #2196F3;
    }
    .save-box {
        background-color: #1a3a2a; border-radius: 8px;
        padding: 15px; margin: 10px 0;
        border-left: 4px solid #4CAF50;
    }
</style>
""", unsafe_allow_html=True)

# ── Storage file ──────────────────────────────────────────────────────────
# Uses a local CSV file to store all user submissions
# When deployed on Streamlit Cloud this persists within the session
# For permanent storage across sessions see the note at the bottom
SAVE_FILE = "user_submissions.csv"
FIELDNAMES = [
    "timestamp", "text_preview", "full_text",
    "prediction", "confidence", "word_count",
    "number_count", "lexical_diversity",
    "exclamation_count", "avg_sent_len",
    "user_feedback"
]

def load_submissions():
    """Load all saved submissions from CSV."""
    if not os.path.exists(SAVE_FILE):
        return []
    with open(SAVE_FILE, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)

def save_submission(data: dict):
    """Append a single submission to the CSV file."""
    file_exists = os.path.exists(SAVE_FILE)
    with open(SAVE_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if not file_exists:
            writer.writeheader()   # write header only once
        writer.writerow(data)

# ── Feature extraction ────────────────────────────────────────────────────
def extract_features(text):
    try:
        from nltk.tokenize import word_tokenize, sent_tokenize
        from nltk.corpus import stopwords
        import nltk
        nltk.download('punkt',     quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('punkt_tab', quiet=True)
        words = word_tokenize(text.lower())
        sents = sent_tokenize(text)
        stops = set(stopwords.words('english'))
    except Exception:
        words = text.lower().split()
        sents = [s for s in text.split('.') if s.strip()]
        stops = set()

    alpha  = [w for w in words if w.isalpha()]
    unique = set(alpha)
    stop_w = [w for w in alpha if w in stops]
    nums   = re.findall(r'\d+\.?\d*', text)
    caps   = re.findall(r'\b[A-Z]{2,}\b', text)
    punct  = [c for c in text if c in string.punctuation]

    return {
        'char_count':        len(text),
        'word_count':        len(alpha),
        'sent_count':        max(len(sents), 1),
        'avg_word_len':      round(np.mean([len(w) for w in alpha]), 3) if alpha else 0,
        'avg_sent_len':      round(len(alpha) / max(len(sents), 1), 2),
        'lexical_diversity': round(len(unique) / max(len(alpha), 1), 3),
        'stopword_ratio':    round(len(stop_w)  / max(len(alpha), 1), 3),
        'punct_ratio':       round(len(punct)   / max(len(text), 1), 3),
        'number_count':      len(nums),
        'exclamation_count': text.count('!'),
        'question_count':    text.count('?'),
        'caps_count':        len(caps),
    }

def rule_based_predict(text):
    f = extract_features(text)
    score = 0.0

    # Numerical claims — strongest AI signal
    if f['number_count'] >= 3:   score += 0.35
    elif f['number_count'] >= 1: score += 0.15

    # Lexical diversity
    if f['lexical_diversity'] > 0.85:   score += 0.20
    elif f['lexical_diversity'] > 0.75: score += 0.10

    # Sentence length
    if f['avg_sent_len'] > 30:   score += 0.20
    elif f['avg_sent_len'] > 20: score += 0.10

    # Exclamation marks — human signal
    if f['exclamation_count'] >= 2: score -= 0.15
    elif f['exclamation_count'] == 1: score -= 0.05

    # Nigerian Pidgin — strong human signal
    pidgin = ['don','dey','dem','na','wetin','abeg','wahala',
              'oga','naija','sha','ehn','nau','joor','comot','abi']
    pidgin_hits = sum(1 for w in pidgin if w in text.lower())
    if pidgin_hits >= 3:   score -= 0.30
    elif pidgin_hits >= 1: score -= 0.15

    # Authority/formal misinformation markers — AI signal
    authority = [
        'classified','verified','intelligence','exclusively',
        'conclusively','definitively','authenticated','forensic',
        'according to multiple','credible sources','leaked',
        'secret','whistleblower','statistical analysis',
        'comprehensive','investigation confirms'
    ]
    auth_hits = sum(1 for phrase in authority if phrase in text.lower())
    if auth_hits >= 3:   score += 0.30
    elif auth_hits >= 1: score += 0.15

    # CAPS
    if f['caps_count'] >= 5: score += 0.10

    # Length
    if f['char_count'] > 400:   score += 0.10
    elif f['char_count'] < 150: score -= 0.10

    score      = max(0.05, min(0.95, score))
    prediction = 1 if score >= 0.50 else 0
    confidence = score if prediction == 1 else (1 - score)

    return prediction, round(confidence, 4), f

# ── Header ────────────────────────────────────────────────────────────────
st.markdown("# 🇳🇬 Nigerian Political Misinformation Detector")
st.markdown("### AI-Powered Detection of AI-Generated Political Content")
st.markdown("""
> **AISIP Capstone — Pathway 4: AI Engineering**
> Detects whether Nigerian political text was written by a **human**
> or generated by an **AI** (e.g. ChatGPT, GPT-4).
""")
st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔍 Analyse Text", "📊 Saved Submissions", "ℹ️ About"])

# ════════════════════════════════════════════════════════════════════════════
# TAB 1: ANALYSE
# ════════════════════════════════════════════════════════════════════════════
with tab1:

    # Sample buttons
    sample_human = "My people abeg make una go out come vote! This election don reach o. We see wetin INEC do for Kano, e no try at all. Labour Party movement don shake this country, even my mama wey always vote APC don change. Na we the youth go decide this thing!"
    sample_ai    = "According to classified intelligence documents exclusively obtained from verified sources, the Independent National Electoral Commission systematically coordinated with foreign agents to manipulate the 2023 Nigerian presidential election results. Statistical analysis definitively confirms that 12.4 million fraudulent votes were recorded."

    col1, col2 = st.columns(2)
    with col1:
        if st.button("📋 Load Human Example", use_container_width=True):
            st.session_state['input_text'] = sample_human
    with col2:
        if st.button("🤖 Load AI Example", use_container_width=True):
            st.session_state['input_text'] = sample_ai

    st.markdown("### 📝 Enter Nigerian Political Text")
    input_text = st.text_area(
        "Paste or type text here:",
        value=st.session_state.get('input_text', ''),
        height=180,
        placeholder="E.g. 'This election don finish o! We see wetin happen for Kano...'"
    )

    # Optional: user name for tracking
    col_name, col_btn = st.columns([2, 1])
    with col_name:
        user_name = st.text_input(
            "Your name (optional — for submission record):",
            placeholder="e.g. Eleazar"
        )

    st.divider()

    if st.button("🔍 Analyse & Save Text", use_container_width=True, type="primary"):
        if not input_text or len(input_text.strip()) < 20:
            st.warning("⚠️ Please enter at least 20 characters of text.")
        else:
            with st.spinner("Analysing text..."):
                pred, conf, feats = rule_based_predict(input_text)

            # ── Result display ────────────────────────────────────────────
            st.markdown("## 🎯 Detection Result")
            label = "AI-GENERATED" if pred == 1 else "HUMAN-AUTHORED"
            css   = "ai" if pred == 1 else "human"
            icon  = "🤖" if pred == 1 else "✅"
            st.markdown(
                f'<div class="result-box {css}">{icon} {label} — {conf*100:.1f}% confidence</div>',
                unsafe_allow_html=True
            )

            if pred == 1:
                st.error("⚠️ This text shows patterns consistent with AI-generated political misinformation.")
            else:
                st.success("✅ This text shows patterns consistent with genuine human political discourse.")

            # Confidence bar
            st.progress(conf)
            st.caption(f"{label} confidence: {conf*100:.1f}%")

            # ── Feature metrics ───────────────────────────────────────────
            st.markdown("### 📊 Stylometric Feature Analysis")
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Word Count",        feats['word_count'])
                st.metric("Avg Sentence Len",  feats['avg_sent_len'])
                st.metric("Numerical Claims",  feats['number_count'])
                st.metric("Exclamation Marks", feats['exclamation_count'])
            with c2:
                st.metric("Lexical Diversity", feats['lexical_diversity'])
                st.metric("Avg Word Length",   feats['avg_word_len'])
                st.metric("Stopword Ratio",    feats['stopword_ratio'])
                st.metric("Question Marks",    feats['question_count'])
            with c3:
                st.metric("Character Count",   feats['char_count'])
                st.metric("Sentence Count",    feats['sent_count'])
                st.metric("CAPS Words",        feats['caps_count'])
                st.metric("Punct Ratio",       feats['punct_ratio'])

            # ── Signals ───────────────────────────────────────────────────
            st.markdown("### 🔍 Key Signals Detected")
            signals = []
            if feats['number_count'] >= 3:
                signals.append("🔴 High number of numerical claims — strong AI signal")
            if feats['lexical_diversity'] > 0.85:
                signals.append("🔴 Very high lexical diversity — AI signal")
            if feats['avg_sent_len'] > 25:
                signals.append("🔴 Long average sentence length — AI signal")
            if feats['exclamation_count'] >= 2:
                signals.append("🟢 Multiple exclamation marks — human signal")
            pidgin_words = ['don','dey','dem','na','wetin','abeg','wahala','oga']
            pc = sum(1 for w in pidgin_words if w in input_text.lower())
            if pc >= 2:
                signals.append(f"🟢 {pc} Nigerian Pidgin expressions — human signal")
            auth_words = ['classified','verified','intelligence','conclusively','forensic']
            ac = sum(1 for w in auth_words if w in input_text.lower())
            if ac >= 1:
                signals.append(f"🔴 {ac} formal authority marker(s) — AI signal")
            for s in (signals or ["No strong individual signals — prediction based on combined feature pattern"]):
                st.markdown(f"- {s}")

            # ── User Feedback ─────────────────────────────────────────────
            st.markdown("### 💬 Was This Prediction Correct?")
            st.caption("Your feedback helps improve the model.")
            feedback_col1, feedback_col2, feedback_col3 = st.columns(3)
            user_feedback = "No feedback"
            with feedback_col1:
                if st.button("👍 Yes, correct", use_container_width=True):
                    user_feedback = "Correct"
            with feedback_col2:
                if st.button("👎 No, wrong", use_container_width=True):
                    user_feedback = "Incorrect"
            with feedback_col3:
                if st.button("🤔 Not sure", use_container_width=True):
                    user_feedback = "Unsure"

            # ── SAVE SUBMISSION ───────────────────────────────────────────
            submission = {
                "timestamp":        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "text_preview":     input_text[:80].replace("\n", " ") + "...",
                "full_text":        input_text.replace("\n", " "),
                "prediction":       "AI-Generated" if pred == 1 else "Human",
                "confidence":       f"{conf*100:.1f}%",
                "word_count":       feats['word_count'],
                "number_count":     feats['number_count'],
                "lexical_diversity":feats['lexical_diversity'],
                "exclamation_count":feats['exclamation_count'],
                "avg_sent_len":     feats['avg_sent_len'],
                "user_feedback":    user_feedback
            }

            save_submission(submission)

            # Store in session state so Saved Submissions tab updates
            if 'all_submissions' not in st.session_state:
                st.session_state['all_submissions'] = []
            st.session_state['all_submissions'].append(submission)

            st.markdown(
                '<div class="save-box">✅ <strong>Submission saved!</strong> '
                'View all saved inputs in the <strong>📊 Saved Submissions</strong> tab.</div>',
                unsafe_allow_html=True
            )

# ════════════════════════════════════════════════════════════════════════════
# TAB 2: SAVED SUBMISSIONS
# ════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("## 📊 Saved User Submissions")
    st.markdown("Every text analysed is automatically saved here for research and model improvement.")
    st.divider()

    submissions = load_submissions()

    if not submissions:
        st.info("No submissions yet. Analyse some text in the **🔍 Analyse Text** tab to get started!")
    else:
        # Summary stats
        total      = len(submissions)
        ai_count   = sum(1 for s in submissions if s['prediction'] == 'AI-Generated')
        human_count= total - ai_count

        m1, m2, m3 = st.columns(3)
        m1.metric("Total Submissions", total)
        m2.metric("AI-Generated",      ai_count)
        m3.metric("Human-Authored",    human_count)

        st.divider()

        # Filter
        filter_col1, filter_col2 = st.columns(2)
        with filter_col1:
            filter_pred = st.selectbox(
                "Filter by prediction:",
                ["All", "AI-Generated", "Human"]
            )
        with filter_col2:
            filter_feedback = st.selectbox(
                "Filter by feedback:",
                ["All", "Correct", "Incorrect", "Unsure", "No feedback"]
            )

        # Apply filters
        filtered = submissions
        if filter_pred != "All":
            filtered = [s for s in filtered if s['prediction'] == filter_pred]
        if filter_feedback != "All":
            filtered = [s for s in filtered if s['user_feedback'] == filter_feedback]

        st.markdown(f"**Showing {len(filtered)} of {total} submissions**")
        st.divider()

        # Display each submission
        for i, sub in enumerate(reversed(filtered), 1):
            color = "🤖" if sub['prediction'] == 'AI-Generated' else "✅"
            with st.expander(
                f"{color} [{sub['timestamp']}] {sub['prediction']} "
                f"({sub['confidence']}) — {sub['text_preview'][:60]}..."
            ):
                st.markdown(f"**Timestamp:** {sub['timestamp']}")
                st.markdown(f"**Prediction:** {sub['prediction']} ({sub['confidence']} confidence)")
                st.markdown(f"**User Feedback:** {sub['user_feedback']}")
                st.divider()
                st.markdown("**Full Text:**")
                st.markdown(f"> {sub['full_text']}")
                st.divider()
                feat_col1, feat_col2, feat_col3 = st.columns(3)
                feat_col1.metric("Word Count",       sub['word_count'])
                feat_col2.metric("Numeric Claims",   sub['number_count'])
                feat_col3.metric("Lexical Diversity",sub['lexical_diversity'])
                feat_col1.metric("Exclamations",     sub['exclamation_count'])
                feat_col2.metric("Avg Sent Len",     sub['avg_sent_len'])

        st.divider()

        # Download button
        if submissions:
            import pandas as pd
            df_export = pd.DataFrame(submissions)
            csv_data  = df_export.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="⬇️ Download All Submissions as CSV",
                data=csv_data,
                file_name=f"misinfo_submissions_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True
            )
            st.caption("Download the CSV to analyse submission patterns or retrain the model.")

        # Clear data option
        st.divider()
        st.markdown("#### ⚠️ Data Management")
        if st.button("🗑️ Clear All Submissions", use_container_width=False):
            if os.path.exists(SAVE_FILE):
                os.remove(SAVE_FILE)
            st.session_state.pop('all_submissions', None)
            st.success("All submissions cleared!")
            st.rerun()

# ════════════════════════════════════════════════════════════════════════════
# TAB 3: ABOUT
# ════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("## ℹ️ About This Tool")

    st.markdown("""
    ### How the Detection Works
    This tool uses **stylometric analysis** to distinguish AI-generated political
    misinformation from human-authored Nigerian political content.

    **Key signals analysed:**
    - 🔴 **Numerical claims** — AI misinformation cites more statistics
    - 🔴 **Lexical diversity** — AI uses a wider, more formal vocabulary
    - 🔴 **Sentence length** — AI generates longer, structured sentences
    - 🔴 **Authority language** — 'classified', 'verified', 'conclusively' = AI signals
    - 🟢 **Exclamation marks** — emotional informal language = human signal
    - 🟢 **Nigerian Pidgin** — Pidgin expressions = strong human authorship signal

    ### How Saving Works
    Every text you analyse is automatically saved to a local CSV file with:
    - The full text and a preview
    - The prediction and confidence score
    - 12 stylometric features
    - Your feedback (correct / incorrect / unsure)
    - A timestamp

    You can **download the CSV** from the Saved Submissions tab and use it to:
    - Retrain a better model with real user data
    - Analyse patterns in Nigerian political misinformation
    - Build a labelled dataset for future research

    ### ⚠️ Permanent Storage Note
    On **Streamlit Cloud**, files reset when the app restarts.
    For permanent storage across sessions, consider:
    - **Google Sheets** via the `gspread` library
    - **Supabase** (free PostgreSQL database)
    - **Firebase** Firestore
    - **AWS S3** bucket

    ### ⚠️ Ethical Notice
    - This is a **research prototype** — not for production use
    - **Do not use** for automated censorship or content removal
    - Human review is always required before any moderation action
    - Intended users: fact-checkers, researchers, civil society organisations

    ### Research Background
    - Zellers et al. (2019) — Neural Fake News Detection
    - Adelani et al. (2020) — NaijaSenti Nigerian Twitter Dataset
    - Ojo et al. (2022) — Fake News Detection in African Languages
    """)

    st.divider()
    st.markdown("""
    <div style='text-align:center; color:#666; font-size:13px;'>
    🇳🇬 Nigerian Political Misinformation Detector |
    AISIP Capstone — Pathway 4: AI Engineering |
    Eleazar Ogidi | May 2026
    </div>
    """, unsafe_allow_html=True)
