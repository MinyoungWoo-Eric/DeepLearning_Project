import re
import time
import streamlit as st
import torch
from transformers import (
    pipeline as hf_pipeline,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
)"""
Yelp Review Intelligence Dashboard
===================================
A Streamlit application that combines two NLP pipelines to analyze
Yelp customer reviews:
  - Pipeline 1: Sentiment Analysis (fine-tuned DistilBERT, 3-class)
  - Pipeline 2: Text Summarization (DistilBART-CNN-samsum with chunking)

Models:
  - Sentiment: ex1619dd/yelp-sentiment-3class-finetuned (Hugging Face)
  - Summarization: philschmid/distilbart-cnn-12-6-samsum (Hugging Face)

Author : Group XX – ISOM5240
Date   : 2025
"""

import time
import streamlit as st
import torch
from transformers import (
    pipeline as hf_pipeline,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
)


# ────────────────────────────────────────────────────────────────────
# Constants
# ────────────────────────────────────────────────────────────────────
SENTIMENT_MODEL = "ex1619dd/yelp-sentiment-3class-finetuned"
SUMMARIZATION_MODEL = "philschmid/distilbart-cnn-12-6-samsum"

LABEL_EMOJI = {"Negative": "😠", "Neutral": "😐", "Positive": "😊"}
LABEL_COLOR = {"Negative": "#e74c3c", "Neutral": "#f39c12", "Positive": "#27ae60"}

# Example reviews from the Yelp/yelp_review_full test split (seed=42)
EXAMPLE_REVIEWS = {
    "-- Select an example --": "",
    "🥩 Steakhouse": (
        "Wow. And this is not a good wow at all. Did NOT expect such bad food here. "
        "My fellow diners said they had better steaks from Outback, possibly Sizzler "
        "and Cocos. The waiters seem programmed to talk about the specials and that "
        "is it."
    ),
    "🍕 Pizza & Wings": (
        "Came here because they have a whole vegan menu but also regular pizza as well "
        "which is perfect because I am vegan and my boyfriend isn't. We came during "
        "happy hour so he got half off wings and then a pizza. Everything was delicious "
        "and the staff were super friendly. Would definitely come back."
    ),
    "🍣 Sushi Bar": (
        "BAD! But that's what we get for straying away from our favorite sushi spot. "
        "We walked into Sushi Fever about 40 minutes before they were due to close "
        "and they made us feel extremely unwelcome. The fish was not fresh and the "
        "service was dismissive. Save your money and go somewhere else."
    ),
    "🌯 Casual Diner": (
        "This is a great place to eat and hang out with friends. I always get the "
        "pretzel appetizer and the angry chicken sandwich. When I am eating them, "
        "I believe there is nothing better I could be eating at that moment. "
        "The staff are friendly and the prices are very reasonable."
    ),
    "🍔 Burger Joint": (
        "This was our best meal in Vegas and the cheapest. Sweet potato fries were "
        "amazing and we loved all the different ketchups to try. The hamburgers were "
        "huge and amazing. We will be going back."
    ),
    "🥗 Breakfast Spot": (
        "This hidden gem is my new favorite breakfast spot! The staff is SO friendly, "
        "and their menu has the right balance of unique choices along with the classics. "
        "They make everything from scratch and try to source local ingredients when "
        "possible. The portions are generous and the prices are very fair."
    ),
    "🍽️ Fine Dining": (
        "I know I will upset some fellow Yelp friends with this review, but the "
        "Ruth Chris on Camelback was a big disappointment for me. I went with a friend "
        "who is a huge fan of Ruth Chris, but I was let down by both the food and "
        "the service. For that price point I expected a lot more."
    ),
    "🏨 Hotel Stay": (
        "I think the rooms need an update. I mean, for the price, they're not "
        "spectacular. Staff were really nice though. The casino and bar/club/restaurant "
        "choices are fine, typical Las Vegas. The lobby is beautiful but once you get "
        "to your room it's a bit of a letdown."
    ),
}


# ────────────────────────────────────────────────────────────────────
# Model Loading (cached across reruns)
# ────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_sentiment_pipeline():
    """Load the fine-tuned 3-class sentiment analysis pipeline (Pipeline 1)."""
    return hf_pipeline(
        "sentiment-analysis",
        model=SENTIMENT_MODEL,
        device=-1,  # CPU for Streamlit Cloud
        truncation=True,
        max_length=512,
    )

@st.cache_resource(show_spinner=False)
def load_summarization_model():
    """Load the DistilBART summarization tokenizer and model (Pipeline 2)."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(SUMMARIZATION_MODEL)
        model = AutoModelForSeq2SeqLM.from_pretrained(SUMMARIZATION_MODEL)
        model.eval()
        return tokenizer, model
    except Exception as e:
        st.error(f"Failed to load summarization model: {e}")
        raise

# ────────────────────────────────────────────────────────────────────
# Pipeline 1 – Sentiment Analysis
# ────────────────────────────────────────────────────────────────────
def analyze_sentiment(text: str) -> dict:
    """
    Run the fine-tuned DistilBERT sentiment model on *text*.

    Returns
    -------
    dict  {"label": str, "score": float, "all_scores": list[dict]}
          where *all_scores* contains Negative / Neutral / Positive
          probabilities via `top_k=None`.
    """
    pipe = load_sentiment_pipeline()
    start = time.time()

    # top_k=None returns scores for all three classes
    all_scores = pipe(text[:512], top_k=None)
    best = max(all_scores, key=lambda x: x["score"])

    elapsed = time.time() - start
    return {
        "label": best["label"],
        "score": best["score"],
        "all_scores": sorted(all_scores, key=lambda x: x["score"], reverse=True),
        "runtime": elapsed,
    }


# ────────────────────────────────────────────────────────────────────
# Pipeline 2 – Text Summarization (with chunking for long reviews)
# ────────────────────────────────────────────────────────────────────
def _summarize_chunk(text: str, tokenizer, model) -> str:
    """
    Summarize a single text chunk using beam search.
    Truncates output at the last complete sentence boundary.
    """
    if not text.strip():
        return ""

    try:
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=80,
                min_length=10,
                num_beams=4,
            )

        raw = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
        last_end = max(raw.rfind("."), raw.rfind("!"), raw.rfind("?"))
        return raw[:last_end + 1].strip() if last_end > 0 else raw

    except Exception:
        # Fall back to returning the first sentence of the original text
        first = text.split(".")[0].strip()
        return first + "." if first else text[:150]

def _split_by_sentences(text: str, max_words_per_chunk: int = 100) -> list[str]:
    """
    Split text into chunks that respect sentence boundaries.
    Each chunk stays under max_words_per_chunk words where possible.

    Word-boundary splitting can break sentences mid-way, losing context.
    Sentence-aware splitting keeps each chunk semantically coherent,
    which leads to more accurate per-chunk summaries.
    """
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    chunks, current_sentences, current_count = [], [], 0

    for sentence in sentences:
        word_count = len(sentence.split())
        if current_count + word_count > max_words_per_chunk and current_sentences:
            chunks.append(" ".join(current_sentences))
            current_sentences, current_count = [sentence], word_count
        else:
            current_sentences.append(sentence)
            current_count += word_count

    if current_sentences:
        chunks.append(" ".join(current_sentences))

    return chunks
  
def summarize_text(text: str) -> dict:
    """
    Summarize a Yelp review using DistilBART with sentence-aware chunking.

    Reviews up to 200 words are summarized in a single pass.
    Longer reviews are split into sentence-coherent chunks, each chunk
    summarized independently, then the combined result is summarized again
    (hierarchical summarization).

    Returns a dict with summary text, word counts, compression rate,
    method description, and inference runtime.
    """
    if not text or not text.strip():
        return {
            "summary": "No input provided.",
            "word_count_original": 0,
            "word_count_summary": 0,
            "compression": 0,
            "method": "n/a",
            "runtime": 0.0,
        }

    tokenizer, model = load_summarization_model()
    words = text.split()
    word_count = len(words)
    start = time.time()

    if word_count <= 200:
        summary = _summarize_chunk(text, tokenizer, model)
        method = "single-pass"
    else:
        chunks = _split_by_sentences(text, max_words_per_chunk=100)
        chunk_summaries = [_summarize_chunk(c, tokenizer, model) for c in chunks]
        combined = " ".join(s for s in chunk_summaries if s)
        summary = _summarize_chunk(combined, tokenizer, model)
        method = f"sentence-chunked ({len(chunks)} chunks)"

    elapsed = time.time() - start
    summary_words = len(summary.split())
    compression = round((1 - summary_words / max(word_count, 1)) * 100)

    return {
        "summary": summary,
        "word_count_original": word_count,
        "word_count_summary": summary_words,
        "compression": compression,
        "method": method,
        "runtime": elapsed,
    }


# ────────────────────────────────────────────────────────────────────
# UI Helpers
# ────────────────────────────────────────────────────────────────────
def render_sentiment_results(result: dict):
    """Display sentiment analysis results with visual indicators."""
    label = result["label"]
    score = result["score"]
    emoji = LABEL_EMOJI.get(label, "")
    color = LABEL_COLOR.get(label, "#333")

    # Header card
    st.markdown("### 📊 Sentiment Analysis Results")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            f"""
            <div style="background:linear-gradient(135deg,{color}22,{color}11);
                        border-left:4px solid {color};padding:16px;border-radius:8px;">
                <div style="font-size:0.85rem;color:#666;">Predicted Sentiment</div>
                <div style="font-size:1.6rem;font-weight:700;color:{color};">
                    {emoji} {label.upper()}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"""
            <div style="background:#f8f9fa;border-left:4px solid #3498db;
                        padding:16px;border-radius:8px;">
                <div style="font-size:0.85rem;color:#666;">Confidence Score</div>
                <div style="font-size:1.6rem;font-weight:700;color:#2c3e50;">
                    {score:.2%}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            f"""
            <div style="background:#f8f9fa;border-left:4px solid #9b59b6;
                        padding:16px;border-radius:8px;">
                <div style="font-size:0.85rem;color:#666;">Inference Time</div>
                <div style="font-size:1.6rem;font-weight:700;color:#2c3e50;">
                    {result['runtime']:.2f}s
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Detailed class probabilities
    st.markdown("**Class Probabilities**")
    for item in result["all_scores"]:
        lbl = item["label"]
        scr = item["score"]
        clr = LABEL_COLOR.get(lbl, "#333")
        emo = LABEL_EMOJI.get(lbl, "")
        st.markdown(
            f"{emo} **{lbl}**"
        )
        st.progress(scr, text=f"{scr:.2%}")


def render_summary_results(result: dict):
    """Display text summarization results."""
    st.markdown("### 📝 Review Summary")

    if not result["summary"] or result["summary"] == "No input provided.":
        st.warning("No summary generated — please enter a valid review.")
        return

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Original Length", f"{result['word_count_original']} words")
    with col2:
        st.metric("Summary Length", f"{result['word_count_summary']} words")
    with col3:
        st.metric("Compression", f"{result['compression']}%")

    st.info(result["summary"])

    st.caption(
        f"Method: {result['method']}  •  "
        f"Inference time: {result['runtime']:.2f}s"
    )

# ────────────────────────────────────────────────────────────────────
# Page Configuration & Main
# ────────────────────────────────────────────────────────────────────
def setup_page():
    """Configure page metadata and inject custom CSS."""
    st.set_page_config(
        page_title="Yelp Review Intelligence",
        page_icon="💬",
        layout="wide",
    )

    st.markdown(
        """
        <style>
            .block-container {padding-top:2rem;}
            div[data-testid="stMetricValue"] {font-size:1.1rem;}
            .stProgress > div > div > div {height:12px; border-radius:6px;}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar():
    """Render the sidebar with user-facing app information."""
    with st.sidebar:
        st.markdown("## 🍽️ Yelp Review Intelligence")
        st.divider()

        st.markdown(
            "Understand any customer review in seconds — "
            "without reading the whole thing."
        )

        st.markdown("### What this app does")
        st.markdown(
            "- 😠😐😊 Detects the **overall sentiment** of a review\n"
            "- 📝 Generates a **concise summary** of the key points\n"
            "- ⚡ Works on reviews of any length"
        )

        st.divider()

        with st.expander("📖 How to Use"):
            st.markdown(
                "1. Select an example review from the dropdown "
                "**or** paste your own below\n"
                "2. Click **Analyze Review**\n"
                "3. View the sentiment verdict and summary side by side"
            )

        with st.expander("💡 Who is this for?"):
            st.markdown(
                "- **Consumers** who want a quick take before visiting\n"
                "- **Business owners** monitoring customer feedback\n"
                "- **Yelp editors** processing large volumes of reviews"
            )

        st.divider()
        st.caption("Powered by Yelp Review Intelligence · v1.0")

def main():
    """Application entry point."""
    setup_page()
    render_sidebar()

    # ── Header ──────────────────────────────────────────────────────
    st.markdown(
        """
        <h1 style="margin-bottom:0;">💬 Yelp Review Intelligence Dashboard</h1>
        <p style="color:#666;font-size:1.05rem;margin-top:4px;">
            Analyze customer sentiment and generate concise review summaries
            to support data-driven decision-making at Yelp.
        </p>
        """,
        unsafe_allow_html=True,
    )
    st.divider()

    # ── Model loading (once) ────────────────────────────────────────
    with st.spinner("Loading models — this may take a minute on first run…"):
        load_sentiment_pipeline()
        load_summarization_model()

    # ── Input Section ───────────────────────────────────────────────
    st.markdown("#### 📥 Input a Customer Review")

    selected_example = st.selectbox(
        "Choose an example from the Yelp test set:",
        options=list(EXAMPLE_REVIEWS.keys()),
        index=0,
    )

    default_text = EXAMPLE_REVIEWS[selected_example]

    review_text = st.text_area(
        "Enter or paste a review below:",
        value=default_text,
        height=180,
        max_chars=5000,
        placeholder="Type or paste a Yelp customer review here…",
    )

    word_count = len(review_text.split()) if review_text.strip() else 0
    st.caption(f"{word_count} words  •  {len(review_text)}/5,000 characters")

    # ── Analyze Button ──────────────────────────────────────────────
    analyze_clicked = st.button(
        "🚀 Analyze Review",
        type="primary",
        use_container_width=True,
        disabled=(not review_text.strip()),
    )

    if analyze_clicked and review_text.strip():
        st.divider()

        col_sent, col_summ = st.columns(2)

        # Run Pipeline 1
        with col_sent:
            with st.spinner("Running sentiment analysis…"):
                sentiment_result = analyze_sentiment(review_text)
            render_sentiment_results(sentiment_result)

        # Run Pipeline 2
        with col_summ:
            with st.spinner("Generating summary…"):
                summary_result = summarize_text(review_text)
            render_summary_results(summary_result)


if __name__ == "__main__":
    main()
