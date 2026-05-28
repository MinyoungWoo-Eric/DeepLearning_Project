"""
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
    "-- Select an example review --": "",
    "⭐ Negative – Bad breakfast experience": (
        "Will never come here again!!! We went there for breakfast a Saturday morning. "
        "They took forever to take our order, bring us drink and to bring our order. "
        "Waitress didn't know menu, she look so lost!!! when our order came back my "
        "eggs were cold and my house potatoes burned!!! Bad experience overall."
    ),
    "⭐ Negative – Disappointing hotel stay": (
        "Stayed here recently for a conference. Check-in took forever. Nearly every "
        "person in line had a problem, everything from inaccurate quotes from the "
        "website, to unexpected room charges to who knows what.\n\nThe rooms are nice, "
        "full-wall windows, and bathrooms are absolutely huge. Though not that "
        "soundproofed, I could hear all sorts of activity from the hallway.\n\nFor a "
        "resort/spa hotel I really expected them to filter the water. Vegas is a desert "
        "and the water here is overly chlorinated, which means it's not that refreshing "
        "and very drying to the skin, which only exacerbates the dryness of the desert."
        "\n\nThe food was good and the buffet is a great deal, except for the overpriced "
        "Sunday brunch.\n\nThe casino area is nice, though it could use some more "
        "smoke-eaters and again, there's the smell of chlorine from the waterworks.\n\n"
        "The acoustics aren't the best, the sound gets totally lost in the registration "
        "area. And the wind literally howls through the complex, especially noticable "
        "in the meeting rooms with patios, the wind forcing through gaps in the glass "
        "doors.\n\nIf you do book a conference here be very aware that nearly anything "
        "you'd want is an extra charge, and often ridiculously so (ie $2000/day for "
        "wifi per room).\n\nYou can do just about everything here, see a movie, gamble, "
        "go bowling, work out, eat, shop, spa, swim, golf, and of course, convene."
    ),
    "⭐⭐⭐ Neutral – Mixed feelings about resort": (
        "My boyfriend said he would give The Boulders 5 stars, I thought it was more "
        "like 3 and since I'm writing the review, it's getting 3.\n\nUpon checking in, "
        "I was told that we would be given a room with two double beds. Say what?? I "
        "specifically requested a king sized bed - it was supposed to be a \"romantic\" "
        "getaway after all. I wanted it to be romantic, but my boyfriend just wanted to "
        "golf. That pretty much sums it up. Anyway, they quickly fixed the problem and "
        "gave us a casita with a king sized bed and a fireplace. The casita was really "
        "cute and was clean and comfortable. The grounds are beautiful and the pool area "
        "was nice. We ate at the Palo Verde restaurant and both thought it was just ok. "
        "The food wasn't bad, just not memorable. The service on the other hand was "
        "excellent.\n\nI would not necessarily go back to The Boulders. It was fine, "
        "but nothing extraordinary."
    ),
    "⭐⭐⭐ Neutral – Average food, decent price": (
        "I dig Hanlon's. I love that pretty much all of their dishes come with a side "
        "and the price is right. Everything on their menu sounds delish and it takes a "
        "little while for me to decide. For any hoagie, you can make it a wrap at no "
        "extra charge. I usually order the buffalo chicken wrap & its fabb!! The chicken "
        "is always tender and the wrap is stuffed with flavor. Only downside is parking "
        "can be tricky during lunch rush."
    ),
    "⭐⭐⭐⭐⭐ Positive – Best meal in years": (
        "Absolutely fantastic! Best meal I've had in years. The steak was cooked to "
        "perfection, the sides were fresh and flavorful, and the dessert was heavenly. "
        "Our server was attentive without being overbearing, and the ambiance was warm "
        "and inviting. We celebrated our anniversary here and it was worth every penny. "
        "The wine list is extensive and reasonably priced. I've already recommended this "
        "place to all my friends. If you're looking for a truly special dining "
        "experience, look no further. Five stars without hesitation!"
    ),
    "⭐⭐⭐⭐⭐ Positive – Outstanding sushi spot": (
        "Hidden gem! This tiny sushi spot on the corner has the freshest fish in town. "
        "The chef is a master — each piece of nigiri is a work of art. The omakase "
        "experience is incredible value for what you get. We had the 12-piece set with "
        "miso soup and it was divine. The tuna melted in my mouth and the uni was "
        "buttery perfection. Staff are so friendly and the atmosphere is cozy and "
        "authentic. We've been coming back every week since we discovered it. Highly "
        "recommend making a reservation as it fills up fast."
    ),
    "⭐⭐⭐⭐⭐ Positive – Amazing pizza place": (
        "I'm from New York, so I know good pizza. This place rivals the best in NYC, "
        "and that's saying something. The crust is thin, crispy, and perfectly charred. "
        "The sauce is tangy and fresh. The mozzarella is top quality. We ordered the "
        "Margherita and the white pizza with truffle oil. Both were phenomenal. The "
        "garlic knots are also a must-try — they come out piping hot with just the "
        "right amount of garlic butter. Prices are super reasonable for the quality. "
        "The only downside is the wait — it gets packed on weekends. But honestly, "
        "it's worth every minute of the wait. This is now our go-to pizza spot."
    ),
    "⭐ Negative – Terrible customer service": (
        "Worst customer service I have ever experienced. I placed an order online and "
        "it never arrived. When I called, I was put on hold for 45 minutes. The manager "
        "was rude and dismissive, offering no solution. I asked for a refund and was "
        "told it would take 10 business days. It's been three weeks and still no "
        "refund. I've filed a complaint with my credit card company. Do not waste your "
        "time or money here. They clearly don't care about their customers."
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
    """Render the sidebar with model information and instructions."""
    st.image(
    "https://www.yelp.com/favicon.ico",
    width=60,
)
st.markdown("## 🍽️ Yelp Review Intelligence")
        st.markdown(
            "This application analyzes Yelp customer reviews using two "
            "deep-learning pipelines to support **data-driven decision-making** "
            "for service quality monitoring."
        )

        with st.expander("🔬 Pipeline 1 — Sentiment Analysis", expanded=False):
            st.markdown(
                "**Model:** `ex1619dd/yelp-sentiment-3class-finetuned`\n\n"
                "A DistilBERT model fine-tuned on the Yelp Review Full dataset, "
                "mapped to three sentiment classes: **Negative**, **Neutral**, "
                "and **Positive**.\n\n"
                "- Base model: `nlptown/bert-base-multilingual-uncased-sentiment`\n"
                "- Fine-tuning: 9,000 balanced training samples (3,000/class)\n"
                "- Evaluation: 2,000 balanced test samples"
            )

        with st.expander("🔬 Pipeline 2 — Text Summarization", expanded=False):
            st.markdown(
                "**Model:** `philschmid/distilbart-cnn-12-6-samsum`\n\n"
                "A DistilBART model trained on CNN/DailyMail + SAMSum for "
                "dialogue-style summarization.\n\n"
                "- Beam search (num_beams=4)\n"
                "- Post-processing: sentence-boundary truncation\n"
                "- Long reviews (> 200 words): hierarchical chunk-then-summarize"
            )

        with st.expander("📖 How to Use", expanded=False):
            st.markdown(
                "1. Select an example review from the dropdown **or** paste your own.\n"
                "2. Click **Analyze Review**.\n"
                "3. View the sentiment prediction with confidence scores "
                "and the auto-generated summary side by side."
            )

        st.divider()
        st.caption(
            "ISOM5240 Group Project · Built with Streamlit · "
            "Models hosted on Hugging Face 🤗"
        )


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
