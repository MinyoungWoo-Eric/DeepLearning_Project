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

import re
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
    "🥩 Steakhouse (Vegas)": (
        "Wow. And this is not a good wow at all.\n\n"
        "Did NOT expect such bad food here. My fellow diners said they had better steaks "
        "from Outback, possibly Sizzler and Cocos.\n\n"
        "The waiters seem programmed to try to upsell everything. The only \"amazing\" "
        "food on the menu from the kitchen are the highest priced items.\n\n"
        "Considering how much amazing food is available in Vegas, this place sucks. "
        "The various managers are completely useless. There seems to be too many staff "
        "running around pretending to be busy than actually working. None of them went "
        "up to any tables to inquire about the food or service.\n\n"
        "Now, onto the food. Again, WOW. It was beyond awful -- once it actually came out. "
        "Apparently our ticket got dropped TWICE by the kitchen. And did any manager "
        "come out to apologize? That'd be a no.\n\n"
        "The waiter and his trainee were so sweet and trying to help out, but honestly, "
        "what else can they do? Final report: More style than substance."
    ),
    "🍺 Sports Bar": (
        "A shi-shi sports bar with an appetizing menu executed in a very average manner.\n\n"
        "The current tenants have stuck with the existing deep red on black decor and "
        "subdued lighting. Very Victorian. Being a sports bar, of course there are "
        "wide-screens all over the shop, which seem a bit anachronistic given that "
        "the place is swathed in late 19th-century red velvet drapery.\n\n"
        "The owners have placed a few old-fashioned items on the menu, such as baked brie, "
        "juxtaposed with more modern tasters like deep-fried mashed potato bites. "
        "The menu is fairly enticing. Sadly the food falls short. Everything here is made "
        "to sound impressive, but looks and tastes like it was cooked at Wendy's.\n\n"
        "Today I tried the Chicken Florentine Soup. It definitely came from a can. "
        "Then I got a California Turkey Burger. The lettuce was limp, the avocado was "
        "almost non-existent, and the turkey patty was too salty.\n\n"
        "My advice: if you're going to do gourmet pub food, actually do gourmet pub food."
    ),
    "🎰 Nightclub (Vegas)": (
        "My first 21+ clubbing experience! I turned 21 on Monday, so we decided to hit "
        "up XS on a Sunday night, right before 12. It was literally 11:50pm and the guy "
        "wouldn't even let me through the door to wait in the second line. We were in a "
        "group of 12, and they had us wait on the side until it was midnight. They were "
        "also pretty rude, even though we were nothing but polite.\n\n"
        "Once we got in, it was hella cold so the outdoor area was a no-go. "
        "Disgustingly packed inside -- like a rave. The dancefloor was just a swaying, "
        "smelly mosh pit.\n\n"
        "All our girls got free drink cards, but it took forever to get to the bar. "
        "I think my bartender forgot to put Midori in my Midori sour -- I didn't taste "
        "any alcohol at all.\n\n"
        "I'm glad I got in for free, or I'd be so disappointed. People barfing along "
        "the walkway and a fight broke out right as we were leaving. Real classy."
    ),
    "🥪 Sandwich Diner (Pittsburgh)": (
        "Had to visit the \"Famous\" Primanti Brother's Bar and Grill during my first "
        "visit to Pittsburgh, but most likely will only go back if my friends/family "
        "really want to eat there.\n\n"
        "Food: The portions are huge, not very greasy, served nice and hot but overall "
        "very bland. The fries were limp and had no seasoning whatsoever. Put everything "
        "together between two slices of white bread and it made for a bland pile of "
        "flavorless starch.\n\n"
        "Price: For the size of the portions you get, the price is great. I got two "
        "sandwiches and a soda for about $16 -- good value since you will definitely "
        "leave full.\n\n"
        "Staff: During my visit, the staff was great. The waitress was attentive and "
        "friendly. The cook also thanked me for patronizing there while I was leaving.\n\n"
        "Overall, I basically came here just to say that I visited a famous Pittsburgh "
        "restaurant, but I most likely will not go again. The blandness of the food "
        "is what kills it for me."
    ),
    "🥩 Fine Dining (Phoenix)": (
        "I know I will upset some fellow Yelp friends with this review, but the "
        "Ruth Chris on Camelback was a big disappointment for me.\n\n"
        "Atmosphere: This location makes a better Chili's than Ruth Chris. The setup "
        "is very awkward and having to leave the restaurant to use the bathroom in the "
        "lobby is not very high end.\n\n"
        "Food: The steak quality was very good, but I have had many more filets elsewhere "
        "that have given me much more flavor. My filet was served in a pool of butter. "
        "The sides are very average.\n\n"
        "Service: This is the only thing worthy of mentioning. Devin at the bar served "
        "us and he was 5 stars plus. Great service and great Martinis. "
        "Devin saved Ruth Chris from getting 2 stars."
    ),
    "🌮 Mexican Restaurant": (
        "We've been by this place a number of times since we live right by it and we "
        "finally decided to try it. I ordered some chicken and beef tacos, tostadas, "
        "a cheese quesadilla, chips/salsa, and flan to go.\n\n"
        "The meat was great! Everyone agreed that the tacos were better than the tostadas. "
        "The kids tore through the quesadilla so I'm assuming it was pretty good. "
        "The chips and salsa were very good. They also give you a salsa bar to select "
        "different kinds of salsa -- we found them all to be very tasty.\n\n"
        "The only issue: they forgot the flan but charged us for it. I was really looking "
        "forward to trying it. Despite that, I decided to give it a 4. I really liked "
        "the flavor of the meat and will probably return for more tacos. "
        "If you get takeout, give your bag a couple of extra checks before leaving."
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
