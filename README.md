# 🍽️ Yelp Review Intelligence Dashboard

A Streamlit web application that analyzes Yelp customer reviews using two complementary NLP pipelines — delivering both an overall sentiment verdict and a concise summary in seconds.

🔗 **Live App:** [deeplearningproject-erics.streamlit.app](https://deeplearningproject-erics.streamlit.app)

---

## What It Does

| Pipeline | Task | Model |
|---|---|---|
| Pipeline 1 | 3-class Sentiment Analysis (Negative / Neutral / Positive) | Fine-tuned DistilBERT |
| Pipeline 2 | Extractive-Abstractive Summarization | DistilBART-CNN-SAMSum |

**Why both?**
Sentiment alone tells you *how* a customer feels. The summary tells you *why*. Together they give a complete, actionable picture of any review — without reading the whole thing.

---

## Project Structure

```
DeepLearning_Project/
├── app.py              # Streamlit application (Pipeline 1 + Pipeline 2)
├── requirements.txt    # Python dependencies
└── README.md           # Project documentation
```

---

## Pipeline 1 — Sentiment Analysis

**Dataset:** `Yelp/yelp_review_full` (HuggingFace)
- Original 5-class star ratings remapped to 3 classes:
  - 1–2 stars → Negative
  - 3 stars → Neutral
  - 4–5 stars → Positive

**Model Selection:** 3 pre-trained models evaluated on 2,000 balanced test samples

| Model | Accuracy | Runtime |
|---|---|---|
| `Soha/sentiment-analysis` | 76.70% | 34s |
| `GHonem/sentiment_analysis` | 86.70% | 55s |
| `yj2773/hinglish11k-sentiment-analysis` ✅ | **90.10%** | 32s |

**Fine-tuning:**
- Training set: 9,000 samples (3,000 per class)
- Validation set: 1,500 samples
- Epochs: 2
- Result: **+5.1 percentage points** improvement over baseline

---

## Pipeline 2 — Text Summarization

**Dataset:** Same `Yelp/yelp_review_full` test set (no fine-tuning required)

**Model Selection:** 3 pre-trained models evaluated on 200 balanced samples using BERTScore (semantic similarity metric — more suitable than ROUGE for informal review text)

| Model | BERTScore-F1 | Runtime |
|---|---|---|
| `philschmid/distilbart-cnn-12-6-samsum` ✅ | **0.891** | 54s |
| `philschmid/bart-large-cnn-samsum` | 0.889 | 82s |
| `facebook/bart-large-cnn` | 0.898 | 79s |

**Why DistilBART-SAMSum?**
Although `bart-large-cnn` achieves marginally higher F1 (+0.007), DistilBART-SAMSum runs ~32% faster with comparable quality — a deliberate speed-accuracy tradeoff for production deployment.

**Long Review Handling:**
Reviews over 200 words are processed via hierarchical sentence-aware chunking:
1. Split into sentence-coherent chunks (~100 words each)
2. Summarize each chunk independently
3. Combine chunk summaries and summarize again

---

