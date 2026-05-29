# 🍽️ Yelp Review Intelligence Dashboard

A Streamlit web application that analyzes Yelp customer reviews using two complementary NLP pipelines — delivering both an overall sentiment verdict and a concise summary in seconds.

🔗 **Live App:** [deeplearningproject-erics.streamlit.app](https://deeplearningproject-erics.streamlit.app)

---

## What It Does

| Pipeline | Task | Model |
|---|---|---|
| Pipeline 1 | 3-class Sentiment Analysis (Negative / Neutral / Positive) | Fine-tuned DistilBERT |
| Pipeline 2 | Extractive-Abstractive Summarization | DistilBART-CNN-SAMSum |

Paste any Yelp review and instantly get:

- 😠😐😊 **Overall sentiment** — Negative, Neutral, or Positive with confidence score
- 📝 **Auto-generated summary** — key points extracted without reading the full review

Useful for consumers who want a quick take, business owners monitoring feedback, or anyone processing large volumes of reviews.

---

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


