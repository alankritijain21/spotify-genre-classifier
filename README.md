# 🎵 Spotify Genre Classifier — Advanced Data Analytics Project

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange?logo=scikit-learn)
![Plotly](https://img.shields.io/badge/Plotly-Interactive-green?logo=plotly)
![Kaggle](https://img.shields.io/badge/Kaggle-Dataset-20BEFF?logo=kaggle)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

🔗 **[▶ View Live Kaggle Notebook](https://www.kaggle.com/code/alankriti21/spotify-kaggle-notebook)** | **[📊 Dataset](https://www.kaggle.com/datasets/mrmorj/dataset-of-songs-in-spotify)**

> **University of Delhi | Shyama Prasad Mukherji College (For Women)**
> Course: Data Mining II | Group Project

---

## 📌 Project Overview

This project performs an **end-to-end data analytics study** on 42,305 Spotify tracks across 15 music genres. It covers the full data science pipeline — from raw data ingestion to machine learning model comparison and business insight generation.

**Key Question:** *Can we predict a song's genre from its audio features alone?*

**Answer:** Yes — with **66.27% accuracy** using Gradient Boosting, where `tempo` alone accounts for 46.6% of the classification power.

---

## 📊 Dataset

| Property | Detail |
|---|---|
| Source | [Dataset of Songs in Spotify — Kaggle](https://www.kaggle.com/datasets/mrmorj/dataset-of-songs-in-spotify) |
| File | `genres_v2.csv` |
| Size | 42,305 rows × 22 columns |
| Genres | 15 (Dark Trap, Hiphop, EDM, Rap, Pop, RnB, and more) |
| Missing Values | None in audio features |

### Features Used
```
danceability, energy, loudness, speechiness, acousticness,
instrumentalness, liveness, valence, tempo, duration_ms
```

---

## 🔬 Project Pipeline

```
Raw Data
   │
   ▼
Data Cleaning ──────── Remove duplicates, encode columns, drop irrelevant fields
   │
   ▼
Feature Engineering ── energy_density, dance_mood_score, vocal_presence,
   │                    loudness_normalized, tempo_category
   ▼
EDA ────────────────── Correlation heatmap, distributions, genre-level analysis,
   │                    interactive Plotly charts
   ▼
Outlier Detection ───── Z-score (threshold=3), boxplots before/after, capping
   │
   ▼
Machine Learning ────── AdaBoost, Gradient Boosting, Random Forest
   │                    (GridSearchCV + 5-fold cross-validation)
   ▼
Model Comparison ────── Accuracy, Precision, Recall, F1-Score
   │
   ▼
Business Insights ───── Feature importance, genre patterns, recommendations
```

---

## 🤖 Model Results

| Model | Accuracy | Notes |
|---|---|---|
| **Gradient Boosting** | **66.27%** | ★ Best — sequential error correction |
| Random Forest | 64.29% | Stable, less overfitting |
| AdaBoost | 34.80% | Weak on overlapping genres |

### Top 5 Features (Gradient Boosting)

| Rank | Feature | Importance |
|---|---|---|
| 1 | `tempo` | 46.6% |
| 2 | `danceability` | 10.3% |
| 3 | `instrumentalness` | 9.8% |
| 4 | `loudness_normalized` | 7.0% |
| 5 | `vocal_presence` | 6.8% |

---

## 💡 Key Business Insights

- **Electronic genres** (dnb, hardstyle, psytrance, techhouse) classified with **86–97% accuracy** — very distinct audio fingerprints
- **Rap subgenres** (Dark Trap, Underground Rap, Trap Metal) overlap heavily — audio features alone are insufficient to distinguish them
- **Tempo is the #1 predictor** — electronic genres cluster at fast BPMs, hip-hop at mid range
- **Slow songs are happier** — tracks < 90 BPM have highest valence (0.437)
- **Mid-tempo = most danceable** — 90–130 BPM tracks score highest danceability (0.724)

---

## 📁 Repository Structure

```
spotify-genre-classifier/
│
├── spotify_kaggle_notebook.ipynb  # Full Kaggle notebook with outputs
├── spotify_kaggle_notebook.py     # Python script version
├── README.md                      # This file
├── requirements.txt               # Python dependencies
│
└── outputs/                       # Generated charts
    ├── correlation_heatmap.png
    ├── distributions.png
    ├── genre_counts.png
    ├── energy_by_genre.png
    ├── boxplots_before.png
    ├── boxplots_after.png
    ├── feature_importance.png
    ├── bagging_mse.png
    ├── model_comparison.png
    └── interactive_dashboard.html
```

---

## ⚙️ How to Run

### Option 1 — Kaggle (Recommended, No Setup Needed)
🔗 **Direct Link:** https://www.kaggle.com/code/alankriti21/spotify-kaggle-notebook

1. Open the link above
2. Click **Copy & Edit** to run it yourself
3. Dataset is pre-attached — just click **Run All**

### Option 2 — Local Machine
```bash
# Clone the repo
git clone https://github.com/alankritijain21/spotify-genre-classifier.git
cd spotify-genre-classifier

# Install dependencies
pip install -r requirements.txt

# Download dataset from Kaggle
# Place genres_v2.csv in the project folder
# Update file path in code:
# df = pd.read_csv('genres_v2.csv')

# Run
python spotify_kaggle_notebook.py
```

---

## 📦 Requirements

```
pandas>=1.5.0
numpy>=1.23.0
matplotlib>=3.6.0
seaborn>=0.12.0
scikit-learn>=1.2.0
plotly>=5.11.0
scipy>=1.9.0
```

---

## 👥 Team

| Name |
|---|
| Alankriti Jain | 
| Riya | 
| Manasvi Arora | 
| Archi Aggarwal | 

**Institution:** Shyama Prasad Mukherji College for Women, University of Delhi
**Course:** Data Mining II (DM2)

---

## 📚 References

- [GeeksforGeeks — Outlier Detection](https://www.geeksforgeeks.org/challenges-of-outlier-detection-in-data-mining/)
- [IBM Topics — Bagging](https://www.ibm.com/topics/bagging)
- [GeeksforGeeks — Boosting & AdaBoost](https://www.geeksforgeeks.org/boosting-in-machine-learning-boosting-and-adaboost/)
- [Scikit-learn — Ensemble Methods](https://scikit-learn.org/stable/modules/ensemble.html)
- [Kaggle Dataset by mrmorj](https://www.kaggle.com/datasets/mrmorj/dataset-of-songs-in-spotify)

---

⭐ *If you found this project useful, please give it a star!*
