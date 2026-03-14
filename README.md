# 🎵 Spotify Genre Classifier — Advanced Data Analytics Project

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange?logo=scikit-learn)
![Plotly](https://img.shields.io/badge/Plotly-Interactive-green?logo=plotly)
![SHAP](https://img.shields.io/badge/SHAP-Explainability-red)
![Kaggle](https://img.shields.io/badge/Kaggle-Dataset-20BEFF?logo=kaggle)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

🔗 **[▶ View Live Kaggle Notebook](https://www.kaggle.com/code/alankriti21/spotify-kaggle-notebook)** | **[📊 Dataset](https://www.kaggle.com/datasets/mrmorj/dataset-of-songs-in-spotify)**

> **University of Delhi | Shyama Prasad Mukherji College (For Women)**
> Course: Data Mining II | Group Project

---

## 📌 Project Overview

This project performs an **end-to-end data analytics study** on 42,305 Spotify tracks across 15 music genres. It covers the full data science pipeline — from raw data ingestion to machine learning model comparison, business insight generation, and SHAP explainability analysis.

**Key Question:** *Can we predict a song's genre from its audio features alone?*

**Answer:** Yes — with **66.27% accuracy** using Gradient Boosting, where `tempo` alone accounts for 46.6% of the classification power.

> 🏆 *This is the only notebook on this Kaggle dataset (50+ notebooks) to include SHAP explainability analysis across all 15 genres.*

---

## 📊 Dataset

| Property | Detail |
|---|---|
| Source | [Dataset of Songs in Spotify — Kaggle (mrmorj)](https://www.kaggle.com/datasets/mrmorj/dataset-of-songs-in-spotify) |
| Files | `genres_v2.csv` + `playlists.csv` |
| Size | 42,305 rows × 22 columns |
| Genres | 15 (Dark Trap, Hiphop, EDM, Rap, Pop, RnB, and more) |
| Missing Values | None in audio features |

### Raw Features Used
```
danceability  energy        loudness      speechiness   acousticness
instrumentalness  liveness  valence       tempo         duration_ms
```

---

## 🔬 Project Pipeline

```
Raw Data (42,305 tracks)
   │
   ▼
Data Cleaning ──────────── Remove duplicates, encode columns, drop junk fields
   │
   ▼
Feature Engineering ─────── 6 new features:
   │                         energy_density, dance_mood_score, vocal_presence,
   │                         duration_min, loudness_normalized, tempo_category
   ▼
EDA ────────────────────── Correlation heatmap, distributions, genre-level
   │                        analysis, interactive Plotly charts
   ▼
Outlier Detection ────────── Z-score (threshold=3), boxplots before/after,
   │                          capping strategy (preserves data volume)
   ▼
Machine Learning ──────────  AdaBoost      (GridSearchCV, 5-fold CV)
   │                          Gradient Boosting
   │                          Random Forest
   ▼
Bagging Regressor ─────────  With replacement vs Without replacement
   │                          Comparison on energy prediction
   ▼
Model Comparison ──────────  Accuracy, Precision, Recall, F1, Confusion Matrix
   │
   ▼
SHAP Explainability ───────  Global importance, per-genre breakdown,
   │                          single-song waterfall chart  ← UNIQUE
   ▼
Business Insights ─────────  Genre patterns, tempo analysis, recommendations
```

---

## 🤖 Model Results

### Classification — Genre Prediction

| Model | Accuracy | Macro F1 | Notes |
|---|---|---|---|
| **Gradient Boosting** | **66.27%** | **0.61** | ★ Best overall |
| Random Forest | 64.29% | 0.61 | Stable, less overfit |
| AdaBoost | 34.80% | 0.29 | Weak on overlapping genres |

### Genre-level Performance (Gradient Boosting)

| Genre | Precision | Recall | F1 | Difficulty |
|---|---|---|---|---|
| dnb | 0.95 | 0.99 | **0.97** | Easy — distinct tempo |
| hardstyle | 0.83 | 0.93 | **0.88** | Easy — high energy |
| psytrance | 0.84 | 0.88 | **0.86** | Easy — fast tempo |
| techhouse | 0.86 | 0.90 | **0.88** | Easy — high danceability |
| Dark Trap | 0.55 | 0.44 | 0.49 | Hard — overlaps with rap |
| Pop | 0.19 | 0.10 | 0.13 | Hard — too few samples |

### Regression — Energy Prediction (Bagging)

| Model | MSE | RMSE | R² |
|---|---|---|---|
| **Bagging (With Replacement)** | **~0.000** | **0.000069** | **~1.000** |
| Bagging (Without Replacement) | ~0.000 | 0.000084 | ~1.000 |
| Gradient Boosting Regressor | ~0.000 | 0.0014 | 0.9999 |

> Note: Near-perfect R² (0.9999) confirms that energy is strongly determined by audio features — a genuine data science finding.

---

## 🔎 Feature Importance

### Gradient Boosting (Classification)

| Rank | Feature | Importance |
|---|---|---|
| 1 | `tempo` | **46.6%** |
| 2 | `danceability` | 10.3% |
| 3 | `instrumentalness` | 9.8% |
| 4 | `loudness_normalized` | 7.0% |
| 5 | `vocal_presence` | 6.8% |

### SHAP (Random Forest — Global)

| Rank | Feature | SHAP Value |
|---|---|---|
| 1 | `tempo` | 0.03606 |
| 2 | `instrumentalness` | 0.01921 |
| 3 | `danceability` | 0.01916 |
| 4 | `vocal_presence` | 0.01659 |
| 5 | `energy` | 0.01459 |

---

## 🔮 SHAP Explainability (Unique Contribution)

This is the **only notebook on this Kaggle dataset** to include SHAP (SHapley Additive exPlanations) analysis, explaining *why* the model makes each prediction.

| SHAP Plot | Description |
|---|---|
| `shap_summary_bar.png` | Global feature importance ranked by mean \|SHAP\| across all 15 genres |
| `shap_by_genre.png` | Per-genre breakdown — what matters for each specific genre |
| `shap_waterfall.png` | Step-by-step explanation of a single song prediction |

**Key SHAP Findings:**
- `tempo` is the single most decisive feature globally
- Electronic genres (dnb, techno, hardstyle) are separated almost entirely by tempo
- Rap genres (Underground Rap, Dark Trap) rely on speechiness + danceability
- RnB/Pop are differentiated by valence + dance_mood_score

---

## 💡 Business Insights

- **Electronic genres** (dnb, hardstyle, psytrance, techhouse) classified with **86–97% accuracy** — very distinct audio fingerprints
- **Rap subgenres** (Dark Trap, Underground Rap, Trap Metal) overlap heavily — audio features alone are insufficient to distinguish them
- **Tempo is the #1 predictor** — electronic genres cluster at fast BPMs (>130), hip-hop at mid range (90–130)
- **Slow songs are happier** — tracks < 90 BPM have highest valence (0.437)
- **Mid-tempo = most danceable** — 90–130 BPM tracks score highest danceability (0.724)
- **Energy is perfectly predictable** (R²=0.9999) from loudness, acousticness, and tempo — these three features define energy

---

## 📁 Repository Structure

```
spotify-genre-classifier/
│
├── spotify_kaggle_final.py        # Complete Python code (16 sections)
├── README.md                      # This file
├── requirements.txt               # Python dependencies
│
└── outputs/                       # Generated charts & plots
    ├── correlation_heatmap.png        # Feature correlations
    ├── distributions.png              # Feature distributions
    ├── genre_counts.png               # Genre track counts
    ├── energy_by_genre.png            # Energy boxplot by genre
    ├── boxplots_before.png            # Outliers before capping
    ├── boxplots_after.png             # Outliers after capping
    ├── feature_importance.png         # GB feature importances
    ├── bagging_comparison.png         # Bagging RMSE + scatter
    ├── model_comparison.png           # Accuracy comparison bar
    ├── genre_energy.png               # Avg energy by genre
    ├── confusion_matrix.png           # GB confusion matrix
    ├── shap_summary_bar.png           # ★ SHAP global importance
    ├── shap_by_genre.png              # ★ SHAP per genre
    ├── shap_waterfall.png             # ★ SHAP single prediction
    ├── energy_vs_danceability_interactive.html
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
# Update file path in code to:
# df = pd.read_csv('genres_v2.csv')

# Run
python spotify_kaggle_final.py
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
shap>=0.41.0
```

---

## 🆚 Comparison with Other Notebooks

| Notebook | Accuracy | SHAP | Feature Eng. | Genres |
|---|---|---|---|---|
| Most notebooks (50+) | 45–58% | ❌ | ❌ | 5–7 |
| XG_TF_Classification (2024) | ~61% | ❌ | ❌ | 15 |
| **This Project** | **66.27%** | **✅ 3 plots** | **✅ 6 features** | **15** |

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

- [GeeksforGeeks — Outlier Detection in Data Mining](https://www.geeksforgeeks.org/challenges-of-outlier-detection-in-data-mining/)
- [IBM Topics — Bootstrap Aggregating (Bagging)](https://www.ibm.com/topics/bagging)
- [GeeksforGeeks — Boosting & AdaBoost](https://www.geeksforgeeks.org/boosting-in-machine-learning-boosting-and-adaboost/)
- [Scikit-learn — Ensemble Methods](https://scikit-learn.org/stable/modules/ensemble.html)
- [SHAP Documentation](https://shap.readthedocs.io/en/latest/)
- [Kaggle Dataset by mrmorj](https://www.kaggle.com/datasets/mrmorj/dataset-of-songs-in-spotify)

---

⭐ *If you found this project useful, please give it a star!*
