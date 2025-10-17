# 🧠 Sentiment / Emotion Classification with GoEmotions (27 Labels)

**1st difference → Classification → 27 different types of emotions**

This project builds a **clean dataset and baseline analysis** for multi-label emotion classification using the **GoEmotions corpus** by Google.  
It covers everything from **data preprocessing and EDA** to **modeling recommendations**, giving you a complete foundation for emotion recognition tasks in NLP.

---

## 📋 Overview

The aim is to develop a machine learning model that can **predict emotional attitudes** in a given text comment by analyzing linguistic patterns.  
After cleaning and exploring the data, the next step is to **train and evaluate models** capable of classifying comments into **27 different emotion categories** such as `joy`, `anger`, `admiration`, `gratitude`, `disgust`, and more.

---

## 📁 Project Structure

├── analyze_data.py # EDA + correlation heatmap
├── clean_text.py # Full cleaning pipeline (functions you can import)
├── main.py # (reserved for training/inference)
├── emotions.txt # 27 emotion labels (one per line, in column order)
├── goemotions_1.csv
├── goemotions_2.csv
├── goemotions_3.csv
└── README.md


---

## 📊 1) Exploratory Data Analysis (EDA)

**Script:** `analyze_data.py`

### What it does
- Loads all `goemotions_*.csv` files  
- Reads emotion names from `emotions.txt`  
- Prints:
  - Total number of samples  
  - Per-emotion sample counts  
  - Text length statistics  
- Creates and saves a **correlation heatmap** (`emotion_correlation_heatmap.png`) showing relationships among emotions.

🧹 2) Data Cleaning Pipeline

Module: clean_text.py

This module provides a comprehensive cleaning pipeline to prepare GoEmotions for training.

🧽 Text Normalization

Lowercase conversion

Punctuation removal

Extra whitespace removal

🧩 Row Filtering

Remove unclear examples (example_very_unclear == True)

Drop very short texts (len(text) <= 10)

Optionally remove multi-label rows (for single-label setups)

Remove rows with no labels

🏷️ Label Index Creation

Adds a labels column containing the index of the first active emotion — required for single-label classification.

🧪 3) Next Steps — Modeling

After cleaning, the dataset is ready for modeling.
You can choose between single-label or multi-label setups.

🔹 Single-Label Classification

Each text expresses only one emotion (e.g., joy or anger, not both).

| text                   | joy | anger | sadness | ... |
| ---------------------- | --- | ----- | ------- | --- |
| "I love my job"        | 1   | 0     | 0       | ... |
| "He ruined everything" | 0   | 1     | 0       | ... |


Recommended Models

Logistic Regression

SVM (Support Vector Machine)

LightGBM / XGBoost

Neural Networks (MLP, BERT fine-tuning)

Feature Representations

TF-IDF vectors (classic)

Embeddings (sentence-transformers, BERT, DistilBERT)

Metrics

Accuracy

Precision / Recall

Macro-F1 (useful for imbalanced labels)

Multi-Label Classification

Each text can express multiple emotions simultaneously (e.g., joy + surprise).
| text                                 | joy | surprise | anger | sadness |
| ------------------------------------ | --- | -------- | ----- | ------- |
| "I got promoted but now I’m nervous" | 1   | 1        | 0     | 0       |
| "This is unfair and makes me sad"    | 0   | 0        | 1     | 1       |

Suggested Models

Binary-Relevance Logistic Regression / SVM

LightGBM / XGBoost (multi-output)

Neural Networks with sigmoid activation
Metrics

Macro-F1: averages each label equally

Micro-F1: aggregates all predictions (better for imbalance)

Data Splitting

Use stratified or time-based splits depending on your data. train and test

from sklearn.model_selection import train_test_split
train, test = train_test_split(df, stratify=df['labels'], test_size=0.2)

summary: 
| Setup            | Labels per Text | Model Type   | Loss                 | Metrics        | Typical Models                           |
| ---------------- | --------------- | ------------ | -------------------- | -------------- | ---------------------------------------- |
| **Single-Label** | 1               | Multi-class  | Cross-Entropy        | Accuracy, F1   | Logistic Regression, SVM, LightGBM, BERT |
| **Multi-Label**  | >1              | Multi-output | Binary Cross-Entropy | Macro/Micro-F1 | Binary Relevance, LightGBM, Sigmoid NN   |

feature engineering and model selection after understanding the data?

**Data Understanding: Explored dataset size, label frequencies, correlations, and text distributions.
**
Feature Engineering:

Cleaned text (lowercase, punctuation removal).

Encoded text with TF-IDF and embeddings for richer representations.

Transformed multi-hot emotion labels for modeling.

Model Selection:

For single-label, compared interpretable baselines (Logistic Regression, SVM) vs. boosted models (LightGBM).

For multi-label, used binary cross-entropy and tracked macro/micro-F1.

Evaluated transformer fine-tuning (BERT) for contextual performance.

Validation Strategy: Used stratified or temporal splits to prevent data leakage and ensure robust generalization.


