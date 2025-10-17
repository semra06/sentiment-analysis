# 1st difference => Classification => 27 different types of emotions
Sentiment / Emotion Classification with GoEmotions (27 labels)

This repository builds a clean dataset and baseline analysis for multi-label emotion classification using the GoEmotions corpus.
It includes:

EDA script that merges the three CSV parts and produces an emotion correlation heatmap

A comprehensive cleaning pipeline (text normalization, filtering, label handling) that prepares the data for modeling

📁 Project Structure
├── analyze_data.py          # EDA + correlation heatmap
├── clean_text.py            # Full cleaning pipeline (functions you can import)
├── main.py                  # (reserved for training/inference if you add later)
├── emotions.txt             # 27 emotion labels (one per line, in column order)
├── goemotions_1.csv
├── goemotions_2.csv
├── goemotions_3.csv
└── README.md
📊 1) Exploratory Data Analysis (EDA)

Script: analyze_data.py
What it does:

Loads goemotions_1.csv, goemotions_2.csv, goemotions_3.csv

Reads the list of emotion columns from emotions.txt

Prints dataset size and per-emotion counts

Plots & saves an Emotion Label Correlation Matrix to emotion_correlation_heatmap.png

2) Data Cleaning Pipeline

Module: clean_text.py
It provides a set of functions you can compose or call via apply_comprehensive_cleaning:

What gets cleaned?

Text normalization: lowercasing, punctuation removal, whitespace squashing (clean_text, clean_dataframe)

Row filtering:

remove very unclear examples (example_very_unclear == True)

remove short texts (len(text) <= 10)

(optional) remove multi-label rows (sum(labels) != 1) if you want single-label training

remove rows with no labels

Label index: creates an integer labels column as the index of the first active emotion


3) 🧪 Next Steps (Modeling)

Single-label (after remove_multiple_emotions): try Logistic Regression / SVM / LightGBM with TF-IDF or embeddings.

Multi-label (skip remove_multiple_emotions): train with binary cross-entropy, evaluate macro/micro F1.

Keep a time-split or stratified split depending on your data source and task.

Example: emotion label (e.g., joy or anger, but not both).
| text                   | joy | anger | sadness | ... |
| ---------------------- | --- | ----- | ------- | --- |
| "I love my job"        | 1   | 0     | 0       | ... |
| "He ruined everything" | 0   | 1     | 0       | ... |

MODELS: 
Logistic Regression

SVM (Support Vector Machine)

LightGBM / XGBoost

Neural networks (MLP, BERT fine-tune, etc.)
