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
