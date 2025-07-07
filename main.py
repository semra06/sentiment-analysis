# https://github.com/google-research/google-research/tree/master/goemotions/data/full_dataset
# Buradkai veri setiyle metin işleme yaparak. Gelen yorumdan o yorumdaki genel duygu tutumunu tahmin eden modeli geliştirelim.
# 1. fark => Sınıflandırma => 27 farklı duygu türü

import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import torch
import warnings
import subprocess
from sklearn.metrics import accuracy_score
from clean_text import clean_text, apply_comprehensive_cleaning
warnings.filterwarnings('ignore')

# 1. Veri setlerini oku ve birleştir
csv_files = ['goemotions_1.csv', 'goemotions_2.csv', 'goemotions_3.csv']
df_list = [pd.read_csv(f) for f in csv_files]
df = pd.concat(df_list, ignore_index=True)

## Veri analizi
subprocess.run(['python', 'analyze_data.py'])

# Sütun isimlerini kontrol et ve gerekirse düzelt
if 'comment_text' in df.columns:
    df.rename(columns={'comment_text': 'text'}, inplace=True)
if 'emotion' in df.columns:
    df.rename(columns={'emotion': 'labels'}, inplace=True)

print("Sütunlar:", df.columns.tolist())

# emotions.txt dosyasından duygu sütunlarını al
with open('emotions.txt', 'r', encoding='utf-8') as f:
    emotion_columns = [line.strip() for line in f if line.strip()]

# --- VERİ TEMİZLİĞİ ---
df = apply_comprehensive_cleaning(df, emotion_columns)

# 3. Etiket sayısı
num_labels = df['labels'].nunique()

# 4. Tokenizer ve veri seti hazırlığı
tokenizer = BertTokenizer.from_pretrained('distilbert-base-uncased')

df = df.sample(5000)  # veya 10000

class GoEmotionsDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=100)
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_texts, val_texts, train_labels, val_labels = train_test_split(df['text'], df['labels'], test_size=0.1)
train_dataset = GoEmotionsDataset(train_texts.tolist(), train_labels.tolist())
val_dataset = GoEmotionsDataset(val_texts.tolist(), val_labels.tolist())

# 5. Model
model = BertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=num_labels)

# 6. Eğitim
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=10,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=2e-5,
    warmup_steps=100,
    logging_dir='./logs',
    logging_steps=10,
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=1)
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()
eval_result = trainer.evaluate()
print(f"Validation accuracy: {eval_result['eval_accuracy']:.4f}")

# 7. Tahmin fonksiyonu
def predict_sentiment(text):
    # Gelen metni de temizle
    text = clean_text(text)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=100)
    outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=1)
    pred = torch.argmax(probs, dim=1).item()
    
    # Duygu adını al
    emotion_name = emotion_columns[pred]
    confidence = probs[0][pred].item()
    
    return f"Predict emotion: {emotion_name} (Güven: {confidence:.3f})"

# Örnek kullanım
user_text = input("Duygu analizi için bir cümle girin: ")
print(predict_sentiment(user_text))

# Duygu dağılımını göster
print("Duygu dağılımı:")
for i, emotion in enumerate(emotion_columns):
    count = (df['labels'] == i).sum()
    print(f"{emotion}: {count}")
