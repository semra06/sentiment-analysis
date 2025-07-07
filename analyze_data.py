import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Veri yükleme
csv_files = ['goemotions_1.csv', 'goemotions_2.csv', 'goemotions_3.csv']
df_list = [pd.read_csv(f) for f in csv_files]
df = pd.concat(df_list, ignore_index=True)

# 2. Duygu sütunlarını al
with open('emotions.txt', 'r', encoding='utf-8') as f:
    emotion_columns = [line.strip() for line in f if line.strip()]

# 3. Temel istatistikler
print(f"Toplam örnek sayısı: {len(df)}")
print("Her duygunun örnek sayısı:")
for col in emotion_columns:
    print(f"  {col}: {df[col].sum()}")

print("\nMetin uzunluğu (karakter bazında):")
print(df['text'].str.len().describe())

# 4. Korelasyon matrisi
corr = df[emotion_columns].corr()
plt.figure(figsize=(14, 12))
sns.heatmap(corr, annot=False, cmap='coolwarm')
plt.title('Emotion Label Correlation Matrix')
plt.tight_layout()
plt.savefig('emotion_correlation_heatmap.png')
plt.show() 