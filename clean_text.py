import re
import string
import pandas as pd

# Noktalama karakterlerini ayarla
punct_chars = list((set(string.punctuation) | {
    "’", "‘", "–", "—", "~", "|", """, """, "…", "'", "`", "_"
}) - set(["#"]))
punct_chars.sort()
punctuation = "".join(punct_chars)
replace = re.compile("[%s]" % re.escape(punctuation))

def clean_text(text):
    """
    Metni temizler: küçük harfe çevirir, noktalama işaretlerini kaldırır ve fazla boşlukları temizler.
    
    Args:
        text (str): Temizlenecek metin
        
    Returns:
        str: Temizlenmiş metin
    """
    if isinstance(text, str):
        text = text.lower()  # Küçük harfe çevir
        text = replace.sub(" ", text)  # Noktalama temizle
        text = re.sub(r'\s+', ' ', text).strip()  # Fazla boşlukları temizle
        return text
    return text

def clean_dataframe(df):
    """
    DataFrame'deki text sütununu temizler.
    
    Args:
        df (pandas.DataFrame): Temizlenecek DataFrame
        
    Returns:
        pandas.DataFrame: Temizlenmiş DataFrame
    """
    if 'text' in df.columns:
        df['text'] = df['text'].apply(clean_text)
    return df

def remove_unclear_examples(df):
    """
    Belirsiz örnekleri çıkarır.
    
    Args:
        df (pandas.DataFrame): DataFrame
        
    Returns:
        pandas.DataFrame: Temizlenmiş DataFrame
    """
    if 'example_very_unclear' in df.columns:
        df = df[df['example_very_unclear'] == False]
    return df

def remove_short_texts(df, min_length=10):
    """
    Çok kısa metinleri çıkarır.
    
    Args:
        df (pandas.DataFrame): DataFrame
        min_length (int): Minimum metin uzunluğu
        
    Returns:
        pandas.DataFrame: Temizlenmiş DataFrame
    """
    if 'text' in df.columns:
        df = df[df['text'].notnull() & (df['text'].str.len() > min_length)]
    return df

def remove_multiple_emotions(df, emotion_columns):
    """
    Birden fazla duyguya sahip satırları çıkarır.
    
    Args:
        df (pandas.DataFrame): DataFrame
        emotion_columns (list): Duygu sütunları listesi
        
    Returns:
        pandas.DataFrame: Temizlenmiş DataFrame
    """
    if all(col in df.columns for col in emotion_columns):
        df = df[df[emotion_columns].sum(axis=1) == 1]
    return df

def remove_no_emotion(df, emotion_columns):
    """
    Hiç duygu etiketi olmayan satırları çıkarır.
    
    Args:
        df (pandas.DataFrame): DataFrame
        emotion_columns (list): Duygu sütunları listesi
        
    Returns:
        pandas.DataFrame: Temizlenmiş DataFrame
    """
    if all(col in df.columns for col in emotion_columns):
        df = df[df[emotion_columns].sum(axis=1) > 0]
    return df

def get_first_label(row, emotion_columns):
    """
    Her satırda 1 olan ilk duygunun index'ini bulur.
    
    Args:
        row: DataFrame satırı
        emotion_columns (list): Duygu sütunları listesi
        
    Returns:
        int: Duygu index'i
    """
    for i, col in enumerate(emotion_columns):
        if row[col] == 1:
            return i
    return -1

def apply_comprehensive_cleaning(df, emotion_columns):
    """
    Kapsamlı veri temizliği uygular.
    
    Args:
        df (pandas.DataFrame): DataFrame
        emotion_columns (list): Duygu sütunları listesi
        
    Returns:
        pandas.DataFrame: Temizlenmiş DataFrame
    """
    # 1. Belirsiz örnekleri çıkar
    df = remove_unclear_examples(df)
    
    # 2. Boş veya çok kısa metinleri çıkar
    df = remove_short_texts(df, min_length=10)
    
    # 3. Metin temizliği
    df = clean_dataframe(df)
    
    # 4. Birden fazla duyguya sahip satırları çıkar
    df = remove_multiple_emotions(df, emotion_columns)
    
    # 5. Hiç duygu etiketi olmayanları çıkar
    df = remove_no_emotion(df, emotion_columns)
    
    # 6. Etiketleri oluştur
    df['labels'] = df.apply(lambda row: get_first_label(row, emotion_columns), axis=1)
    
    # 7. Etiketsiz satırları çıkar
    df = df[df['labels'] != -1]
    
    return df 