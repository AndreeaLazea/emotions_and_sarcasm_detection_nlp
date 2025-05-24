import kagglehub
import pandas as pd
import re
import nltk
import os

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split


class NLPDatasetLoader:
    def __init__(self):
        nltk.download('stopwords')
        nltk.download('punkt')
        nltk.download('wordnet')
        nltk.download('omw-1.4')

        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()

    # -------------------
    # Load Sarcasm Data
    # -------------------
    def load_sarcasm_data(self):
        path = kagglehub.dataset_download("danofer/sarcasm")
        file_path = f"{path}/train-balanced-sarcasm.csv"
        df = pd.read_csv(file_path, low_memory=False)
        print("[Sarcasm] Shape:", df.shape)
        print("[Sarcasm] Columns:", df.columns)
        return df

    # -------------------
    # Load Emotion Data
    # -------------------
    def load_emotion_data(self):
        path = kagglehub.dataset_download("shivamb/go-emotions-google-emotions-dataset")

        for f in os.listdir(path):
            print("[Emotion] File found:", f)

        file_path = f"{path}/go_emotions_dataset.csv"
        df = pd.read_csv(file_path, low_memory=False)
        print("[Emotion] Shape:", df.shape)
        print("[Emotion] Columns:", df.columns)
        return df

    # -------------------
    # General Text Preprocessing
    # -------------------
    def preprocess(self, text):
        text = text.lower()
        text = re.sub(r"http\S+|www\S+|https\S+", '', text)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        tokens = word_tokenize(text)
        tokens = [word for word in tokens if word not in self.stop_words]
        lemmatized = [self.lemmatizer.lemmatize(word) for word in tokens]
        return ' '.join(lemmatized)

    # -------------------
    # Apply Preprocessing to a DataFrame
    # -------------------
    def preprocess_dataframe(self, df, text_column):
        df = df.copy()
        df[text_column] = df[text_column].astype(str).apply(self.preprocess)
        return df

    # -------------------
    # Train/Test Split
    # -------------------
    def get_train_test_split(self, df, text_column, label_column, test_size=0.2):
        X = df[text_column]
        y = df[label_column]
        return train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
