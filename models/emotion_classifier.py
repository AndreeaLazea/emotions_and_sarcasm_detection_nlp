from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

from transformers import BertTokenizer, BertModel
import torch
import numpy as np


class EmotionClassifier:
    def __init__(self, embedding_type='tfidf'):
        self.embedding_type = embedding_type
        self.label_encoder = LabelEncoder()

        if self.embedding_type == 'tfidf':
            self.vectorizer = TfidfVectorizer(max_features=10000)
            self.model = LogisticRegression(max_iter=1000)

        elif self.embedding_type == 'bert':
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            self.bert_model = BertModel.from_pretrained('bert-base-uncased')
            self.model = LogisticRegression(max_iter=1000)

        else:
            raise ValueError("Unsupported embedding type: choose 'tfidf' or 'bert'")

    def _get_bert_embeddings(self, texts, batch_size=32):
        self.bert_model.eval()
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]

            with torch.no_grad():
                encoded = self.tokenizer(batch, padding=True, truncation=True,
                                         return_tensors='pt', max_length=128)

                # Optional: move to device if using GPU
                # encoded = {k: v.to(device) for k, v in encoded.items()}

                output = self.bert_model(**encoded)
                cls_embeddings = output.last_hidden_state[:, 0, :]  # [CLS]
                all_embeddings.append(cls_embeddings.cpu().numpy())

        return np.vstack(all_embeddings)

    def train(self, X, y):
        # Encode emotion labels (e.g., joy → 0)
        y_encoded = self.label_encoder.fit_transform(y)

        if self.embedding_type == 'tfidf':
            X_vect = self.vectorizer.fit_transform(X)
            self.model.fit(X_vect, y_encoded)

        elif self.embedding_type == 'bert':
            X_embed = self._get_bert_embeddings(list(X), batch_size=32)

            self.model.fit(X_embed, y_encoded)

    def predict(self, X):
        if self.embedding_type == 'tfidf':
            X_vect = self.vectorizer.transform(X)
            y_pred = self.model.predict(X_vect)

        elif self.embedding_type == 'bert':
            X_embed = self._get_bert_embeddings(list(X))
            y_pred = self.model.predict(X_embed)

        return self.label_encoder.inverse_transform(y_pred)

    def predict_proba(self, X):
        if self.embedding_type == 'tfidf':
            X_vect = self.vectorizer.transform(X)
            return self.model.predict_proba(X_vect)

        elif self.embedding_type == 'bert':
            X_embed = self._get_bert_embeddings(list(X))
            return self.model.predict_proba(X_embed)
