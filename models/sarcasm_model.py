from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from transformers import BertTokenizer, BertModel
import torch
import numpy as np


class SarcasmClassifier:
    def __init__(self, embedding_type='tfidf'):
        self.embedding_type = embedding_type

        if self.embedding_type == 'tfidf':
            self.vectorizer = TfidfVectorizer(max_features=10000)
            self.model = LogisticRegression(max_iter=1000)

        elif self.embedding_type == 'bert':
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            self.bert_model = BertModel.from_pretrained('bert-base-uncased')
            self.model = LogisticRegression(max_iter=1000)

        else:
            raise ValueError("Unsupported embedding type: choose 'tfidf' or 'bert'")

    def _get_bert_embeddings(self, texts):
        self.bert_model.eval()
        with torch.no_grad():
            encoded = self.tokenizer(texts, padding=True, truncation=True,
                                     return_tensors='pt', max_length=128)
            output = self.bert_model(**encoded)
            cls_embeddings = output.last_hidden_state[:, 0, :]  # [CLS] token
            return cls_embeddings.numpy()

    def train(self, X, y):
        if self.embedding_type == 'tfidf':
            X_vect = self.vectorizer.fit_transform(X)
            self.model.fit(X_vect, y)

        elif self.embedding_type == 'bert':
            X_embed = self._get_bert_embeddings(list(X))
            self.model.fit(X_embed, y)

    def predict(self, X):
        if self.embedding_type == 'tfidf':
            X_vect = self.vectorizer.transform(X)
            return self.model.predict(X_vect)

        elif self.embedding_type == 'bert':
            X_embed = self._get_bert_embeddings(list(X))
            return self.model.predict(X_embed)

    def predict_proba(self, X):
        if self.embedding_type == 'tfidf':
            X_vect = self.vectorizer.transform(X)
            return self.model.predict_proba(X_vect)

        elif self.embedding_type == 'bert':
            X_embed = self._get_bert_embeddings(list(X))
            return self.model.predict_proba(X_embed)
