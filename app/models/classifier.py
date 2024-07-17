from sentence_transformers import SentenceTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
import torch

class QueryClassifier:
    def __init__(self):
        if torch.backends.mps.is_available():
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
        
        try:
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device=self.device)
        except RuntimeError:
            print("Warning: MPS acceleration failed for SentenceTransformer. Falling back to CPU.")
            self.device = "cpu"
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device=self.device)
        
        self.classifier = Pipeline([
            ('features', FeatureExtractor(self.embedding_model)),
            ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
        ])
        self.is_fitted = False

    def fit(self, X, y):
        self.classifier.fit(X, y)
        self.is_fitted = True

    def classify(self, query, threshold=0.4):
        if not self.is_fitted:
            raise ValueError("Classifier not fitted yet. Call fit() before classify().")
        
        probs = self.classifier.predict_proba([query])[0]
        print(probs)
        best_class = self.classifier.classes_[probs.argmax()]
        # print best_class
        print(f"Best class: {best_class}")
        if probs.max() < threshold:
            return 'general'
        return best_class

    
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class FeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, embedding_model):
        self.embedding_model = embedding_model

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        features = []
        for query in X:
            length = len(query)
            word_count = len(query.split())
            contains_code = int('code' in query.lower() or 'function' in query.lower())
            contains_creative = int('story' in query.lower() or 'write' in query.lower())
            embedding = self.embedding_model.encode(query).tolist()

            # Flatten the embedding and append other features
            query_features = [length, word_count, contains_code, contains_creative] + embedding
            features.append(query_features)

        # Convert the list of features to a NumPy array
        return np.array(features)