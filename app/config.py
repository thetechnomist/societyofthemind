import os

class Config:
    REDIS_HOST = os.environ.get('REDIS_HOST', 'localhost')
    REDIS_PORT = int(os.environ.get('REDIS_PORT', 6379))
    REDIS_DB = int(os.environ.get('REDIS_DB', 0))

# app/models/language_models.py
from transformers import pipeline

class LanguageModels:
    def __init__(self):
        self.general_lm = pipeline("text-generation", model="gpt2")
        self.code_lm = pipeline("text-generation", model="codeparrot/codeparrot-small")
        self.creative_lm = pipeline("text-generation", model="distilgpt2")

    def generate_response(self, query, model_type):
        if model_type == 'code':
            return self.code_lm(query, max_length=100)[0]['generated_text']
        elif model_type == 'creative':
            return self.creative_lm(query, max_length=100)[0]['generated_text']
        else:
            return self.general_lm(query, max_length=100)[0]['generated_text']