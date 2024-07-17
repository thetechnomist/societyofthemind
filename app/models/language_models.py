import torch
from transformers import pipeline


class LanguageModels:
    def __init__(self):
        # self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = "cpu"
        try:
            self.general_lm = pipeline("text-generation", model="gpt2", device=self.device)
            self.code_lm = pipeline("text-generation", model="codeparrot/codeparrot-small", device=self.device)
            self.creative_lm = pipeline("text-generation", model="distilgpt2", device=self.device)
        except RuntimeError:
            print("Warning: MPS acceleration failed. Falling back to CPU.")
            self.device = "cpu"
            self.general_lm = pipeline("text-generation", model="gpt2", device=self.device)
            self.code_lm = pipeline("text-generation", model="codeparrot/codeparrot-small", device=self.device)
            self.creative_lm = pipeline("text-generation", model="distilgpt2", device=self.device)

    def generate_response(self, query, model_type):
        try:
            if model_type == 'code':
                return self.code_lm(query, max_length=100)[0]['generated_text']
            elif model_type == 'creative':
                return self.creative_lm(query, max_length=100)[0]['generated_text']
            else:
                return self.general_lm(query, max_length=100)[0]['generated_text']
        except RuntimeError:
            print(f"Warning: Error using {self.device}. Falling back to CPU for this operation.")
            with torch.device('cpu'):
                if model_type == 'code':
                    return self.code_lm(query, max_length=100)[0]['generated_text']
                elif model_type == 'creative':
                    return self.creative_lm(query, max_length=100)[0]['generated_text']
                else:
                    return self.general_lm(query, max_length=100)[0]['generated_text']
