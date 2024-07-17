# Function to search for the best model on Hugging Face

from huggingface_hub import HfApi

def search_model(task_type):
    from huggingface_hub import HfApi

    api = HfApi()
    models = api.list_models(filter="object-detection")
    for model in models:
        print(model.modelId, model.downloads)
        
search_model("answer questions")