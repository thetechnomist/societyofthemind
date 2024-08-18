import json
import logging
import random
import requests
from typing import List, Dict, Any
from openai import OpenAI

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize OpenAI client
client = OpenAI(api_key="your-openai-api-key-here")

class AIModelRepository:
    def __init__(self):
        self.api_url = "https://huggingface.co/api/models"
        self.cache = {}

    def search_models(self, query: str, task: str = None, limit: int = 5) -> List[Dict[str, Any]]:
        if (query, task) in self.cache:
            return self.cache[(query, task)]

        params = {"search": query, "limit": limit}
        if task:
            params["filter"] = task

        response = requests.get(self.api_url, params=params)
        if response.status_code != 200:
            logger.error(f"Failed to fetch models: {response.status_code}")
            return []

        models = response.json()
        processed_models = self._process_models(models)
        self.cache[(query, task)] = processed_models
        return processed_models

    def _process_models(self, models: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        processed = []
        for model in models:
            processed_model = {
                "name": model["modelId"],
                "task": model.get("pipeline_tag", "Unknown"),
                "downloads": model.get("downloads", 0),
                "likes": model.get("likes", 0),
                "tags": model.get("tags", []),
            }
            # Estimate metrics based on available data
            processed_model["estimated_accuracy"] = min(0.5 + (processed_model["likes"] / 1000) * 0.5, 0.99)
            processed_model["estimated_latency"] = max(5 - (processed_model["downloads"] / 10000), 1)
            processed_model["estimated_reliability"] = min(0.9 + (processed_model["downloads"] / 100000) * 0.1, 0.99)
            processed.append(processed_model)
        return processed

class ModelSelector:
    @staticmethod
    def select_model(models: List[Dict[str, Any]], min_accuracy: float = 0.8, 
                     max_latency: float = 5.0, min_reliability: float = 0.9) -> Dict[str, Any]:
        candidates = [model for model in models 
                      if model['estimated_accuracy'] >= min_accuracy
                      and model['estimated_latency'] <= max_latency
                      and model['estimated_reliability'] >= min_reliability]
        
        if not candidates:
            return None  # No model meets the criteria

        return max(candidates, key=lambda x: x['estimated_accuracy'])

class ReasoningEngine:
    def __init__(self):
        self.model_repo = AIModelRepository()

    def find_and_evaluate_model(self, query: str, task: str = None, 
                                min_accuracy: float = 0.8, max_latency: float = 5.0, 
                                min_reliability: float = 0.9) -> str:
        models = self.model_repo.search_models(query, task)
        selected_model = ModelSelector.select_model(models, min_accuracy, max_latency, min_reliability)

        if not selected_model:
            return f"No suitable model found for the query: {query}"

        return (f"Selected model: {selected_model['name']}\n"
                f"Task: {selected_model['task']}\n"
                f"Estimated Accuracy: {selected_model['estimated_accuracy']:.2f}\n"
                f"Estimated Latency: {selected_model['estimated_latency']:.2f} seconds\n"
                f"Estimated Reliability: {selected_model['estimated_reliability']:.2f}\n"
                f"Tags: {', '.join(selected_model['tags'])}")

    def process_query(self, user_question: str):
        messages = [
            {"role": "system", "content": "You are an AI assistant that can search for and evaluate AI models "
                                          "based on user queries. Analyze the user's question to determine the "
                                          "appropriate AI task and any specific requirements for accuracy, "
                                          "latency, or reliability."},
            {"role": "user", "content": user_question}
        ]

        response = chat_with_gpt(messages)
        content = response.content

        # Extract task and requirements from the LLM's response
        task = extract_task(content)
        requirements = extract_requirements(content)

        # Use the extracted information to find and evaluate a model
        model_info = self.find_and_evaluate_model(
            user_question, 
            task=task,
            min_accuracy=requirements.get('accuracy', 0.8),
            max_latency=requirements.get('latency', 5.0),
            min_reliability=requirements.get('reliability', 0.9)
        )

        # Generate a final response incorporating the model information
        final_response = chat_with_gpt([
            {"role": "system", "content": "You are an AI assistant that provides information about AI models. "
                                          "Use the following model information to answer the user's question."},
            {"role": "user", "content": user_question},
            {"role": "assistant", "content": f"Based on your question, I've found a suitable AI model. Here's the information:\n\n{model_info}\n\nNow, to answer your question:"}
        ])

        return final_response.content

logger = logging.getLogger(__name__)

def generate_insights(post_data, comments_df):
    insights = {}
    
    # Combine post body and comments
    all_text = post_data['body'] + '\n\n' + '\n\n'.join(comments_df['body'])
    
    # Prepare the messages for the GPT model
    messages = [
        {"role": "system", "content": "You are an AI assistant that analyzes Reddit threads. Provide insights including key questions, top answer, key topics, and a summary."},
        {"role": "user", "content": f"Analyze this Reddit thread:\n\n{all_text}"}
    ]
    
    try:
        # Use the chat_with_gpt function to get the analysis
        response = chat_with_gpt(messages, model="gpt-4o-mini")
        
        analysis = response.content
        
        # Parse the analysis to extract insights
        sections = analysis.split('\n\n')
        for section in sections:
            if section.startswith("Key Questions:"):
                insights['key_questions'] = section.split('\n')[1:]
            elif section.startswith("Top Answer:"):
                insights['top_answer'] = section.split('\n', 1)[1]
            elif section.startswith("Key Topics:"):
                insights['key_topics'] = section.split('\n')[1].split(', ')
            elif section.startswith("Summary:"):
                insights['summary'] = section.split('\n', 1)[1]
        
    except Exception as e:
        logger.error(f"Error using GPT for analysis: {e}")
        insights = {
            'key_questions': [],
            'top_answer': "Error in analysis",
            'key_topics': [],
            'summary': "Unable to generate summary due to an error."
        }
    
    return insights

# You might want to add this function if it's not already defined elsewhere in your code
def chat_with_gpt(messages: List[Dict[str, str]], model: str = "gpt-3.5-turbo"):
    logger.info(f"Sending request to OpenAI API with messages: {json.dumps(messages, indent=2)}")
    response = client.chat.completions.create(
        model=model,
        messages=messages
    )
    logger.info(f"Received response from OpenAI API: {response}")
    return response.choices[0].message

def extract_task(content: str) -> str:
    # This is a simplistic extraction. In a real-world scenario, you might use
    # a more sophisticated NLP technique or another LLM call for this.
    tasks = ["text-classification", "question-answering", "summarization", "translation"]
    for task in tasks:
        if task in content.lower():
            return task
    return None

def extract_requirements(content: str) -> Dict[str, float]:
    # Again, this is a simplistic extraction. A more robust solution would be needed
    # for a production system.
    requirements = {}
    if "high accuracy" in content.lower():
        requirements["accuracy"] = 0.9
    if "low latency" in content.lower():
        requirements["latency"] = 2.0
    if "high reliability" in content.lower():
        requirements["reliability"] = 0.95
    return requirements

# Usage
engine = ReasoningEngine()

user_question = "I need a model for sentiment analysis of product reviews. It should be highly accurate and respond quickly."
final_response = engine.process_query(user_question)

print(f"User: {user_question}")
print(f"Assistant: {final_response}")
