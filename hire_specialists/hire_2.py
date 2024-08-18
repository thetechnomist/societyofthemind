import json
import logging
import random
from typing import List, Dict, Any
from openai import OpenAI

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize OpenAI client
client = OpenAI(api_key="your-openai-api-key-here")

# Define specialized LLMs with extended properties
specialized_llms = {
    "legal_expert_premium": {
        "model": "gpt-4",
        "system_prompt": "You are a premium AI legal expert. Provide highly accurate and detailed legal advice.",
        "cost_per_token": 0.06 / 1000,  # $0.06 per 1K tokens
        "accuracy": 0.98,
        "latency": 2.5,  # seconds
        "reliability": 0.99,
        "specialization": ["corporate law", "intellectual property"],
    },
    "legal_expert_standard": {
        "model": "gpt-3.5-turbo",
        "system_prompt": "You are an AI legal expert. Provide accurate legal advice.",
        "cost_per_token": 0.002 / 1000,
        "accuracy": 0.92,
        "latency": 1.0,
        "reliability": 0.97,
        "specialization": ["general law", "contract law"],
    },
    "medical_expert_premium": {
        "model": "gpt-4",
        "system_prompt": "You are a premium AI medical expert. Provide highly accurate medical information and advice.",
        "cost_per_token": 0.06 / 1000,
        "accuracy": 0.99,
        "latency": 3.0,
        "reliability": 0.995,
        "specialization": ["diagnostics", "treatment plans"],
    },
    "medical_expert_standard": {
        "model": "gpt-3.5-turbo",
        "system_prompt": "You are an AI medical expert. Provide accurate medical information and advice.",
        "cost_per_token": 0.002 / 1000,
        "accuracy": 0.94,
        "latency": 1.2,
        "reliability": 0.98,
        "specialization": ["general health", "first aid"],
    },
    "general_assistant": {
        "model": "gpt-3.5-turbo",
        "system_prompt": "You are a helpful AI assistant.",
        "cost_per_token": 0.002 / 1000,
        "accuracy": 0.90,
        "latency": 1.0,
        "reliability": 0.99,
        "specialization": ["general knowledge"],
    }
}

# Define functions for the main LLM to call
functions = [
    {
        "name": "hire_specialist",
        "description": "Hire a specialist LLM based on the query and constraints",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The question to ask the specialist"
                },
                "domain": {
                    "type": "string",
                    "description": "The domain of expertise required (e.g., legal, medical)"
                },
                "min_accuracy": {
                    "type": "number",
                    "description": "The minimum required accuracy (0-1)"
                },
                "max_latency": {
                    "type": "number",
                    "description": "The maximum acceptable latency in seconds"
                },
                "min_reliability": {
                    "type": "number",
                    "description": "The minimum required reliability (0-1)"
                },
                "specific_expertise": {
                    "type": "string",
                    "description": "Any specific area of expertise required"
                }
            },
            "required": ["query", "domain"]
        }
    }
]

def chat_with_gpt(messages: List[Dict[str, str]], model: str = "gpt-3.5-turbo", functions: List[Dict[str, Any]] = None):
    logger.info(f"Sending request to OpenAI API with messages: {json.dumps(messages, indent=2)}")
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        functions=functions,
        function_call="auto",
    )
    logger.info(f"Received response from OpenAI API: {response}")
    return response.choices[0].message

class CostManager:
    def __init__(self, budget: float):
        self.budget = budget
        self.total_cost = 0

    def can_afford(self, estimated_tokens: int, cost_per_token: float) -> bool:
        estimated_cost = estimated_tokens * cost_per_token
        return (self.total_cost + estimated_cost) <= self.budget

    def update_cost(self, tokens_used: int, cost_per_token: float):
        cost = tokens_used * cost_per_token
        self.total_cost += cost
        logger.info(f"Cost updated. New total: ${self.total_cost:.4f}")

class SpecialistSelector:
    @staticmethod
    def select_specialist(domain: str, min_accuracy: float = 0.9, max_latency: float = 3.0, 
                          min_reliability: float = 0.95, specific_expertise: str = None) -> str:
        candidates = [name for name, llm in specialized_llms.items() 
                      if domain.lower() in name.lower() 
                      and llm['accuracy'] >= min_accuracy
                      and llm['latency'] <= max_latency
                      and llm['reliability'] >= min_reliability
                      and (specific_expertise is None or specific_expertise.lower() in [s.lower() for s in llm['specialization']])]
        
        if not candidates:
            return "general_assistant"  # Fallback to general assistant if no specialist meets the criteria
        
        return max(candidates, key=lambda x: specialized_llms[x]['accuracy'])  # Select the most accurate among candidates

class ReasoningEngine:
    def __init__(self, budget: float):
        self.environment = {}
        self.cost_manager = CostManager(budget)

    def hire_specialist(self, query: str, domain: str, min_accuracy: float = 0.9, 
                        max_latency: float = 3.0, min_reliability: float = 0.95, 
                        specific_expertise: str = None) -> str:
        specialist = SpecialistSelector.select_specialist(domain, min_accuracy, max_latency, min_reliability, specific_expertise)
        llm = specialized_llms[specialist]
        messages = [
            {"role": "system", "content": llm["system_prompt"]},
            {"role": "user", "content": query}
        ]

        estimated_tokens = sum(len(m["content"].split()) for m in messages) * 1.3

        if not self.cost_manager.can_afford(estimated_tokens, llm["cost_per_token"]):
            return f"Cannot afford to consult {specialist} due to budget constraints."

        # Simulate latency
        import time
        time.sleep(llm["latency"])

        # Simulate reliability
        if random.random() > llm["reliability"]:
            return f"The {specialist} is currently unavailable due to technical issues. Please try again later."

        response = chat_with_gpt(messages, model=llm["model"])
        
        self.cost_manager.update_cost(response.usage.total_tokens, llm["cost_per_token"])
        
        return response.content

    def process_query(self, user_question: str):
        messages = [
            {"role": "system", "content": "You are an AI assistant that can hire specialists when needed. "
                                          "Consider the nature of the query and specify appropriate constraints "
                                          "for accuracy, latency, reliability, and specific expertise required."},
            {"role": "user", "content": user_question}
        ]

        while True:
            response = chat_with_gpt(messages, functions=functions)

            if response.function_call:
                function_name = response.function_call.name
                function_args = json.loads(response.function_call.arguments)
                
                if function_name == "hire_specialist":
                    specialist_response = self.hire_specialist(**function_args)
                    messages.append({
                        "role": "function",
                        "name": function_name,
                        "content": specialist_response
                    })
                else:
                    raise ValueError(f"Unknown function: {function_name}")
            else:
                return response.content

# Usage
engine = ReasoningEngine(budget=1.0)  # Set a budget of $1.0

user_question = "What are the legal implications of using AI-generated content in a commercial product?"
final_response = engine.process_query(user_question)

print(f"User: {user_question}")
print(f"Assistant: {final_response}")
print(f"Total cost: ${engine.cost_manager.total_cost:.4f}")