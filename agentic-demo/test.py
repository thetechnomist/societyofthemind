import os
from typing import List, Dict, Any
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re
from dotenv import load_dotenv
from openai import OpenAI
import logging
import math

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Download necessary NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Load environment variables from .env file
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Check if the API key is set
if not client.api_key:
    raise ValueError("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")

class AdvancedConversationMemory:
    def __init__(self, max_history: int = 1000):
        self.history: List[Dict[str, Any]] = []
        self.max_history = max_history
        self.embeddings = []

    def add(self, entry: Dict[str, Any]):
        logger.info(f"Adding entry to memory: {entry['role']}: {entry['content']}")
        self.history.append(entry)
        if len(self.history) > self.max_history:
            logger.info("Memory limit reached. Removing oldest entry.")
            self.history.pop(0)
            self.embeddings.pop(0)
        
        embedding = self.get_embedding(entry['content'])
        self.embeddings.append(embedding)

    def get_embedding(self, text: str) -> List[float]:
        logger.info(f"Getting embedding for text: {text}")
        response = client.embeddings.create(
            model="text-embedding-ada-002",
            input=[text]
        )
        logger.info("Embedding retrieved successfully.")
        return response.data[0].embedding

    def get_relevant_context(self, query: str, k: int = 5) -> str:
        logger.info(f"Getting relevant context for query: {query}")
        query_embedding = self.get_embedding(query)
        
        similarities = cosine_similarity([query_embedding], self.embeddings)[0]
        top_k_indices = np.argsort(similarities)[-k:][::-1]
        
        relevant_entries = [self.history[i] for i in top_k_indices]
        context = "\n".join([f"{entry['role']}: {entry['content']}" for entry in relevant_entries])
        
        logger.info(f"Retrieved {k} relevant context entries.")
        return context

class Tool:
    def __init__(self, name: str, func: callable):
        self.name = name
        self.func = func

    def execute(self, *args, **kwargs):
        logger.info(f"Executing tool: {self.name} with args: {args}, kwargs: {kwargs}")
        result = self.func(*args, **kwargs)
        logger.info(f"Tool execution result: {result}")
        return result

def llm_call(prompt: str) -> str:
    logger.info(f"Making LLM call with prompt: {prompt}")
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=150
        )
        result = response.choices[0].message.content.strip()
        logger.info(f"LLM call successful. Response: {result}")
        return result
    except Exception as e:
        logger.error(f"An error occurred while calling the OpenAI API: {e}")
        return "I apologize, but I'm having trouble accessing my language model at the moment. Could you please try again later?"

class AdvancedLLMAssistant:
    def __init__(self):
        logger.info("Initializing AdvancedLLMAssistant")
        self.memory = AdvancedConversationMemory()
        self.tools = self.load_tools()

    def load_tools(self) -> Dict[str, Tool]:
        logger.info("Loading tools")
        return {
            "calculator": Tool("calculator", self.safe_eval),
            "web_search": Tool("web_search", lambda x: f"Simulated web search for: {x}")
        }

    def safe_eval(self, expression: str) -> float:
        logger.info(f"Safely evaluating expression: {expression}")
        try:
            # Replace '^' with '**' for exponentiation
            expression = expression.replace('^', '**')
            # Only allow safe mathematical operations
            allowed_names = {
                'abs': abs, 'round': round, 'min': min, 'max': max,
                'sum': sum, 'pow': pow, 'sqrt': math.sqrt,
                'sin': math.sin, 'cos': math.cos, 'tan': math.tan,
                'pi': math.pi, 'e': math.e
            }
            return float(eval(expression, {"__builtins__": None}, allowed_names))
        except Exception as e:
            logger.error(f"Error in safe_eval: {e}")
            return float('nan')

    def process_query(self, query: str) -> str:
        logger.info(f"Processing query: {query}")
        self.memory.add({
            "role": "user", 
            "content": query,
            "reasoning": "Capturing the user's input to maintain conversation context and inform the response generation process."
        })

        context = self.memory.get_relevant_context(query)
        logger.info(f"Retrieved relevant context: {context}")
        
        plan = self.plan_query(query, context)
        logger.info(f"Generated plan: {plan}")
        
        result = self.execute_plan(plan)
        logger.info(f"Executed plan. Final result: {result}")

        self.memory.add({
            "role": "assistant", 
            "content": result,
            "reasoning": "Storing the final response to maintain conversation flow and provide context for future interactions."
        })

        return result

    def plan_query(self, query: str, context: str) -> List[Dict[str, Any]]:
        logger.info("Planning query execution")
        prompt = f"""
        Given the following query and context, create a plan to answer the query.
        Each step should be either a 'react' step for reasoning, or a 'tool' step for using a tool.
        Available tools: calculator, web_search

        Context: {context}
        Query: {query}

        Plan:
        """
        plan_str = llm_call(prompt)
        logger.info(f"Generated plan string: {plan_str}")
        # In a real implementation, you'd parse the plan string into a structured format
        # For simplicity, we'll use a dummy plan here
        return [
            {"type": "react", "query": query},
            {"type": "tool", "tool": "calculator", "parameters": "10 + 6"},
            {"type": "tool", "tool": "calculator", "parameters": "sqrt(16)"},
            {"type": "react", "query": query}
        ]

    def execute_plan(self, plan: List[Dict[str, Any]]) -> str:
        logger.info("Executing plan")
        results = []
        for step in plan:
            logger.info(f"Executing step: {step}")
            if step["type"] == "react":
                step_result = self.react_loop(step["query"])
            elif step["type"] == "tool":
                step_result = self.tools[step["tool"]].execute(step["parameters"])
            
            self.memory.add({
                "role": "system", 
                "content": f"Intermediate result: {step_result}",
                "reasoning": "Capturing intermediate results to track the problem-solving process and provide context for multi-step reasoning."
            })
            
            results.append(step_result)
        
        final_result = self.synthesize_results(results)
        return self.verify_result(final_result, plan[0]["query"])

    def react_loop(self, query: str, max_steps: int = 3) -> str:
        logger.info(f"Starting react loop for query: {query}")
        thought = ""
        for step in range(max_steps):
            logger.info(f"React loop step {step + 1}")
            thought_prompt = f"""
            Query: {query}
            Current thought: {thought}
            Next thought: Focus on directly answering the query. If you have a numerical answer, state it clearly.
            If you're confident in your answer, state that the task is complete.
            """
            thought = llm_call(thought_prompt)
            
            self.memory.add({
                "role": "system", 
                "content": f"Thought: {thought}",
                "reasoning": "Storing the reasoning process to explain decision-making and improve transparency of the system."
            })
            
            if "task complete" in thought.lower():
                logger.info("Task complete. Exiting react loop.")
                return self.extract_answer(thought)
            
            action_prompt = f"""
            Query: {query}
            Thought: {thought}
            Decide on an action: Choose between providing a final answer or using a tool (calculator or web_search).
            If you have a final answer, state it clearly and mark the task as complete.
            """
            action = llm_call(action_prompt)
            
            if "task complete" in action.lower():
                logger.info("Task complete. Exiting react loop.")
                return self.extract_answer(action)
            
            result = self.execute_action(action)
            
            self.memory.add({
                "role": "system", 
                "content": f"Action: {action}\nResult: {result}",
                "reasoning": "Recording the chosen action and its result to track the problem-solving strategy and enable analysis of decision-making patterns."
            })
            
            reflection_prompt = f"""
            Action: {action}
            Result: {result}
            Reflection: Evaluate if the result answers the query. If it does, state the answer clearly and mark the task as complete.
            If not, explain what's missing and what the next step should be.
            """
            reflection = llm_call(reflection_prompt)
            
            self.memory.add({
                "role": "system", 
                "content": f"Reflection: {reflection}",
                "reasoning": "Capturing self-evaluation to enable learning from experience and improve future performance."
            })
            
            if "task complete" in reflection.lower():
                logger.info("Task complete. Exiting react loop.")
                return self.extract_answer(reflection)
            
            thought += f"\nReflection: {reflection}"
        
        logger.info("Max steps reached in react loop without resolution.")
        return "Max steps reached without resolution"

    def execute_action(self, action: str) -> str:
        logger.info(f"Executing action: {action}")
        if "calculator" in action.lower():
            match = re.search(r'calculate\s+([\d\+\-\*/\^\(\)\s]+)', action, re.IGNORECASE)
            if match:
                expression = match.group(1)
                return str(self.safe_eval(expression))
        elif "web_search" in action.lower():
            match = re.search(r'search for\s+(.+)', action, re.IGNORECASE)
            if match:
                query = match.group(1)
                return self.tools["web_search"].execute(query)
        return f"Executed: {action}"

    def synthesize_results(self, results: List[str]) -> str:
        logger.info("Synthesizing results")
        prompt = f"""
        Synthesize the following results into a coherent response:
        {results}
        Provide a clear and concise answer to the original query. If a numerical answer is appropriate, state it clearly.
        """
        return llm_call(prompt)

    def verify_result(self, result: str, query: str) -> str:
        logger.info("Verifying result")
        verification_prompt = f"""
        Original query: {query}
        Generated answer: {result}
        
        Verify if the generated answer correctly and completely addresses the original query.
        If it does, return the answer as is.
        If it doesn't, provide a corrected answer that directly addresses the query.
        """
        verified_result = llm_call(verification_prompt)
        return verified_result

    def extract_answer(self, text: str) -> str:
        logger.info("Extracting answer from text")
        lines = text.split('\n')
        for line in lines:
            if "answer" in line.lower() or "result" in line.lower():
                return line.split(":", 1)[1].strip() if ":" in line else line.strip()
        return text  # Return the full text if no clear answer is found

# Usage example
if __name__ == "__main__":
    try:
        logger.info("Starting AdvancedLLMAssistant")
        assistant = AdvancedLLMAssistant()
        query = "What's the square root of the sum of 10 and 6?"
        logger.info(f"Processing query: {query}")
        result = assistant.process_query(query)
        logger.info(f"Final result: {result}")
        print(result)
    except ValueError as e:
        logger.error(f"Error: {e}")
        print(f"Error: {e}")
        print("Please make sure you have set the OPENAI_API_KEY environment variable.")
    except Exception as e:
        logger.exception("An unexpected error occurred")
        print(f"An unexpected error occurred: {e}")