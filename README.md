```markdown
# Society of Mind

## Project Description

**Society of Mind** is a project designed to facilitate interactions with Language Models (LLMs). The underlying concept draws from the idea of a "society" of minds, where each LLM acts as an individual entity contributing to a larger conversation. This application allows users to communicate with LLMs, manage conversations, and efficiently classify and respond to queries.

## Features

- User-friendly CLI for making queries to LLMs.
- Dynamic conversation ID management, allowing users to maintain context across queries.
- Built with Flask as the web framework to handle incoming requests.
- Redis integration for caching and managing conversation states.

## Installation Instructions

To set up the Society of Mind application, follow these steps:

1. **Python Installation**: Ensure you have Python 3.8 or higher installed on your system. You can verify this with the command:
   ```bash
   python --version
   ```

2. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/societyofmind.git
   cd societyofmind
   ```

3. **Install Poetry**: If you don't have Poetry installed, you can do so with:
   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   ```

4. **Install Dependencies**:
   ```bash
   poetry install
   ```

5. **Setting Up Environment Variables**: Create a `.env` file in the root directory following the example below:
   ```
   REDIS_HOST=localhost
   REDIS_PORT=6379
   REDIS_DB=0
   ```

6. **Run the Application**: Before running the application, ensure your Redis server is up and running. Then you can use Poetry to run the Flask service:
   ```bash
   poetry run flask run
   ```

## Usage Examples

To use the Society of Mind interface, you can run the command line interface as follows:

1. **Start the CLI**:
   Ensure your server is running, then start the CLI:
   ```bash
   python cli.py
   ```

2. **Interact with the LLM**:
   Follow the prompts in the CLI:
   - **Enter the Port Number**: (default is 5001 unless specified).
   - **Conversation ID**: Press Enter for a new conversation or enter an existing one.
   - **Type your Query**: Enter queries until you type `quit` to exit the CLI.

### Example Interaction

```bash
Welcome to the LLM Router Test Interface!
Enter the port number (default is 5001): 
Enter a conversation ID (or press Enter for a new conversation): 
Enter your query (or 'quit' to exit): What is the capital of France?
Sending request to: http://localhost:5001/query
Request data: {
  "query": "What is the capital of France?",
  "conversation_id": null
}
Status Code: 200
Headers: {...}
Content: {...}

Response: The capital of France is Paris.
Classified as: Geography
```

## Development

For development, you can use the provided tools in the `dev-dependencies`. Commonly used commands include:

- **Run Tests**: 
   ```bash
   poetry run pytest
   ```

- **Code Formatting**:
   Use [Black](https://pypi.org/project/black/) to format your code:
   ```bash
   poetry run black .
   ```

- **Linting**:
   Use [Flake8](https://pypi.org/project/flake8/) for linting:
   ```bash
   poetry run flake8
   ```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Author

Adel Zaalouk

---

For further information or to report issues, please open an issue in this repository.
```
