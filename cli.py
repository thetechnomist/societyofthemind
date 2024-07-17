import requests
import json
import sys

def main():
    print("Welcome to the LLM Router Test Interface!")
    port = input("Enter the port number (default is 5001): ") or "5001"
    conversation_id = input("Enter a conversation ID (or press Enter for a new conversation): ").strip() or None
    
    while True:
        query = input("\nEnter your query (or 'quit' to exit): ")
        if query.lower() == 'quit':
            break
        
        data = {
            'query': query,
            'conversation_id': conversation_id
        }
        
        try:
            url = f'http://localhost:{port}/query'
            print(f"Sending request to: {url}")
            print(f"Request data: {json.dumps(data, indent=2)}")
            response = requests.post(url, json=data, timeout=10)
            print(f"Status Code: {response.status_code}")
            print(f"Headers: {response.headers}")
            print(f"Content: {response.text}")
            
            response.raise_for_status()
            result = response.json()
            
            print(f"\nResponse: {result.get('response', 'No response')}")
            print(f"Classified as: {result.get('classified_as', 'Not classified')}")
            
            conversation_id = result.get('conversation_id', conversation_id)
        except requests.RequestException as e:
            print(f"An error occurred: {type(e).__name__}")
            print(f"Error details: {str(e)}")
            if hasattr(e, 'response'):
                if e.response is not None:
                    print(f"Status code: {e.response.status_code}")
                    print(f"Response content: {e.response.text}")
                else:
                    print("No response received from server")
            else:
                print("No response object available")
        except Exception as e:
            print(f"An unexpected error occurred: {type(e).__name__}")
            print(f"Error details: {str(e)}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nExiting the program.")
        sys.exit(0)
    except Exception as e:
        print(f"An unexpected error occurred: {type(e).__name__}")
        print(f"Error details: {str(e)}")
        sys.exit(1)