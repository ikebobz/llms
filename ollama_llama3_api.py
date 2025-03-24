import requests
import json

def get_llama3_completion(prompt, model="llama3"):
    """Gets completion from Ollama's Llama3."""
    url = "http://localhost:11434/api/generate"  # Ollama's default API endpoint
    data = {
        "model": model,
        "prompt": prompt,
        "stream": False,  # Set to True for streaming responses
    }
    try:
        response = requests.post(url, json=data)
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
        return response.json()["response"]
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
        return None
    except KeyError:
        print("Error: Could not extract response from Ollama's reply")
        return None

if __name__ == "__main__":
    # Test the function
    prompt = "Can you summarize with simple terms how an airplane generates lift."
    completion = get_llama3_completion(prompt)
    if completion:
        print(completion)