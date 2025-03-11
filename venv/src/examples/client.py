import requests
import json

class AISupportClient:
    def __init__(self, api_key: str, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.headers = {
            "X-API-Key": api_key,
            "Content-Type": "application/json"
        }

    def chat(self, message: str, context: dict = None, company_id: str = "default"):
        """Send a chat message to the AI"""
        endpoint = f"{self.base_url}/api/v1/chat"
        
        payload = {
            "message": message,
            "context": context or {},
            "company_id": company_id
        }

        response = requests.post(
            endpoint,
            headers=self.headers,
            json=payload
        )

        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Error: {response.status_code} - {response.text}")

# Example usage
if __name__ == "__main__":
    client = AISupportClient(api_key="your_api_key_here")
    
    response = client.chat(
        message="How do I reset my password?",
        context={"user_id": "12345"},
        company_id="company_1"
    )
    
    print(json.dumps(response, indent=2)) 