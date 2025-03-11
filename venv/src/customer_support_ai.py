from src.faq_data import faq_data  # Correct import path to faq_data

class CustomerSupportAI:
    def __init__(self):
        # Use the imported faq_data dictionary
        self.faq_data = faq_data

    def get_response(self, query: str) -> str:
        """Retrieve the response for a given query."""
        query_lower = query.lower()

        # Go through each question and check if the query matches the question
        for question, answer in self.faq_data.items():
            if query_lower in question.lower():  # Case-insensitive match
                return answer
        
        return "Sorry, I couldn't find an answer to that. Please contact customer support directly for further assistance."

    def start_conversation(self):
        """Start the conversation with the user."""
        print("Welcome to the Advanced Customer Support AI System!")
        print("Type 'exit' to end the conversation or 'reset' to start over.")
        
        while True:
            user_input = input("You: ").strip()
            
            if user_input.lower() == "exit":
                print("Goodbye! Have a great day.")
                break
            elif user_input.lower() == "reset":
                print("Starting over...")
            else:
                response = self.get_response(user_input)
                print(f"AI: {response}")

# Example usage
if __name__ == "__main__":
    ai = CustomerSupportAI()
    ai.start_conversation()
