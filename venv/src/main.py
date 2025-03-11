import asyncio
from src.models.faq_database import FAQDatabase
from src.models.response_generator import ResponseGenerator
from src.faq_data import faq_data
from src.logger import setup_logger

async def chatbot_main():
    logger = setup_logger(__name__, "logs/app.log")
    
    try:
        # Initialize components
        faq_db = FAQDatabase()
        faq_db.setup(faq_data)
        
        response_gen = ResponseGenerator(faq_database=faq_db)

        # Interactive loop
        while True:
            try:
                user_input = input("\nCustomer: ").strip()
                if user_input.lower() in ["exit", "quit"]:
                    break
                
                response = await response_gen.generate_response(user_input)
                print(f"\nSupport: {response}")
                
            except KeyboardInterrupt:
                print("\nConversation ended.")
                break

    except Exception as e:
        logger.error(f"Critical failure: {str(e)}")
        print("System unavailable. Please try again later.")

def main():
    asyncio.run(chatbot_main())

if __name__ == "__main__":
    main()