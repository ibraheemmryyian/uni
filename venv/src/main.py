import asyncio
import logging
import os
from src.models.faq_database import FAQDatabase
from src.models.response_generator import ResponseGenerator
from src.faq_data import faq_data
from src.logger import setup_logger

async def chatbot_main():
    # Create necessary directories
    os.makedirs("logs", exist_ok=True)
    
    logger = setup_logger(__name__, "logs/app.log", level=logging.ERROR)
    
    try:
        # Initialize FAQ database
        print("🔄 Initializing FAQ database...")
        faq_db = FAQDatabase()
        faq_db.setup(faq_data)
        
        # Initialize Response Generator
        print("🚀 Initializing ResponseGenerator...")
        response_gen = ResponseGenerator(faq_database=faq_db)
        
        print("✅ System initialized successfully!\n")
        print("Type 'exit' to end the conversation\n" + "-"*50)

        while True:
            try:
                user_input = input("\nCustomer: ").strip()
                if user_input.lower() in ["exit", "quit"]:
                    break
                
                response = await response_gen.generate_response(user_input)
                print(f"\nSupport: {response}")
                
            except KeyboardInterrupt:
                print("\n🚫 Conversation ended by user")
                break

    except Exception as e:
        logger.error(f"Critical failure: {str(e)}")
        print("\n❌ System unavailable. Please try again later.")

def main():
    asyncio.run(chatbot_main())

if __name__ == "__main__":
    main()
