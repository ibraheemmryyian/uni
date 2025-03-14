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
    os.makedirs("models", exist_ok=True)
    
    logger = setup_logger(__name__, "logs/app.log", level=logging.ERROR)
    
    try:
        # Initialize FAQ database
        print("Initializing FAQ database...")
        faq_db = FAQDatabase()
        faq_db.setup(faq_data)
        
        # Initialize Response Generator with proper error handling
        print("Initializing Response Generator...")
        try:
            # Set environment variable for model path - using the correct path
            os.environ['FINE_TUNED_MODEL_PATH'] = 'venv/fine_tuned_phi2_model'
            
            response_gen = ResponseGenerator(faq_database=faq_db)
        except Exception as e:
            logger.error(f"Failed to initialize Response Generator: {str(e)}")
            raise
        
        print("System initialized successfully!")
        
        while True:
            user_input = input("\nCustomer: ").strip()
            if user_input.lower() in ["exit", "quit"]:
                break
                
            try:
                response = await response_gen.generate_response(user_input)
                print(f"\nSupport: {response}")
            except Exception as e:
                logger.error(f"Response generation error: {str(e)}")
                print("\nSupport: I apologize, but I'm having trouble processing your request. Please try again.")

    except Exception as e:
        logger.error(f"Critical failure: {str(e)}")
        print("System unavailable. Please try again later.")
        return

def main():
    asyncio.run(chatbot_main())

if __name__ == "__main__":
    main()
