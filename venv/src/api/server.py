from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.security.api_key import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Optional
from pydantic import BaseModel
import uvicorn
import os
import torch
from dotenv import load_dotenv
from src.models.faq_database import FAQDatabase
from src.models.response_generator import ResponseGenerator
import gc
from transformers import AutoConfig

# Load environment variables
load_dotenv()

# Force garbage collection
gc.collect()

# Initialize components with CPU-only configuration
print("Initializing components...")
try:
    print("Loading model configuration...")
    model_path = os.getenv('FINE_TUNED_MODEL_PATH', './models/fine_tuned_phi2_model')
    
    if os.path.exists(model_path):
        print(f"Using fine-tuned model from {model_path}")
        config = AutoConfig.from_pretrained(
            model_path,
            trust_remote_code=True
        )
    else:
        print("Fine-tuned model not found, using base model temporarily")
        config = AutoConfig.from_pretrained(
            "microsoft/phi-2",
            trust_remote_code=True
        )

    print("Initializing components for CPU...")
    faq_db = FAQDatabase()
    response_generator = ResponseGenerator(faq_database=faq_db)
    
    print("Components initialized successfully!")
except Exception as e:
    print(f"Initialization error: {str(e)}")
    raise

# FastAPI initialization
app = FastAPI(title="Enterprise AI Support System")

# API Key header verification
api_key_header = APIKeyHeader(name="X-API-Key")

# CORS configuration to allow requests from your website
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://website-rouge-one-41.vercel.app"],  # Link to your website
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Key verification function
async def verify_api_key(api_key: str = Security(api_key_header)):
    if api_key != os.getenv("API_KEY"):
        raise HTTPException(
            status_code=403,
            detail="Invalid API key"
        )
    return api_key

# Define the chat request model
class ChatRequest(BaseModel):
    message: str
    context: Optional[Dict] = None
    company_id: str

@app.post("/api/v1/chat")
async def chat_endpoint(
    request: ChatRequest,
    api_key: str = Depends(verify_api_key)
):
    try:
        # Generate response using your ResponseGenerator
        response = await response_generator.generate_response(request.message)
        
        return {
            "status": "success",
            "response": response,
            "confidence": 0.95  # You can implement actual confidence scoring if needed
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
