from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.security.api_key import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Optional
from pydantic import BaseModel
import uvicorn
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI(title="Enterprise AI Support System")
api_key_header = APIKeyHeader(name="X-API-Key")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update with your allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API key verification
async def verify_api_key(api_key: str = Security(api_key_header)):
    if api_key != os.getenv("API_KEY"):
        raise HTTPException(
            status_code=403,
            detail="Invalid API key"
        )
    return api_key

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
        response = response_generator.generate_response(
            request.message,
            request.context
        )
        return {
            "status": "success",
            "response": response["response"],
            "confidence": response["confidence"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True) 