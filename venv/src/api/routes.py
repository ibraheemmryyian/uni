from fastapi import FastAPI, HTTPException, Depends
from typing import Dict, Optional
from pydantic import BaseModel

app = FastAPI(title="Enterprise AI Support System")

class Query(BaseModel):
    text: str
    context: Optional[Dict] = None
    company_id: str

@app.post("/api/v1/chat")
async def chat_endpoint(query: Query):
    try:
        response = response_generator.generate_response(
            query.text,
            query.context
        )
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 