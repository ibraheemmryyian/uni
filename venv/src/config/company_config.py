from dataclasses import dataclass
from typing import Dict, List, Optional

@dataclass
class CompanyConfig:
    """Configuration for customizing the AI for different companies"""
    company_name: str
    support_email: str = ""
    phone_number: str = ""
    business_hours: Dict[str, str] = None
    language: str = "en"
    
    # Model settings
    model_name: str = "HuggingFaceH4/zephyr-7b-beta"
    temperature: float = 0.7
    max_tokens: int = 512
    
    # Support settings
    escalation_threshold: float = 0.7
    max_conversation_turns: int = 10
    sensitive_topics: List[str] = None
    
    # Custom prompts
    greeting_template: str = "Hello! How can I assist you today?"
    farewell_template: str = "Thank you for contacting us. Is there anything else I can help you with?" 