import torch
import logging
import asyncio
import re
import numpy as np
from typing import Dict, List, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM, T5ForConditionalGeneration, T5Tokenizer
from src.logger import setup_logger
from src.utils.sentiment_analyzer import SentimentAnalyzer
from src.utils.text_processing import normalize_query, is_greeting
from .faq_database import FAQDatabase
from src.models.prompt.prompt_builder import PromptBuilder
import os

class TokenLengthError(Exception):
    """Custom exception for token length issues"""
    pass

class RetryError(Exception):
    """Custom exception for retry attempts"""
    pass

class ConversationState:
    def __init__(self):
        self.current_topic = None
        self.payment_mentioned = False
        self.dispute_mentioned = False
        self.aggression_level = 0
        self.last_sentiment = 0

class ResponseGenerator:
    def __init__(self, faq_database: FAQDatabase):
        """Initialize the ResponseGenerator with CPU configuration"""
        self.faq_db = faq_database
        self.sentiment_analyzer = SentimentAnalyzer()
        
        # Create necessary directories
        os.makedirs("logs", exist_ok=True)
        os.makedirs("models", exist_ok=True)
        
        self.logger = setup_logger(__name__, "logs/response_generator.log", level=logging.INFO)
        
        # Initialize conversation tracking before model loading
        self.conversation_history = []
        self.max_history_length = 5
        self.context_window = 1000
        self.conversation_state = ConversationState()
        self.device = torch.device("cpu")
        
        try:
            # Define model path - use environment variable or fallback to a default path
            model_path = os.getenv('FINE_TUNED_MODEL_PATH', '/content/uni/venv/fine_tuned_phi2_model/')
            print(f"Model path from env: {model_path}")  # Debugging line
            
            if not os.path.exists(model_path):
                print(f"Model not found at {model_path}")  # Debugging line
                model_path = "microsoft/phi-2"  # Fallback to base model
            else:
                print(f"Model found at {model_path}")  # Debugging line
            
            print(f"Loading tokenizer from {model_path}...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                use_fast=True,
                model_max_length=512,
                trust_remote_code=True
            )
            
            print(f"Loading main model from {model_path}...")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                low_cpu_mem_usage=True,
                torch_dtype=torch.float32,
                trust_remote_code=True
            )
            
            # Ensure pad token is set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            print("Loading paraphrase model...")
            # Load smaller T5 model for paraphrasing
            self.paraphrase_tokenizer = T5Tokenizer.from_pretrained(
                't5-small',
                legacy=False,
                model_max_length=512
            )
            self.paraphrase_model = T5ForConditionalGeneration.from_pretrained(
                't5-small',
                low_cpu_mem_usage=True,
                torch_dtype=torch.float32
            )
            
            # Initialize other configurations...
            self._initialize_configurations()
            
            print("Models loaded successfully!")
            
        except Exception as e:
            print(f"Model initialization failed: {str(e)}")
            raise

    def _initialize_configurations(self):
        """Initialize all configurations"""
        self.support_resources = {
            "financial": [
            "National Financial Counseling: 1-800-555-HELP",
            "Consumer Credit Assistance: www.credithelp.org"
            ],
            "emotional": [
                "Mental Health Support: 1-800-273-8255",
                "Stress Management Hotline: 1-800-784-2433"
            ]
        }

        self.intent_keywords = {
            "payment_plan": ["payment plan", "installment", "pay later", "extend payment"],
            "help": ["help", "support", "assistance", "guidance"],
            "complaint": ["problem", "issue", "can't pay", "struggling"],
            "information": ["tell me", "explain", "what is", "how do"]
        }

        # Initialize PromptBuilder
        self.prompt_builder = PromptBuilder()
        
        # Optimize for CPU
        self.model.eval()
        torch.set_num_threads(4)
        
        # Configuration settings
        self.batch_size = 1
        self.max_length = 1024
        self.memory_config = {
            'max_tokens_per_response': 1000,
            'history_token_limit': 2048,
            'context_window_size': 4096
        }
        
        # Response caching
        self.response_cache = {}
        self.cache_ttl = 3600  # 1 hour cache lifetime

    async def generate_response(self, user_input: str) -> str:
        """Advanced response generation pipeline"""
        try:
            # Clean input
            clean_input = self._sanitize_input(user_input)
            
            # Add user input to history
            self.conversation_history.append({"role": "user", "content": clean_input})
            
            # Trim history if too long
            if len(self.conversation_history) > self.max_history_length * 2:
                self.conversation_history = self.conversation_history[-self.max_history_length * 2:]

            # Generate contextual response
            response = await self._generate_contextual_response(clean_input)
            
            # Add response to history if valid
            if response:
                self.conversation_history.append({"role": "assistant", "content": response})
            
            return response

        except Exception as e:
            self.logger.error(f"Critical generation error: {str(e)}")
            return "I apologize for the inconvenience. Our system is experiencing difficulties."

    def _detect_intent(self, query: str) -> str:
        """Advanced intent detection"""
        query = query.lower()
        for intent, keywords in self.intent_keywords.items():
            if any(keyword in query for keyword in keywords):
                return intent
        return "general"

    def _handle_payment_plan_request(self, query: str) -> str:
        """Specialized handling for payment plan requests"""
        return (
            "I understand you're looking for a payment plan. "
            "Could you provide more details about your current financial situation? "
            "This will help me recommend the most suitable options."
        )

    def _provide_targeted_help(self, query: str) -> str:
        """Contextual help based on query"""
        return (
            "I'm here to help! What specific aspect of your financial situation "
            "would you like assistance with? Payment plans, financial counseling, "
            "or understanding your options?"
        )

    def _handle_complaint(self, query: str, sentiment: Dict) -> str:
        """Empathetic response to complaints"""
        resources = "\n".join(self.support_resources["financial"] + self.support_resources["emotional"])
        
        if sentiment['compound'] < -0.3:
            return (
                f"I hear that you're going through a difficult time. "
                f"Here are some resources that might provide additional support:\n{resources}"
            )
        
        return (
            "I understand you're experiencing challenges. "
            "Let's work together to find a solution. Can you tell me more about your specific situation?"
        )

    def _generate_greeting(self) -> str:
        """Enhanced greeting with context"""
        greetings = [
            "Hello! I'm your financial support assistant. How can I help you today?",
            "Hi there! I'm here to assist you with payment plans and financial guidance.",
            "Welcome! What financial questions or concerns can I help you with?"
        ]
        return np.random.choice(greetings)

    def _generate_fallback_response(self, query: str) -> str:
        """Intelligent fallback response with context"""
        fallback_responses = [
            "I'm not quite sure I understand. Could you rephrase that?",
            "Could you provide more context about what you're looking for?",
            "I want to help you effectively. Can you clarify your request?",
            "I'm here to assist you. Could you tell me more about your specific situation?"
        ]
        
        # Context-specific fallbacks
        context_fallbacks = {
            "help": "I'm here to help. What specific assistance do you need?",
            "payment": "Regarding payment, could you provide more details about your situation?",
            "options": "I can help you understand your options. What specific area are you inquiring about?"
        }
        
        # Check for context-specific keywords
        for keyword, response in context_fallbacks.items():
            if keyword in query.lower():
                return response
        
        return np.random.choice(fallback_responses)

    async def _get_faq_response(self, query: str) -> Optional[str]:
        """Fetch response from FAQ database"""
        try:
            if exact := self.faq_db.get_exact_match(query):
                return exact

            results = self.faq_db.query_similar(query, threshold=0.65)
            if results:
                return max(results, key=lambda x: x['similarity'])['response']

            if keyword := self.faq_db.keyword_search(query):
                return keyword

            return None
        except Exception as e:
            self.logger.error(f"FAQ lookup error: {str(e)}")
            return None

    def _process_response(self, response: str, sentiment: Dict) -> str:
        """Adjust FAQ response based on sentiment analysis"""
        # Skip paraphrasing for identity and greeting responses
        if any(keyword in response.lower() for keyword in ['who made you', 'who created you', 'hello', 'hi there']):
            return self._apply_sentiment_tuning(response, sentiment)
        
        paraphrased = self._paraphrase_response(response)
        if not paraphrased or paraphrased.lower() in ['false', 'paraphrase']:
            return self._sanitize_output(response)  # Return original if paraphrase fails
        
        return self._sanitize_output(paraphrased)

    def _paraphrase_response(self, text: str) -> str:
        """Paraphrase the response to sound more natural"""
        try:
            # Don't paraphrase if text is too short
            if len(text.split()) < 5:
                return text

            inputs = self.paraphrase_tokenizer(
                text,  # Remove "paraphrase:" prefix
                return_tensors="pt",
                max_length=512,
                truncation=True
            )
            outputs = self.paraphrase_model.generate(
                inputs.input_ids,
                max_length=1024,
                num_beams=5,
                early_stopping=True,
                no_repeat_ngram_size=2,
                repetition_penalty=1.2
            )
            result = self.paraphrase_tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Return original text if result is too short or contains unwanted words
            if len(result.split()) < 3 or any(word in result.lower() for word in ['false', 'paraphrase']):
                return text
            
            return result
        except Exception as e:
            self.logger.error(f"Paraphrasing failed: {str(e)}")
            return text

    def _apply_sentiment_tuning(self, response: str, sentiment: Dict) -> str:
        """Adjust the response tone based on sentiment"""
        # Skip sentiment prefix for certain types of responses
        if any(keyword in response.lower() for keyword in [
            'who made you', 'who created you', 'hello', 'hi there',
            'i was developed by', 'i was created by'
        ]):
            return response
        
        # Only add sentiment prefix for strong sentiment scores
        if sentiment['compound'] < -0.5:  # High negative sentiment
            return (f"I understand this is a very difficult situation, especially "
                    f"given your circumstances. {response}")
        elif sentiment['compound'] > 0.5:
            return response  # Skip positive sentiment prefix
        return response  # Return without prefix for neutral sentiment

    def _sanitize_input(self, text: str) -> str:
        """Sanitize user input"""
        # Remove special characters, keep alphanumeric and basic punctuation
        return re.sub(r'[^\w\s,.?!-]', '', text).strip().lower()

    def _sanitize_output(self, text: str) -> str:
        """Enhanced output sanitization"""
        # Remove unwanted patterns and words
        unwanted_patterns = [
            r'```.*?```',
            r'<?\w+\s*=\s*.*?>',
            r'OUTPUT:',
            r'\b(?:false|paraphrase)\b',
            r'(?:^|\s)(?:Bitte|Pour)\b',  # Remove foreign language fragments
            r'A:',
            r'Q:',
            r'Response:'
        ]
        
        for pattern in unwanted_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        # Clean up punctuation and whitespace
        text = re.sub(r'([!?.]){2,}', r'\1', text)
        text = ' '.join(text.split())
        
        # Ensure complete sentences
        if text and not text[-1] in '.!?':
            text += '.'
        
        return text.strip()

    async def _generate_ai_response(self, query: str, sentiment: Dict) -> Optional[str]:
        """Generate response using fine-tuned Phi-2 model"""
        try:
            # Handle specific identity queries first
            if "who" in query.lower():
                return self._handle_identity_query(query)

            # Get FAQ response first for common scenarios
            faq_response = await self._get_faq_response(query)
            if faq_response:
                return self._process_response(faq_response, sentiment)

            # Build prompt using PromptBuilder
            prompt_builder = PromptBuilder()
            full_prompt = prompt_builder.build_prompt(
                query=query,
                faq_matches=[],  # Add FAQ matches if available
                conversation_history={"history": ""}  # Add conversation history if tracking
            )
            
            # Tokenize input
            inputs = self.tokenizer(
                full_prompt,
                return_tensors='pt',
                add_special_tokens=True,
                padding=True,
                truncation=True,
                max_length=512
            )

            # Generate response with stricter parameters
            outputs = self.model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_length=1000,
                min_length=20,
                do_sample=True,
                top_k=30,
                top_p=0.85,
                temperature=0.3,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.2
            )
            
            # Decode and clean response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Validate response
            if not response or len(response.split()) < 3:
                return self._get_default_response(query)
            
            # Clean up response
            response = self._clean_response(response, full_prompt)
            
            # Final validation
            if not self._is_valid_response(response, query, full_prompt):
                return self._get_default_response(query)
            
            return response

        except Exception as e:
            self.logger.error(f"LLM response generation failed: {str(e)}")
            return self._get_default_response(query)

    def _clean_response(self, response: str, prompt: str) -> str:
        """Clean up the generated response"""
        # Remove the prompt from the response
        response = response.replace(prompt, "").strip()
        
        # Remove any system instruction fragments
        response = re.sub(r"You are .*?assistant\.", "", response, flags=re.DOTALL)
        response = re.sub(r"Customer query:.*?\n", "", response, flags=re.DOTALL)
        
        # Clean up common artifacts
        response = self._sanitize_output(response)
        
        # Ensure response is complete
        response = self._ensure_complete_response(response)
        
        return response

    def _get_default_response(self, query: str) -> str:
        """Get a default response based on query type"""
        if any(word in query.lower() for word in ['time', 'extension', 'plan', 'later']):
            return ("We cannot offer payment extensions or plans. Full payment is required immediately "
                    "to avoid further consequences. Please make the payment as soon as possible.")
        
        if any(word in query.lower() for word in ["can't", 'cant', 'cannot', 'dont', "don't"]):
            return ("While I understand your situation, immediate payment is required. "
                    "Delays will only lead to additional complications. "
                    "Please arrange to make the payment as soon as possible.")
        
        return ("I understand your situation, however, full payment is required immediately. "
                "Please make the payment as soon as possible to avoid any further consequences.")

    def _is_finance_related(self, response: str) -> bool:
        """Enhanced check for finance-related content"""
        finance_keywords = {
            'payment', 'financial', 'money', 'debt', 'balance', 'pay', 
            'account', 'credit', 'debit', 'fund', 'fee', 'charge',
            'settlement', 'installment', 'plan', 'support', 'due',
            'bill', 'amount', 'loan', 'owe', 'payment', 'finance',
            'collection', 'outstanding', 'resolve', 'situation'
        }
        
        response_lower = response.lower()
        words = set(response_lower.split())
        
        # Check for keyword presence
        if not bool(words & finance_keywords):
            return False
        
        # Check for context relevance
        relevant_phrases = [
            'payment plan', 'financial situation', 'outstanding balance',
            'debt collection', 'payment options', 'financial support'
        ]
        
        return any(phrase in response_lower for phrase in relevant_phrases)

    def _remove_input_echo(self, query: str, response: str) -> str:
        """Remove direct echoes of the input from the response"""
        # Remove any instances of 'paraphrase:' from the response
        response = re.sub(r'paraphrase:\s*', '', response, flags=re.IGNORECASE)
        
        query_words = set(query.lower().split())
        response_words = response.split()
        
        # Remove initial words that match the query
        while response_words and response_words[0].lower() in query_words:
            response_words.pop(0)
        
        cleaned_response = ' '.join(response_words)
        return cleaned_response if cleaned_response else response

    def _handle_identity_query(self, query: str) -> str:
        """Enhanced identity query handling"""
        identity_queries = {
            "who are you": [
                "I'm an AI assistant designed to help you with financial support and guidance.",
                "I'm a digital assistant created to provide intelligent customer support.",
                "I'm an AI-powered support system focused on helping you with financial queries."
            ],
            "who made you": [
                "I was developed by Ibraheem Mryyian as an advanced AI-driven customer support assistant.",
                "I was created by a talented developer to provide intelligent and empathetic support.",
                "My creator is Ibraheem Mryyian, who designed me to enhance customer service experiences."
            ],
            "are you human": [
                "No, I'm an AI assistant. I'm here to help you with your financial queries.",
                "I'm an artificial intelligence, not a human. My goal is to provide you with the best possible support.",
                "I'm an AI - a digital assistant designed to assist you effectively."
            ]
        }
        
        query_lower = query.lower()
        for key, responses in identity_queries.items():
            if key in query_lower:
                return np.random.choice(responses)
        
        return "I'm an AI assistant created to provide support and information."

    def _is_valid_response(self, response: str, query: str, context: str) -> bool:
        """Enhanced response validation"""
        try:
            # Basic checks
            if not response or len(response.split()) < 3:
                return False
            
            # Check response coherence
            if not self._is_coherent_response(response):
                return False
            
            # Check topic consistency
            if not self._is_topic_consistent(response):
                return False
            
            # Check for circular references
            if self._has_circular_reference(response, self.conversation_history):
                return False
            
            # Check for appropriate tone based on state
            if not self._has_appropriate_tone(response):
                return False
            
            return True
        except Exception as e:
            self.logger.error(f"Response validation failed: {str(e)}")
            return False

    def _is_coherent_response(self, response: str) -> bool:
        """Check if response is logically coherent"""
        # Check for contradictions
        contradictions = [
            (r'can(?:not|\'t)\s+pay.*can\s+pay', 'payment contradiction'),
            (r'no\s+extension.*offer\s+extension', 'extension contradiction'),
            (r'must\s+pay.*don\'t\s+pay', 'payment instruction contradiction')
        ]
        
        for pattern, _ in contradictions:
            if re.search(pattern, response.lower()):
                return False
        
        return True

    def _is_topic_consistent(self, response: str) -> bool:
        """Check if response stays on topic"""
        if self.conversation_state.current_topic == 'payment':
            return any(word in response.lower() for word in ['pay', 'payment', 'money', 'balance'])
        elif self.conversation_state.current_topic == 'dispute':
            return any(word in response.lower() for word in ['dispute', 'claim', 'documentation'])
        return True

    def _has_appropriate_tone(self, response: str) -> bool:
        """Check if response has appropriate tone based on context"""
        if self.conversation_state.aggression_level > 0:
            return 'legal' in response.lower() or 'consequences' in response.lower()
        if self.conversation_state.last_sentiment < -0.5:
            return 'understand' in response.lower() or 'however' in response.lower()
        return True

    def _is_conversation_end(self, text: str) -> bool:
        """Enhanced conversation end detection with state reset"""
        end_phrases = {'bye', 'goodbye', 'thank you', 'thanks', 'end', 'quit', 'exit'}
        is_end = any(phrase in text.lower() for phrase in end_phrases)
        if is_end:
            self.reset_conversation()  # Reset state when conversation ends
        return is_end

    def _format_conversation_history(self) -> str:
        """Format conversation history for context"""
        formatted_history = []
        for msg in self.conversation_history[-self.max_history_length * 2:]:  # Get recent history
            role = "Customer" if msg["role"] == "user" else "Assistant"
            formatted_history.append(f"{role}: {msg['content']}")
        return "\n".join(formatted_history)

    async def _generate_contextual_response(self, query: str) -> str:
        try:
            # Check for greetings first
            if greeting_response := self._handle_greeting(query):
                return greeting_response
            
            # Check for identity queries BEFORE sentiment analysis
            if any(phrase in query.lower() for phrase in ['who are you', 'who made you', 'who coded you']):
                return self._handle_identity_query(query)
            
            # Get sentiment and update state
            sentiment = await asyncio.to_thread(
                self.sentiment_analyzer.analyze_sentiment,
                query
            )
            self._update_conversation_state(query, sentiment)
            
            # Check for hardship evidence
            has_evidence, hardship_type = self._check_for_hardship_evidence(query)
            if has_evidence:
                return self._handle_hardship_case(query, hardship_type)
            
            # Check for payment confirmation with proof
            if self._is_payment_confirmation(query):
                return ("Thank you for confirming your payment. Please allow 48 hours for it to reflect. "
                       "If you have any proof of payment, please share it with us.")
            
            # Check for conversation end
            if self._is_conversation_end(query):
                return "Thank you for contacting us. Remember that immediate payment is required to avoid further consequences. Have a good day."
            
            # Check for evasive responses
            if evasive_response := self._handle_evasive_response(query):
                return evasive_response
            
            # Generate base response with context
            response = await self._generate_base_response(query, sentiment)
            
            # Ensure response quality and consistency
            response = self._ensure_response_quality(response, query)
            response = self._enforce_response_consistency(response)
            response = self._personalize_response(response)
            
            return response
        except Exception as e:
            self.logger.error(f"Response generation failed: {str(e)}")
            return self._get_default_response(query)

    def reset_conversation(self):
        """Reset the conversation history"""
        self.conversation_history = []

    def _handle_dispute_claim(self, query: str) -> str:
        dispute_keywords = ['not me', 'not my debt', 'wrong person']
        if any(keyword in query.lower() for keyword in dispute_keywords):
            return ("If you believe this debt is not yours, please: \n"
                    "1. Contact {Lender} immediately\n"
                    "2. File a police report if needed\n"
                    "3. Gather documentation to support your claim")

    def _format_response(self, response: str) -> str:
        # Standardize formatting
        response = self._replace_placeholders(response)
        response = self._ensure_complete_sentences(response)
        response = self._standardize_contact_info(response)
        return response

    def _handle_aggressive_response(self, query: str, sentiment: Dict) -> str:
        if sentiment['compound'] < -0.7 or self._contains_threats(query):
            return ("I understand you're frustrated. However, I must inform you that "
                    "threats or abusive language will be documented and may have legal "
                    "consequences. Let's maintain professional communication.")

    def _is_contextually_relevant(self, response: str, context: str) -> bool:
        """Check if response is relevant to conversation context"""
        try:
            # Get key terms from context
            context_terms = set(context.lower().split())
            response_terms = set(response.lower().split())
            
            # Check term overlap
            overlap = context_terms & response_terms
            
            # Check for topic consistency
            if len(overlap) < 2:
                return False
                
            # Check for context continuity
            last_context = self.conversation_history[-1]["content"].lower() if self.conversation_history else ""
            if not any(term in response.lower() for term in last_context.split()):
                return False
                
            return True
        except Exception as e:
            self.logger.error(f"Context relevance check failed: {str(e)}")
            return True  # Default to true on error

    async def _generate_response_with_truncated_context(self, query: str) -> str:
        """Retry response generation with truncated context"""
        try:
            # Temporarily reduce history length
            original_length = self.max_history_length
            self.max_history_length = 2
            
            # Get truncated context
            conversation_context = self._format_conversation_history()
            
            # Build new prompt
            full_prompt = self.prompt_builder.build_prompt(
                query=query,
                faq_matches=[],
                conversation_history={"history": conversation_context}
            )
            
            # Generate response with truncated context
            response = await self._generate_ai_response(query, {})
            
            # Restore original history length
            self.max_history_length = original_length
            
            return response
        except Exception as e:
            self.logger.error(f"Truncated context generation failed: {str(e)}")
            return self._get_default_response(query)

    def _get_contextual_fallback_response(self, query: str) -> str:
        """Generate context-aware fallback response"""
        try:
            # Get recent context
            recent_context = self.conversation_history[-2:] if len(self.conversation_history) >= 2 else []
            
            # Check for specific scenarios
            if any("payment" in msg["content"].lower() for msg in recent_context):
                return ("Regarding your payment inquiry, please note that full payment is required. "
                       "Let me know if you need clarification about the payment process.")
            
            if any("dispute" in msg["content"].lower() for msg in recent_context):
                return ("About your dispute, please contact {Lender} directly and ensure "
                       "you have documentation to support your claim.")
            
            # Default contextual fallback
            return ("I want to ensure I address your concern properly. "
                   "Could you please rephrase your question about " +
                   (recent_context[-1]["content"] if recent_context else "your situation") + "?")
            
        except Exception as e:
            self.logger.error(f"Contextual fallback failed: {str(e)}")
            return self._get_default_response(query)

    def _contains_threats(self, text: str) -> bool:
        """Check for threatening or abusive language"""
        threat_patterns = [
            r'\b(?:kill|death|hurt|harm|destroy)\b',
            r'\b(?:fuck|shit|damn|bitch)\b',
            r'\b(?:lawsuit|sue|lawyer|legal)\b'
        ]
        
        return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in threat_patterns)

    def _replace_placeholders(self, text: str) -> str:
        """Replace placeholder values with appropriate content"""
        replacements = {
            "{Lender}": "the lender",
            "{reference ID}": "your account reference number",
            "{outstanding balance}": "your outstanding balance",
            "{Registered Number}": "your registered contact number"
        }
        
        for placeholder, replacement in replacements.items():
            text = text.replace(placeholder, replacement)
        
        return text

    def _ensure_complete_sentences(self, text: str) -> str:
        """Ensure text consists of complete, grammatical sentences"""
        # Split into sentences
        sentences = re.split(r'([.!?]+)', text)
        
        # Recombine ensuring proper punctuation
        formatted = []
        for i in range(0, len(sentences)-1, 2):
            sentence = sentences[i].strip()
            punctuation = sentences[i+1] if i+1 < len(sentences) else "."
            
            # Capitalize first letter
            if sentence:
                sentence = sentence[0].upper() + sentence[1:]
                formatted.append(sentence + punctuation)
        
        return " ".join(formatted)

    def _standardize_contact_info(self, text: str) -> str:
        """Standardize format of contact information"""
        # Standardize phone numbers
        text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', 
                     lambda m: m.group(0).replace('.', '-').replace(' ', '-'),
                     text)
        
        # Standardize email addresses
        text = re.sub(r'\b[\w\.-]+@[\w\.-]+\.\w+\b',
                     lambda m: m.group(0).lower(),
                     text)
        
        return text

    def _handle_evasive_response(self, query: str) -> Optional[str]:
        """Enhanced handling of evasive responses"""
        evasion_patterns = {
            r'\b(?:leaving|left)\s+(?:uae|country)\b': (
                "Please note that leaving the UAE does not eliminate your debt obligations. "
                "This could lead to legal complications and international enforcement. "
                "We strongly recommend settling your payment before departure to avoid these consequences."
            ),
            r'\b(?:never|won\'t|wont|refuse)\s+pay\b': (
                "I must inform you that refusing to pay will result in immediate legal action "
                "and damage to your credit rating. This can affect your future financial opportunities "
                "worldwide. Let's discuss how you can resolve this payment immediately."
            ),
            r'\b(?:not|dont|don\'t)\s+want\s+(?:to\s+)?pay\b': (
                "While I understand your reluctance, this debt is a legal obligation that requires "
                "immediate attention. Avoiding payment will only lead to more severe consequences. "
                "Let's focus on resolving this matter now."
            ),
            r'\b(?:why|what|how)\s+(?:should|must)\s+(?:i|we)\s+pay\b': (
                "This debt is a legal obligation registered under your name. Failure to pay can result "
                "in legal action, credit rating damage, and other enforcement measures. "
                "The best course of action is to settle the payment immediately."
            )
        }
        
        for pattern, response in evasion_patterns.items():
            if re.search(pattern, query.lower()):
                return response
        
        return None

    def _ensure_complete_response(self, response: str) -> str:
        """Ensure response is complete and meaningful"""
        # Check for incomplete sentences
        if response.endswith((',', '...')) or len(response.split()) < 5:
            return self._get_default_response("")
        
        # Check for truncated responses
        if any(response.lower().startswith(word) for word in ['and', 'but', 'or', 'so', 'because']):
            return self._get_default_response("")
        
        # Remove hanging commas and incomplete phrases
        response = re.sub(r',\s*$', '.', response)
        
        return response

    async def _retry_with_backoff(self, func, *args, max_retries=3, initial_delay=1):
        """Retry mechanism with exponential backoff"""
        delay = initial_delay
        last_exception = None
        
        for attempt in range(max_retries):
            try:
                # Handle both async and sync functions
                if asyncio.iscoroutinefunction(func):
                    return await func(*args)
                else:
                    return func(*args)
            except Exception as e:
                last_exception = e
                self.logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                await asyncio.sleep(delay)
                delay *= 2
        
        self.logger.error(f"All retry attempts failed: {str(last_exception)}")
        raise RetryError(f"Failed after {max_retries} attempts")

    def _update_conversation_state(self, query: str, sentiment: Dict):
        """Update conversation state based on current interaction"""
        # Update topic
        if any(word in query.lower() for word in ['payment', 'pay', 'money']):
            self.conversation_state.payment_mentioned = True
            self.conversation_state.current_topic = 'payment'
        elif any(word in query.lower() for word in ['dispute', 'not me', 'wrong']):
            self.conversation_state.dispute_mentioned = True
            self.conversation_state.current_topic = 'dispute'
            
        # Update sentiment tracking
        self.conversation_state.last_sentiment = sentiment['compound']
        
        # Track aggression
        if self._contains_threats(query):
            self.conversation_state.aggression_level += 1

    async def cleanup(self):
        """Cleanup resources"""
        try:
            # Clear GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Reset conversation state
            self.conversation_state = ConversationState()
            self.conversation_history.clear()
            
            # Close any open resources
            await self.faq_db.close()  # Assuming FAQ database needs cleanup
            
        except Exception as e:
            self.logger.error(f"Cleanup failed: {str(e)}")

    def _ensure_response_quality(self, response: str, query: str) -> str:
        """Comprehensive quality check and enhancement"""
        try:
            # Check for payment information accuracy
            if 'payment' in response.lower():
                response = self._validate_payment_info(response)
            
            # Ensure proper handling of sensitive information
            response = self._mask_sensitive_info(response)
            
            # Check for response completeness
            if not self._is_response_complete(response):
                return self._get_default_response(query)
            
            # Ensure proper formatting
            response = self._format_response(response)
            
            # Add urgency for payment-related responses
            if self.conversation_state.payment_mentioned:
                response = self._add_urgency(response)
            
            return response
        except Exception as e:
            self.logger.error(f"Quality assurance failed: {str(e)}")
            return self._get_default_response(query)

    def _validate_payment_info(self, response: str) -> str:
        """Ensure payment information is accurate and consistent"""
        payment_corrections = {
            r'payment\s+plan': "immediate payment is required",
            r'later\s+payment': "immediate payment is required",
            r'partial\s+payment': "full payment is required",
            r'installments?': "full immediate payment"
        }
        
        for pattern, correction in payment_corrections.items():
            response = re.sub(pattern, correction, response, flags=re.IGNORECASE)
        
        return response

    def _add_urgency(self, response: str) -> str:
        """Add urgency to payment-related responses without repetition"""
        urgency_phrases = [
            "This matter requires immediate attention.",
            "Delay in payment will lead to further consequences.",
            "Your immediate action is required.",
            "This must be resolved without delay."
        ]
        
        # Check if response already contains urgency phrases
        if any(phrase.lower() in response.lower() for phrase in urgency_phrases):
            return response
        
        return f"{response} {np.random.choice(urgency_phrases)}"

    def _prevent_response_errors(self, response: str) -> bool:
        """Check for common response errors"""
        error_patterns = {
            r'\{[^}]*\}': "unresolved placeholder",
            r'\b(?:undefined|null|nan)\b': "programming artifact",
            r'\b(?:error|exception|traceback)\b': "error message",
            r'\b(?:todo|fixme|xxx)\b': "development artifact",
            r'\b(?:test|debug|sample)\b': "testing artifact"
        }
        
        for pattern, error_type in error_patterns.items():
            if re.search(pattern, response, re.IGNORECASE):
                self.logger.error(f"Response error detected: {error_type}")
                return False
        return True

    def _manage_conversation_flow(self, query: str) -> Optional[str]:
        """Manage conversation flow and context"""
        try:
            # Check for conversation derailment
            if self._is_conversation_derailed():
                return self._get_back_on_track()
            
            # Check for circular conversations
            if self._is_circular_conversation():
                return self._break_conversation_loop()
            
            # Check for topic switching
            if self._is_topic_switch(query):
                return self._handle_topic_switch(query)
            
            return None
        except Exception as e:
            self.logger.error(f"Conversation flow management failed: {str(e)}")
            return None

    def _is_conversation_derailed(self) -> bool:
        """Check if conversation has derailed from financial focus"""
        if len(self.conversation_history) < 2:
            return False
        
        recent_messages = self.conversation_history[-3:]
        finance_related = sum(1 for msg in recent_messages 
                             if self._is_finance_related(msg['content']))
        return finance_related == 0

    def _enforce_response_consistency(self, response: str) -> str:
        """Ensure response consistency with previous statements"""
        try:
            # Check for contradictions with previous responses
            if self._has_contradiction(response):
                return self._resolve_contradiction(response)
            
            # Ensure consistent tone
            response = self._maintain_consistent_tone(response)
            
            # Ensure consistent stance on payment
            response = self._enforce_payment_stance(response)
            
            return response
        except Exception as e:
            self.logger.error(f"Consistency enforcement failed: {str(e)}")
            return response

    def _enforce_payment_stance(self, response: str) -> str:
        """Ensure consistent stance on payment requirements"""
        payment_stance = {
            'required': ['must', 'required', 'necessary', 'mandatory'],
            'immediate': ['immediate', 'now', 'right away', 'without delay'],
            'full': ['full', 'complete', 'entire', 'total']
        }
        
        if any(word in response.lower() for word in ['payment', 'pay', 'settle']):
            for stance_type, words in payment_stance.items():
                if not any(word in response.lower() for word in words):
                    response = f"{response} Full payment is required immediately."
                    break
        
        return response

    def _personalize_response(self, response: str) -> str:
        """Add appropriate personalization to response"""
        try:
            # Add contextual acknowledgment
            if self.conversation_state.last_sentiment < -0.3:
                response = self._add_empathetic_prefix(response)
            
            # Add professional closing
            if self._is_final_response(response):
                response = self._add_professional_closing(response)
            
            # Add urgency based on payment history
            if self.conversation_state.payment_mentioned:
                response = self._add_payment_urgency(response)
            
            return response
        except Exception as e:
            self.logger.error(f"Response personalization failed: {str(e)}")
            return response

    def _is_circular_conversation(self) -> bool:
        """Check for repetitive conversation patterns"""
        if len(self.conversation_history) < 4:
            return False
        
        # Get last few messages
        recent_messages = [msg["content"].lower() for msg in self.conversation_history[-4:]]
        
        # Check for repeated content
        unique_messages = set(recent_messages)
        if len(unique_messages) < len(recent_messages) / 2:
            return True
        
        return False

    def _break_conversation_loop(self) -> str:
        """Handle circular conversations"""
        return ("Let me redirect our conversation. The immediate requirement is full payment. "
                "How would you like to proceed with settling your outstanding balance?")

    async def _generate_base_response(self, query: str, sentiment: Dict) -> str:
        """Generate base response before enhancements"""
        try:
            # Check for identity queries
            if "who" in query.lower():
                return self._handle_identity_query(query)
            
            # Check FAQ database
            if faq_response := self.faq_db.get_exact_match(query):
                return self._process_response(faq_response, sentiment)
            
            # Generate AI response
            return await self._generate_ai_response(query, sentiment)
            
        except Exception as e:
            self.logger.error(f"Base response generation failed: {str(e)}")
            return self._get_default_response(query)

    def _is_topic_switch(self, query: str) -> bool:
        """Detect if user is switching topics"""
        if not self.conversation_state.current_topic:
            return False
        
        current_topic = self.conversation_state.current_topic
        new_topic = self._detect_topic(query)
        
        return current_topic != new_topic

    def _detect_topic(self, query: str) -> str:
        """Detect the topic of the query"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['payment', 'pay', 'money']):
            return 'payment'
        elif any(word in query_lower for word in ['dispute', 'wrong', 'not me']):
            return 'dispute'
        elif any(word in query_lower for word in ['help', 'support', 'assist']):
            return 'support'
        else:
            return 'general'

    def _handle_topic_switch(self, query: str) -> Optional[str]:
        """Handle topic switching in conversation"""
        new_topic = self._detect_topic(query)
        
        if new_topic == 'payment':
            return ("Regarding payment, please note that full payment is required immediately. "
                    "How would you like to proceed with the payment?")
        elif new_topic == 'dispute':
            return self._handle_dispute_claim(query)
        
        return None

    def _get_back_on_track(self) -> str:
        """Redirect conversation back to financial focus"""
        return ("Let's focus on resolving your financial situation. "
                "The outstanding balance requires immediate attention. "
                "How would you like to proceed with the payment?")

    def _has_circular_reference(self, response: str, history: List[Dict]) -> bool:
        """Check if response repeats content from recent history"""
        if not history:
            return False
        
        recent_responses = [msg["content"].lower() for msg in history[-3:] 
                           if msg["role"] == "assistant"]
        response_lower = response.lower()
        
        # Check for substantial overlap with recent responses
        for past_response in recent_responses:
            if self._calculate_similarity(response_lower, past_response) > 0.8:
                return True
        return False

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity using word overlap"""
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)

    def _maintain_consistent_tone(self, response: str) -> str:
        """Ensure consistent professional tone"""
        # Remove casual language
        casual_words = ['yeah', 'nah', 'hey', 'okay', 'ok', 'sure']
        for word in casual_words:
            response = re.sub(r'\b' + word + r'\b', '', response, flags=re.IGNORECASE)
        
        # Ensure professional language
        response = response.replace("don't", "do not")
        response = response.replace("can't", "cannot")
        response = response.replace("won't", "will not")
        
        return response.strip()

    def _add_empathetic_prefix(self, response: str) -> str:
        """Add empathetic acknowledgment to response"""
        empathy_prefixes = [
            "I understand this is a challenging situation. ",
            "I acknowledge your concerns. ",
            "I appreciate you sharing your situation. "
        ]
        
        if not any(response.startswith(prefix) for prefix in empathy_prefixes):
            return f"{np.random.choice(empathy_prefixes)}{response}"
        return response

    def _is_final_response(self, response: str) -> bool:
        """Check if this is likely a final response in the conversation"""
        final_indicators = [
            'goodbye', 'thank you', 'have a', 'contact us', 
            'reach out', 'further assistance'
        ]
        return any(indicator in response.lower() for indicator in final_indicators)

    def _add_professional_closing(self, response: str) -> str:
        """Add professional closing to final responses"""
        closings = [
            " Please don't hesitate to reach out if you need further assistance.",
            " We're here to help if you need anything else.",
            " Please contact us if you have any other questions."
        ]
        
        if not any(response.endswith(closing) for closing in closings):
            return f"{response}{np.random.choice(closings)}"
        return response

    def _add_payment_urgency(self, response: str) -> str:
        """Add payment urgency to response"""
        urgency_phrases = [
            " This matter requires immediate attention.",
            " Please note that immediate payment is required.",
            " Your prompt action is necessary to avoid further consequences."
        ]
        
        if not any(phrase in response for phrase in urgency_phrases):
            return f"{response}{np.random.choice(urgency_phrases)}"
        return response

    def _mask_sensitive_info(self, response: str) -> str:
        """Mask any potentially sensitive information"""
        # Mask potential account numbers
        response = re.sub(r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b', 'XXXX-XXXX-XXXX-XXXX', response)
        
        # Mask potential phone numbers
        response = re.sub(r'\b\d{3}[-\s]?\d{3}[-\s]?\d{4}\b', 'XXX-XXX-XXXX', response)
        
        # Mask potential email addresses
        response = re.sub(r'\b[\w\.-]+@[\w\.-]+\.\w+\b', 'email@masked.com', response)
        
        return response

    def _has_contradiction(self, response: str) -> bool:
        """Check for contradictions in response"""
        contradiction_patterns = [
            (r'can(?:not|\'t)\s+pay.*can\s+pay', 'payment ability'),
            (r'no\s+extension.*offer\s+extension', 'payment extension'),
            (r'must\s+pay.*don\'t\s+pay', 'payment requirement'),
            (r'full\s+payment.*partial\s+payment', 'payment amount'),
            (r'immediate.*later', 'payment timing')
        ]
        
        return any(re.search(pattern, response.lower()) 
                  for pattern, _ in contradiction_patterns)

    def _resolve_contradiction(self, response: str) -> str:
        """Resolve contradictions with clear stance"""
        return ("Full payment is required immediately. "
                "We cannot offer extensions or partial payment options. "
                "Please arrange to settle the outstanding balance as soon as possible.")

    def _manage_context(self, query: str) -> Dict:
        """Advanced context management system"""
        context = {
            'user_intent': self._detect_intent(query),
            'payment_status': self._check_payment_status(),
            'conversation_stage': self._determine_conversation_stage(),
            'user_sentiment': self._get_user_sentiment_history(),
            'critical_points': self._extract_critical_points()
        }
        
        # Update conversation metrics
        self._update_conversation_metrics(context)
        
        return context

    def _determine_conversation_stage(self) -> str:
        """Determine the current stage of conversation"""
        if len(self.conversation_history) < 2:
            return 'initial_contact'
        
        if self.conversation_state.payment_mentioned:
            return 'payment_negotiation'
        elif self.conversation_state.dispute_mentioned:
            return 'dispute_resolution'
        elif self._contains_threats(self.conversation_history[-1]['content']):
            return 'conflict_management'
        
        return 'information_gathering'

    def _evaluate_response_quality(self, response: str, context: Dict) -> float:
        """Evaluate response quality using multiple metrics"""
        metrics = {
            'relevance': self._calculate_relevance_score(response, context),
            'clarity': self._calculate_clarity_score(response),
            'professionalism': self._calculate_professionalism_score(response),
            'assertiveness': self._calculate_assertiveness_score(response),
            'completeness': self._calculate_completeness_score(response)
        }
        
        # Weight and combine metrics
        weights = {
            'relevance': 0.3,
            'clarity': 0.2,
            'professionalism': 0.2,
            'assertiveness': 0.2,
            'completeness': 0.1
        }
        
        return sum(score * weights[metric] for metric, score in metrics.items())

    def _calculate_assertiveness_score(self, response: str) -> float:
        """Calculate assertiveness score of response"""
        assertive_phrases = [
            'must', 'required', 'necessary', 'immediate',
            'cannot offer', 'will not accept', 'full payment'
        ]
        
        score = sum(phrase in response.lower() for phrase in assertive_phrases)
        return min(score / 3, 1.0)  # Normalize score between 0 and 1

    async def _generate_safe_response(self, query: str, context: Dict) -> str:
        """Generate response with advanced error recovery"""
        try:
            # Try primary response generation
            response = await self._generate_ai_response(query, context)
            
            # Validate response quality
            quality_score = self._evaluate_response_quality(response, context)
            
            if quality_score < 0.7:  # Quality threshold
                # Try fallback methods in order
                for method in [
                    self._get_faq_response,
                    self._generate_template_response,
                    self._get_default_response
                ]:
                    try:
                        fallback_response = await method(query)
                        if fallback_response:
                            return fallback_response
                    except Exception as e:
                        self.logger.warning(f"Fallback method failed: {str(e)}")
                        continue
            
            return response
        except Exception as e:
            self.logger.error(f"Safe response generation failed: {str(e)}")
            return self._get_emergency_response()

    def _analyze_payment_intent(self, query: str) -> str:
        """Analyze payment intent from the query"""
        # Implement payment intent analysis logic here
        # This is a placeholder and should be replaced with actual implementation
        return "delay"

    def _handle_payment_discussion(self, query: str, context: Dict) -> str:
        """Advanced payment discussion handling"""
        payment_intent = self._analyze_payment_intent(query)
        
        responses = {
            'delay': (
                "I understand your situation, however, immediate full payment is required. "
                "Delays will result in additional consequences. "
                "Let's proceed with the payment now to avoid any complications."
            ),
            'partial': (
                "We cannot accept partial payments. The full outstanding balance of "
                "{outstanding balance} must be paid immediately. "
                "How would you like to proceed with the full payment?"
            ),
            'dispute': (
                "While I understand your concerns, the debt is registered under your name "
                "and requires immediate attention. Full payment is required while any "
                "dispute is being investigated."
            ),
            'willing': (
                "Excellent decision. Let me guide you through the immediate payment process. "
                "Please note that only full payment will be accepted."
            )
        }
        
        return responses.get(payment_intent, self._get_default_payment_response())

    def _analyze_conversation_metrics(self) -> Dict:
        """Analyze conversation performance metrics"""
        metrics = {
            'response_times': self._calculate_response_times(),
            'topic_coherence': self._calculate_topic_coherence(),
            'user_satisfaction': self._estimate_user_satisfaction(),
            'goal_progress': self._calculate_goal_progress(),
            'conversation_efficiency': self._calculate_efficiency()
        }
        
        # Log metrics for monitoring
        self.logger.info(f"Conversation metrics: {metrics}")
        
        return metrics

    def _estimate_user_satisfaction(self) -> float:
        """Estimate user satisfaction based on conversation indicators"""
        indicators = {
            'sentiment_trend': self._calculate_sentiment_trend(),
            'user_engagement': self._calculate_user_engagement(),
            'issue_resolution': self._check_issue_resolution(),
            'conversation_flow': self._analyze_conversation_flow()
        }
        
        return sum(indicators.values()) / len(indicators)

    def _is_response_complete(self, response: str) -> bool:
        """Check if response is complete and valid"""
        try:
            # Check minimum length
            if len(response.split()) < 5:
                return False
            
            # Check for incomplete endings
            if response.endswith(('...', ',', '-', ':', ';')):
                return False
            
            # Check for broken sentences
            if any(response.lower().startswith(word) for word in ['and', 'but', 'or', 'so', 'because']):
                return False
            
            # Check for proper sentence structure
            sentences = re.split(r'[.!?]+', response)
            for sentence in sentences:
                if len(sentence.strip().split()) < 3:  # Minimum words per sentence
                    return False
                
            return True
        except Exception as e:
            self.logger.error(f"Response completion check failed: {str(e)}")
            return False

    def _is_payment_confirmation(self, query: str) -> bool:
        """Enhanced payment confirmation detection"""
        confirmation_patterns = [
            r'\b(?:i|we)\s+(?:have|just|already)\s+paid\b',
            r'\bpayment\s+(?:done|complete|sent|made)\b',
            r'\b(?:have|has)\s+paid\b',
            r'\bpaid\s+(?:it|all|everything)\b'
        ]
        
        future_payment_patterns = [
            r'\b(?:will|going to|gonna)\s+pay\b',
            r'\bpromise\s+to\s+pay\b'
        ]
        
        query_lower = query.lower()
        if any(re.search(pattern, query_lower) for pattern in confirmation_patterns):
            return True
        if any(re.search(pattern, query_lower) for pattern in future_payment_patterns):
            return False
        return False

    def _get_payment_confirmation_response(self, query: str) -> str:
        """Get appropriate response for payment confirmation"""
        if re.search(r'\b(?:will|gonna|going\s+to)\s+pay\b', query.lower()):
            return ("Thank you for your commitment to pay. Please proceed with the payment "
                    "as soon as possible to avoid any further consequences. Let me know "
                    "once the payment is completed.")
        else:
            return ("Thank you for confirming your payment. Please allow 48 hours for it to reflect. "
                    "If you have any proof of payment, please share it with us.")

    def _get_varied_response(self, response_type: str) -> str:
        """Get varied responses to avoid repetition"""
        payment_responses = {
            'default': [
                "Full payment is required immediately to avoid further consequences. How would you like to proceed with the payment?",
                "Your outstanding balance requires immediate attention. Please arrange the payment now to prevent additional complications.",
                "It's crucial to settle your payment immediately. Any delay will lead to escalated measures. How can we assist you with the payment process?",
                "Your account requires immediate payment. Let's proceed with settling the outstanding balance to avoid any adverse actions."
            ],
            'urgent': [
                "This matter requires your immediate attention. Full payment must be made now to avoid legal consequences.",
                "Your situation is becoming critical. Immediate payment is required to prevent further escalation.",
                "We must stress the urgency of this matter. Full payment is required now to avoid severe consequences.",
                "This is a final reminder that your payment must be made immediately to avoid legal action."
            ]
        }
        
        # Get the last used response
        last_response = self.conversation_history[-1]['content'] if self.conversation_history else ""
        
        # Select a different response than the last one
        responses = payment_responses.get(response_type, payment_responses['default'])
        available_responses = [r for r in responses if r != last_response]
        
        return np.random.choice(available_responses) if available_responses else responses[0]

    def _calculate_relevance_score(self, response: str, context: Dict) -> float:
        """Calculate relevance score based on context matching"""
        try:
            # Extract key terms from context
            context_terms = set()
            for value in context.values():
                if isinstance(value, str):
                    context_terms.update(value.lower().split())
            
            # Get response terms
            response_terms = set(response.lower().split())
            
            # Calculate overlap
            overlap = len(context_terms & response_terms)
            total_terms = len(response_terms)
            
            return min(overlap / max(total_terms, 1), 1.0)
        except Exception as e:
            self.logger.error(f"Relevance score calculation failed: {str(e)}")
            return 0.5

    def _calculate_clarity_score(self, response: str) -> float:
        """Calculate clarity score based on sentence structure"""
        try:
            sentences = re.split(r'[.!?]+', response)
            scores = []
            
            for sentence in sentences:
                words = sentence.strip().split()
                if not words:
                    continue
                
                # Check sentence length (not too short, not too long)
                length_score = min(len(words) / 10, 1.0) if len(words) < 20 else 20 / len(words)
                
                # Check for proper structure (starts with capital, has subject-verb)
                structure_score = 1.0 if (words[0][0].isupper() and len(words) >= 3) else 0.5
                
                scores.append((length_score + structure_score) / 2)
            
            return sum(scores) / max(len(scores), 1)
        except Exception as e:
            self.logger.error(f"Clarity score calculation failed: {str(e)}")
            return 0.5

    def _check_payment_status(self) -> str:
        """Check current payment status based on conversation history"""
        try:
            # Check recent history for payment indicators
            recent_messages = self.conversation_history[-5:]
            
            if any('paid' in msg['content'].lower() for msg in recent_messages):
                return 'payment_claimed'
            elif any('refuse' in msg['content'].lower() for msg in recent_messages):
                return 'payment_refused'
            elif any('later' in msg['content'].lower() for msg in recent_messages):
                return 'payment_delayed'
            elif any('how' in msg['content'].lower() and 'pay' in msg['content'].lower() 
                    for msg in recent_messages):
                return 'payment_inquiry'
            
            return 'payment_pending'
        except Exception as e:
            self.logger.error(f"Payment status check failed: {str(e)}")
            return 'unknown'

    def _get_user_sentiment_history(self) -> List[float]:
        """Track sentiment history over conversation"""
        try:
            sentiment_history = []
            for msg in self.conversation_history[-5:]:
                if msg['role'] == 'user':
                    sentiment = self.sentiment_analyzer.analyze_sentiment(msg['content'])
                    sentiment_history.append(sentiment['compound'])
            return sentiment_history
        except Exception as e:
            self.logger.error(f"Sentiment history tracking failed: {str(e)}")
            return []

    def _get_emergency_response(self) -> str:
        """Generate safe response for emergency situations"""
        try:
            # Check conversation state
            if self.conversation_state.payment_mentioned:
                return ("I apologize for any confusion. Your immediate attention to the outstanding payment "
                       "is required. Please proceed with the full payment to avoid further consequences.")
            elif self.conversation_state.dispute_mentioned:
                return ("I understand there may be a dispute. However, this debt is registered under your "
                       "name and requires immediate attention. Please contact the lender directly while "
                       "arranging the payment.")
            else:
                return ("I apologize for any technical difficulties. Your outstanding balance requires "
                       "immediate payment. How would you like to proceed with settling this matter?")
        except Exception:
            return "Your immediate attention to the outstanding payment is required. How would you like to proceed?"

    def _generate_template_response(self, query: str) -> str:
        """Generate template-based response as fallback"""
        templates = {
            'payment': (
                "Full payment of your outstanding balance is required immediately. "
                "This matter requires your urgent attention to avoid further consequences. "
                "How would you like to proceed with the payment?"
            ),
            'dispute': (
                "While I understand your concerns, this debt is registered under your name. "
                "Full payment is required while any dispute is being investigated. "
                "Please contact the lender directly regarding your dispute."
            ),
            'default': (
                "Your immediate attention to this matter is required. "
                "Please arrange to make the full payment as soon as possible "
                "to avoid any additional consequences."
            )
        }
        
        # Select appropriate template based on conversation state
        if self.conversation_state.dispute_mentioned:
            return templates['dispute']
        elif self.conversation_state.payment_mentioned:
            return templates['payment']
        else:
            return templates['default']

    def _calculate_response_times(self) -> Dict:
        """Calculate response time metrics"""
        try:
            response_times = []
            for i in range(1, len(self.conversation_history), 2):
                if i+1 < len(self.conversation_history):
                    response_times.append(1.0)  # Placeholder for actual timing
            
            return {
                'average': sum(response_times) / max(len(response_times), 1),
                'max': max(response_times) if response_times else 0,
                'min': min(response_times) if response_times else 0
            }
        except Exception as e:
            self.logger.error(f"Response time calculation failed: {str(e)}")
            return {'average': 0, 'max': 0, 'min': 0}

    def _calculate_efficiency(self) -> float:
        """Calculate conversation efficiency score"""
        try:
            if not self.conversation_history:
                return 1.0
            
            # Check progress towards payment
            payment_progress = 0.0
            for msg in self.conversation_history:
                if 'pay' in msg['content'].lower():
                    payment_progress += 0.2
                if 'paid' in msg['content'].lower():
                    payment_progress += 0.5
                
            return min(payment_progress, 1.0)
        except Exception as e:
            self.logger.error(f"Efficiency calculation failed: {str(e)}")
            return 0.5

    def _handle_greeting(self, query: str) -> Optional[str]:
        """Handle greeting messages appropriately"""
        greeting_patterns = ['hi', 'hey', 'hello']
        if any(query.lower().strip() == pattern for pattern in greeting_patterns):
            return self._generate_greeting()
        return None

    def _check_for_hardship_evidence(self, query: str) -> tuple[bool, str]:
        """Enhanced check for hardship situations"""
        hardship_patterns = {
            'illness': r'\b(?:sick|hospital|medical|illness|disease|treatment|doctor)\b',
            'death': r'\b(?:death|passed away|funeral|deceased|died|lost.*father|lost.*mother|lost.*parent)\b',
            'job_loss': r'\b(?:lost.*job|unemployed|no.*job|fired|laid off)\b'
        }
        
        query_lower = query.lower()
        for hardship_type, pattern in hardship_patterns.items():
            if re.search(pattern, query_lower):
                return True, hardship_type
        return False, ''

    def _handle_hardship_case(self, query: str, hardship_type: str) -> str:
        """Generate appropriate response for verified hardship cases"""
        responses = {
            'illness': (
                "I understand you're offering medical evidence of your condition. "
                "While payment is still required, we can: \n"
                "1. Put your account on hold for 48 hours while we verify your documentation\n"
                "2. Work with you to explore suitable arrangements once verified\n"
                "Please provide your medical documentation for review."
            ),
            'death': (
                "I'm very sorry for your loss. If you can provide the death certificate, "
                "we can: \n"
                "1. Put your account on hold for 48 hours while we verify the documentation\n"
                "2. Work with you to explore suitable arrangements once verified\n"
                "Please share the documentation for review."
            )
        }
        return responses.get(hardship_type, self._get_default_response(query))

# Usage example remains unchanged
