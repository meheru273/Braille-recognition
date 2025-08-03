# assistant.py - Lightweight Braille Assistant (No LangGraph/LangChain)
import os
import json
import requests
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime

@dataclass
class BrailleResult:
    """Result from braille processing"""
    text: str
    explanation: str
    confidence: float

class LightweightLLM:
    """Lightweight LLM client for OpenAI/Groq APIs"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY")
        
        if not self.api_key:
            raise ValueError("API key required: GROQ_API_KEY or OPENAI_API_KEY")
        
        # Configure based on API key type
        if self.api_key.startswith("gsk_"):  # Groq
            self.base_url = "https://api.groq.com/openai/v1"
            self.model = "llama-3.1-8b-instant"
        else:  # OpenAI
            self.base_url = "https://api.openai.com/v1"
            self.model = "gpt-3.5-turbo"
    
    def generate_response(self, messages: List[Dict], max_tokens: int = 500) -> str:
        """Generate response using direct API call"""
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": self.model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": 0.3,
                "timeout": 20
            }
            
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=25
            )
            
            if response.status_code == 200:
                data = response.json()
                return data["choices"][0]["message"]["content"].strip()
            else:
                print(f"API error: {response.status_code} - {response.text}")
                return "I apologize, but I'm unable to process your request right now."
                
        except requests.exceptions.Timeout:
            return "Request timed out. Please try again."
        except Exception as e:
            print(f"LLM error: {e}")
            return "I encountered an error processing your request."

def search_wikipedia_simple(query: str, max_chars: int = 300) -> str:
    """Simple Wikipedia search without heavy dependencies"""
    try:
        # Use Wikipedia API directly
        search_url = "https://en.wikipedia.org/api/rest_v1/page/summary/"
        # Clean query for URL
        clean_query = query.replace(" ", "_").replace(",", "")
        
        response = requests.get(f"{search_url}{clean_query}", timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            extract = data.get("extract", "")
            return extract[:max_chars] if extract else f"No information found for {query}"
        else:
            return f"Could not find information about {query}"
            
    except Exception as e:
        return f"Wikipedia search unavailable: {str(e)}"

class BrailleAssistant:
    """Lightweight Braille Assistant without heavy dependencies"""
    
    def __init__(self, api_key: str = None):
        self.llm = LightweightLLM(api_key)
        self.conversation_memory = {}  # Simple in-memory storage
    
    def process_braille_strings(self, detected_strings: List[str]) -> BrailleResult:
        """Process braille detection results"""
        
        if not detected_strings:
            return BrailleResult(
                text="",
                explanation="No braille characters detected.",
                confidence=0.0
            )
        
        try:
            # Step 1: Process raw text
            raw_text = " ".join(detected_strings).strip()
            
            process_prompt = [
                {
                    "role": "system", 
                    "content": "You are a braille text interpreter. Convert detected braille characters into meaningful text."
                },
                {
                    "role": "user", 
                    "content": f"""
                    Braille characters detected: "{raw_text}"
                    
                    Task:
                    1. Form meaningful words/phrases from these characters
                    2. Correct obvious OCR errors
                    3. Return only the corrected/interpreted text
                    
                    Processed text:"""
                }
            ]
            
            processed_text = self.llm.generate_response(process_prompt, max_tokens=200)
            
            if not processed_text or len(processed_text.strip()) < 2:
                processed_text = raw_text
            
            # Step 2: Generate explanation
            explanation = self._generate_explanation(processed_text)
            
            # Calculate confidence
            confidence = min(0.9, len([s for s in detected_strings if s.strip()]) / max(1, len(detected_strings)))
            
            return BrailleResult(
                text=processed_text,
                explanation=explanation,
                confidence=confidence
            )
            
        except Exception as e:
            print(f"Braille processing error: {e}")
            fallback_text = " ".join(detected_strings)
            return BrailleResult(
                text=fallback_text,
                explanation=f"Basic text assembly: {fallback_text}",
                confidence=0.3
            )
    
    def _generate_explanation(self, text: str) -> str:
        """Generate explanation for processed text"""
        try:
            # Try direct explanation first
            explain_prompt = [
                {
                    "role": "system",
                    "content": "Provide brief, helpful explanations about topics."
                },
                {
                    "role": "user",
                    "content": f'Explain this topic in 2-3 sentences: "{text}"'
                }
            ]
            
            explanation = self.llm.generate_response(explain_prompt, max_tokens=150)
            
            # If explanation is too short or generic, try Wikipedia
            if not explanation or len(explanation) < 20 or "I don't" in explanation:
                wiki_info = search_wikipedia_simple(text, 200)
                
                if wiki_info and "unavailable" not in wiki_info.lower():
                    enhanced_prompt = [
                        {
                            "role": "user",
                            "content": f"""
                            Topic: "{text}"
                            Wikipedia info: {wiki_info}
                            
                            Explain this topic in 2-3 clear sentences:"""
                        }
                    ]
                    
                    enhanced_explanation = self.llm.generate_response(enhanced_prompt, max_tokens=150)
                    if enhanced_explanation:
                        explanation = enhanced_explanation
            
            return explanation or f"This text discusses: {text}"
            
        except Exception as e:
            return f"This appears to be about: {text}"
    
    def chat(self, user_message: str, thread_id: str = "default") -> str:
        """Handle chat messages with simple memory"""
        
        if not user_message.strip():
            return "Please provide a message."
        
        try:
            # Get conversation history (keep last 6 messages for context)
            history = self.conversation_memory.get(thread_id, [])
            
            # Add system message if new conversation
            if not history:
                history = [
                    {
                        "role": "system",
                        "content": "You are a helpful AI assistant. Provide clear, concise, and helpful responses."
                    }
                ]
            
            # Add user message
            history.append({"role": "user", "content": user_message})
            
            # Keep only recent messages (to avoid token limits)
            if len(history) > 7:  # System + 6 messages
                history = [history[0]] + history[-6:]
            
            # Check if we need Wikipedia info
            needs_info = any(keyword in user_message.lower() for keyword in 
                           ['what is', 'tell me about', 'explain', 'define', 'information about'])
            
            enhanced_messages = history.copy()
            
            if needs_info:
                # Extract key terms for Wikipedia search
                search_terms = self._extract_search_terms(user_message)
                if search_terms:
                    wiki_info = search_wikipedia_simple(search_terms)
                    if wiki_info and "unavailable" not in wiki_info.lower():
                        enhanced_messages.append({
                            "role": "system",
                            "content": f"Additional context: {wiki_info}"
                        })
            
            # Generate response
            response = self.llm.generate_response(enhanced_messages, max_tokens=300)
            
            # Update conversation memory
            history.append({"role": "assistant", "content": response})
            self.conversation_memory[thread_id] = history
            
            return response
            
        except Exception as e:
            return f"I apologize, but I encountered an error: {str(e)}"
    
    def _extract_search_terms(self, message: str) -> str:
        """Extract search terms from user message"""
        # Simple extraction - get terms after common question words
        message_lower = message.lower()
        
        patterns = [
            "what is ", "tell me about ", "explain ", "define ",
            "information about ", "what are ", "who is ", "where is "
        ]
        
        for pattern in patterns:
            if pattern in message_lower:
                start_idx = message_lower.find(pattern) + len(pattern)
                # Get next few words
                remaining = message[start_idx:].strip()
                # Take first phrase (up to punctuation or next sentence)
                search_term = remaining.split('.')[0].split('?')[0].split(',')[0]
                return search_term.strip()[:50]  # Limit length
        
        # Fallback: return first few meaningful words
        words = message.split()
        meaningful_words = [w for w in words if len(w) > 2 and w.lower() not in 
                          ['the', 'and', 'but', 'for', 'you', 'can', 'will', 'this', 'that']]
        
        return ' '.join(meaningful_words[:3]) if meaningful_words else message[:30]
    
    def process_text(self, text: str, task: str = "explain", max_length: int = 400) -> str:
        """Process text with specific task"""
        
        task_prompts = {
            "explain": f"Explain this text clearly: '{text}'",
            "summarize": f"Summarize this text briefly: '{text}'",
            "correct": f"Correct any errors and improve clarity: '{text}'",
            "enhance": f"Improve this text while keeping its meaning: '{text}'",
            "analyze": f"Analyze the content and meaning: '{text}'"
        }
        
        prompt = task_prompts.get(task, task_prompts["explain"])
        
        if max_length < 500:
            prompt += f" Keep response under {max_length} characters."
        
        try:
            messages = [
                {"role": "system", "content": "You are a helpful text processing assistant."},
                {"role": "user", "content": prompt}
            ]
            
            response = self.llm.generate_response(messages, max_tokens=min(300, max_length//2))
            
            # Truncate if needed
            if len(response) > max_length:
                response = response[:max_length-3] + "..."
            
            return response
            
        except Exception as e:
            return f"Error processing text: {str(e)}"
    
    def clear_conversation(self, thread_id: str = None):
        """Clear conversation memory"""
        if thread_id:
            self.conversation_memory.pop(thread_id, None)
        else:
            self.conversation_memory.clear()