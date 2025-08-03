# assistant.py - Enhanced Braille Assistant with Better Error Handling
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
    """Enhanced LLM client with better error handling and fallbacks"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY")
        
        # Configure based on API key type
        if self.api_key and self.api_key.startswith("gsk_"):  # Groq
            self.base_url = "https://api.groq.com/openai/v1"
            self.model = "llama-3.1-8b-instant"
            self.provider = "groq"
        elif self.api_key:  # OpenAI
            self.base_url = "https://api.openai.com/v1"
            self.model = "gpt-3.5-turbo"
            self.provider = "openai"
        else:
            self.provider = "fallback"
            print("Warning: No API key found. Using fallback responses.")
    
    def generate_response(self, messages: List[Dict], max_tokens: int = 500) -> str:
        """Generate response with fallback for no API key"""
        
        # Fallback mode if no API key
        if not self.api_key:
            return self._fallback_response(messages)
        
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": self.model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": 0.3
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
                return self._fallback_response(messages)
                
        except requests.exceptions.Timeout:
            print("Request timed out, using fallback")
            return self._fallback_response(messages)
        except Exception as e:
            print(f"LLM error: {e}")
            return self._fallback_response(messages)
    
    def _fallback_response(self, messages: List[Dict]) -> str:
        """Provide intelligent fallback responses without API"""
        user_message = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                user_message = msg.get("content", "").lower()
                break
        
        # Simple pattern matching for common queries
        if "hello" in user_message or "hi" in user_message:
            return "Hello! I'm your Braille Recognition Assistant. I can help you process braille text and answer questions. How can I assist you today?"
        
        elif "help" in user_message:
            return "I can help you with:\n1. Processing braille text into readable format\n2. Explaining topics and concepts\n3. General conversation\n4. Text analysis and correction\n\nWhat would you like to do?"
        
        elif "braille" in user_message:
            return "I can process braille characters and convert them to readable text. Please provide the braille characters you'd like me to process."
        
        elif any(word in user_message for word in ["what", "explain", "tell me"]):
            # Extract topic after question words
            for phrase in ["what is", "explain", "tell me about"]:
                if phrase in user_message:
                    topic = user_message.split(phrase, 1)[1].strip()
                    if topic:
                        return f"I'd be happy to explain {topic}, but I currently don't have access to my full knowledge base. Could you provide more specific details about what you'd like to know?"
            return "I'd be happy to help explain something. Could you be more specific about what you'd like to know?"
        
        elif "thank" in user_message:
            return "You're welcome! Is there anything else I can help you with?"
        
        elif "bye" in user_message or "goodbye" in user_message:
            return "Goodbye! Feel free to come back anytime if you need help with braille processing or have questions."
        
        else:
            return f"I understand you're asking about: {user_message[:50]}{'...' if len(user_message) > 50 else ''}. I'm currently operating in limited mode. Could you rephrase your question or ask about braille processing?"

def search_wikipedia_simple(query: str, max_chars: int = 300) -> str:
    """Simple Wikipedia search with better error handling"""
    try:
        # Use Wikipedia API directly
        search_url = "https://en.wikipedia.org/api/rest_v1/page/summary/"
        # Clean query for URL
        clean_query = query.replace(" ", "_").replace(",", "").replace("?", "")
        
        response = requests.get(f"{search_url}{clean_query}", timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            extract = data.get("extract", "")
            if extract:
                return extract[:max_chars] if len(extract) > max_chars else extract
            else:
                return f"No detailed information found for {query}"
        else:
            return f"Could not find information about {query}"
            
    except Exception as e:
        print(f"Wikipedia search error: {e}")
        return f"Wikipedia search unavailable for {query}"

class BrailleAssistant:
    """Enhanced Braille Assistant with better error handling"""
    
    def __init__(self, api_key: str = None):
        self.llm = LightweightLLM(api_key)
        self.conversation_memory = {}
        
        # Status check
        if not self.llm.api_key:
            print("Assistant initialized in fallback mode (no API key)")
        else:
            print(f"Assistant initialized with {self.llm.provider} API")
    
    def process_braille_strings(self, detected_strings: List[str]) -> BrailleResult:
        """Process braille detection results with fallback"""
        
        if not detected_strings:
            return BrailleResult(
                text="",
                explanation="No braille characters detected.",
                confidence=0.0
            )
        
        try:
            # Step 1: Process raw text
            raw_text = " ".join(detected_strings).strip()
            
            # Simple fallback processing if no API
            if not self.llm.api_key:
                processed_text = self._fallback_braille_processing(raw_text)
                explanation = f"Processed braille text: {processed_text}. (Using basic processing - API not available)"
                confidence = 0.6
            else:
                # Use LLM processing
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
                
                # Generate explanation
                explanation = self._generate_explanation(processed_text)
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
    
    def _fallback_braille_processing(self, text: str) -> str:
        """Basic braille processing without API"""
        # Simple cleaning and formatting
        cleaned = text.strip()
        
        # Basic word formation (this is very simplified)
        if len(cleaned) > 10:
            # Try to identify word boundaries (very basic)
            words = []
            current_word = ""
            
            for char in cleaned:
                if char.isspace() or char in ".,!?":
                    if current_word:
                        words.append(current_word)
                        current_word = ""
                    if char in ".,!?":
                        words.append(char)
                else:
                    current_word += char
            
            if current_word:
                words.append(current_word)
            
            return " ".join(words)
        
        return cleaned
    
    def _generate_explanation(self, text: str) -> str:
        """Generate explanation with fallback"""
        try:
            if not self.llm.api_key:
                return f"This appears to be braille text that reads: '{text}'. For detailed explanations, please configure an API key."
            
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
                    if enhanced_explanation and len(enhanced_explanation) > 20:
                        explanation = enhanced_explanation
            
            return explanation or f"This text discusses: {text}"
            
        except Exception as e:
            return f"This appears to be about: {text}"
    
    def chat(self, user_message: str, thread_id: str = "default") -> str:
        """Enhanced chat with better fallback handling"""
        
        if not user_message.strip():
            return "Please provide a message."
        
        try:
            # Get conversation history (keep last 6 messages for context)
            history = self.conversation_memory.get(thread_id, [])
            
            # Add system message if new conversation
            if not history:
                system_msg = "You are a helpful AI assistant specializing in braille recognition and general assistance. Provide clear, concise, and helpful responses."
                if not self.llm.api_key:
                    system_msg += " You are currently operating in fallback mode with limited capabilities."
                
                history = [{"role": "system", "content": system_msg}]
            
            # Add user message
            history.append({"role": "user", "content": user_message})
            
            # Keep only recent messages (to avoid token limits)
            if len(history) > 7:  # System + 6 messages
                history = [history[0]] + history[-6:]
            
            # Check if we need Wikipedia info (only if we have API access)
            needs_info = any(keyword in user_message.lower() for keyword in 
                           ['what is', 'tell me about', 'explain', 'define', 'information about'])
            
            enhanced_messages = history.copy()
            
            if needs_info and self.llm.api_key:
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
            print(f"Chat error: {e}")
            return f"I apologize, but I encountered an error. Please try rephrasing your question."
    
    def _extract_search_terms(self, message: str) -> str:
        """Extract search terms from user message"""
        message_lower = message.lower()
        
        patterns = [
            "what is ", "tell me about ", "explain ", "define ",
            "information about ", "what are ", "who is ", "where is "
        ]
        
        for pattern in patterns:
            if pattern in message_lower:
                start_idx = message_lower.find(pattern) + len(pattern)
                remaining = message[start_idx:].strip()
                search_term = remaining.split('.')[0].split('?')[0].split(',')[0]
                return search_term.strip()[:50]
        
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
    
    def get_status(self) -> Dict[str, Any]:
        """Get assistant status"""
        return {
            "api_available": bool(self.llm.api_key),
            "provider": self.llm.provider,
            "conversations": len(self.conversation_memory),
            "mode": "full" if self.llm.api_key else "fallback"
        }

# Test function for debugging
def test_assistant():
    """Test the assistant functionality"""
    print("Testing Braille Assistant...")
    
    assistant = BrailleAssistant()
    status = assistant.get_status()
    print(f"Status: {status}")
    
    # Test chat
    test_messages = ["hello", "what is python", "help me with braille"]
    
    for msg in test_messages:
        print(f"\nUser: {msg}")
        response = assistant.chat(msg)
        print(f"Assistant: {response}")
    
    # Test braille processing
    print("\nTesting braille processing...")
    braille_test = ["⠓⠑⠇⠇⠕", "⠺⠕⠗⠇⠙"]
    result = assistant.process_braille_strings(braille_test)
    print(f"Text: {result.text}")
    print(f"Explanation: {result.explanation}")
    print(f"Confidence: {result.confidence}")

if __name__ == "__main__":
    test_assistant()