import os
from typing import List, Dict, Any, TypedDict, Annotated, Literal
from dataclasses import dataclass
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools import WikipediaQueryRun
from dotenv import load_dotenv
import traceback

# Load environment variables
load_dotenv()

@dataclass
class BrailleResult:
    """Result from braille processing"""
    text: str
    explanation: str
    confidence: float

@tool
def search_wikipedia(query: str) -> str:
    """Search Wikipedia for context about a topic"""
    try:
        wiki_wrapper = WikipediaAPIWrapper(
            top_k_results=2, 
            doc_content_chars_max=300
        )
        wiki_tool = WikipediaQueryRun(api_wrapper=wiki_wrapper)
        return wiki_tool.run(query)
    except Exception as e:
        return f"Wikipedia search unavailable: {str(e)}"

class ChatState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    detected_strings: List[str]
    processed_text: str
    explanation: str
    confidence: float
    input_type: str  # "braille" or "chat"
    user_message: str  # For chat input

class BrailleAssistant:
    def __init__(self, api_key: str = None):
        """Initialize the Braille Assistant with proper error handling"""
        print("Initializing BrailleAssistant...")
        
        # Get API key from parameter or environment
        self.api_key = api_key or os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY")
        
        if not self.api_key:
            raise ValueError("No API key found. Please set GROQ_API_KEY or OPENAI_API_KEY environment variable.")
        
        print(f"Using API key starting with: {self.api_key[:10]}...")
        
        try:
            # Configure LLM based on API key type
            if self.api_key.startswith("gsk_"):  # Groq API key
                print("Configuring Groq LLM...")
                self.llm = ChatOpenAI(
                    api_key=self.api_key,
                    base_url="https://api.groq.com/openai/v1",
                    model="llama-3.1-8b-instant",
                    temperature=0.3,
                    timeout=30,
                    max_retries=2
                )
            elif self.api_key.startswith("sk-"):  # OpenAI API key
                print("Configuring OpenAI LLM...")
                self.llm = ChatOpenAI(
                    api_key=self.api_key, 
                    model="gpt-3.5-turbo",
                    temperature=0.3,
                    timeout=30,
                    max_retries=2
                )
            else:
                raise ValueError("Invalid API key format. Use Groq (gsk_) or OpenAI (sk_) key.")
            
            # Test the LLM connection
            print("Testing LLM connection...")
            test_response = self.llm.invoke([HumanMessage(content="Hello")])
            print(f"LLM test successful: {test_response.content[:50]}...")
            
            # Bind tools to LLM
            self.llm_with_tools = self.llm.bind_tools([search_wikipedia])
            
            # Build graph
            print("Building LangGraph workflow...")
            self._build_graph()
            print("✅ BrailleAssistant initialized successfully")
            
        except Exception as e:
            error_msg = f"Failed to initialize LLM: {str(e)}"
            print(f"❌ {error_msg}")
            print(traceback.format_exc())
            raise ValueError(error_msg)
    
    def _build_graph(self):
        """Build the LangGraph workflow"""
        
        def route_input(state: ChatState) -> Literal["process_braille", "process_chat"]:
            """Route based on input type"""
            input_type = state.get('input_type', 'chat')
            return "process_braille" if input_type == 'braille' else "process_chat"
        
        def process_braille_node(state: ChatState):
            """Process detected braille strings into readable text"""
            detected_strings = state.get('detected_strings', [])
            
            if not detected_strings:
                return {
                    'processed_text': '',
                    'confidence': 0.0,
                    'messages': [SystemMessage(content="No braille strings detected")]
                }
            
            try:
                # Create prompt for text processing
                raw_text = " ".join(detected_strings)
                prompt = f"""You are a braille text interpreter. 

Detected braille characters: "{raw_text}"

Please:
1. Interpret these characters as meaningful text
2. Correct any obvious OCR errors
3. Form coherent words or phrases

Respond with only the corrected/interpreted text."""
                
                response = self.llm.invoke([HumanMessage(content=prompt)])
                processed_text = response.content.strip()
                
                # Calculate confidence based on detection quality
                non_empty_strings = [s for s in detected_strings if s.strip()]
                confidence = min(0.9, len(non_empty_strings) / max(1, len(detected_strings)))
                
                return {
                    'processed_text': processed_text,
                    'confidence': confidence
                }
                
            except Exception as e:
                print(f"Error in process_braille_node: {e}")
                return {
                    'processed_text': ' '.join(detected_strings),
                    'confidence': 0.3,
                    'messages': [SystemMessage(content=f"Processing error: {str(e)}")]
                }
        
        def generate_braille_explanation(state: ChatState):
            """Generate explanation for braille text"""
            processed_text = state.get('processed_text', '')
            
            if not processed_text:
                return {'explanation': 'No text available for explanation.'}
            
            try:
                # Generate explanation
                prompt = f"""The detected braille text says: "{processed_text}"

Provide a helpful 2-3 sentence explanation about what this text means or discusses. Include any relevant context that would help someone understand the topic."""
                
                response = self.llm.invoke([HumanMessage(content=prompt)])
                explanation = response.content.strip()
                
                # If explanation is too short, try to enhance it
                if len(explanation) < 30:
                    try:
                        wiki_result = search_wikipedia(processed_text)
                        if wiki_result and "Wikipedia search unavailable" not in wiki_result:
                            enhanced_prompt = f"""Topic: "{processed_text}"
Additional context: {wiki_result[:200]}

Based on this information, provide a clear explanation about this topic in 2-3 sentences."""
                            response = self.llm.invoke([HumanMessage(content=enhanced_prompt)])
                            explanation = response.content.strip()
                    except:
                        pass  # Use original explanation
                
                return {'explanation': explanation or f"The text appears to be about: {processed_text}"}
                
            except Exception as e:
                print(f"Error in generate_braille_explanation: {e}")
                return {'explanation': f"Detected text: {processed_text}. Unable to generate detailed explanation."}
        
        def process_chat_node(state: ChatState):
            """Handle regular chat messages"""
            user_message = state.get('user_message', '')
            messages = state.get('messages', [])
            
            if not user_message:
                return {'messages': [SystemMessage(content="Please provide a message.")]}
            
            # Add user message to conversation
            messages.append(HumanMessage(content=user_message))
            
            try:
                # Check if message might benefit from Wikipedia search
                search_keywords = ['what is', 'tell me about', 'explain', 'define', 'information about', 'how does', 'history of']
                needs_search = any(keyword in user_message.lower() for keyword in search_keywords)
                
                if needs_search and len(user_message.split()) <= 10:  # Simple queries only
                    try:
                        response = self.llm_with_tools.invoke(messages)
                    except:
                        # Fallback to regular LLM if tools fail
                        response = self.llm.invoke(messages)
                else:
                    response = self.llm.invoke(messages)
                
                return {'messages': [response]}
                
            except Exception as e:
                print(f"Error in process_chat_node: {e}")
                return {'messages': [SystemMessage(content=f"I encountered an error: {str(e)}. Please try again.")]}
        
        # Build the state graph
        self.graph = StateGraph(ChatState)
        
        # Add nodes
        self.graph.add_node("process_braille", process_braille_node)
        self.graph.add_node("generate_braille_explanation", generate_braille_explanation)
        self.graph.add_node("process_chat", process_chat_node)
        
        # Add conditional routing from START
        self.graph.add_conditional_edges(
            START,
            route_input,
            {
                "process_braille": "process_braille",
                "process_chat": "process_chat"
            }
        )
        
        # Add edges for braille processing workflow
        self.graph.add_edge("process_braille", "generate_braille_explanation")
        self.graph.add_edge("generate_braille_explanation", END)
        
        # Add edge for chat workflow
        self.graph.add_edge("process_chat", END)
        
        # Compile with memory
        checkpointer = MemorySaver()
        self.chatbot = self.graph.compile(checkpointer=checkpointer)
    
    def process_braille_strings(self, detected_strings: List[str]) -> BrailleResult:
        """Process braille detection results"""
        
        if not detected_strings:
            return BrailleResult(
                text="",
                explanation="No braille characters detected.",
                confidence=0.0
            )
        
        print(f"Processing {len(detected_strings)} detected strings: {detected_strings}")
        
        # Initial state for braille processing
        initial_state = {
            'messages': [SystemMessage(content="Processing braille detection results.")],
            'detected_strings': detected_strings,
            'processed_text': '',
            'explanation': '',
            'confidence': 0.0,
            'input_type': 'braille'
        }
        
        try:
            config = {'configurable': {'thread_id': f'braille_{hash(str(detected_strings))}'}}
            result = self.chatbot.invoke(initial_state, config=config)
            
            return BrailleResult(
                text=result.get('processed_text', ' '.join(detected_strings)),
                explanation=result.get('explanation', 'Unable to generate explanation.'),
                confidence=result.get('confidence', 0.5)
            )
        
        except Exception as e:
            print(f"Error in process_braille_strings: {e}")
            print(traceback.format_exc())
            return BrailleResult(
                text=' '.join(detected_strings),
                explanation=f"Processing error: {str(e)}",
                confidence=0.3
            )
    
    def chat(self, user_message: str, thread_id: str = "default_chat") -> str:
        """Handle regular chat messages"""
        
        if not user_message.strip():
            return "Please provide a message."
        
        print(f"Processing chat message: {user_message[:50]}...")
        
        # Initial state for chat processing
        initial_state = {
            'messages': [SystemMessage(content="You are a helpful AI assistant specializing in accessibility, braille, and assistive technologies.")],
            'user_message': user_message,
            'input_type': 'chat'
        }
        
        try:
            config = {'configurable': {'thread_id': thread_id}}
            result = self.chatbot.invoke(initial_state, config=config)
            
            # Extract the last message content
            messages = result.get('messages', [])
            if messages and hasattr(messages[-1], 'content'):
                return messages[-1].content
            else:
                return "I couldn't process your message properly. Please try again."
        
        except Exception as e:
            print(f"Error in chat: {e}")
            print(traceback.format_exc())
            return f"I encountered an error: {str(e)}. Please try again."