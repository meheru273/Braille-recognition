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
load_dotenv()  # Loads .env file

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
        return f"Could not search Wikipedia: {str(e)}"


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
        # Use provided key or environment variable
        self.api_key = os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY")
        
        # Configure LLM based on API key type
        if self.api_key.startswith("gsk_"):  # Groq API key
            self.llm = ChatOpenAI(
                api_key=self.api_key,
                base_url="https://api.groq.com/openai/v1",
                model="llama-3.1-8b-instant",
                temperature=0.3
            )
        else:  # OpenAI API key
            self.llm = ChatOpenAI(
                api_key=self.api_key, 
                model="gpt-3.5-turbo",
                temperature=0.3
            )
        
        # Bind tools to LLM
        self.llm_with_tools = self.llm.bind_tools([search_wikipedia])
        
        # Build graph
        self._build_graph()
    
    def _build_graph(self):
        """Build the LangGraph workflow with parallel processing"""
        
        def route_input(state: ChatState) -> Literal["process_braille", "process_chat"]:
            """Route based on input type"""
            input_type = state.get('input_type', 'chat')
            if input_type == 'braille':
                return "process_braille"
            else:
                return "process_chat"
        
        def process_braille_node(state: ChatState):
            """Process detected braille strings into readable text"""
            detected_strings = state.get('detected_strings', [])
            
            if not detected_strings:
                return {
                    'processed_text': '',
                    'confidence': 0.0,
                    'messages': [SystemMessage(content="No braille strings detected")]
                }
            
            # Create prompt for text processing
            raw_text = " ".join(detected_strings)
            prompt = f"""
            You are a braille text interpreter. I have braille characters detected by OCR: "{raw_text}"
            
            Task:
            1. Form meaningful keywords/phrases from these characters
            2. Correct obvious OCR errors if any
            3. Extract the main topic or subject matter
            
            Respond with the corrected/interpreted text only.
            """
            
            response = self.llm.invoke([HumanMessage(content=prompt)])
            processed_text = response.content.strip()
            
            # Calculate confidence
            confidence = min(0.9, len([s for s in detected_strings if s.strip()]) / max(1, len(detected_strings)))
            
            return {
                'processed_text': processed_text,
                'confidence': confidence
            }
        
        def generate_braille_explanation(state: ChatState):
            """Generate explanation for braille text with tools if needed"""
            processed_text = state.get('processed_text', '')
            
            if not processed_text:
                return {'explanation': 'No text to explain.'}
            
            try:
                # First attempt: direct explanation
                prompt = f"""
                Braille text detected: "{processed_text}"
                
                Provide a brief 2-3 sentence explanation about this topic.
                Include key concepts and context that would help someone understand what this is about.
                """
                
                response = self.llm.invoke([HumanMessage(content=prompt)])
                explanation = response.content.strip()
                
                # If explanation is too short, enhance with Wikipedia
                if not explanation or len(explanation) < 20:
                    try:
                        wiki_result = search_wikipedia(processed_text)
                        enhanced_prompt = f"""
                        Topic: "{processed_text}"
                        Wikipedia context: {wiki_result[:300]}
                        
                        Based on this information, provide a clear 2-3 sentence explanation about this topic.
                        """
                        response = self.llm.invoke([HumanMessage(content=enhanced_prompt)])
                        explanation = response.content.strip()
                    except:
                        explanation = f"This text appears to discuss {processed_text}. This is a topic that may require further research for detailed understanding."
                
                return {'explanation': explanation or f"Topic identified: {processed_text}"}
                
            except Exception as e:
                return {'explanation': f"This text discusses {processed_text}. Unable to provide detailed explanation due to processing error."}
        
        def process_chat_node(state: ChatState):
            """Handle regular chat messages with tool access"""
            user_message = state.get('user_message', '')
            messages = state.get('messages', [])
            
            if not user_message:
                return {'messages': [SystemMessage(content="No message provided")]}
            
            # Add user message to conversation
            messages.append(HumanMessage(content=user_message))
            
            try:
                # Check if the message might benefit from Wikipedia search
                needs_search = any(keyword in user_message.lower() for keyword in 
                                 ['what is', 'tell me about', 'explain', 'define', 'information about'])
                
                if needs_search:
                    # Use LLM with tools for potentially complex queries
                    response = self.llm_with_tools.invoke(messages)
                else:
                    # Use regular LLM for simple conversation
                    response = self.llm.invoke(messages)
                
                return {'messages': [response]}
                
            except Exception as e:
                return {'messages': [SystemMessage(content=f"I apologize, but I encountered an error: {str(e)}")]}
        
        # Build the graph
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
        
        # Braille processing chain
        self.graph.add_edge("process_braille", "generate_braille_explanation")
        self.graph.add_edge("generate_braille_explanation", END)
        
        # Chat processing chain
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
        
        # Initial state for braille processing
        initial_state = {
            'messages': [SystemMessage(content="Processing braille text detection results.")],
            'detected_strings': detected_strings,
            'processed_text': '',
            'explanation': '',
            'confidence': 0.0,
            'input_type': 'braille'
        }
        
        try:
            config = {'configurable': {'thread_id': f'braille_session_{hash(str(detected_strings))}'}}
            result = self.chatbot.invoke(initial_state, config=config)
            
            return BrailleResult(
                text=result.get('processed_text', ' '.join(detected_strings)),
                explanation=result.get('explanation', 'Could not generate explanation.'),
                confidence=result.get('confidence', 0.5)
            )
        
        except Exception as e:
            return BrailleResult(
                text=' '.join(detected_strings),
                explanation=f"Error processing: {str(e)}",
                confidence=0.3
            )
    
    def chat(self, user_message: str, thread_id: str = "default_chat") -> str:
        """Handle regular chat messages"""
        
        if not user_message.strip():
            return "Please provide a message."
        
        # Initial state for chat processing
        initial_state = {
            'messages': [SystemMessage(content="You are a helpful AI assistant.")],
            'user_message': user_message,
            'input_type': 'chat'
        }
        
        try:
            config = {'configurable': {'thread_id': thread_id}}
            result = self.chatbot.invoke(initial_state, config=config)
            
            # Extract the last message content
            messages = result.get('messages', [])
            if messages:
                return messages[-1].content
            else:
                return "I couldn't process your message."
        
        except Exception as e:
            return f"Error: {str(e)}"
