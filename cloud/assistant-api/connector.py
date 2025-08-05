# connector.py - Controller Layer for Braille System
"""
Controller layer that coordinates between the detector and assistant
Following MVC pattern for clean separation of concerns
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import traceback

# Import our models
from detector import BrailleDetector
from assistant import BrailleAssistant, BrailleResult

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DetectionResult:
    """Structured result from detection phase"""
    success: bool
    predictions: List[Dict]
    text_rows: List[str]
    detection_count: int
    error: Optional[str] = None

@dataclass
class ProcessingResult:
    """Structured result from processing phase"""
    success: bool
    text: str
    explanation: str
    confidence: float
    error: Optional[str] = None

@dataclass
class CompleteResult:
    """Combined result from detection + processing"""
    detection: DetectionResult
    processing: ProcessingResult
    timestamp: datetime
    total_time_ms: int

class BrailleController:
    """
    Controller that coordinates between BrailleDetector and BrailleAssistant
    Implements the orchestration logic and error handling
    """
    
    def __init__(self, detector_config: Dict = None, assistant_config: Dict = None):
        """Initialize controller with optional configurations"""
        logger.info("Initializing BrailleController")
        
        # Initialize models with configurations
        try:
            self.detector = BrailleDetector()
            logger.info("✅ BrailleDetector initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize BrailleDetector: {e}")
            raise
        
        try:
            # Pass API key from config if provided
            api_key = assistant_config.get('api_key') if assistant_config else None
            self.assistant = BrailleAssistant(api_key=api_key)
            logger.info("✅ BrailleAssistant initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize BrailleAssistant: {e}")
            raise
        
        # Controller state
        self.processing_stats = {
            'total_requests': 0,
            'successful_detections': 0,
            'successful_processings': 0,
            'errors': 0
        }
        
        logger.info("🎯 BrailleController ready")
    
    def detect_only(self, image_bytes: bytes) -> DetectionResult:
        """
        Perform braille detection only
        
        Args:
            image_bytes: Raw image bytes
            
        Returns:
            DetectionResult with detection information
        """
        logger.info("🔍 Starting detection-only operation")
        start_time = datetime.now()
        
        try:
            self.processing_stats['total_requests'] += 1
            
            # Step 1: Run detection
            logger.info("Step 1: Running braille detection")
            detection_result = self.detector.detect_braille(image_bytes)
            
            if "error" in detection_result:
                logger.error(f"Detection failed: {detection_result['error']}")
                self.processing_stats['errors'] += 1
                return DetectionResult(
                    success=False,
                    predictions=[],
                    text_rows=[],
                    detection_count=0,
                    error=detection_result['error']
                )
            
            # Step 2: Extract predictions
            logger.info("Step 2: Extracting predictions")
            predictions = self.detector.extract_predictions(detection_result)
            logger.info(f"Extracted {len(predictions)} predictions")
            
            # Step 3: Organize into text rows
            logger.info("Step 3: Organizing text rows")
            text_rows = self.detector.organize_text_by_rows(predictions)
            logger.info(f"Organized into {len(text_rows)} text rows")
            
            # Success
            self.processing_stats['successful_detections'] += 1
            
            result = DetectionResult(
                success=True,
                predictions=predictions,
                text_rows=text_rows,
                detection_count=len(predictions)
            )
            
            elapsed = (datetime.now() - start_time).total_seconds() * 1000
            logger.info(f"✅ Detection completed in {elapsed:.0f}ms")
            
            return result
            
        except Exception as e:
            logger.error(f"Detection exception: {str(e)}")
            logger.error(traceback.format_exc())
            self.processing_stats['errors'] += 1
            
            return DetectionResult(
                success=False,
                predictions=[],
                text_rows=[],
                detection_count=0,
                error=f"Detection error: {str(e)}"
            )
    
    def process_only(self, text_rows: List[str]) -> ProcessingResult:
        """
        Process text rows with AI only
        
        Args:
            text_rows: List of detected text rows
            
        Returns:
            ProcessingResult with AI processing results
        """
        logger.info(f"🤖 Starting processing-only operation with {len(text_rows)} rows")
        
        try:
            if not text_rows:
                logger.warning("No text rows to process")
                return ProcessingResult(
                    success=True,
                    text="",
                    explanation="No braille text detected to process.",
                    confidence=0.0
                )
            
            # Process with assistant
            logger.info("Processing with BrailleAssistant")
            braille_result = self.assistant.process_braille_strings(text_rows)
            
            self.processing_stats['successful_processings'] += 1
            
            result = ProcessingResult(
                success=True,
                text=braille_result.text,
                explanation=braille_result.explanation,
                confidence=braille_result.confidence
            )
            
            logger.info("✅ Processing completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Processing exception: {str(e)}")
            logger.error(traceback.format_exc())
            self.processing_stats['errors'] += 1
            
            return ProcessingResult(
                success=False,
                text="",
                explanation=f"Processing error: {str(e)}",
                confidence=0.0,
                error=str(e)
            )
    
    def detect_and_process(self, image_bytes: bytes) -> CompleteResult:
        """
        Complete pipeline: detect braille + process with AI
        
        Args:
            image_bytes: Raw image bytes
            
        Returns:
            CompleteResult with both detection and processing results
        """
        logger.info("🚀 Starting complete detect and process operation")
        start_time = datetime.now()
        
        # Step 1: Detection
        detection_result = self.detect_only(image_bytes)
        
        # Step 2: Processing (even if detection had issues)
        if detection_result.success and detection_result.text_rows:
            processing_result = self.process_only(detection_result.text_rows)
        else:
            # Create appropriate processing result based on detection outcome
            if detection_result.error:
                processing_result = ProcessingResult(
                    success=False,
                    text="",
                    explanation=f"Cannot process: {detection_result.error}",
                    confidence=0.0,
                    error=detection_result.error
                )
            else:
                processing_result = ProcessingResult(
                    success=True,
                    text="",
                    explanation="No braille characters detected in the image.",
                    confidence=0.0
                )
        
        # Calculate total time
        total_time = int((datetime.now() - start_time).total_seconds() * 1000)
        
        # Create complete result
        complete_result = CompleteResult(
            detection=detection_result,
            processing=processing_result,
            timestamp=datetime.now(),
            total_time_ms=total_time
        )
        
        logger.info(f"✅ Complete operation finished in {total_time}ms")
        return complete_result
    
    def chat_with_context(self, message: str, context: Dict = None, thread_id: str = "default") -> str:
        """
        Chat with optional braille context
        
        Args:
            message: User message
            context: Optional context from previous detection/processing
            thread_id: Conversation thread ID
            
        Returns:
            Assistant response
        """
        logger.info(f"💬 Chat request: '{message[:50]}...' (thread: {thread_id})")
        
        try:
            # Add context to message if provided
            enhanced_message = message
            if context and context.get('detected_text'):
                enhanced_message = f"Context: Recently detected braille text: '{context['detected_text']}'\n\nUser question: {message}"
                logger.info("Added braille context to chat message")
            
            response = self.assistant.chat(enhanced_message, thread_id)
            logger.info("✅ Chat response generated")
            return response
            
        except Exception as e:
            logger.error(f"Chat error: {str(e)}")
            return f"Sorry, I encountered an error: {str(e)}"
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        try:
            assistant_status = self.assistant.get_status()
            
            return {
                'controller': {
                    'status': 'healthy',
                    'stats': self.processing_stats.copy()
                },
                'detector': {
                    'api_configured': bool(self.detector.api_key),
                    'workspace': self.detector.workspace_name,
                    'model_version': self.detector.model_version
                },
                'assistant': assistant_status,
                'overall_health': all([
                    bool(self.detector.api_key),  # Detection capability
                    True  # Assistant always works (fallback mode)
                ])
            }
        except Exception as e:
            logger.error(f"Status check error: {str(e)}")
            return {
                'controller': {'status': 'error', 'error': str(e)},
                'detector': {'status': 'unknown'},
                'assistant': {'status': 'unknown'},
                'overall_health': False
            }
    
    def reset_stats(self):
        """Reset processing statistics"""
        logger.info("🔄 Resetting processing statistics")
        self.processing_stats = {
            'total_requests': 0,
            'successful_detections': 0,
            'successful_processings': 0,
            'errors': 0
        }
    
    def process_text_only(self, text: str, task: str = "explain") -> str:
        """
        Process arbitrary text with AI (not braille-specific)
        
        Args:
            text: Text to process
            task: Processing task (explain, summarize, correct, etc.)
            
        Returns:
            Processed text result
        """
        logger.info(f"📝 Text processing request: task='{task}', text='{text[:50]}...'")
        
        try:
            result = self.assistant.process_text(text, task)
            logger.info("✅ Text processing completed")
            return result
        except Exception as e:
            logger.error(f"Text processing error: {str(e)}")
            return f"Processing error: {str(e)}"

# Factory function for easy initialization
def create_braille_controller(detector_config: Dict = None, assistant_config: Dict = None) -> BrailleController:
    """
    Factory function to create and configure a BrailleController
    
    Args:
        detector_config: Configuration for detector (API keys, etc.)
        assistant_config: Configuration for assistant (API keys, etc.)
        
    Returns:
        Configured BrailleController instance
    """
    logger.info("🏭 Creating BrailleController via factory")
    return BrailleController(detector_config, assistant_config)

# Example usage and testing
if __name__ == "__main__":
    # Example usage
    print("Testing BrailleController...")
    
    try:
        # Create controller
        controller = create_braille_controller()
        
        # Check status
        status = controller.get_system_status()
        print(f"System Status: {status}")
        
        # Test chat
        response = controller.chat_with_context("Hello, how does braille detection work?")
        print(f"Chat Response: {response}")
        
        print("✅ Controller test completed successfully")
        
    except Exception as e:
        print(f"❌ Controller test failed: {e}")
        traceback.print_exc()