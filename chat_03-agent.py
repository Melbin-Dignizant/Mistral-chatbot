import base64
import os
import json
import logging
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import requests
import concurrent.futures
from threading import Lock

# External dependencies
from mistralai import Mistral
from dotenv import load_dotenv
from PIL import Image
import pytesseract
from pdf2image import convert_from_path


# ============================================================================
# Configuration and Data Classes
# ============================================================================

class QueryType(Enum):
    """Enumeration for different types of queries."""
    DOCUMENT = "document"
    WEB_SEARCH = "web_search"
    GENERAL = "general"


@dataclass
class ProcessingResult:
    """Result container for processing operations."""
    success: bool
    data: Optional[Any] = None
    error: Optional[str] = None
    metadata: Optional[Dict] = None


@dataclass
class PageAnalysisResult:
    """Result container for individual page analysis."""
    page_number: int
    success: bool
    content: Optional[str] = None
    relevance_score: float = 0.0
    error: Optional[str] = None


@dataclass
class DocumentState:
    """Container for current document state."""
    pdf_path: Optional[str] = None
    image_paths: List[str] = None
    current_page: int = 1
    total_pages: int = 0
    is_loaded: bool = False
    page_summaries: Dict[int, str] = None  # Cache for page summaries

    def __post_init__(self):
        if self.image_paths is None:
            self.image_paths = []
        if self.page_summaries is None:
            self.page_summaries = {}


class Config:
    """Configuration class for the document processor."""
    
    def __init__(self, base_dir: Optional[str] = None):
        self.base_dir = Path(base_dir) if base_dir else Path.cwd()
        self.documents_dir = self.base_dir / "documents"
        self.output_dir = self.documents_dir / "output"
        self.ocr_output_file = self.base_dir / "ocr_extracted_text.txt"
        self.env_file = self.base_dir / ".env"
        
        # AI Configuration
        self.model_name = "mistral-medium-latest"
        self.image_dpi = 300
        self.image_format = "JPEG"
        
        # Multi-page search configuration
        self.max_concurrent_pages = 3  # Limit concurrent page analysis
        self.relevance_threshold = 0.3  # Minimum relevance score to include page
        self.max_pages_in_response = 100  # Maximum pages to include in final response
        
        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.documents_dir.mkdir(parents=True, exist_ok=True)


# ============================================================================
# Utility Classes
# ============================================================================

class Logger:
    """Centralized logging utility."""
    
    def __init__(self, name: str = "DocumentProcessor"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def info(self, message: str):
        self.logger.info(f"✅ {message}")
    
    def error(self, message: str):
        self.logger.error(f"❌ {message}")
    
    def warning(self, message: str):
        self.logger.warning(f"⚠️ {message}")
    
    def processing(self, message: str):
        self.logger.info(f"🔄 {message}")


class FileManager:
    """Handle file operations and validations."""
    
    @staticmethod
    def validate_file_exists(file_path: str) -> bool:
        """Check if file exists."""
        return Path(file_path).exists()
    
    @staticmethod
    def validate_pdf(file_path: str) -> bool:
        """Validate if file is a PDF."""
        return (FileManager.validate_file_exists(file_path) and 
                Path(file_path).suffix.lower() == '.pdf')
    
    @staticmethod
    def clean_directory(directory: Path) -> None:
        """Remove all files from a directory."""
        for file in directory.glob("*"):
            if file.is_file():
                file.unlink()
    
    @staticmethod
    def encode_image_to_base64(image_path: str) -> Optional[str]:
        """Encode image file to base64 string."""
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            return None


# ============================================================================
# Core Processing Classes
# ============================================================================

class PDFProcessor:
    """Handle PDF to image conversion and OCR operations."""
    
    def __init__(self, config: Config, logger: Logger):
        self.config = config
        self.logger = logger
    
    def convert_pdf_to_images(self, pdf_path: str) -> ProcessingResult:
        """Convert PDF pages to images."""
        try:
            self.logger.processing(f"Converting PDF '{Path(pdf_path).name}' to images")
            
            if not FileManager.validate_pdf(pdf_path):
                return ProcessingResult(
                    success=False, 
                    error=f"Invalid PDF file: {pdf_path}"
                )
            
            # Clear previous images
            FileManager.clean_directory(self.config.output_dir)
            
            # Convert PDF to images
            images = convert_from_path(
                pdf_path, 
                dpi=self.config.image_dpi
            )
            
            image_paths = []
            for i, image in enumerate(images):
                output_path = (
                    self.config.output_dir / 
                    f'page_{i+1}.{self.config.image_format.lower()}'
                )
                image.save(output_path, self.config.image_format)
                image_paths.append(str(output_path))
            
            self.logger.info(f"Conversion complete. {len(image_paths)} pages saved")
            
            return ProcessingResult(
                success=True,
                data=image_paths,
                metadata={"total_pages": len(image_paths)}
            )
            
        except Exception as e:
            return ProcessingResult(
                success=False,
                error=f"PDF conversion failed: {str(e)}"
            )
    
    def extract_text_via_ocr(self, image_paths: List[str]) -> ProcessingResult:
        """Extract text from images using OCR."""
        try:
            self.logger.processing("Extracting text via OCR")
            
            with open(self.config.ocr_output_file, "w", encoding="utf-8") as f:
                for image_path in image_paths:
                    try:
                        filename = Path(image_path).name
                        img = Image.open(image_path)
                        text = pytesseract.image_to_string(img)
                        f.write(f"--- Text from {filename} ---\n")
                        f.write(text + "\n\n")
                    except Exception as e:
                        self.logger.warning(f"OCR failed for {image_path}: {e}")
            
            self.logger.info(f"OCR text saved to '{self.config.ocr_output_file}'")
            
            return ProcessingResult(
                success=True,
                data=str(self.config.ocr_output_file)
            )
            
        except Exception as e:
            return ProcessingResult(
                success=False,
                error=f"OCR extraction failed: {str(e)}"
            )


class AIProcessor:
    """Handle AI-based document analysis and web search."""

    def __init__(self, config: Config, logger: Logger):
        self.config = config
        self.logger = logger
        self.client = None
        self.web_search_agent = None
        self._initialize_client()
        self._initialize_web_search_agent()
        self._analysis_lock = Lock()

    def _initialize_client(self) -> None:
        """Initialize the Mistral AI client."""
        try:
            load_dotenv(self.config.env_file)
            api_key = os.getenv("MISTRAL_AI_API_KEY")

            if not api_key:
                raise ValueError("MISTRAL_AI_API_KEY not found in environment variables")

            self.client = Mistral(api_key=api_key)
            self.logger.info("AI client initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize AI client: {e}")
            raise

    def _initialize_web_search_agent(self) -> None:
        """Initialize the web search agent."""
        try:
            self.web_search_agent = self.client.beta.agents.create(
                model=self.config.model_name,
                name="WebSearch Assistant",
                description="Assistant with real-time web search capabilities",
                instructions="When you need current information, use the web_search tool.",
                tools=[{"type": "web_search"}],
                completion_args={"temperature": 0.3, "top_p": 0.95}
            )
            self.logger.info("Web search agent initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize web search agent: {e}")
            self.web_search_agent = None

    def _get_document_analysis_prompt(self) -> str:
        """Return a structured system prompt for document analysis."""
        return '''You are an expert document understanding assistant. When analyzing documents:

1. **Refine the Query**: Rewrite the user's question for clarity and specificity.
2. **Analyze Relevance**: Determine if the question relates to the document content.
3. **Provide Answer**:
   - If related: Analyze the image and provide detailed, accurate answers.
   - If unrelated: State that the information isn't in the document, then use general knowledge.

**Output Format:**
Refined Query: <Clear, specific version of the question>
Reasoning: <Brief explanation of document relevance>
Answer: <Structured answer with document references where applicable>'''

    def _get_relevance_scoring_prompt(self) -> str:
        """Return prompt for scoring page relevance."""
        return '''You are a document relevance analyzer. Your task is to:

1. Analyze the document page image
2. Determine how relevant this page is to the user's question
3. Provide a relevance score from 0.0 to 1.0 where:
   - 0.0 = Completely irrelevant
   - 0.3 = Somewhat related
   - 0.7 = Highly relevant
   - 1.0 = Directly answers the question

**Output Format:**
Relevance Score: <0.0-1.0>
Brief Summary: <One sentence summary of page content>
Key Topics: <Comma-separated list of main topics on this page>'''

    def _get_multi_page_synthesis_prompt(self) -> str:
        """Return prompt for synthesizing multi-page results."""
        return '''You are a document synthesis expert. You will receive:
1. A user's question
2. Analysis results from multiple relevant pages of a document

Your task is to:
1. Synthesize information from all relevant pages
2. Provide a comprehensive answer that references specific pages
3. Organize the information logically
4. Indicate which pages contain which information

**Output Format:**
Comprehensive Answer: <Detailed answer synthesizing all relevant information>
Page References: <List which pages contain what information>
Additional Notes: <Any important context or limitations>'''

    def analyze_single_page(self, question: str, image_path: str, page_number: int) -> PageAnalysisResult:
        """Analyze a single page for relevance and content."""
        try:
            base64_image = FileManager.encode_image_to_base64(image_path)
            if not base64_image:
                return PageAnalysisResult(
                    page_number=page_number,
                    success=False,
                    error="Failed to encode image to base64"
                )

            # First, get relevance score
            relevance_messages = [
                {
                    "role": "system",
                    "content": self._get_relevance_scoring_prompt()
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"Question: {question}\n\nAnalyze this page and provide relevance score:"},
                        {"type": "image_url", "image_url": f"data:image/jpeg;base64,{base64_image}"}
                    ]
                }
            ]

            with self._analysis_lock:
                relevance_response = self.client.chat.complete(
                    model=self.config.model_name, 
                    messages=relevance_messages
                )
            
            relevance_text = relevance_response.choices[0].message.content.strip()
            
            # Extract relevance score
            relevance_score = self._extract_relevance_score(relevance_text)
            
            # If relevance is too low, don't analyze further
            if relevance_score < self.config.relevance_threshold:
                return PageAnalysisResult(
                    page_number=page_number,
                    success=True,
                    content=relevance_text,
                    relevance_score=relevance_score
                )

            # If relevant enough, get detailed analysis
            analysis_messages = [
                {
                    "role": "system",
                    "content": self._get_document_analysis_prompt()
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": question},
                        {"type": "image_url", "image_url": f"data:image/jpeg;base64,{base64_image}"}
                    ]
                }
            ]

            with self._analysis_lock:
                analysis_response = self.client.chat.complete(
                    model=self.config.model_name,
                    messages=analysis_messages
                )
            
            analysis_content = analysis_response.choices[0].message.content.strip()

            return PageAnalysisResult(
                page_number=page_number,
                success=True,
                content=analysis_content,
                relevance_score=relevance_score
            )

        except Exception as e:
            return PageAnalysisResult(
                page_number=page_number,
                success=False,
                error=f"Page analysis failed: {str(e)}"
            )

    def _extract_relevance_score(self, text: str) -> float:
        """Extract relevance score from AI response."""
        try:
            lines = text.split('\n')
            for line in lines:
                if 'relevance score:' in line.lower():
                    # Extract number from the line
                    parts = line.split(':')
                    if len(parts) > 1:
                        score_text = parts[1].strip()
                        # Try to extract float
                        import re
                        score_match = re.search(r'(\d+\.?\d*)', score_text)
                        if score_match:
                            score = float(score_match.group(1))
                            return min(max(score, 0.0), 1.0)  # Clamp between 0 and 1
            return 0.0
        except:
            return 0.0

    def analyze_document_multipage(self, question: str, image_paths: List[str]) -> ProcessingResult:
        """Analyze document across multiple pages to find relevant content."""
        try:
            self.logger.processing(f"Analyzing {len(image_paths)} pages for relevance")
            
            page_results = []
            
            # Analyze pages in batches to avoid overwhelming the API
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.config.max_concurrent_pages) as executor:
                future_to_page = {
                    executor.submit(self.analyze_single_page, question, image_path, i + 1): i + 1
                    for i, image_path in enumerate(image_paths)
                }
                
                for future in concurrent.futures.as_completed(future_to_page):
                    page_num = future_to_page[future]
                    try:
                        result = future.result()
                        page_results.append(result)
                        self.logger.info(f"Page {page_num} analyzed - Relevance: {result.relevance_score:.2f}")
                    except Exception as e:
                        self.logger.error(f"Page {page_num} analysis failed: {e}")

            # Sort by relevance score
            page_results.sort(key=lambda x: x.relevance_score, reverse=True)
            
            # Filter relevant pages
            relevant_pages = [
                page for page in page_results 
                if page.success and page.relevance_score >= self.config.relevance_threshold
            ]
            
            if not relevant_pages:
                self.logger.info("No relevant content found in document, falling back to general knowledge")
                return self.answer_general_question(question)

            # Limit to top relevant pages
            top_relevant_pages = relevant_pages[:self.config.max_pages_in_response]
            
            # Synthesize results from multiple pages
            synthesis_result = self._synthesize_multipage_results(question, top_relevant_pages)
            
            return ProcessingResult(
                success=True,
                data=synthesis_result,
                metadata={
                    "pages_analyzed": len(page_results),
                    "relevant_pages": len(relevant_pages),
                    "pages_used": len(top_relevant_pages),
                    "page_numbers": [p.page_number for p in top_relevant_pages]
                }
            )

        except Exception as e:
            return ProcessingResult(
                success=False,
                error=f"Multi-page document analysis failed: {str(e)}"
            )

    def _synthesize_multipage_results(self, question: str, page_results: List[PageAnalysisResult]) -> str:
        """Synthesize results from multiple pages into a comprehensive answer."""
        try:
            # Prepare synthesis data
            synthesis_data = []
            for page_result in page_results:
                synthesis_data.append(f"Page {page_result.page_number} (Relevance: {page_result.relevance_score:.2f}):\n{page_result.content}\n")
            
            synthesis_input = f"Question: {question}\n\nRelevant Pages Analysis:\n" + "\n".join(synthesis_data)
            
            messages = [
                {
                    "role": "system",
                    "content": self._get_multi_page_synthesis_prompt()
                },
                {
                    "role": "user",
                    "content": synthesis_input
                }
            ]

            response = self.client.chat.complete(model=self.config.model_name, messages=messages)
            synthesized_answer = response.choices[0].message.content.strip()
            
            # Add page reference summary
            page_numbers = [str(p.page_number) for p in page_results]
            page_summary = f"\n\n📄 Information found on pages: {', '.join(page_numbers)}"
            
            return synthesized_answer + page_summary

        except Exception as e:
            # Fallback to simple concatenation if synthesis fails
            self.logger.warning(f"Synthesis failed, using fallback: {e}")
            result = f"Based on analysis of {len(page_results)} relevant pages:\n\n"
            
            for page_result in page_results:
                result += f"**Page {page_result.page_number}:**\n{page_result.content}\n\n"
            
            return result

    def analyze_document_image(self, question: str, image_path: str) -> ProcessingResult:
        """Analyze document image and answer questions (single page - legacy method)."""
        try:
            self.logger.processing(f"Analyzing image '{Path(image_path).name}' with AI")

            base64_image = FileManager.encode_image_to_base64(image_path)
            if not base64_image:
                return ProcessingResult(success=False, error="Failed to encode image to base64")

            messages = [
                {
                    "role": "system",
                    "content": self._get_document_analysis_prompt()
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": question},
                        {"type": "image_url", "image_url": f"data:image/jpeg;base64,{base64_image}"}
                    ]
                }
            ]

            response = self.client.chat.complete(model=self.config.model_name, messages=messages)
            answer = response.choices[0].message.content.strip()

            return ProcessingResult(success=True, data=answer)

        except Exception as e:
            return ProcessingResult(success=False, error=f"Document analysis failed: {str(e)}")

    def perform_web_search(self, question: str) -> ProcessingResult:
        """Perform web search using Mistral's agent API."""
        try:
            if not self.web_search_agent:
                self._initialize_web_search_agent()
                if not self.web_search_agent:
                    return ProcessingResult(
                        success=False,
                        error="Web search agent not available"
                    )

            self.logger.processing("Performing web search via Mistral Agent API")
            
            response = self.client.beta.conversations.start(
                agent_id=self.web_search_agent.id,
                inputs=question
            )
            
            if response.outputs:
                # Get the last output's content
                content = response.outputs[-1].content
                
                # Handle case where content might be a list
                if isinstance(content, list):
                    content = "\n".join([str(item) for item in content])
                
                if content and str(content).strip():
                    return ProcessingResult(
                        success=True,
                        data=str(content).strip()
                    )
            
            return ProcessingResult(
                success=False,
                error="No valid response from web search agent"
            )

        except Exception as e:
            return ProcessingResult(
                success=False,
                error=f"Web search failed: {str(e)}"
            )

    def answer_general_question(self, question: str) -> ProcessingResult:
        """Answer general questions using Mistral's knowledge."""
        try:
            self.logger.processing("Answering with general AI knowledge")

            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful AI assistant. Answer questions clearly and comprehensively using your knowledge."
                },
                {
                    "role": "user",
                    "content": question
                }
            ]

            response = self.client.chat.complete(model=self.config.model_name, messages=messages)
            answer = response.choices[0].message.content.strip()

            return ProcessingResult(success=True, data=answer)

        except Exception as e:
            return ProcessingResult(success=False, error=f"General AI response failed: {str(e)}")


# ============================================================================
# Main Document Processor Class
# ============================================================================

class DocumentProcessor:
    """Main document processing system orchestrator."""
    
    def __init__(self, base_dir: Optional[str] = None):
        self.config = Config(base_dir)
        self.logger = Logger()
        self.pdf_processor = PDFProcessor(self.config, self.logger)
        self.ai_processor = AIProcessor(self.config, self.logger)
        self.document_state = DocumentState()
        
        self.logger.info("DocumentProcessor initialized successfully")
        self.logger.info(f"Output directory: {self.config.output_dir}")
        self.logger.info(f"OCR fallback file: {self.config.ocr_output_file}")
        self.logger.info(f"Multi-page search enabled with {self.config.max_concurrent_pages} concurrent analyses")
    
    def load_document(self, pdf_path: str) -> ProcessingResult:
        """Load and process a PDF document."""
        try:
            # Skip if already loaded
            if (self.document_state.pdf_path == pdf_path and 
                self.document_state.is_loaded):
                self.logger.info(f"Document '{Path(pdf_path).name}' already loaded")
                return ProcessingResult(success=True, data="Already loaded")
            
            # Convert PDF to images
            conversion_result = self.pdf_processor.convert_pdf_to_images(pdf_path)
            if not conversion_result.success:
                return conversion_result
            
            # Extract text via OCR
            ocr_result = self.pdf_processor.extract_text_via_ocr(
                conversion_result.data
            )
            if not ocr_result.success:
                self.logger.warning(f"OCR failed: {ocr_result.error}")
            
            # Update document state
            self.document_state.pdf_path = pdf_path
            self.document_state.image_paths = conversion_result.data
            self.document_state.total_pages = len(conversion_result.data)
            self.document_state.current_page = 1
            self.document_state.is_loaded = True
            self.document_state.page_summaries = {}  # Reset summaries
            
            self.logger.info(f"Successfully loaded '{Path(pdf_path).name}' with {self.document_state.total_pages} pages")
            
            return ProcessingResult(
                success=True,
                data=self.document_state,
                metadata=conversion_result.metadata
            )
            
        except Exception as e:
            self._reset_document_state()
            return ProcessingResult(
                success=False,
                error=f"Document loading failed: {str(e)}"
            )
    
    def set_current_page(self, page_number: int) -> ProcessingResult:
        """Set the current page for document analysis (legacy method)."""
        if not self.document_state.is_loaded:
            return ProcessingResult(
                success=False,
                error="No document loaded"
            )
        
        if not (1 <= page_number <= self.document_state.total_pages):
            return ProcessingResult(
                success=False,
                error=f"Invalid page. Available: 1-{self.document_state.total_pages}"
            )
        
        self.document_state.current_page = page_number
        return ProcessingResult(
            success=True,
            data=f"Switched to page {page_number}"
        )
    
    def answer_question(self, question: str, query_type: QueryType = None) -> ProcessingResult:
        """Route and answer questions based on context."""
        if not question.strip():
            return ProcessingResult(
                success=False,
                error="Empty question provided"
            )
        
        # Auto-detect query type if not specified
        if query_type is None:
            query_type = self._detect_query_type(question)
        
        # Route to appropriate handler
        if query_type == QueryType.DOCUMENT:
            if not self.document_state.is_loaded:
                return ProcessingResult(
                    success=False,
                    error="No document loaded for document-based questions"
                )
            
            # Use multi-page analysis instead of single page
            return self.ai_processor.analyze_document_multipage(
                question, 
                self.document_state.image_paths
            )
        
        elif query_type == QueryType.WEB_SEARCH:
            return self.ai_processor.perform_web_search(question)
        
        else:
            return ProcessingResult(
                success=False,
                error=f"Unsupported query type: {query_type}"
            )
    
    def answer_question_single_page(self, question: str, page_number: int = None) -> ProcessingResult:
        """Answer question from a specific page (legacy method for backward compatibility)."""
        if not self.document_state.is_loaded:
            return ProcessingResult(
                success=False,
                error="No document loaded"
            )
        
        if page_number is None:
            page_number = self.document_state.current_page
        
        if not (1 <= page_number <= self.document_state.total_pages):
            return ProcessingResult(
                success=False,
                error=f"Invalid page. Available: 1-{self.document_state.total_pages}"
            )
        
        image_path = self.document_state.image_paths[page_number - 1]
        return self.ai_processor.analyze_document_image(question, image_path)
    
    def get_status(self) -> Dict[str, Any]:
        """Get current system status."""
        return {
            "document_loaded": self.document_state.is_loaded,
            "document_name": (
                Path(self.document_state.pdf_path).name 
                if self.document_state.pdf_path else None
            ),
            "current_page": self.document_state.current_page,
            "total_pages": self.document_state.total_pages,
            "output_directory": str(self.config.output_dir),
            "ocr_file": str(self.config.ocr_output_file),
            "multi_page_search": True,
            "max_concurrent_analyses": self.config.max_concurrent_pages,
            "relevance_threshold": self.config.relevance_threshold
        }
    
    def _detect_query_type(self, question: str) -> QueryType:
        """Automatically detect the type of query."""
        if self.document_state.is_loaded:
            return QueryType.DOCUMENT
        else:
            return QueryType.WEB_SEARCH
    
    def _reset_document_state(self) -> None:
        """Reset document state on errors."""
        self.document_state = DocumentState()


# ============================================================================
# Command Line Interface
# ============================================================================

class CommandLineInterface:
    """Interactive command-line interface for the document processor."""
    
    def __init__(self):
        self.processor = DocumentProcessor()
        self.logger = Logger("CLI")
        self.commands = {
            'load': self._handle_load,
            'page': self._handle_page,
            'status': self._handle_status,
            'help': self._handle_help,
            'quit': self._handle_quit,
            'exit': self._handle_quit
        }
    
    def run(self) -> None:
        """Run the interactive CLI."""
        self.logger.info("Document & Web Assistant Started")
        self._show_welcome()
        
        while True:
            try:
                status = self.processor.get_status()
                prompt = self._build_prompt(status)
                user_input = input(prompt).strip()
                
                if not user_input:
                    continue
                
                if not self._process_input(user_input):
                    break
                    
            except KeyboardInterrupt:
                self.logger.info("Goodbye!")
                break
            except Exception as e:
                self.logger.error(f"Unexpected error: {e}")
    
    def _build_prompt(self, status: Dict[str, Any]) -> str:
        """Build the input prompt based on current status."""
        if status['document_loaded']:
            doc_info = f"Doc: {status['document_name']} | Page: {status['current_page']}/{status['total_pages']}"
        else:
            doc_info = "No document loaded"
        
        return f"\n[{doc_info}]\n> Enter question, command, or 'help': "
    
    def _process_input(self, user_input: str) -> bool:
        """Process user input and return True to continue, False to quit."""
        words = user_input.split()
        
        # Check for explicit web search command
        if words and words[-1].lower() == 'search':
            question = ' '.join(words[:-1])
            if question:
                result = self.processor.answer_question(question, QueryType.WEB_SEARCH)
                self._display_result(result, "Web Search")
            else:
                self.logger.warning("Please provide a question before 'search'")
            return True
        
        # Check for commands
        if words and words[0].lower() in self.commands:
            return self.commands[words[0].lower()](user_input)
        
        # Regular question
        result = self.processor.answer_question(user_input)
        self._display_result(result, "AI Response")
        return True
    
    def _display_result(self, result: ProcessingResult, title: str) -> None:
        """Display processing results."""
        print(f"\n--- {title} ---")
        if result.success:
            print(result.data)
        else:
            self.logger.error(result.error)
        print("-------------------")
    
    def _handle_load(self, user_input: str) -> bool:
        """Handle load command."""
        parts = user_input.split(' ', 1)
        if len(parts) < 2:
            self.logger.warning("Usage: load <pdf_path>")
            return True
        
        pdf_path = parts[1].strip().strip('"\'')
        result = self.processor.load_document(pdf_path)
        
        if not result.success:
            self.logger.error(result.error)
        
        return True
    
    def _handle_page(self, user_input: str) -> bool:
        """Handle page command."""
        parts = user_input.split(' ', 1)
        if len(parts) < 2:
            self.logger.warning("Usage: page <number>")
            return True
        
        try:
            page_num = int(parts[1].strip())
            result = self.processor.set_current_page(page_num)
            
            if result.success:
                self.logger.info(result.data)
            else:
                self.logger.error(result.error)
                
        except ValueError:
            self.logger.error("Invalid page number format")
        
        return True
    
    def _handle_status(self, user_input: str) -> bool:
        """Handle status command."""
        status = self.processor.get_status()
        print("\n--- System Status ---")
        for key, value in status.items():
            print(f"{key.replace('_', ' ').title()}: {value}")
        print("--------------------")
        return True
    
    def _handle_help(self, user_input: str) -> bool:
        """Handle help command."""
        help_text = """
Available Commands:
- load <pdf_path>     : Load a PDF document
- page <number>       : Switch to specific page
- status              : Show system status
- help                : Show this help message
- quit/exit           : Exit the program

Question Types:
- Regular question    : Analyze current document page
- <question> search   : Perform web search
- Any question when no document loaded performs web search

Examples:
- load "documents/report.pdf"
- page 3
- What is the main topic of this document?
- What is machine learning search
        """
        print(help_text)
        return True
    
    def _handle_quit(self, user_input: str) -> bool:
        """Handle quit command."""
        self.logger.info("Goodbye!")
        return False
    
    def _show_welcome(self) -> None:
        """Show welcome message."""
        print("\n" + "="*50)
        print("   📄 Geometra Document & Web Assistant 2.0")
        print("="*50)
        print("Type 'help' for available commands")


# ============================================================================
# Entry Point and Setup
# ============================================================================

def setup_environment(base_dir: Optional[str] = None) -> None:
    """Set up the environment and required files."""
    config = Config(base_dir)
    
    # Create .env file if it doesn't exist
    if not config.env_file.exists():
        with open(config.env_file, "w") as f:
            f.write("MISTRAL_AI_API_KEY='YOUR_API_KEY_HERE'\n")
        print(f"Created .env file at {config.env_file}")
        print("Please replace 'YOUR_API_KEY_HERE' with your actual Mistral API key.")


def main():
    """Main entry point."""
    setup_environment()
    
    try:
        cli = CommandLineInterface()
        cli.run()
    except Exception as e:
        print(f"Failed to start application: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())