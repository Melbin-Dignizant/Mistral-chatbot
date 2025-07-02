import base64
import os
import json
import logging
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import requests
import concurrent.futures
from threading import Lock
from datetime import datetime

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
    MULTI_DOCUMENT = "multi_document"
    CONVERSATION = "conversation"


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
    document_id: str
    success: bool
    content: Optional[str] = None
    relevance_score: float = 0.0
    error: Optional[str] = None


@dataclass
class ConversationMessage:
    """Container for conversation messages."""
    role: str  # 'user' or 'assistant'
    content: str
    timestamp: datetime
    document_context: Optional[List[str]] = None  # List of document IDs referenced
    metadata: Optional[Dict] = None

    def __post_init__(self):
        if self.document_context is None:
            self.document_context = []
        if self.metadata is None:
            self.metadata = {}


@dataclass
class DocumentInfo:
    """Container for individual document information."""
    document_id: str
    pdf_path: str
    name: str
    image_paths: List[str] = field(default_factory=list)
    total_pages: int = 0
    is_loaded: bool = False
    page_summaries: Dict[int, str] = field(default_factory=dict)
    load_timestamp: Optional[datetime] = None
    tags: List[str] = field(default_factory=list)  # User-defined tags for organization


@dataclass
class MultiDocumentState:
    """Container for multi-document state."""
    documents: Dict[str, DocumentInfo] = field(default_factory=dict)
    active_document_id: Optional[str] = None
    conversation_history: List[ConversationMessage] = field(default_factory=list)
    global_context: Dict[str, Any] = field(default_factory=dict)
    
    def get_active_document(self) -> Optional[DocumentInfo]:
        """Get the currently active document."""
        if self.active_document_id and self.active_document_id in self.documents:
            return self.documents[self.active_document_id]
        return None
    
    def get_loaded_documents(self) -> List[DocumentInfo]:
        """Get all loaded documents."""
        return [doc for doc in self.documents.values() if doc.is_loaded]
    
    def get_document_by_name(self, name: str) -> Optional[DocumentInfo]:
        """Find document by name (case-insensitive partial match)."""
        name_lower = name.lower()
        for doc in self.documents.values():
            if name_lower in doc.name.lower():
                return doc
        return None


class Config:
    """Configuration class for the document processor."""
    
    def __init__(self, base_dir: Optional[str] = None):
        self.base_dir = Path(base_dir) if base_dir else Path.cwd()
        self.documents_dir = self.base_dir / "documents"
        self.output_dir = self.documents_dir / "output"
        self.conversation_dir = self.base_dir / "conversations"
        self.ocr_output_file = self.base_dir / "ocr_extracted_text.txt"
        self.env_file = self.base_dir / ".env"
        
        # AI Configuration
        self.model_name = "mistral-medium-latest"
        self.image_dpi = 300
        self.image_format = "JPEG"
        
        # Multi-page search configuration
        self.max_concurrent_pages = 3
        self.relevance_threshold = 0.3
        self.max_pages_in_response = 100
        
        # Conversation configuration
        self.max_conversation_history = 50  # Maximum messages to keep in memory
        self.auto_save_conversations = True
        
        # Multi-document configuration
        self.max_concurrent_documents = 5
        self.cross_document_search = True
        
        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.documents_dir.mkdir(parents=True, exist_ok=True)
        self.conversation_dir.mkdir(parents=True, exist_ok=True)


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
    
    @staticmethod
    def generate_document_id(pdf_path: str) -> str:
        """Generate a unique document ID from PDF path."""
        return f"doc_{hash(pdf_path) % 10000:04d}_{Path(pdf_path).stem}"


class ConversationManager:
    """Manage conversation history and context."""
    
    def __init__(self, config: Config, logger: Logger):
        self.config = config
        self.logger = logger
    
    def add_message(self, state: MultiDocumentState, role: str, content: str, 
                   document_context: Optional[List[str]] = None, 
                   metadata: Optional[Dict] = None) -> None:
        """Add a message to conversation history."""
        message = ConversationMessage(
            role=role,
            content=content,
            timestamp=datetime.now(),
            document_context=document_context or [],
            metadata=metadata or {}
        )
        
        state.conversation_history.append(message)
        
        # Trim history if too long
        if len(state.conversation_history) > self.config.max_conversation_history:
            state.conversation_history = state.conversation_history[-self.config.max_conversation_history:]
        
        # Auto-save if enabled
        if self.config.auto_save_conversations:
            self.save_conversation(state)
    
    def get_conversation_context(self, state: MultiDocumentState, 
                               last_n_messages: int = 5) -> str:
        """Get recent conversation context for AI."""
        recent_messages = state.conversation_history[-last_n_messages:]
        context_parts = []
        
        for msg in recent_messages:
            role_prefix = "User" if msg.role == "user" else "Assistant"
            doc_context = ""
            if msg.document_context:
                doc_names = [state.documents[doc_id].name for doc_id in msg.document_context 
                           if doc_id in state.documents]
                if doc_names:
                    doc_context = f" [Documents: {', '.join(doc_names)}]"
            
            context_parts.append(f"{role_prefix}{doc_context}: {msg.content}")
        
        return "\n".join(context_parts)
    
    def save_conversation(self, state: MultiDocumentState) -> None:
        """Save conversation to file."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"conversation_{timestamp}.json"
            filepath = self.config.conversation_dir / filename
            
            # Convert to serializable format
            conversation_data = {
                "timestamp": timestamp,
                "documents": {doc_id: {
                    "name": doc.name,
                    "pdf_path": doc.pdf_path,
                    "total_pages": doc.total_pages,
                    "tags": doc.tags
                } for doc_id, doc in state.documents.items()},
                "messages": [{
                    "role": msg.role,
                    "content": msg.content,
                    "timestamp": msg.timestamp.isoformat(),
                    "document_context": msg.document_context,
                    "metadata": msg.metadata
                } for msg in state.conversation_history]
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(conversation_data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            self.logger.warning(f"Failed to save conversation: {e}")


# ============================================================================
# Core Processing Classes
# ============================================================================

class PDFProcessor:
    """Handle PDF to image conversion and OCR operations."""
    
    def __init__(self, config: Config, logger: Logger):
        self.config = config
        self.logger = logger
    
    def convert_pdf_to_images(self, pdf_path: str, document_id: str) -> ProcessingResult:
        """Convert PDF pages to images."""
        try:
            self.logger.processing(f"Converting PDF '{Path(pdf_path).name}' to images")
            
            if not FileManager.validate_pdf(pdf_path):
                return ProcessingResult(
                    success=False, 
                    error=f"Invalid PDF file: {pdf_path}"
                )
            
            # Create document-specific output directory
            doc_output_dir = self.config.output_dir / document_id
            doc_output_dir.mkdir(parents=True, exist_ok=True)
            
            # Clear previous images for this document
            FileManager.clean_directory(doc_output_dir)
            
            # Convert PDF to images
            images = convert_from_path(
                pdf_path, 
                dpi=self.config.image_dpi
            )
            
            image_paths = []
            for i, image in enumerate(images):
                output_path = (
                    doc_output_dir / 
                    f'page_{i+1}.{self.config.image_format.lower()}'
                )
                image.save(output_path, self.config.image_format)
                image_paths.append(str(output_path))
            
            self.logger.info(f"Conversion complete. {len(image_paths)} pages saved for {document_id}")
            
            return ProcessingResult(
                success=True,
                data=image_paths,
                metadata={"total_pages": len(image_paths), "document_id": document_id}
            )
            
        except Exception as e:
            return ProcessingResult(
                success=False,
                error=f"PDF conversion failed: {str(e)}"
            )
    
    def extract_text_via_ocr(self, image_paths: List[str], document_id: str) -> ProcessingResult:
        """Extract text from images using OCR."""
        try:
            self.logger.processing(f"Extracting text via OCR for {document_id}")
            
            ocr_file = self.config.output_dir / document_id / "ocr_text.txt"
            
            with open(ocr_file, "w", encoding="utf-8") as f:
                for image_path in image_paths:
                    try:
                        filename = Path(image_path).name
                        img = Image.open(image_path)
                        text = pytesseract.image_to_string(img)
                        f.write(f"--- Text from {filename} ---\n")
                        f.write(text + "\n\n")
                    except Exception as e:
                        self.logger.warning(f"OCR failed for {image_path}: {e}")
            
            self.logger.info(f"OCR text saved for {document_id}")
            
            return ProcessingResult(
                success=True,
                data=str(ocr_file)
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
        self._initialize_client()
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

    def _get_conversation_aware_prompt(self, conversation_context: str = "") -> str:
        """Return conversation-aware system prompt."""
        base_prompt = '''You are an expert document understanding assistant with conversation memory. 

Key capabilities:
1. **Document Analysis**: Analyze document images and provide detailed answers
2. **Multi-Document Context**: Work with multiple documents simultaneously
3. **Conversation Continuity**: Remember previous interactions and build upon them
4. **Cross-Document Insights**: Draw connections between different documents

When analyzing documents:
- Reference specific pages and documents by name
- Build upon previous conversation context
- Highlight connections between documents when relevant
- Provide comprehensive answers that consider all available context

**Current Conversation Context:**
{context}

**Response Format:**
- Start with brief acknowledgment of context if relevant
- Provide detailed analysis referencing specific documents/pages
- Include cross-document insights when applicable
- End with follow-up suggestions if appropriate'''

        return base_prompt.format(context=conversation_context if conversation_context else "No previous context")

    def analyze_multi_document_query(self, question: str, state: MultiDocumentState, 
                                   conversation_context: str = "") -> ProcessingResult:
        """Analyze query across multiple documents."""
        try:
            loaded_docs = state.get_loaded_documents()
            if not loaded_docs:
                return ProcessingResult(
                    success=False,
                    error="No documents loaded for analysis"
                )

            self.logger.processing(f"Analyzing query across {len(loaded_docs)} documents")

            # Collect all relevant pages from all documents
            all_page_results = []
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.config.max_concurrent_pages) as executor:
                future_to_page = {}
                
                for doc in loaded_docs:
                    for i, image_path in enumerate(doc.image_paths):
                        future = executor.submit(
                            self.analyze_single_page, 
                            question, 
                            image_path, 
                            i + 1, 
                            doc.document_id
                        )
                        future_to_page[future] = (doc.document_id, i + 1)
                
                for future in concurrent.futures.as_completed(future_to_page):
                    doc_id, page_num = future_to_page[future]
                    try:
                        result = future.result()
                        all_page_results.append(result)
                        self.logger.info(f"Analyzed {doc_id} page {page_num} - Relevance: {result.relevance_score:.2f}")
                    except Exception as e:
                        self.logger.error(f"Analysis failed for {doc_id} page {page_num}: {e}")

            # Sort by relevance and filter
            all_page_results.sort(key=lambda x: x.relevance_score, reverse=True)
            relevant_pages = [
                page for page in all_page_results 
                if page.success and page.relevance_score >= self.config.relevance_threshold
            ]

            if not relevant_pages:
                return ProcessingResult(
                    success=True,
                    data="No relevant content found across the loaded documents for your question.",
                    metadata={"documents_searched": len(loaded_docs), "relevant_pages": 0}
                )

            # Group by document for better organization
            pages_by_doc = {}
            for page in relevant_pages[:self.config.max_pages_in_response]:
                if page.document_id not in pages_by_doc:
                    pages_by_doc[page.document_id] = []
                pages_by_doc[page.document_id].append(page)

            # Synthesize multi-document results
            synthesis_result = self._synthesize_multi_document_results(
                question, pages_by_doc, state, conversation_context
            )

            # Track which documents were referenced
            referenced_docs = list(pages_by_doc.keys())

            return ProcessingResult(
                success=True,
                data=synthesis_result,
                metadata={
                    "documents_searched": len(loaded_docs),
                    "relevant_pages": len(relevant_pages),
                    "referenced_documents": referenced_docs,
                    "pages_by_document": {doc_id: len(pages) for doc_id, pages in pages_by_doc.items()}
                }
            )

        except Exception as e:
            return ProcessingResult(
                success=False,
                error=f"Multi-document analysis failed: {str(e)}"
            )

    def analyze_single_page(self, question: str, image_path: str, page_number: int, 
                          document_id: str) -> PageAnalysisResult:
        """Analyze a single page for relevance and content."""
        try:
            base64_image = FileManager.encode_image_to_base64(image_path)
            if not base64_image:
                return PageAnalysisResult(
                    page_number=page_number,
                    document_id=document_id,
                    success=False,
                    error="Failed to encode image to base64"
                )

            # Get relevance score first
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
            relevance_score = self._extract_relevance_score(relevance_text)
            
            if relevance_score < self.config.relevance_threshold:
                return PageAnalysisResult(
                    page_number=page_number,
                    document_id=document_id,
                    success=True,
                    content=relevance_text,
                    relevance_score=relevance_score
                )

            # Get detailed analysis for relevant pages
            analysis_messages = [
                {
                    "role": "system",
                    "content": self._get_document_analysis_prompt()
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"Document ID: {document_id}, Page: {page_number}\nQuestion: {question}"},
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
                document_id=document_id,
                success=True,
                content=analysis_content,
                relevance_score=relevance_score
            )

        except Exception as e:
            return PageAnalysisResult(
                page_number=page_number,
                document_id=document_id,
                success=False,
                error=f"Page analysis failed: {str(e)}"
            )

    def _synthesize_multi_document_results(self, question: str, pages_by_doc: Dict[str, List[PageAnalysisResult]], 
                                         state: MultiDocumentState, conversation_context: str = "") -> str:
        """Synthesize results from multiple documents."""
        try:
            # Prepare synthesis data organized by document
            synthesis_parts = []
            synthesis_parts.append(f"Question: {question}")
            
            if conversation_context:
                synthesis_parts.append(f"Previous Context: {conversation_context}")
            
            for doc_id, pages in pages_by_doc.items():
                doc_info = state.documents[doc_id]
                synthesis_parts.append(f"\n=== Document: {doc_info.name} ===")
                
                for page_result in pages:
                    synthesis_parts.append(f"Page {page_result.page_number} (Relevance: {page_result.relevance_score:.2f}):")
                    synthesis_parts.append(page_result.content)
                    synthesis_parts.append("")

            synthesis_input = "\n".join(synthesis_parts)

            messages = [
                {
                    "role": "system",
                    "content": self._get_conversation_aware_prompt(conversation_context)
                },
                {
                    "role": "user",
                    "content": synthesis_input
                }
            ]

            response = self.client.chat.complete(model=self.config.model_name, messages=messages)
            synthesized_answer = response.choices[0].message.content.strip()

            # Add document reference summary
            doc_summary = []
            for doc_id, pages in pages_by_doc.items():
                doc_name = state.documents[doc_id].name
                page_numbers = [str(p.page_number) for p in pages]
                doc_summary.append(f"{doc_name} (pages: {', '.join(page_numbers)})")

            reference_summary = f"\n\n📚 Information found in: {' | '.join(doc_summary)}"
            
            return synthesized_answer + reference_summary

        except Exception as e:
            self.logger.warning(f"Synthesis failed, using fallback: {e}")
            # Fallback to simple organization
            result_parts = [f"Analysis of {len(pages_by_doc)} documents:\n"]
            
            for doc_id, pages in pages_by_doc.items():
                doc_name = state.documents[doc_id].name
                result_parts.append(f"**{doc_name}:**")
                for page in pages:
                    result_parts.append(f"  Page {page.page_number}: {page.content}")
                result_parts.append("")
            
            return "\n".join(result_parts)

    # Include other necessary methods from the original AIProcessor
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

    def _extract_relevance_score(self, text: str) -> float:
        """Extract relevance score from AI response."""
        try:
            lines = text.split('\n')
            for line in lines:
                if 'relevance score:' in line.lower():
                    parts = line.split(':')
                    if len(parts) > 1:
                        score_text = parts[1].strip()
                        import re
                        score_match = re.search(r'(\d+\.?\d*)', score_text)
                        if score_match:
                            score = float(score_match.group(1))
                            return min(max(score, 0.0), 1.0)
            return 0.0
        except:
            return 0.0

    def perform_web_search(self, question: str) -> ProcessingResult:
        """Perform web search with multiple fallback strategies."""
        try:
            agents_result = self._try_agents_api(question)
            if agents_result.success:
                return agents_result
            return self._try_beta_conversations(question)
        except Exception as e:
            return ProcessingResult(success=False, error=f"Web search failed: {str(e)}")

    def _try_agents_api(self, question: str) -> ProcessingResult:
        """Try Mistral Agents API."""
        self.logger.processing("Trying Agents API for web search")
        try:
            url = "https://api.mistral.ai/v1/agents/completions"
            headers = {
                "Authorization": f"Bearer {os.getenv('MISTRAL_AI_API_KEY')}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": self.config.model_name,
                "messages": [{"role": "user", "content": question}],
                "tool": "web_search"
            }
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            data = response.json()
            if "choices" in data and data["choices"]:
                content = data["choices"][0].get("message", {}).get("content", "")
                if content.strip():
                    return ProcessingResult(success=True, data=content.strip())
            return ProcessingResult(success=False, error="No valid response from Agents API")
        except Exception as e:
            return ProcessingResult(success=False, error=f"Agents API failed: {str(e)}")

    def _try_beta_conversations(self, question: str) -> ProcessingResult:
        """Fallback to beta conversations API."""
        self.logger.processing("Trying beta conversations API with streaming")
        try:
            stream = self.client.beta.conversations.start_stream(
                inputs=[{"role": "user", "content": question}],
                model=self.config.model_name,
                tools=[{"type": "web_search"}],
            )
            full_response = ""
            for event in stream:
                if hasattr(event, 'delta') and event.delta and event.delta.content:
                    full_response += event.delta.content
                elif hasattr(event, 'content') and event.content:
                    full_response += event.content
            if full_response.strip():
                return ProcessingResult(success=True, data=full_response.strip())
            return ProcessingResult(success=False, error="No valid response from beta conversations API")
        except Exception as e:
            return ProcessingResult(success=False, error=f"Beta conversations API failed: {str(e)}")

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

class MultiDocumentProcessor:
    """Main multi-document processing system orchestrator."""
    
    def __init__(self, base_dir: Optional[str] = None):
        self.config = Config(base_dir)
        self.logger = Logger()
        self.pdf_processor = PDFProcessor(self.config, self.logger)
        self.ai_processor = AIProcessor(self.config, self.logger)
        self.conversation_manager = ConversationManager(self.config, self.logger)
        self.state = MultiDocumentState()
        
        self.logger.info("Multi-Document Processor initialized")
    
    def load_document(self, pdf_path: str, tags: Optional[List[str]] = None) -> ProcessingResult:
        """Load a PDF document into the system."""
        try:
            if not FileManager.validate_pdf(pdf_path):
                return ProcessingResult(
                    success=False,
                    error=f"Invalid PDF file: {pdf_path}"
                )
            
            document_id = FileManager.generate_document_id(pdf_path)
            document_name = Path(pdf_path).stem
            
            # Check if document already loaded
            if document_id in self.state.documents:
                self.logger.info(f"Document '{document_name}' already loaded")
                self.state.active_document_id = document_id
                return ProcessingResult(
                    success=True,
                    data=f"Document '{document_name}' is already loaded and set as active",
                    metadata={"document_id": document_id, "action": "reactivated"}
                )
            
            # Convert PDF to images
            conversion_result = self.pdf_processor.convert_pdf_to_images(pdf_path, document_id)
            if not conversion_result.success:
                return conversion_result
            
            # Extract text via OCR
            ocr_result = self.pdf_processor.extract_text_via_ocr(
                conversion_result.data, document_id
            )
            
            # Create document info
            doc_info = DocumentInfo(
                document_id=document_id,
                pdf_path=pdf_path,
                name=document_name,
                image_paths=conversion_result.data,
                total_pages=len(conversion_result.data),
                is_loaded=True,
                load_timestamp=datetime.now(),
                tags=tags or []
            )
            
            # Add to state
            self.state.documents[document_id] = doc_info
            self.state.active_document_id = document_id
            
            self.logger.info(f"Document '{document_name}' loaded successfully ({doc_info.total_pages} pages)")
            
            return ProcessingResult(
                success=True,
                data=f"Document '{document_name}' loaded successfully with {doc_info.total_pages} pages",
                metadata={
                    "document_id": document_id,
                    "total_pages": doc_info.total_pages,
                    "ocr_file": ocr_result.data if ocr_result.success else None
                }
            )
            
        except Exception as e:
            return ProcessingResult(
                success=False,
                error=f"Document loading failed: {str(e)}"
            )
    
    def unload_document(self, document_identifier: str) -> ProcessingResult:
        """Unload a document from the system."""
        try:
            # Find document by ID or name
            doc_to_remove = None
            if document_identifier in self.state.documents:
                doc_to_remove = self.state.documents[document_identifier]
            else:
                doc_to_remove = self.state.get_document_by_name(document_identifier)
            
            if not doc_to_remove:
                return ProcessingResult(
                    success=False,
                    error=f"Document '{document_identifier}' not found"
                )
            
            document_id = doc_to_remove.document_id
            document_name = doc_to_remove.name
            
            # Clean up files
            doc_output_dir = self.config.output_dir / document_id
            if doc_output_dir.exists():
                FileManager.clean_directory(doc_output_dir)
                doc_output_dir.rmdir()
            
            # Remove from state
            del self.state.documents[document_id]
            
            # Update active document if needed
            if self.state.active_document_id == document_id:
                remaining_docs = list(self.state.documents.keys())
                self.state.active_document_id = remaining_docs[0] if remaining_docs else None
            
            self.logger.info(f"Document '{document_name}' unloaded successfully")
            
            return ProcessingResult(
                success=True,
                data=f"Document '{document_name}' unloaded successfully"
            )
            
        except Exception as e:
            return ProcessingResult(
                success=False,
                error=f"Document unloading failed: {str(e)}"
            )
    
    def set_active_document(self, document_identifier: str) -> ProcessingResult:
        """Set the active document for single-document operations."""
        try:
            # Find document by ID or name
            doc = None
            if document_identifier in self.state.documents:
                doc = self.state.documents[document_identifier]
            else:
                doc = self.state.get_document_by_name(document_identifier)
            
            if not doc:
                return ProcessingResult(
                    success=False,
                    error=f"Document '{document_identifier}' not found"
                )
            
            self.state.active_document_id = doc.document_id
            
            return ProcessingResult(
                success=True,
                data=f"Active document set to '{doc.name}'"
            )
            
        except Exception as e:
            return ProcessingResult(
                success=False,
                error=f"Setting active document failed: {str(e)}"
            )
    
    def list_documents(self) -> ProcessingResult:
        """List all loaded documents."""
        try:
            if not self.state.documents:
                return ProcessingResult(
                    success=True,
                    data="No documents currently loaded"
                )
            
            doc_list = []
            for doc_id, doc in self.state.documents.items():
                status = "ACTIVE" if doc_id == self.state.active_document_id else "LOADED"
                tags_str = f" [Tags: {', '.join(doc.tags)}]" if doc.tags else ""
                doc_list.append(
                    f"• {doc.name} ({doc.total_pages} pages) - {status}{tags_str}"
                )
            
            result = "Loaded Documents:\n" + "\n".join(doc_list)
            
            return ProcessingResult(
                success=True,
                data=result,
                metadata={
                    "total_documents": len(self.state.documents),
                    "active_document": self.state.active_document_id
                }
            )
            
        except Exception as e:
            return ProcessingResult(
                success=False,
                error=f"Listing documents failed: {str(e)}"
            )
    
    def query(self, question: str, query_type: Optional[QueryType] = None) -> ProcessingResult:
        """Process a query with automatic type detection."""
        try:
            # Determine query type if not specified
            if query_type is None:
                query_type = self._detect_query_type(question)
            
            self.logger.info(f"Processing {query_type.value} query: {question[:50]}...")
            
            # Get conversation context
            conversation_context = self.conversation_manager.get_conversation_context(self.state)
            
            # Add user message to conversation
            active_docs = [self.state.active_document_id] if self.state.active_document_id else []
            if query_type in [QueryType.MULTI_DOCUMENT, QueryType.DOCUMENT]:
                active_docs = [doc.document_id for doc in self.state.get_loaded_documents()]
            
            self.conversation_manager.add_message(
                self.state, "user", question, 
                document_context=active_docs
            )
            
            # Process query based on type
            result = None
            if query_type == QueryType.WEB_SEARCH:
                result = self.ai_processor.perform_web_search(question)
            elif query_type == QueryType.DOCUMENT:
                result = self._process_single_document_query(question, conversation_context)
            elif query_type == QueryType.MULTI_DOCUMENT:
                result = self.ai_processor.analyze_multi_document_query(
                    question, self.state, conversation_context
                )
            else:  # GENERAL or CONVERSATION
                result = self.ai_processor.answer_general_question(question)
            
            if result.success:
                # Add assistant response to conversation
                self.conversation_manager.add_message(
                    self.state, "assistant", result.data,
                    document_context=active_docs,
                    metadata=result.metadata
                )
            
            return result
            
        except Exception as e:
            return ProcessingResult(
                success=False,
                error=f"Query processing failed: {str(e)}"
            )
    
    def _detect_query_type(self, question: str) -> QueryType:
        """Automatically detect the type of query."""
        question_lower = question.lower()
        
        # Web search indicators
        web_indicators = [
            'latest', 'recent', 'current', 'today', 'news', 'update',
            'search for', 'look up', 'find online', 'web search',
            'what happened', 'current events'
        ]
        
        if any(indicator in question_lower for indicator in web_indicators):
            return QueryType.WEB_SEARCH
        
        # Multi-document indicators
        multi_doc_indicators = [
            'compare', 'contrast', 'between', 'across documents',
            'all documents', 'multiple', 'documents', 'both'
        ]
        
        loaded_doc_count = len(self.state.get_loaded_documents())
        if loaded_doc_count > 1 and any(indicator in question_lower for indicator in multi_doc_indicators):
            return QueryType.MULTI_DOCUMENT
        
        # Document query if documents are loaded
        if loaded_doc_count > 0:
            return QueryType.DOCUMENT if loaded_doc_count == 1 else QueryType.MULTI_DOCUMENT
        
        # Default to general
        return QueryType.GENERAL
    
    def _process_single_document_query(self, question: str, conversation_context: str = "") -> ProcessingResult:
        """Process a query against the active document."""
        active_doc = self.state.get_active_document()
        if not active_doc:
            return ProcessingResult(
                success=False,
                error="No active document set. Use 'load_document()' or 'set_active_document()'"
            )
        
        # Create a temporary multi-document state with just the active document
        temp_state = MultiDocumentState()
        temp_state.documents[active_doc.document_id] = active_doc
        
        return self.ai_processor.analyze_multi_document_query(
            question, temp_state, conversation_context
        )
    
    def get_conversation_history(self, last_n: int = 10) -> ProcessingResult:
        """Get recent conversation history."""
        try:
            recent_messages = self.state.conversation_history[-last_n:]
            
            if not recent_messages:
                return ProcessingResult(
                    success=True,
                    data="No conversation history available"
                )
            
            history_parts = []
            for i, msg in enumerate(recent_messages, 1):
                role = "You" if msg.role == "user" else "Assistant"
                timestamp = msg.timestamp.strftime("%H:%M:%S")
                
                doc_context = ""
                if msg.document_context:
                    doc_names = [
                        self.state.documents[doc_id].name 
                        for doc_id in msg.document_context 
                        if doc_id in self.state.documents
                    ]
                    if doc_names:
                        doc_context = f" [{', '.join(doc_names)}]"
                
                history_parts.append(f"{i}. [{timestamp}] {role}{doc_context}: {msg.content}")
            
            return ProcessingResult(
                success=True,
                data="\n\n".join(history_parts),
                metadata={"message_count": len(recent_messages)}
            )
            
        except Exception as e:
            return ProcessingResult(
                success=False,
                error=f"Getting conversation history failed: {str(e)}"
            )
    
    def clear_conversation(self) -> ProcessingResult:
        """Clear conversation history."""
        try:
            message_count = len(self.state.conversation_history)
            self.state.conversation_history.clear()
            
            return ProcessingResult(
                success=True,
                data=f"Cleared {message_count} messages from conversation history"
            )
            
        except Exception as e:
            return ProcessingResult(
                success=False,
                error=f"Clearing conversation failed: {str(e)}"
            )
    
    def add_document_tags(self, document_identifier: str, tags: List[str]) -> ProcessingResult:
        """Add tags to a document."""
        try:
            # Find document
            doc = None
            if document_identifier in self.state.documents:
                doc = self.state.documents[document_identifier]
            else:
                doc = self.state.get_document_by_name(document_identifier)
            
            if not doc:
                return ProcessingResult(
                    success=False,
                    error=f"Document '{document_identifier}' not found"
                )
            
            # Add new tags (avoid duplicates)
            for tag in tags:
                if tag not in doc.tags:
                    doc.tags.append(tag)
            
            return ProcessingResult(
                success=True,
                data=f"Tags added to '{doc.name}': {', '.join(tags)}"
            )
            
        except Exception as e:
            return ProcessingResult(
                success=False,
                error=f"Adding tags failed: {str(e)}"
            )
    
    def search_documents_by_tags(self, tags: List[str]) -> ProcessingResult:
        """Search documents by tags."""
        try:
            matching_docs = []
            
            for doc in self.state.documents.values():
                if any(tag in doc.tags for tag in tags):
                    matching_docs.append(doc)
            
            if not matching_docs:
                return ProcessingResult(
                    success=True,
                    data=f"No documents found with tags: {', '.join(tags)}"
                )
            
            result_parts = [f"Documents matching tags {tags}:"]
            for doc in matching_docs:
                tags_str = ', '.join(doc.tags)
                result_parts.append(f"• {doc.name} [Tags: {tags_str}]")
            
            return ProcessingResult(
                success=True,
                data="\n".join(result_parts),
                metadata={"matching_documents": len(matching_docs)}
            )
            
        except Exception as e:
            return ProcessingResult(
                success=False,
                error=f"Tag search failed: {str(e)}"
            )
    
    def get_system_status(self) -> ProcessingResult:
        """Get system status and statistics."""
        try:
            loaded_docs = self.state.get_loaded_documents()
            total_pages = sum(doc.total_pages for doc in loaded_docs)
            
            status_parts = [
                f"📊 System Status:",
                f"• Loaded Documents: {len(loaded_docs)}",
                f"• Total Pages: {total_pages}",
                f"• Active Document: {self.state.get_active_document().name if self.state.get_active_document() else 'None'}",
                f"• Conversation Messages: {len(self.state.conversation_history)}",
                f"• Output Directory: {self.config.output_dir}",
                ""
            ]
            
            if loaded_docs:
                status_parts.append("📚 Loaded Documents:")
                for doc in loaded_docs:
                    load_time = doc.load_timestamp.strftime("%Y-%m-%d %H:%M:%S") if doc.load_timestamp else "Unknown"
                    tags_str = f" [Tags: {', '.join(doc.tags)}]" if doc.tags else ""
                    status_parts.append(f"• {doc.name} ({doc.total_pages} pages) - Loaded: {load_time}{tags_str}")
            
            return ProcessingResult(
                success=True,
                data="\n".join(status_parts),
                metadata={
                    "loaded_documents": len(loaded_docs),
                    "total_pages": total_pages,
                    "conversation_messages": len(self.state.conversation_history)
                }
            )
            
        except Exception as e:
            return ProcessingResult(
                success=False,
                error=f"Getting system status failed: {str(e)}"
            )


# ============================================================================
# Command Line Interface
# ============================================================================

def main():
    """Main command line interface."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Multi-Document Processor with AI Analysis")
    parser.add_argument("--base-dir", help="Base directory for the processor", default=None)
    parser.add_argument("--load", help="Load a PDF document", action="append")
    parser.add_argument("--query", help="Query the loaded documents")
    parser.add_argument("--web-search", help="Perform web search")
    parser.add_argument("--list", help="List loaded documents", action="store_true")
    parser.add_argument("--status", help="Show system status", action="store_true")
    
    args = parser.parse_args()
    
    try:
        processor = MultiDocumentProcessor(args.base_dir)
        
        # Load documents if specified
        if args.load:
            for pdf_path in args.load:
                result = processor.load_document(pdf_path)
                print(f"Load Result: {result.data if result.success else result.error}")
        
        # List documents if requested
        if args.list:
            result = processor.list_documents()
            print(result.data if result.success else result.error)
        
        # Show status if requested
        if args.status:
            result = processor.get_system_status()
            print(result.data if result.success else result.error)
        
        # Process query if specified
        if args.query:
            result = processor.query(args.query)
            print(f"Query Result: {result.data if result.success else result.error}")
        
        # Perform web search if specified
        if args.web_search:
            result = processor.query(args.web_search, QueryType.WEB_SEARCH)
            print(f"Web Search Result: {result.data if result.success else result.error}")
        
        # If no specific action, start interactive mode
        if not any([args.load, args.query, args.web_search, args.list, args.status]):
            interactive_mode(processor)
            
    except Exception as e:
        print(f"Error: {e}")


def interactive_mode(processor: MultiDocumentProcessor):
    """Interactive command line mode."""
    print("\n🤖 Multi-Document Processor Interactive Mode")
    print("Available commands:")
    print("  load <pdf_path> [tags] - Load a PDF document")
    print("  unload <doc_name>      - Unload a document")
    print("  list                   - List loaded documents")
    print("  active <doc_name>      - Set active document")
    print("  query <question>       - Query documents")
    print("  web <question>         - Web search")
    print("  history [n]            - Show conversation history")
    print("  clear                  - Clear conversation")
    print("  tags <doc_name> <tags> - Add tags to document")
    print("  search-tags <tags>     - Search documents by tags")
    print("  status                 - Show system status")
    print("  help                   - Show this help")
    print("  quit                   - Exit")
    print()
    
    while True:
        try:
            user_input = input("📝 Enter command: ").strip()
            
            if not user_input:
                continue
            
            parts = user_input.split()
            command = parts[0].lower()
            
            if command == "quit":
                break
            elif command == "help":
                print("Available commands: load, unload, list, active, query, web, history, clear, tags, search-tags, status, help, quit")
            elif command == "load" and len(parts) >= 2:
                pdf_path = parts[1]
                tags = parts[2:] if len(parts) > 2 else None
                result = processor.load_document(pdf_path, tags)
                print(result.data if result.success else f"Error: {result.error}")
            elif command == "unload" and len(parts) >= 2:
                doc_name = " ".join(parts[1:])
                result = processor.unload_document(doc_name)
                print(result.data if result.success else f"Error: {result.error}")
            elif command == "list":
                result = processor.list_documents()
                print(result.data if result.success else f"Error: {result.error}")
            elif command == "active" and len(parts) >= 2:
                doc_name = " ".join(parts[1:])
                result = processor.set_active_document(doc_name)
                print(result.data if result.success else f"Error: {result.error}")
            elif command == "query" and len(parts) >= 2:
                question = " ".join(parts[1:])
                result = processor.query(question)
                print(result.data if result.success else f"Error: {result.error}")
            elif command == "web" and len(parts) >= 2:
                question = " ".join(parts[1:])
                result = processor.query(question, QueryType.WEB_SEARCH)
                print(result.data if result.success else f"Error: {result.error}")
            elif command == "history":
                n = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 10
                result = processor.get_conversation_history(n)
                print(result.data if result.success else f"Error: {result.error}")
            elif command == "clear":
                result = processor.clear_conversation()
                print(result.data if result.success else f"Error: {result.error}")
            elif command == "tags" and len(parts) >= 3:
                doc_name = parts[1]
                tags = parts[2:]
                result = processor.add_document_tags(doc_name, tags)
                print(result.data if result.success else f"Error: {result.error}")
            elif command == "search-tags" and len(parts) >= 2:
                tags = parts[1:]
                result = processor.search_documents_by_tags(tags)
                print(result.data if result.success else f"Error: {result.error}")
            elif command == "status":
                result = processor.get_system_status()
                print(result.data if result.success else f"Error: {result.error}")
            else:
                print("Invalid command. Type 'help' for available commands.")
                
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")
    
    print("Goodbye! 👋")


if __name__ == "__main__":
    main()
        