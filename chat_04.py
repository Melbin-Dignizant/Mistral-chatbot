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

        # Multi-document configuration
        self.max_concurrent_documents = 5
        self.cross_document_search = True

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

    @staticmethod
    def generate_document_id(pdf_path: str) -> str:
        """Generate a unique document ID from PDF path."""
        return f"doc_{hash(pdf_path) % 10000:04d}_{Path(pdf_path).stem}"


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
        return  '''You are an expert document understanding assistant. When analyzing documents:

                1. **Refine the Query**: Rewrite the user's question for clarity and specificity.
                2. **Analyze Relevance**: Carefully determine if the question relates to the document content.
                3. **Provide Answer**:
                - If related: Analyze the image and provide detailed, accurate answers. Clearly indicate when information comes from the document.
                - If unrelated: Explicitly state "This information is not found in the document. Here's what I know:" before providing general knowledge answer.

                **Important Rules**:
                - Never claim information comes from the document if it doesn't
                - Clearly distinguish between document content and general knowledge
                - When using general knowledge, always preface with clear disclaimer

                **Output Format:**
                Refined Query: <Clear, specific version of the question>
                Document Relevance: <"Found in document" or "Not found in document">
                Answer: <Structured answer with clear source attribution>'''

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
                self.logger.info("No relevant content found in documents, falling back to general knowledge")
                return self.answer_general_question(question)  # Fall back to general knowledge

            # Rest of the method remains the same...
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
        self.state = MultiDocumentState()

        self.logger.info("Document Processor initialized")

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

            # Process query based on type
            if query_type == QueryType.WEB_SEARCH:
                result = self.ai_processor.perform_web_search(question)
            elif query_type == QueryType.DOCUMENT:
                result = self._process_single_document_query(question)
            elif query_type == QueryType.MULTI_DOCUMENT:
                result = self.ai_processor.analyze_multi_document_query(question, self.state)
            else:  # GENERAL
                result = self.ai_processor.answer_general_question(question)

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

    def _process_single_document_query(self, question: str) -> ProcessingResult:
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

        return self.ai_processor.analyze_multi_document_query(question, temp_state)

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
                    "loaded_documents": loaded_docs,  # Return the list of documents, not just count
                    "total_pages": total_pages,
                    "active_document": self.state.active_document_id
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

class CommandLineInterface:
    """Interactive command-line interface for the document processor."""

    def __init__(self):
        self.processor = DocumentProcessor()
        self.logger = Logger("CLI")
        self.commands = {
            'load': self._handle_load,
            'unload': self._handle_unload,
            'list': self._handle_list,
            'active': self._handle_active,
            'status': self._handle_status,
            'tags': self._handle_tags,
            'search-tags': self._handle_search_tags,
            'help': self._handle_help,
            'quit': self._handle_quit,
            'exit': self._handle_quit
        }

    def run(self) -> None:
        """Run the interactive CLI."""
        self.logger.info("Multi-Document Processor Started")
        self._show_welcome()

        while True:
            try:
                result = self.processor.get_system_status()
                if not result.success:
                    print(f"\nSystem Status Error: {result.error}")
                    prompt = "\n[System Error]\n> "
                else:
                    prompt = self._build_prompt(result)

                user_input = input(prompt).strip()

                if not user_input:
                    continue

                if not self._process_input(user_input):
                    break

            except KeyboardInterrupt:
                self.logger.info("\nGoodbye!")
                break
            except Exception as e:
                print(f"\nUnexpected error: {str(e)}")
                continue

        print("Goodbye! 👋")

    def _build_prompt(self, status: ProcessingResult) -> str:
        """Build the input prompt based on current status."""
        if not status.success:
            return "\n[System Error]\n> "

        # Safely get metadata with defaults
        metadata = status.metadata or {}
        loaded_docs = metadata.get("loaded_documents", [])
        active_doc = metadata.get("active_document", None)

        # Get document names for display
        if isinstance(loaded_docs, list):
            loaded_count = len(loaded_docs)
            if active_doc:
                active_doc_name = next(
                    (doc.name for doc in loaded_docs if doc.document_id == active_doc),
                    active_doc
                )
            else:
                active_doc_name = "None"
        else:
            # Fallback if loaded_documents is not a list (shouldn't happen with the fix above)
            loaded_count = 0
            active_doc_name = "None"

        if loaded_count > 0:
            doc_info = f"Docs: {loaded_count} | Active: {active_doc_name}"
        else:
            doc_info = "No documents loaded"

        return f"\n[{doc_info}]\n> Enter question or command ('help' for options): "

    def _process_input(self, user_input: str) -> bool:
        """Process user input and return True to continue, False to quit."""
        words = user_input.split()

        # Check for commands
        if words and words[0].lower() in self.commands:
            return self.commands[words[0].lower()](user_input)

        # Regular question
        result = self.processor.query(user_input)
        self._display_result(result, "Response")
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
            self.logger.warning("Usage: load <pdf_path> [tags]")
            return True

        # Split into path and optional tags
        rest = parts[1].strip().split()
        pdf_path = rest[0].strip('"\'')
        tags = rest[1:] if len(rest) > 1 else None

        result = self.processor.load_document(pdf_path, tags)

        if not result.success:
            self.logger.error(result.error)

        return True

    def _handle_unload(self, user_input: str) -> bool:
        """Handle unload command."""
        parts = user_input.split(' ', 1)
        if len(parts) < 2:
            self.logger.warning("Usage: unload <document_name>")
            return True

        doc_name = parts[1].strip()
        result = self.processor.unload_document(doc_name)

        if result.success:
            self.logger.info(result.data)
        else:
            self.logger.error(result.error)

        return True

    def _handle_list(self, user_input: str) -> bool:
        """Handle list command."""
        result = self.processor.list_documents()
        self._display_result(result, "Documents")
        return True

    def _handle_active(self, user_input: str) -> bool:
        """Handle active command."""
        parts = user_input.split(' ', 1)
        if len(parts) < 2:
            self.logger.warning("Usage: active <document_name>")
            return True

        doc_name = parts[1].strip()
        result = self.processor.set_active_document(doc_name)

        if result.success:
            self.logger.info(result.data)
        else:
            self.logger.error(result.error)

        return True

    def _handle_status(self, user_input: str) -> bool:
        """Handle status command."""
        result = self.processor.get_system_status()
        self._display_result(result, "System Status")
        return True

    def _handle_tags(self, user_input: str) -> bool:
        """Handle tags command."""
        parts = user_input.split(' ', 2)
        if len(parts) < 3:
            self.logger.warning("Usage: tags <document_name> <tag1> [tag2...]")
            return True

        doc_name = parts[1].strip()
        tags = parts[2].strip().split()
        result = self.processor.add_document_tags(doc_name, tags)

        if result.success:
            self.logger.info(result.data)
        else:
            self.logger.error(result.error)

        return True

    def _handle_search_tags(self, user_input: str) -> bool:
        """Handle search-tags command."""
        parts = user_input.split(' ', 1)
        if len(parts) < 2:
            self.logger.warning("Usage: search-tags <tag1> [tag2...]")
            return True

        tags = parts[1].strip().split()
        result = self.processor.search_documents_by_tags(tags)

        self._display_result(result, "Tag Search Results")
        return True

    def _handle_help(self, user_input: str) -> bool:
        """Handle help command."""
        help_text = """
Available Commands:
- load <pdf_path> [tags] : Load a PDF document with optional tags
- unload <doc_name>      : Unload a document
- list                   : List loaded documents
- active <doc_name>      : Set active document
- status                 : Show system status
- tags <doc_name> <tags> : Add tags to document
- search-tags <tags>     : Search documents by tags
- help                   : Show this help message
- quit/exit              : Exit the program

Question Types:
- Regular question       : Automatically analyze documents or use general knowledge
- Questions with words like 'latest', 'current' will trigger web search
- Questions with words like 'compare', 'between' will analyze multiple documents
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
        print("   📄 Multi-Document Processor with AI Analysis")
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