#!/usr/bin/env python3
"""
Enhanced PDF Text Extraction Tool using Mistral AI

Features:
- Preserves raw extracted text exactly as it appears in the document
- Additional processing options for optimized text, questions, and structured data
- Improved error handling and logging
- Configurable output options
"""

import argparse
import base64
import json
import logging
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import dotenv
from mistralai import Mistral
from pdf2image import convert_from_path
from PIL import Image
import tempfile

# ====================== CONFIGURATION CLASSES ======================

@dataclass
class ExtractionConfig:
    """Configuration for PDF extraction settings."""
    max_workers: int = 40
    output_format: str = "JPEG"
    image_quality: int = 95
    model_name: str = "mistral-medium-latest"
    timeout: int = 30
    max_retries: int = 35
    max_text_length: int = 10000
    output_dir: str = "output"
    preserve_raw: bool = True  # Always keep raw extracted text
    generate_optimized: bool = False
    generate_questions: bool = False
    generate_structured: bool = False
    consolidate_output: bool = True

# ====================== CORE FUNCTIONALITY ======================

class PDFProcessor:
    """Handles PDF to image conversion and basic operations."""
    
    @staticmethod
    def convert_pdf_to_images(pdf_path: Path, config: ExtractionConfig) -> List[Image.Image]:
        """Convert PDF pages to images."""
        try:
            logging.info(f"Converting PDF to images: {pdf_path}")
            return convert_from_path(str(pdf_path))
        except Exception as e:
            logging.error(f"Error converting PDF to images: {e}")
            raise

class ImageProcessor:
    """Handles image processing and encoding."""
    
    @staticmethod
    @contextmanager
    def _temp_file(suffix: str = ".jpg"):
        """Context manager for temporary files."""
        temp_file = None
        try:
            temp_file = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
            yield temp_file
        finally:
            if temp_file and temp_file.name and os.path.exists(temp_file.name):
                try:
                    os.unlink(temp_file.name)
                except OSError as e:
                    logging.warning(f"Failed to delete temporary file {temp_file.name}: {e}")
    
    @classmethod
    def encode_image(cls, image: Image.Image, config: ExtractionConfig) -> Optional[str]:
        """Encode image to base64 string."""
        try:
            with cls._temp_file() as temp_file:
                image.save(
                    temp_file.name,
                    format=config.output_format,
                    quality=config.image_quality
                )
                with open(temp_file.name, "rb") as image_file:
                    return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            logging.error(f"Error encoding image: {e}")
            return None

class MistralClient:
    """Wrapper for Mistral AI operations."""
    
    def __init__(self, config: ExtractionConfig):
        self.config = config
        self.client = self._initialize_client()
    
    def _initialize_client(self) -> Mistral:
        """Initialize and return Mistral client."""
        try:
            dotenv.load_dotenv()
            api_key = os.getenv("MISTRAL_API_KEY")
            if not api_key:
                raise ValueError("MISTRAL_API_KEY not found in environment variables")
            return Mistral(api_key=api_key)
        except Exception as e:
            logging.error(f"Failed to initialize Mistral client: {e}")
            raise
    
    def extract_raw_text(self, image: Image.Image, page_num: int) -> Optional[str]:
        """Extract raw text from image using Mistral with strict instructions."""
        for attempt in range(self.config.max_retries):
            try:
                base64_image = ImageProcessor.encode_image(image, self.config)
                if not base64_image:
                    logging.error(f"Failed to encode image for page {page_num + 1}")
                    return None
                
                messages = [{
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": ("Extract ALL text from this image EXACTLY as it appears. "
                                    "Preserve original formatting, spacing, and line breaks. "
                                    "Do not correct errors or modify the text in any way. "
                                    "Include all headers, footers, page numbers, and marginalia.")
                        },
                        {
                            "type": "image_url",
                            "image_url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    ]
                }]
                
                response = self.client.chat.complete(
                    model=self.config.model_name,
                    messages=messages,
                    temperature=0.1  # Lower temperature for more deterministic output
                )
                return response.choices[0].message.content
                
            except Exception as e:
                logging.warning(f"Attempt {attempt + 1} failed for page {page_num + 1}: {e}")
                if attempt == self.config.max_retries - 1:
                    logging.error(f"All attempts failed for page {page_num + 1}")
        return None
    
    def generate_optimized_text(self, text: str) -> Optional[str]:
        """Generate optimized version of extracted text."""
        return self._process_text(
            text,
            ("Create a clean, readable version of this text while preserving all key information. "
             "Correct obvious errors, normalize formatting, and improve readability. "
             "Maintain the original structure and content.")
        )
    
    def generate_related_questions(self, text: str) -> Optional[List[str]]:
        """Generate related questions from extracted text."""
        result = self._process_text(
            text,
            "Generate 10 relevant questions about this content as a numbered list."
        )
        if not result:
            return None
        
        questions = [q.strip() for q in result.split('\n') if q.strip() and q[0].isdigit()]
        return [re.sub(r'^\d+[\.\)]\s*', '', q) for q in questions[:10]]
    
    def extract_structured_data(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract structured data from text."""
        try:
            messages = [{
                "role": "user",
                "content": ("Extract structured information from this text and return as JSON. "
                          "Include key entities, relationships, and important details. "
                          "Preserve all quantitative data and technical specifications.\n\n"
                          f"{text}"),
            }]
            
            response = self.client.chat.complete(
                model=self.config.model_name,
                messages=messages,
                response_format={"type": "json_object"}
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            logging.error(f"Error extracting structured data: {e}")
            return None
    
    def _process_text(self, text: str, instruction: str) -> Optional[str]:
        """Generic text processing method."""
        try:
            messages = [{
                "role": "user",
                "content": f"{instruction}\n\n{text[:self.config.max_text_length]}"
            }]
            
            response = self.client.chat.complete(
                model=self.config.model_name,
                messages=messages
            )
            return response.choices[0].message.content
        except Exception as e:
            logging.error(f"Error processing text: {e}")
            return None

# ====================== MAIN APPLICATION ======================

class PDFTextExtractor:
    """Main application class for PDF text extraction."""
    
    def __init__(self, config: ExtractionConfig):
        self.config = config
        self.mistral = MistralClient(config)
        self._setup_logging()
        self.consolidated_data = {
            'raw_text': {},
            'optimized_text': {},
            'questions': {},
            'structured_data': {}
        }
    
    def _setup_logging(self):
        """Configure logging system."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('pdf_extraction.log'),
                logging.StreamHandler()
            ]
        )
    
    def process_page(self, image: Image.Image, page_num: int, total_pages: int, pdf_name: str) -> Tuple[int, Optional[Dict[str, Any]]]:
        """Process a single page and return multiple results."""
        logging.info(f"Processing page {page_num + 1} of {total_pages} from {pdf_name}...")
        results = {}
        
        # Always extract raw text
        extracted_text = self.mistral.extract_raw_text(image, page_num)
        if not extracted_text:
            return page_num, None
        
        results['raw_text'] = f"=== {pdf_name} - Page {page_num + 1} ===\n{extracted_text}\n\n"
        
        # Conditional processing based on config
        if self.config.generate_optimized:
            if optimized := self.mistral.generate_optimized_text(extracted_text):
                results['optimized_text'] = f"=== {pdf_name} - Page {page_num + 1} ===\n{optimized}\n\n"
        
        if self.config.generate_questions:
            if questions := self.mistral.generate_related_questions(extracted_text):
                results['questions'] = f"=== {pdf_name} - Page {page_num + 1} ===\n" + \
                                     "Questions:\n\n" + \
                                     "\n".join(f"{i+1}. {q}" for i, q in enumerate(questions)) + \
                                     "\n\n"
        
        if self.config.generate_structured:
            if structured := self.mistral.extract_structured_data(extracted_text):
                results['structured_data'] = {
                    'source': pdf_name,
                    'page': page_num + 1,
                    'data': structured
                }
        
        return page_num, results
    
    def extract_text_from_pdf(self, pdf_path: Path) -> Optional[Dict[str, Any]]:
        """Main extraction method for a single PDF."""
        try:
            images = PDFProcessor.convert_pdf_to_images(pdf_path, self.config)
            if not images:
                logging.error("No images extracted from PDF")
                return None
            
            results = {
                'raw_text': [],
                'optimized_text': [],
                'questions': [],
                'structured_data': []
            }
            total_pages = len(images)
            pdf_name = pdf_path.stem
            
            with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                futures = {
                    executor.submit(self.process_page, image, i, total_pages, pdf_name): i
                    for i, image in enumerate(images)
                }
                
                for future in as_completed(futures):
                    try:
                        page_num, page_results = future.result()
                        if page_results:
                            for key in results:
                                if key in page_results:
                                    results[key].append((page_num, page_results[key]))
                    except Exception as e:
                        logging.error(f"Error processing page: {e}")
            
            if not results['raw_text']:
                logging.error("No raw text extracted from any page")
                return None
            
            # Sort all results by page number
            for key in results:
                results[key].sort(key=lambda x: x[0])
                results[key] = [item[1] for item in results[key]]
            
            return results
            
        except Exception as e:
            logging.error(f"Error processing PDF: {e}")
            return None
    
    def save_consolidated_output(self):
        """Save all consolidated data to files."""
        if not self.config.consolidate_output:
            return
        
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save raw text (always preserved)
        if self.consolidated_data['raw_text']:
            raw_content = "\n".join(
                f"{pdf_name}\n{'='*40}\n{content}"
                for pdf_name, content in self.consolidated_data['raw_text'].items()
            )
            raw_path = output_dir / "consolidated_raw.txt"
            raw_path.write_text(raw_content, encoding='utf-8')
            logging.info(f"Consolidated raw text saved to: {raw_path}")
        
        # Save optimized text
        if self.config.generate_optimized and self.consolidated_data['optimized_text']:
            opt_content = "\n".join(
                f"{pdf_name}\n{'='*40}\n{content}"
                for pdf_name, content in self.consolidated_data['optimized_text'].items()
            )
            opt_path = output_dir / "consolidated_optimized.txt"
            opt_path.write_text(opt_content, encoding='utf-8')
            logging.info(f"Consolidated optimized text saved to: {opt_path}")
        
        # Save questions
        if self.config.generate_questions and self.consolidated_data['questions']:
            q_content = "\n".join(
                f"{pdf_name}\n{'='*40}\n{content}"
                for pdf_name, content in self.consolidated_data['questions'].items()
            )
            q_path = output_dir / "consolidated_questions.txt"
            q_path.write_text(q_content, encoding='utf-8')
            logging.info(f"Consolidated questions saved to: {q_path}")
        
        # Save structured data
        if self.config.generate_structured and self.consolidated_data['structured_data']:
            json_data = []
            for pdf_name, data_list in self.consolidated_data['structured_data'].items():
                for data in data_list:
                    json_data.append(data)
            
            json_path = output_dir / "consolidated_structured.json"
            json_path.write_text(json.dumps(json_data, indent=2), encoding='utf-8')
            logging.info(f"Consolidated structured data saved to: {json_path}")
    
    def process_single_pdf(self, pdf_path: Path) -> bool:
        """Full processing pipeline for a single PDF."""
        try:
            logging.info(f"Starting processing for {pdf_path.name}...")
            pdf_name = pdf_path.stem
            
            # Extract content
            extracted_data = self.extract_text_from_pdf(pdf_path)
            if not extracted_data:
                return False
            
            # Store in consolidated data
            if self.config.consolidate_output:
                if extracted_data['raw_text']:
                    self.consolidated_data['raw_text'][pdf_name] = "\n".join(extracted_data['raw_text'])
                
                if self.config.generate_optimized and extracted_data['optimized_text']:
                    self.consolidated_data['optimized_text'][pdf_name] = "\n".join(extracted_data['optimized_text'])
                
                if self.config.generate_questions and extracted_data['questions']:
                    self.consolidated_data['questions'][pdf_name] = "\n".join(extracted_data['questions'])
                
                if self.config.generate_structured and extracted_data['structured_data']:
                    self.consolidated_data['structured_data'][pdf_name] = []
                    for data in extracted_data['structured_data']:
                        if isinstance(data, dict):
                            self.consolidated_data['structured_data'][pdf_name].append(data)
            
            # Save individual files if not consolidating
            if not self.config.consolidate_output:
                output_dir = Path(self.config.output_dir) / pdf_name
                output_dir.mkdir(parents=True, exist_ok=True)
                
                # Always save raw text
                raw_path = output_dir / f"{pdf_name}_raw.txt"
                raw_path.write_text("\n".join(extracted_data['raw_text']), encoding='utf-8')
                
                # Conditional saves
                if self.config.generate_optimized and extracted_data['optimized_text']:
                    opt_path = output_dir / f"{pdf_name}_optimized.txt"
                    opt_path.write_text("\n".join(extracted_data['optimized_text']), encoding='utf-8')
                
                if self.config.generate_questions and extracted_data['questions']:
                    q_path = output_dir / f"{pdf_name}_questions.txt"
                    q_path.write_text("\n".join(extracted_data['questions']), encoding='utf-8')
                
                if self.config.generate_structured and extracted_data['structured_data']:
                    json_path = output_dir / f"{pdf_name}_structured.json"
                    json_path.write_text(json.dumps(extracted_data['structured_data'], indent=2), encoding='utf-8')
            
            return True
            
        except Exception as e:
            logging.error(f"Error processing {pdf_path.name}: {e}")
            return False
    
    def process_multiple_pdfs(self, pdf_paths: List[Path]) -> Dict[Path, bool]:
        """Batch processing for multiple PDFs."""
        results = {}
        
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            future_to_pdf = {
                executor.submit(self.process_single_pdf, pdf_path): pdf_path
                for pdf_path in pdf_paths
            }
            
            for future in as_completed(future_to_pdf):
                pdf_path = future_to_pdf[future]
                try:
                    results[pdf_path] = future.result()
                except Exception as e:
                    results[pdf_path] = False
                    logging.error(f"Error processing {pdf_path.name}: {e}")
        
        # Save consolidated output after all processing is done
        if self.config.consolidate_output:
            self.save_consolidated_output()
        
        return results

# ====================== CLI INTERFACE ======================

class CLI:
    """Command Line Interface handler."""
    
    @staticmethod
    def validate_pdf_paths(paths: List[str]) -> List[Path]:
        """Validate input PDF paths."""
        valid = []
        for path in paths:
            pdf_path = Path(path)
            if not pdf_path.exists():
                print(f"Error: File not found - {path}", file=sys.stderr)
                continue
            if pdf_path.suffix.lower() != '.pdf':
                print(f"Error: Not a PDF file - {path}", file=sys.stderr)
                continue
            valid.append(pdf_path)
        return valid
    
    @staticmethod
    def get_interactive_input() -> List[Path]:
        """Get PDF paths through interactive input."""
        print("\nPDF Text Extractor - Interactive Mode")
        print("Enter PDF file paths (one per line, blank line to finish):")
        
        paths = []
        while True:
            try:
                line = input("> ").strip()
                if not line:
                    break
                path = Path(line)
                if not path.exists():
                    print(f"File not found: {line}", file=sys.stderr)
                    continue
                if path.suffix.lower() != '.pdf':
                    print(f"Not a PDF file: {line}", file=sys.stderr)
                    continue
                paths.append(path)
            except KeyboardInterrupt:
                print("\nOperation cancelled.")
                return []
        
        return paths
    
    @classmethod
    def run(cls):
        """Run the CLI application."""
        parser = argparse.ArgumentParser(
            description="PDF Text Extractor using Mistral AI",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
        
        parser.add_argument(
            "files",
            nargs="*",
            help="PDF files to process",
            metavar="PDF_FILE"
        )
        parser.add_argument(
            "-o", "--output",
            default="output",
            help="Output directory for extracted content"
        )
        parser.add_argument(
            "-w", "--workers",
            type=int,
            default=4,
            help="Number of parallel workers"
        )
        parser.add_argument(
            "-i", "--interactive",
            action="store_true",
            help="Enable interactive file selection"
        )
        parser.add_argument(
            "--no-consolidate",
            action="store_true",
            help="Disable consolidated output and save files per document"
        )
        parser.add_argument(
            "--optimize",
            action="store_true",
            help="Generate optimized/clean versions of the text"
        )
        parser.add_argument(
            "--questions",
            action="store_true",
            help="Generate related questions from the content"
        )
        parser.add_argument(
            "--structured",
            action="store_true",
            help="Extract structured data from the content"
        )
        
        args = parser.parse_args()
        
        # Get PDF paths
        pdf_paths = cls.validate_pdf_paths(args.files)
        if args.interactive or not pdf_paths:
            pdf_paths.extend(cls.get_interactive_input())
        
        if not pdf_paths:
            print("No valid PDF files to process.", file=sys.stderr)
            sys.exit(1)
        
        # Initialize and run processor
        config = ExtractionConfig(
            max_workers=args.workers,
            output_dir=args.output,
            preserve_raw=True,  # Always true
            generate_optimized=args.optimize,
            generate_questions=args.questions,
            generate_structured=args.structured,
            consolidate_output=not args.no_consolidate
        )
        
        extractor = PDFTextExtractor(config)
        results = extractor.process_multiple_pdfs(pdf_paths)
        
        # Display results
        success = sum(1 for r in results.values() if r)
        print(f"\nProcessed {len(results)} files. Successful: {success}")
        
        if success < len(results):
            print("\nFailed files:", file=sys.stderr)
            for path, success in results.items():
                if not success:
                    print(f"- {path}", file=sys.stderr)

# ====================== MAIN ENTRY POINT ======================

if __name__ == "__main__":
    CLI.run()