import base64
import os
import logging
import tempfile
import json
import re
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from contextlib import contextmanager

from mistralai import Mistral
from dotenv import load_dotenv
from pdf2image import convert_from_path
from PIL import Image

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pdf_extraction.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class ExtractionConfig:
    """Configuration for PDF extraction settings."""
    max_workers: int = 6
    output_format: str = "JPEG"
    image_quality: int = 95
    model_name: str = "mistral-medium-latest"
    timeout: int = 30
    max_retries: int = 3
    max_text_length: int = 10000

class PDFTextExtractor:
    """A comprehensive PDF text extraction and processing utility using Mistral AI."""
    
    def __init__(self, config: ExtractionConfig = None):
        self.config = config or ExtractionConfig()
        self.client = None
        self._setup_client()
    
    def _setup_client(self) -> None:
        try:
            load_dotenv()
            api_key = os.getenv("MISTRAL_API_KEY")
            if not api_key:
                raise ValueError("MISTRAL_API_KEY not found in environment variables")
            
            self.client = Mistral(api_key=api_key)
            logger.info("Mistral client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Mistral client: {e}")
            raise
    
    @contextmanager
    def _temp_file(self, suffix: str = ".jpg"):
        temp_file = None
        try:
            temp_file = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
            yield temp_file
        finally:
            if temp_file and temp_file.name and os.path.exists(temp_file.name):
                try:
                    os.unlink(temp_file.name)
                except OSError as e:
                    logger.warning(f"Failed to delete temporary file {temp_file.name}: {e}")
    
    def _encode_image(self, image: Image.Image) -> Optional[str]:
        try:
            with self._temp_file() as temp_file:
                image.save(
                    temp_file.name, 
                    format=self.config.output_format,
                    quality=self.config.image_quality
                )
                
                with open(temp_file.name, "rb") as image_file:
                    return base64.b64encode(image_file.read()).decode('utf-8')
                    
        except Exception as e:
            logger.error(f"Error encoding image: {e}")
            return None
    
    def _extract_text_from_image(self, image: Image.Image, page_num: int) -> Optional[str]:
        for attempt in range(self.config.max_retries):
            try:
                base64_image = self._encode_image(image)
                if not base64_image:
                    logger.error(f"Failed to encode image for page {page_num + 1}")
                    return None
                
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Extract all the text from this image exactly as it appears, including numbers, symbols, and formatting. Return the complete text without any omissions."
                            },
                            {
                                "type": "image_url",
                                "image_url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        ]
                    }
                ]
                
                chat_response = self.client.chat.complete(
                    model=self.config.model_name,
                    messages=messages
                )
                
                return chat_response.choices[0].message.content
                
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed for page {page_num + 1}: {e}")
                if attempt == self.config.max_retries - 1:
                    logger.error(f"All attempts failed for page {page_num + 1}")
        
        return None
    
    def _process_page(self, image: Image.Image, page_num: int, total_pages: int) -> Tuple[int, Optional[str]]:
        logger.info(f"Processing page {page_num + 1} of {total_pages}...")
        extracted_text = self._extract_text_from_image(image, page_num)
        if extracted_text:
            return page_num, f"=== Page {page_num + 1} ===\n{extracted_text}\n\n"
        return page_num, None
    
    def _convert_pdf_to_images(self, pdf_path: Path) -> List[Image.Image]:
        try:
            logger.info(f"Converting PDF to images: {pdf_path}")
            return convert_from_path(str(pdf_path))
        except Exception as e:
            logger.error(f"Error converting PDF to images: {e}")
            raise
    
    def _generate_optimized_text(self, extracted_text: str) -> Optional[str]:
        try:
            messages = [
                {
                    "role": "user",
                    "content": f"""Please optimize the following text for clarity and conciseness while preserving all key information:
                    
                    {extracted_text[:self.config.max_text_length]}
                    
                    Guidelines:
                    1. Remove redundant information
                    2. Organize content logically
                    3. Maintain technical accuracy
                    4. Keep all important details
                    5. Improve readability without changing meaning"""
                }
            ]
            
            chat_response = self.client.chat.complete(
                model=self.config.model_name,
                messages=messages
            )
            
            return chat_response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error generating optimized text: {e}")
            return None

    def _generate_related_questions(self, extracted_text: str) -> Optional[List[str]]:
        try:
            messages = [
                {
                    "role": "user",
                    "content": f"""Based on the following text, generate 10 common and highly relevant questions that someone might ask:
                    
                    {extracted_text[:self.config.max_text_length]}
                    
                    Return the questions as a numbered list, one question per line."""
                }
            ]
            
            chat_response = self.client.chat.complete(
                model=self.config.model_name,
                messages=messages
            )
            
            questions = [
                q.strip() 
                for q in chat_response.choices[0].message.content.split('\n') 
                if q.strip() and q[0].isdigit()
            ]
            
            return [re.sub(r'^\d+[\.\)]\s*', '', q) for q in questions[:10]]
            
        except Exception as e:
            logger.error(f"Error generating related questions: {e}")
            return None

    def _extract_structured_data(self, extracted_text: str) -> Optional[Dict[str, Any]]:
        """Extract structured data from the text with added model insights."""
        try:
            messages = [
                {
                    "role": "user",
                    "content": f"""Extract the following structured information from the text below:
                    
                    {extracted_text[:self.config.max_text_length]}
                    
                    Return a JSON object with these fields:
                    - project_name: The name of the project
                    - project_number: The project number or identifier
                    - client_name: The name of the client (if mentioned)
                    - location: The project location
                    - start_date: Project start date (if mentioned)
                    - end_date: Project end date (if mentioned)
                    - total_area: Total project area
                    - rooms: Array of room objects with:
                    - name: Room name
                    - id: Room identifier
                    - area: Room area
                    - materials: Array of materials used
                    - notes: Any additional notes
                    - materials_used: Array of all unique materials used in project
                    - key_dates: Important dates mentioned
                    - key_contacts: Important contacts mentioned
                    - model_notes: A brief note (1-2 sentences) with insights about the project based on the extracted information
                    
                    For 'model_notes', provide simple observations like:
                    - "This appears to be a [type] project with focus on [features]"
                    - "The project emphasizes [materials/techniques] which suggests [characteristics]"
                    - "Notable aspects include [key features]"
                    
                    If a field cannot be determined, set it to null."""
                }
            ]
            
            chat_response = self.client.chat.complete(
                model=self.config.model_name,
                messages=messages,
                response_format={"type": "json_object"}
            )
            
            return json.loads(chat_response.choices[0].message.content)
            
        except Exception as e:
            logger.error(f"Error extracting structured data: {e}")
            return None

    def extract_text_from_pdf(self, pdf_path: Union[str, Path]) -> Optional[str]:
        try:
            pdf_file = Path(pdf_path)
            if not pdf_file.exists():
                raise FileNotFoundError(f"PDF file not found: {pdf_path}")
            
            if not pdf_file.suffix.lower() == '.pdf':
                raise ValueError(f"File must be a PDF: {pdf_path}")
            
            images = self._convert_pdf_to_images(pdf_file)
            if not images:
                logger.error("No images extracted from PDF")
                return None
            
            results = []
            total_pages = len(images)
            
            with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                futures = {
                    executor.submit(self._process_page, image, i, total_pages): i 
                    for i, image in enumerate(images)
                }
                
                for future in as_completed(futures):
                    try:
                        page_num, page_text = future.result()
                        if page_text:
                            results.append((page_num, page_text))
                        else:
                            logger.warning(f"Failed to extract text from page {page_num + 1}")
                    except Exception as e:
                        logger.error(f"Error processing page: {e}")
            
            if not results:
                logger.error("No text extracted from any page")
                return None
            
            results.sort(key=lambda x: x[0])
            return "".join([text for _, text in results])
            
        except Exception as e:
            logger.error(f"Error processing PDF: {e}")
            return None
    
    def save_text_to_file(self, text: str, output_path: Union[str, Path]) -> bool:
        try:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(text)
            
            logger.info(f"Text saved to: {output_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving text to file: {e}")
            return False

    def process_pdf(self, pdf_path: Union[str, Path], output_base_name: str = None) -> bool:
        try:
            pdf_file = Path(pdf_path)
            if not output_base_name:
                output_base_name = pdf_file.stem
            
            logger.info(f"Starting text extraction for {pdf_file.name}...")
            extracted_text = self.extract_text_from_pdf(pdf_path)
            if not extracted_text:
                logger.error(f"Text extraction failed for {pdf_file.name}")
                return False
            
            # Create output directory with PDF name
            output_dir = Path(f"output_{pdf_file.stem}")
            output_dir.mkdir(exist_ok=True)
            
            raw_text_path = output_dir / f"{output_base_name}_raw.txt"
            if not self.save_text_to_file(extracted_text, raw_text_path):
                logger.error(f"Failed to save raw extracted text for {pdf_file.name}")
                return False
            
            logger.info(f"Generating optimized text for {pdf_file.name}...")
            optimized_text = self._generate_optimized_text(extracted_text)
            if optimized_text:
                optimized_path = output_dir / f"{output_base_name}_optimized.txt"
                if not self.save_text_to_file(optimized_text, optimized_path):
                    logger.error(f"Failed to save optimized text for {pdf_file.name}")
            
            logger.info(f"Generating related questions for {pdf_file.name}...")
            questions = self._generate_related_questions(extracted_text)
            if questions:
                questions_path = output_dir / f"{output_base_name}_questions.txt"
                questions_text = "Frequently Asked Questions:\n\n" + "\n".join(
                    f"{i+1}. {q}" for i, q in enumerate(questions)
                )
                if not self.save_text_to_file(questions_text, questions_path):
                    logger.error(f"Failed to save questions for {pdf_file.name}")
            
            logger.info(f"Extracting structured data for {pdf_file.name}...")
            structured_data = self._extract_structured_data(extracted_text)
            if structured_data:
                json_path = output_dir / f"{output_base_name}_structured.json"
                try:
                    with open(json_path, 'w', encoding='utf-8') as f:
                        json.dump(structured_data, f, indent=2, ensure_ascii=False)
                    logger.info(f"Structured data saved to: {json_path}")
                except Exception as e:
                    logger.error(f"Failed to save JSON data for {pdf_file.name}: {e}")
            
            logger.info(f"PDF processing completed for {pdf_file.name}")
            return True
            
        except Exception as e:
            logger.error(f"Error in PDF processing pipeline for {pdf_path}: {e}")
            return False

    def process_multiple_pdfs(self, pdf_paths: List[Union[str, Path]], output_dir: str = "output") -> Dict[str, bool]:
        """
        Process multiple PDF files in batch.
        
        Args:
            pdf_paths: List of paths to PDF files
            output_dir: Base directory for all outputs
            
        Returns:
            Dictionary with PDF paths as keys and success status as values
        """
        results = {}
        base_output_dir = Path(output_dir)
        base_output_dir.mkdir(exist_ok=True)
        
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            future_to_pdf = {
                executor.submit(
                    self.process_pdf, 
                    pdf_path, 
                    str(base_output_dir / Path(pdf_path).stem)
                ): pdf_path 
                for pdf_path in pdf_paths
            }
            
            for future in as_completed(future_to_pdf):
                pdf_path = future_to_pdf[future]
                try:
                    success = future.result()
                    results[str(pdf_path)] = success
                    if success:
                        logger.info(f"Successfully processed: {pdf_path}")
                    else:
                        logger.error(f"Failed to process: {pdf_path}")
                except Exception as e:
                    results[str(pdf_path)] = False
                    logger.error(f"Error processing {pdf_path}: {e}")
        
        return results

def main():
    config = ExtractionConfig(
        max_workers=40,  # Reduced workers to avoid API rate limits
        max_retries=35,
        timeout=30,
        max_text_length=8000
    )
    
    extractor = PDFTextExtractor(config)
    
    # Example: Process multiple PDFs in a directory
    pdf_directory = r"D:\Melbin\Geometra_Chatbot\cortex_mistral\documents"
    pdf_paths = [
        Path(pdf_directory) / "03.pdf",
        Path(pdf_directory) / "02.pdf",
        # Add more PDFs as needed
    ]
    
    # Filter only existing PDF files
    valid_pdf_paths = [p for p in pdf_paths if p.exists() and p.suffix.lower() == '.pdf']
    
    if not valid_pdf_paths:
        logger.error("No valid PDF files found to process")
        return
    
    logger.info(f"Processing {len(valid_pdf_paths)} PDF files...")
    results = extractor.process_multiple_pdfs(valid_pdf_paths)
    
    # Print summary
    successful = sum(1 for success in results.values() if success)
    logger.info(f"Processing complete. Success: {successful}/{len(results)}")
    
    if successful < len(results):
        failed_files = [pdf for pdf, success in results.items() if not success]
        logger.warning("Failed to process the following files:")
        for pdf in failed_files:
            logger.warning(f"- {pdf}")

if __name__ == "__main__":
    main()