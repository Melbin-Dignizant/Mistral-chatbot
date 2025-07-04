# import base64
# import requests
# import os
# from mistralai import Mistral
# from dotenv import load_dotenv
# from pdf2image import convert_from_path
# import tempfile
# from concurrent.futures import ThreadPoolExecutor, as_completed

# # Load environment variables from .env file
# load_dotenv()

# def encode_image(image):
#     """Encode a PIL image to base64."""
#     try:
#         with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
#             image.save(temp_file.name, format="JPEG")
#             with open(temp_file.name, "rb") as image_file:
#                 return base64.b64encode(image_file.read()).decode('utf-8')
#     except Exception as e:
#         print(f"Error encoding image: {e}")
#         return None
#     finally:
#         if 'temp_file' in locals() and temp_file.name:
#             os.unlink(temp_file.name)

# def process_page(image, page_num, total_pages, client):
#     """Process a single page and return its text."""
#     print(f"Processing page {page_num+1} of {total_pages}...")
    
#     # Encode the image
#     base64_image = encode_image(image)
#     if not base64_image:
#         return None
        
#     # Define the messages for the chat
#     messages = [
#         {
#             "role": "user",
#             "content": [
#                 {
#                     "type": "text",
#                     "text": "Extract all the text from this image exactly as it appears, including numbers, symbols, and formatting. Return the complete text without any omissions."
#                 },
#                 {
#                     "type": "image_url",
#                     "image_url": f"data:image/jpeg;base64,{base64_image}" 
#                 }
#             ]
#         }
#     ]
    
#     # Get the chat response
#     chat_response = client.chat.complete(
#         model="mistral-medium-latest",
#         messages=messages
#     )
    
#     return f"Page {page_num+1}:\n{chat_response.choices[0].message.content}\n\n"

# def process_pdf(pdf_path, max_workers=4):
#     """Convert PDF to images and extract text from each page in parallel."""
#     try:
#         # Convert PDF to list of images (one per page)
#         images = convert_from_path(pdf_path)
        
#         # Retrieve the API key from environment variables
#         api_key = os.getenv("MISTRAL_API_KEY")
#         if not api_key:
#             raise ValueError("MISTRAL_API_KEY not found in environment variables")
            
#         # Initialize the Mistral client
#         client = Mistral(api_key=api_key)
        
#         all_text = []
#         total_pages = len(images)
        
#         # Use ThreadPoolExecutor to process pages in parallel
#         with ThreadPoolExecutor(max_workers=max_workers) as executor:
#             # Submit all pages for processing
#             futures = {
#                 executor.submit(process_page, image, i, total_pages, client): i 
#                 for i, image in enumerate(images)
#             }
            
#             # Process results as they complete
#             for future in as_completed(futures):
#                 try:
#                     page_text = future.result()
#                     if page_text:
#                         all_text.append((futures[future], page_text))
#                 except Exception as e:
#                     print(f"Error processing page: {e}")
        
#         # Sort the results by page number and combine
#         all_text_sorted = [text for _, text in sorted(all_text, key=lambda x: x[0])]
#         return "".join(all_text_sorted)
        
#     except Exception as e:
#         print(f"Error processing PDF: {e}")
#         return None

# # Path to your PDF file
# pdf_path = r"D:\Melbin\Geometra_Chatbot\cortex_mistral\documents\01.pdf"

# # Process the PDF and extract text
# extracted_text = process_pdf(pdf_path, max_workers=10)  # Adjust max_workers based on your API rate limits

# if extracted_text:
#     # Print the extracted text
#     print(extracted_text)
    
#     # Optionally save to a file
#     with open("extracted_text.txt", "w", encoding="utf-8") as f:
#         f.write(extracted_text)
#     print("Text extracted and saved to extracted_text.txt")
# else:
#     print("Failed to extract text from PDF")



#######=========================================================================
import base64
import requests
import os
import logging
import tempfile
from pathlib import Path
from typing import Optional, List, Tuple
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
    """Configuration class for PDF extraction settings."""
    max_workers: int = 4
    output_format: str = "JPEG"
    image_quality: int = 95
    model_name: str = "mistral-medium-latest"
    timeout: int = 30
    max_retries: int = 3

class PDFTextExtractor:
    """A robust PDF text extraction utility using Mistral AI."""
    
    def __init__(self, config: ExtractionConfig = None):
        """Initialize the extractor with configuration."""
        self.config = config or ExtractionConfig()
        self.client = None
        self._setup_client()
    
    def _setup_client(self) -> None:
        """Setup Mistral client with error handling."""
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
        """Context manager for temporary file handling."""
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
        """Encode a PIL image to base64 with error handling."""
        try:
            with self._temp_file() as temp_file:
                image.save(
                    temp_file.name, 
                    format=self.config.output_format,
                    quality=self.config.image_quality
                )
                
                with open(temp_file.name, "rb") as image_file:
                    encoded = base64.b64encode(image_file.read()).decode('utf-8')
                    logger.debug(f"Successfully encoded image (size: {len(encoded)} chars)")
                    return encoded
                    
        except Exception as e:
            logger.error(f"Error encoding image: {e}")
            return None
    
    def _extract_text_from_image(self, image: Image.Image, page_num: int) -> Optional[str]:
        """Extract text from a single image with retry logic."""
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
                
                text_content = chat_response.choices[0].message.content
                logger.info(f"Successfully extracted text from page {page_num + 1}")
                return text_content
                
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed for page {page_num + 1}: {e}")
                if attempt == self.config.max_retries - 1:
                    logger.error(f"All attempts failed for page {page_num + 1}")
                    return None
        
        return None
    
    def _process_page(self, image: Image.Image, page_num: int, total_pages: int) -> Tuple[int, Optional[str]]:
        """Process a single page and return its text with page number."""
        logger.info(f"Processing page {page_num + 1} of {total_pages}...")
        
        extracted_text = self._extract_text_from_image(image, page_num)
        if extracted_text:
            formatted_text = f"=== Page {page_num + 1} ===\n{extracted_text}\n\n"
            return page_num, formatted_text
        
        return page_num, None
    
    def _convert_pdf_to_images(self, pdf_path: Path) -> List[Image.Image]:
        """Convert PDF to images with error handling."""
        try:
            logger.info(f"Converting PDF to images: {pdf_path}")
            images = convert_from_path(str(pdf_path))
            logger.info(f"Successfully converted PDF to {len(images)} images")
            return images
        except Exception as e:
            logger.error(f"Error converting PDF to images: {e}")
            raise
    
    def extract_text_from_pdf(self, pdf_path: str) -> Optional[str]:
        """
        Extract text from PDF file with parallel processing.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Extracted text or None if extraction fails
        """
        try:
            # Validate input
            pdf_file = Path(pdf_path)
            if not pdf_file.exists():
                raise FileNotFoundError(f"PDF file not found: {pdf_path}")
            
            if not pdf_file.suffix.lower() == '.pdf':
                raise ValueError(f"File must be a PDF: {pdf_path}")
            
            # Convert PDF to images
            images = self._convert_pdf_to_images(pdf_file)
            if not images:
                logger.error("No images extracted from PDF")
                return None
            
            # Process pages in parallel
            results = []
            total_pages = len(images)
            
            with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                # Submit all pages for processing
                futures = {
                    executor.submit(self._process_page, image, i, total_pages): i 
                    for i, image in enumerate(images)
                }
                
                # Collect results as they complete
                for future in as_completed(futures):
                    try:
                        page_num, page_text = future.result()
                        if page_text:
                            results.append((page_num, page_text))
                        else:
                            logger.warning(f"Failed to extract text from page {page_num + 1}")
                    except Exception as e:
                        logger.error(f"Error processing page: {e}")
            
            # Sort results by page number and combine
            if not results:
                logger.error("No text extracted from any page")
                return None
            
            results.sort(key=lambda x: x[0])
            combined_text = "".join([text for _, text in results])
            
            logger.info(f"Successfully extracted text from {len(results)} out of {total_pages} pages")
            return combined_text
            
        except Exception as e:
            logger.error(f"Error processing PDF: {e}")
            return None
    
    def save_text_to_file(self, text: str, output_path: str) -> bool:
        """Save extracted text to file with error handling."""
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

def main():
    """Main function to demonstrate usage."""
    # Configuration
    config = ExtractionConfig(
        max_workers=6,  # Adjust based on your API rate limits
        max_retries=3,
        timeout=30
    )
    
    # Initialize extractor
    extractor = PDFTextExtractor(config)
    
    # PDF file path - update this to your actual file path
    pdf_path = r"D:\Melbin\Geometra_Chatbot\cortex_mistral\documents\02.pdf"
    
    # Extract text
    logger.info("Starting PDF text extraction...")
    extracted_text = extractor.extract_text_from_pdf(pdf_path)
    
    if extracted_text:
        # Print preview
        preview_length = 500
        print("=" * 50)
        print("EXTRACTION PREVIEW:")
        print("=" * 50)
        print(extracted_text[:preview_length])
        if len(extracted_text) > preview_length:
            print(f"... (truncated, total length: {len(extracted_text)} characters)")
        print("=" * 50)
        
        # Save to file
        output_path = "extracted_text.txt"
        if extractor.save_text_to_file(extracted_text, output_path):
            logger.info(f"Text extraction completed successfully. Output saved to: {output_path}")
        else:
            logger.error("Failed to save extracted text to file")
    else:
        logger.error("Failed to extract text from PDF")

if __name__ == "__main__":
    main()