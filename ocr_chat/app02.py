import os
import base64
from PIL import Image
import shutil
from datetime import datetime
from mistralai import Mistral
import time
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing
from functools import partial
import logging
import io

# Load environment variables from .env file
load_dotenv()

# Configurations
INPUT_DIR = r"D:\Melbin\Geometra_Chatbot\cortex_mistral\ocr_chat\static"
OUTPUT_BASE_DIR = r"D:\Melbin\Geometra_Chatbot\cortex_mistral\ocr_chat\output"
CONSOLIDATED_TEXT_FILE = "consolidated_extracted_text.txt"
OPTIMIZED_TEXT_FILE = "mistral_optimized_text.txt"

# Mistral AI Configuration
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
MISTRAL_OCR_MODEL = "mistral-ocr-latest"
MISTRAL_CHAT_MODEL = "mistral-medium-latest"

# Performance Configuration
MAX_WORKERS = min(multiprocessing.cpu_count(), 6)  # Slightly reduced for API calls
BATCH_SIZE = 3  # Smaller batches for API processing

# Supported extensions
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')
PDF_EXTENSION = ('.pdf',)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Ensure output base directory exists
os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)

def create_unique_folder(base_dir, prefix="DOC_"):
    """Creates a unique folder with timestamp"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"{prefix}{timestamp}"
    folder_path = os.path.join(base_dir, folder_name)
    os.makedirs(folder_path, exist_ok=True)
    return folder_path

def encode_pdf_to_base64(pdf_path):
    """Encode PDF file to base64 string"""
    try:
        with open(pdf_path, "rb") as pdf_file:
            return base64.b64encode(pdf_file.read()).decode('utf-8')
    except FileNotFoundError:
        logger.error(f"PDF file not found: {pdf_path}")
        return None
    except Exception as e:
        logger.error(f"Error encoding PDF {pdf_path}: {e}")
        return None

def encode_image_to_base64(image_path):
    """Encode image file to base64 string"""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except FileNotFoundError:
        logger.error(f"Image file not found: {image_path}")
        return None
    except Exception as e:
        logger.error(f"Error encoding image {image_path}: {e}")
        return None

def get_file_mime_type(file_path):
    """Get MIME type based on file extension"""
    ext = os.path.splitext(file_path)[1].lower()
    mime_types = {
        '.pdf': 'application/pdf',
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.png': 'image/png',
        '.bmp': 'image/bmp',
        '.tiff': 'image/tiff',
        '.webp': 'image/webp'
    }
    return mime_types.get(ext, 'application/octet-stream')

def extract_text_with_mistral_ocr(file_path, mistral_client):
    """Extract text from PDF or image using Mistral OCR"""
    filename = os.path.basename(file_path)
    logger.info(f"Extracting text from: {filename}")
    
    try:
        # Determine file type and encode
        if file_path.lower().endswith(PDF_EXTENSION):
            base64_content = encode_pdf_to_base64(file_path)
            file_type = "PDF"
        elif file_path.lower().endswith(IMAGE_EXTENSIONS):
            base64_content = encode_image_to_base64(file_path)
            file_type = "Image"
        else:
            logger.error(f"Unsupported file type: {filename}")
            return None
        
        if not base64_content:
            return None
        
        # Get MIME type
        mime_type = get_file_mime_type(file_path)
        
        # Process with Mistral OCR
        logger.info(f"Processing {file_type} with Mistral OCR: {filename}")
        
        ocr_response = mistral_client.ocr.process(
            model=MISTRAL_OCR_MODEL,
            document={
                "type": "document_url",
                "document_url": f"data:{mime_type};base64,{base64_content}"
            },
            include_image_base64=False  # Set to True if you need image data
        )
        
        # Extract text from OCR response
        extracted_text = ""
        if hasattr(ocr_response, 'text'):
            extracted_text = ocr_response.text
        elif hasattr(ocr_response, 'content'):
            extracted_text = ocr_response.content
        elif hasattr(ocr_response, 'choices') and len(ocr_response.choices) > 0:
            extracted_text = ocr_response.choices[0].message.content
        else:
            # Handle different response formats
            extracted_text = str(ocr_response)
        
        logger.info(f"Successfully extracted text from: {filename}")
        return extracted_text.strip()
        
    except Exception as e:
        logger.error(f"Error extracting text from {filename}: {e}")
        return None

def optimize_text_with_mistral_chat(text, source_name, mistral_client):
    """Optimize extracted text using Mistral Chat API"""
    try:
        logger.info(f"Optimizing text with Mistral Chat for: {source_name}")
        
        # Skip if text is too short
        if len(text.strip()) < 50:
            return f"[MINIMAL TEXT - Original]\n{text}"
        
        # Create optimization prompt
        optimization_prompt = f"""
                    Please optimize and improve the following OCR-extracted text. The text may contain errors, formatting issues, or unclear sections due to OCR processing.

                    Your task is to:
                    1. CREATE A STRUCTURED, COHERENT, AND READABLE VERSION of the text.
                    2. MAKE SURE THAT THE RELATIONSHIPS BETWEEN DIFFERENT PARTS OF THE TEXT ARE CLEAR.
                    3. MAKE A DETAILED SUMMARY OF THE TEXT. make a detailed summary of the text.
                    4. CORRECT ANY OCR ERRORS or formatting issues.
                    5. ENSURE THE TEXT IS SUITABLE FOR FURTHER ANALYSIS OR PRESENTATION.
                    6. Maintain the original meaning and context of the text.
                    7. All pages should be included in the final output.



                    Source: {source_name}

                    Original extracted text:
                    {text}

                    Please provide the optimized version:
"""

        # Send request to Mistral Chat
        messages = [
            {
                "role": "user",
                "content": optimization_prompt
            }
        ]
        
        chat_response = mistral_client.chat.complete(
            model=MISTRAL_CHAT_MODEL,
            messages=messages,
            max_tokens=3000,
            temperature=0.2
        )
        
        optimized_text = chat_response.choices[0].message.content
        logger.info(f"Text optimized successfully for: {source_name}")
        return optimized_text
        
    except Exception as e:
        logger.error(f"Error optimizing text with Mistral Chat: {e}")
        return f"[OPTIMIZATION FAILED - Original Text]\n{text}"

def process_single_file(file_path, mistral_client):
    """Process a single file with Mistral OCR and optimization"""
    filename = os.path.basename(file_path)
    file_type = "PDF" if file_path.lower().endswith(PDF_EXTENSION) else "Image"
    
    # Create output folder
    output_folder = create_unique_folder(
        OUTPUT_BASE_DIR, 
        f"{file_type}_{os.path.splitext(filename)[0]}_"
    )
    
    try:
        # Step 1: Extract text using Mistral OCR
        raw_text = extract_text_with_mistral_ocr(file_path, mistral_client)
        
        if not raw_text or len(raw_text.strip()) < 10:
            logger.warning(f"No meaningful text extracted from: {filename}")
            return None, None, filename
        
        # Step 2: Save raw extracted text
        raw_txt_path = os.path.join(output_folder, "mistral_ocr_extracted_text.txt")
        with open(raw_txt_path, "w", encoding="utf-8") as f:
            f.write(f"Source {file_type}: {filename}\n")
            f.write(f"Extracted by: {MISTRAL_OCR_MODEL}\n")
            f.write(f"Processed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(raw_text)
        
        # Step 3: Optimize text with Mistral Chat
        optimized_text = optimize_text_with_mistral_chat(raw_text, filename, mistral_client)
        
        # Step 4: Save optimized text
        optimized_txt_path = os.path.join(output_folder, "mistral_chat_optimized_text.txt")
        with open(optimized_txt_path, "w", encoding="utf-8") as f:
            f.write(f"Source {file_type}: {filename}\n")
            f.write(f"OCR by: {MISTRAL_OCR_MODEL}\n")
            f.write(f"Optimized by: {MISTRAL_CHAT_MODEL}\n")
            f.write(f"Processed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(optimized_text)
        
        logger.info(f"Successfully processed: {filename}")
        logger.info(f"Raw text saved to: {raw_txt_path}")
        logger.info(f"Optimized text saved to: {optimized_txt_path}")
        
        return raw_text, optimized_text, filename
        
    except Exception as e:
        logger.error(f"Error processing file {filename}: {e}")
        return None, None, filename

def process_files_parallel(file_paths, mistral_client):
    """Process multiple files in parallel with rate limiting"""
    results = []
    
    # Process files in batches to respect API rate limits
    for i in range(0, len(file_paths), BATCH_SIZE):
        batch = file_paths[i:i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        total_batches = (len(file_paths) + BATCH_SIZE - 1) // BATCH_SIZE
        
        logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} files)")
        
        # Process batch with ThreadPoolExecutor (for I/O bound API calls)
        with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, len(batch))) as executor:
            # Submit all tasks in the batch
            futures = [
                executor.submit(process_single_file, file_path, mistral_client)
                for file_path in batch
            ]
            
            # Collect results
            for future in futures:
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"Error in batch processing: {e}")
                    results.append((None, None, "unknown"))
        
        # Rate limiting between batches
        if i + BATCH_SIZE < len(file_paths):
            logger.info("Waiting between batches to respect API rate limits...")
            time.sleep(2)
    
    return results

def main():
    start_time = time.time()
    
    # Check if input directory exists
    if not os.path.isdir(INPUT_DIR):
        logger.error(f"'{INPUT_DIR}' is not a valid directory!")
        return
    
    # Initialize Mistral client
    try:
        mistral_client = Mistral(api_key=MISTRAL_API_KEY)
        logger.info(f"Mistral AI client initialized")
        logger.info(f"OCR Model: {MISTRAL_OCR_MODEL}")
        logger.info(f"Chat Model: {MISTRAL_CHAT_MODEL}")
    except Exception as e:
        logger.error(f"Error initializing Mistral client: {e}")
        logger.error("Please check your API key and internet connection.")
        return
    
    # Get all supported files
    supported_files = []
    for filename in os.listdir(INPUT_DIR):
        if filename.lower().endswith(PDF_EXTENSION + IMAGE_EXTENSIONS):
            supported_files.append(os.path.join(INPUT_DIR, filename))
    
    if not supported_files:
        logger.info("No supported files found.")
        return
    
    logger.info(f"Found {len(supported_files)} files to process")
    
    # Process all files
    results = process_files_parallel(supported_files, mistral_client)
    
    # Write consolidated results
    consolidated_raw_path = os.path.join(OUTPUT_BASE_DIR, CONSOLIDATED_TEXT_FILE)
    consolidated_optimized_path = os.path.join(OUTPUT_BASE_DIR, OPTIMIZED_TEXT_FILE)
    
    successful_results = [r for r in results if r[0] is not None and r[1] is not None]
    
    if successful_results:
        with open(consolidated_raw_path, "w", encoding="utf-8") as f_raw, \
             open(consolidated_optimized_path, "w", encoding="utf-8") as f_optimized:
            
            # Write headers
            f_raw.write(f"Extracted by Mistral OCR ({MISTRAL_OCR_MODEL})\n")
            f_raw.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f_raw.write("="*80 + "\n\n")
            
            f_optimized.write(f"OCR by {MISTRAL_OCR_MODEL} | Optimized by {MISTRAL_CHAT_MODEL}\n")
            f_optimized.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f_optimized.write("="*80 + "\n\n")
            
            # Write results
            for raw_text, optimized_text, filename in successful_results:
                f_raw.write(f"\n\n--- File: {filename} ---\n\n")
                f_raw.write(raw_text)
                
                f_optimized.write(f"\n\n--- File: {filename} ---\n\n")
                f_optimized.write(optimized_text)
        
        logger.info(f"Raw extracted text saved to: {consolidated_raw_path}")
        logger.info(f"Optimized text saved to: {consolidated_optimized_path}")
    
    # Summary
    elapsed_time = time.time() - start_time
    successful_count = len(successful_results)
    failed_count = len(results) - successful_count
    
    logger.info(f"\n" + "="*60)
    logger.info(f"PROCESSING COMPLETE!")
    logger.info(f"Total time: {elapsed_time:.2f} seconds")
    logger.info(f"Files processed successfully: {successful_count}")
    logger.info(f"Files failed: {failed_count}")
    logger.info(f"Success rate: {(successful_count/len(results)*100):.1f}%")
    logger.info(f"="*60)

if __name__ == "__main__":
    main()