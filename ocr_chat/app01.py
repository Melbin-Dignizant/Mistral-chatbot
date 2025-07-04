import os
import fitz  # PyMuPDF
from PIL import Image
import pytesseract
import shutil
from datetime import datetime
from mistralai import Mistral
import time
from dotenv import load_dotenv
import asyncio
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing
from functools import partial
import logging

# Load environment variables from .env file
load_dotenv()

# Configurations
INPUT_DIR = r"D:\Melbin\Geometra_Chatbot\cortex_mistral\ocr_chat\static"
OUTPUT_BASE_DIR = r"D:\Melbin\Geometra_Chatbot\cortex_mistral\ocr_chat\output"
CONSOLIDATED_TEXT_FILE = "consolidated_extracted_text.txt"
OPTIMIZED_TEXT_FILE = "mistral_optimized_text.txt"

# Mistral AI Configuration
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
MISTRAL_MODEL = "mistral-medium-latest"

# Performance Configuration
MAX_WORKERS = min(multiprocessing.cpu_count(), 8)  # Limit workers to prevent overload
BATCH_SIZE = 5  # Process files in batches
OCR_DPI = 200  # Reduced DPI for faster processing (was 300)

# Supported extensions
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png')
PDF_EXTENSION = ('.pdf',)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Ensure output base directory exists
os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)

def create_unique_folder(base_dir, prefix="PDF_"):
    """Creates a unique folder with timestamp"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"{prefix}{timestamp}"
    folder_path = os.path.join(base_dir, folder_name)
    os.makedirs(folder_path, exist_ok=True)
    return folder_path

def pdf_to_images_parallel(pdf_path, output_folder, dpi=OCR_DPI):
    """Converts PDF pages to images using parallel processing"""
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
    image_paths = []
    
    try:
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        
        # Use ThreadPoolExecutor for I/O bound operations
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = []
            
            for page_num in range(total_pages):
                future = executor.submit(
                    _convert_single_page, 
                    pdf_path, page_num, output_folder, pdf_name, dpi
                )
                futures.append(future)
            
            # Collect results
            for future in futures:
                img_path = future.result()
                if img_path:
                    image_paths.append(img_path)
        
        doc.close()
        
    except Exception as e:
        logger.error(f"Error converting PDF to images: {e}")
    
    return sorted(image_paths)  # Sort to maintain page order

def _convert_single_page(pdf_path, page_num, output_folder, pdf_name, dpi):
    """Convert a single PDF page to image (for parallel processing)"""
    try:
        doc = fitz.open(pdf_path)
        page = doc.load_page(page_num)
        pix = page.get_pixmap(dpi=dpi)
        img_path = os.path.join(output_folder, f"{pdf_name}_page_{page_num + 1}.png")
        pix.save(img_path)
        doc.close()
        return img_path
    except Exception as e:
        logger.error(f"Error converting page {page_num}: {e}")
        return None

def extract_text_from_image_optimized(image_path):
    """Optimized OCR text extraction with better Tesseract configuration"""
    try:
        # Use optimized Tesseract configuration for better speed/accuracy balance
        custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,!?;:()[]{}"\'-_+=*/<>@#$%^&|~`'
        
        img = Image.open(image_path)
        
        # Resize image if too large (speeds up OCR)
        max_dimension = 2000
        if max(img.size) > max_dimension:
            ratio = max_dimension / max(img.size)
            new_size = tuple(int(dim * ratio) for dim in img.size)
            img = img.resize(new_size, Image.Resampling.LANCZOS)
        
        # Convert to grayscale for faster processing
        if img.mode != 'L':
            img = img.convert('L')
        
        text = pytesseract.image_to_string(img, config=custom_config)
        return text.strip()
        
    except Exception as e:
        logger.error(f"Error processing image {image_path}: {e}")
        return ""

def extract_text_batch_parallel(image_paths):
    """Extract text from multiple images in parallel"""
    texts = []
    
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all OCR tasks
        futures = [executor.submit(extract_text_from_image_optimized, img_path) 
                  for img_path in image_paths]
        
        # Collect results in order
        for future in futures:
            text = future.result()
            texts.append(text)
    
    return texts

def optimize_text_with_mistral_batch(texts_data, mistral_client):
    """Optimize multiple texts with Mistral AI in batch"""
    optimized_results = []
    
    for text_data in texts_data:
        text, source_name = text_data
        
        # Skip if text is too short
        if len(text.strip()) < 50:
            optimized_results.append(f"[MINIMAL TEXT - Original]\n{text}")
            continue
        
        try:
            logger.info(f"Optimizing text with Mistral AI for: {source_name}")
            
            # Simplified optimization prompt for faster processing
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

            messages = [{"role": "user", "content": optimization_prompt}]
            
            chat_response = mistral_client.chat.complete(
                model=MISTRAL_MODEL,
                messages=messages,
                max_tokens=2000,  # Reduced for faster processing
                temperature=0.2
            )
            
            optimized_text = chat_response.choices[0].message.content
            optimized_results.append(optimized_text)
            
        except Exception as e:
            logger.error(f"Error optimizing text with Mistral: {e}")
            optimized_results.append(f"[OPTIMIZATION FAILED - Original Text]\n{text}")
        
        # Reduced delay for batch processing
        time.sleep(0.5)
    
    return optimized_results

def process_pdf_optimized(pdf_path, output_folder, mistral_client):
    """Optimized PDF processing with parallel operations"""
    pdf_name = os.path.basename(pdf_path)
    logger.info(f"Processing PDF: {pdf_name}")
    
    # Step 1: Convert PDF to images (parallel)
    image_paths = pdf_to_images_parallel(pdf_path, output_folder)
    if not image_paths:
        return None, None
    
    # Step 2: Extract text from all images (parallel)
    extracted_texts = extract_text_batch_parallel(image_paths)
    
    # Filter out empty texts
    valid_texts = [text for text in extracted_texts if text.strip()]
    
    if not valid_texts:
        return None, None
    
    raw_text = "\n\n".join(valid_texts)
    
    # Step 3: Save raw extracted text
    raw_txt_path = os.path.join(output_folder, "raw_extracted_text.txt")
    with open(raw_txt_path, "w", encoding="utf-8") as f:
        f.write(f"Source PDF: {pdf_name}\n\n")
        f.write(raw_text)
    
    # Step 4: Optimize text with Mistral AI
    texts_data = [(raw_text, pdf_name)]
    optimized_results = optimize_text_with_mistral_batch(texts_data, mistral_client)
    optimized_text = optimized_results[0]
    
    # Step 5: Save optimized text
    optimized_txt_path = os.path.join(output_folder, "mistral_optimized_text.txt")
    with open(optimized_txt_path, "w", encoding="utf-8") as f:
        f.write(f"Source PDF: {pdf_name}\n")
        f.write(f"Optimized by: {MISTRAL_MODEL}\n")
        f.write(f"Processed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(optimized_text)
    
    logger.info(f"PDF processing completed: {pdf_name}")
    return raw_text, optimized_text

def process_image_optimized(image_path, output_folder, mistral_client):
    """Optimized image processing"""
    img_name = os.path.basename(image_path)
    logger.info(f"Processing image: {img_name}")
    
    # Step 1: Extract text
    raw_text = extract_text_from_image_optimized(image_path)
    if not raw_text or len(raw_text.strip()) < 10:
        return None, None
    
    # Step 2: Save raw extracted text
    raw_txt_path = os.path.join(output_folder, f"{os.path.splitext(img_name)[0]}_raw_extracted.txt")
    with open(raw_txt_path, "w", encoding="utf-8") as f:
        f.write(f"Source Image: {img_name}\n\n")
        f.write(raw_text)
    
    # Step 3: Optimize text with Mistral AI
    texts_data = [(raw_text, img_name)]
    optimized_results = optimize_text_with_mistral_batch(texts_data, mistral_client)
    optimized_text = optimized_results[0]
    
    # Step 4: Save optimized text
    optimized_txt_path = os.path.join(output_folder, f"{os.path.splitext(img_name)[0]}_mistral_optimized.txt")
    with open(optimized_txt_path, "w", encoding="utf-8") as f:
        f.write(f"Source Image: {img_name}\n")
        f.write(f"Optimized by: {MISTRAL_MODEL}\n")
        f.write(f"Processed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(optimized_text)
    
    logger.info(f"Image processing completed: {img_name}")
    return raw_text, optimized_text

def process_files_in_batches(files, mistral_client):
    """Process files in batches for better resource management"""
    all_results = []
    
    for i in range(0, len(files), BATCH_SIZE):
        batch = files[i:i + BATCH_SIZE]
        batch_results = []
        
        logger.info(f"Processing batch {i//BATCH_SIZE + 1}/{(len(files) + BATCH_SIZE - 1)//BATCH_SIZE}")
        
        for file_path in batch:
            filename = os.path.basename(file_path)
            
            if filename.lower().endswith(PDF_EXTENSION):
                pdf_output_folder = create_unique_folder(OUTPUT_BASE_DIR, f"PDF_{os.path.splitext(filename)[0]}_")
                result = process_pdf_optimized(file_path, pdf_output_folder, mistral_client)
                batch_results.append(('PDF', filename, result))
                
            elif filename.lower().endswith(IMAGE_EXTENSIONS):
                img_output_folder = create_unique_folder(OUTPUT_BASE_DIR, f"IMG_{os.path.splitext(filename)[0]}_")
                result = process_image_optimized(file_path, img_output_folder, mistral_client)
                batch_results.append(('IMAGE', filename, result))
        
        all_results.extend(batch_results)
        
        # Brief pause between batches
        time.sleep(1)
    
    return all_results

def main():
    start_time = time.time()
    
    # Check if input directory exists
    if not os.path.isdir(INPUT_DIR):
        logger.error(f"'{INPUT_DIR}' is not a valid directory!")
        return
    
    # Initialize Mistral client
    try:
        mistral_client = Mistral(api_key=MISTRAL_API_KEY)
        logger.info(f"Mistral AI client initialized with model: {MISTRAL_MODEL}")
    except Exception as e:
        logger.error(f"Error initializing Mistral client: {e}")
        return
    
    # Get all files to process
    files_to_process = []
    for filename in os.listdir(INPUT_DIR):
        if filename.lower().endswith(PDF_EXTENSION + IMAGE_EXTENSIONS):
            files_to_process.append(os.path.join(INPUT_DIR, filename))
    
    if not files_to_process:
        logger.info("No files to process found.")
        return
    
    logger.info(f"Found {len(files_to_process)} files to process")
    
    # Process files in batches
    results = process_files_in_batches(files_to_process, mistral_client)
    
    # Write consolidated results
    consolidated_raw_path = os.path.join(OUTPUT_BASE_DIR, CONSOLIDATED_TEXT_FILE)
    consolidated_optimized_path = os.path.join(OUTPUT_BASE_DIR, OPTIMIZED_TEXT_FILE)
    
    with open(consolidated_raw_path, "w", encoding="utf-8") as f_raw, \
         open(consolidated_optimized_path, "w", encoding="utf-8") as f_optimized:
        
        f_optimized.write(f"Optimized by Mistral AI ({MISTRAL_MODEL})\n")
        f_optimized.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f_optimized.write("="*80 + "\n\n")
        
        for file_type, filename, (raw_text, optimized_text) in results:
            if raw_text and optimized_text:
                f_raw.write(f"\n\n--- {file_type}: {filename} ---\n\n")
                f_raw.write(raw_text)
                
                f_optimized.write(f"\n\n--- {file_type}: {filename} ---\n\n")
                f_optimized.write(optimized_text)
    
    elapsed_time = time.time() - start_time
    logger.info(f"All processing completed in {elapsed_time:.2f} seconds!")
    logger.info(f"Raw extracted text saved to: {consolidated_raw_path}")
    logger.info(f"Mistral-optimized text saved to: {consolidated_optimized_path}")

if __name__ == "__main__":
    main()



###================================================================
# import os
# import fitz  # PyMuPDF
# from PIL import Image
# import pytesseract
# import shutil
# from datetime import datetime

# # Configurations
# INPUT_DIR = r"D:\Melbin\Geometra_Chatbot\cortex_mistral\documents\output"  # Must be a directory
# OUTPUT_BASE_DIR = r"D:\Melbin\Geometra_Chatbot\cortex_mistral\ocr_chat\output"
# CONSOLIDATED_TEXT_FILE = "consolidated_extracted_text.txt"

# # Supported extensions
# IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png')
# PDF_EXTENSION = ('.pdf',)

# # Ensure output base directory exists
# os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)

# def create_unique_folder(base_dir, prefix="PDF_"):
#     """Creates a unique folder with timestamp"""
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     folder_name = f"{prefix}{timestamp}"
#     folder_path = os.path.join(base_dir, folder_name)
#     os.makedirs(folder_path, exist_ok=True)
#     return folder_path

# def pdf_to_images(pdf_path, output_folder, dpi=300):
#     """Converts PDF pages to images and returns image paths"""
#     pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
#     image_paths = []
    
#     try:
#         doc = fitz.open(pdf_path)
#         for page_num in range(len(doc)):
#             page = doc.load_page(page_num)
#             pix = page.get_pixmap(dpi=dpi)
#             img_path = os.path.join(output_folder, f"{pdf_name}_page_{page_num + 1}.png")
#             pix.save(img_path)
#             image_paths.append(img_path)
#     except Exception as e:
#         print(f"⚠️ Error converting PDF to images: {e}")
#     return image_paths

# def extract_text_from_image(image_path):
#     """Extracts text from an image using Tesseract OCR"""
#     try:
#         img = Image.open(image_path)
#         text = pytesseract.image_to_string(img)
#         return text.strip()
#     except Exception as e:
#         print(f"⚠️ Error processing image {image_path}: {e}")
#         return ""

# def process_pdf(pdf_path, output_folder):
#     """Processes a PDF: Converts to images → Extracts text → Saves results"""
#     pdf_name = os.path.basename(pdf_path)
#     print(f"\n📄 Processing PDF: {pdf_name}")
    
#     # Step 1: Convert PDF to images
#     image_paths = pdf_to_images(pdf_path, output_folder)
#     if not image_paths:
#         return None
    
#     # Step 2: Extract text from each image
#     extracted_texts = []
#     for img_path in image_paths:
#         text = extract_text_from_image(img_path)
#         if text:
#             extracted_texts.append(text)
    
#     # Step 3: Save extracted text per PDF
#     txt_output_path = os.path.join(output_folder, f"extracted_text.txt")
#     with open(txt_output_path, "w", encoding="utf-8") as f:
#         f.write(f"📄 Source PDF: {pdf_name}\n\n")
#         f.write("\n\n".join(extracted_texts))
    
#     print(f"✅ Extracted text saved to: {txt_output_path}")
#     return "\n\n".join(extracted_texts)

# def process_image(image_path, output_folder):
#     """Processes an image and extracts text"""
#     img_name = os.path.basename(image_path)
#     print(f"\n🖼️ Processing image: {img_name}")
    
#     text = extract_text_from_image(image_path)
#     if not text:
#         return None
    
#     # Save extracted text per image
#     txt_output_path = os.path.join(output_folder, f"{os.path.splitext(img_name)[0]}_extracted.txt")
#     with open(txt_output_path, "w", encoding="utf-8") as f:
#         f.write(f"🖼️ Source Image: {img_name}\n\n")
#         f.write(text)
    
#     print(f"✅ Extracted text saved to: {txt_output_path}")
#     return text

# def main():
#     # Check if input directory exists
#     if not os.path.isdir(INPUT_DIR):
#         print(f"❌ Error: '{INPUT_DIR}' is not a valid directory!")
#         return
    
#     # Open consolidated output file
#     consolidated_text_path = os.path.join(OUTPUT_BASE_DIR, CONSOLIDATED_TEXT_FILE)
#     with open(consolidated_text_path, "w", encoding="utf-8") as f_consolidated:
#         for filename in os.listdir(INPUT_DIR):
#             file_path = os.path.join(INPUT_DIR, filename)
            
#             # Process PDFs
#             if filename.lower().endswith(PDF_EXTENSION):
#                 pdf_output_folder = create_unique_folder(OUTPUT_BASE_DIR, f"PDF_{os.path.splitext(filename)[0]}_")
#                 pdf_text = process_pdf(file_path, pdf_output_folder)
#                 if pdf_text:
#                     f_consolidated.write(f"\n\n--- PDF: {filename} ---\n\n")
#                     f_consolidated.write(pdf_text)
            
#             # Process Images
#             elif filename.lower().endswith(IMAGE_EXTENSIONS):
#                 img_output_folder = create_unique_folder(OUTPUT_BASE_DIR, f"IMG_{os.path.splitext(filename)[0]}_")
#                 img_text = process_image(file_path, img_output_folder)
#                 if img_text:
#                     f_consolidated.write(f"\n\n--- Image: {filename} ---\n\n")
#                     f_consolidated.write(img_text)
    
#     print(f"\n🎉 All done! Consolidated text saved to: {consolidated_text_path}")

# if __name__ == "__main__":
#     main()

