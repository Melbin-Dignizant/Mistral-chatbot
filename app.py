# import base64
# import requests
# import os
# import json
# from mistralai import Mistral
# from dotenv import load_dotenv
# from PIL import Image
# import pytesseract
# from pdf2image import convert_from_path

# # Load environment variables from .env file
# load_dotenv()

# class DocumentProcessor:
#     def __init__(self, base_dir=None):
#         """Initialize the document processor with base directory paths."""
#         self.base_dir = base_dir or r"D:\Melbin\Geometra_Chatbot\cortex_mistral"
#         self.documents_dir = os.path.join(self.base_dir, "documents")
#         self.output_dir = os.path.join(self.documents_dir, "output")
#         self.fallback_text_path = os.path.join(self.base_dir, "sample.txt")
#         self.api_key = os.getenv("MISTRAL_AI_API_KEY")
#         self.model = "mistral-medium-latest"
        
#         # Ensure directories exist
#         os.makedirs(self.output_dir, exist_ok=True)
    
#     def convert_pdf_to_images(self, pdf_path, dpi=300, image_format='JPEG'):
#         """
#         Step 1: Convert PDF to images
#         """
#         print(f"🔄 Converting PDF to images...")
        
#         if not os.path.exists(pdf_path):
#             raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
#         # Convert PDF pages to images
#         images = convert_from_path(pdf_path, dpi=dpi)
#         image_paths = []
        
#         for i, image in enumerate(images):
#             output_path = os.path.join(self.output_dir, f'page_{i+1}.{image_format.lower()}')
#             image.save(output_path, image_format)
#             image_paths.append(output_path)
#             print(f"✅ Saved page {i+1} to {output_path}")
        
#         return image_paths
    
#     def extract_text_from_images(self, image_paths=None):
#         """
#         Step 2: Extract text from all images using OCR
#         """
#         print(f"🔄 Extracting text from images...")
        
#         if image_paths is None:
#             # Process all images in output directory
#             image_extensions = ('.jpg', '.jpeg', '.png')
#             image_paths = [
#                 os.path.join(self.output_dir, f) 
#                 for f in os.listdir(self.output_dir) 
#                 if f.lower().endswith(image_extensions)
#             ]
        
#         with open(self.fallback_text_path, "w", encoding="utf-8") as f_out:
#             for image_path in image_paths:
#                 filename = os.path.basename(image_path)
#                 print(f"Processing {filename}...")
                
#                 img = Image.open(image_path)
#                 text = pytesseract.image_to_string(img)
                
#                 f_out.write(f"--- Text from {filename} ---\n")
#                 f_out.write(text + "\n\n")
        
#         print(f"✅ Text extracted and saved to '{self.fallback_text_path}'")
#         return self.fallback_text_path
    
#     def encode_image(self, image_path):
#         """Encode the image to base64."""
#         try:
#             with open(image_path, "rb") as image_file:
#                 return base64.b64encode(image_file.read()).decode('utf-8')
#         except FileNotFoundError:
#             print(f"Error: The file {image_path} was not found.")
#             return None
#         except Exception as e:
#             print(f"Error: {e}")
#             return None
    
#     def search_fallback_text_file(self, question):
#         """Search the fallback text file for a possible answer."""
#         try:
#             with open(self.fallback_text_path, "r", encoding="utf-8") as f:
#                 content = f.read()
#                 # Simple keyword-based search logic
#                 if question.lower() in content.lower():
#                     return f"Relevant information from fallback file: {question} appears in:\n\n{content}"
#                 else:
#                     return "The question was not found in the fallback file either."
#         except FileNotFoundError:
#             return f"Fallback file {self.fallback_text_path} not found."
#         except Exception as e:
#             return f"Error reading fallback file: {e}"
    
#     def answer_question_with_image(self, question, image_path):
#         """
#         Step 3: Use Mistral AI to answer questions based on document image
#         """
#         print(f"🔄 Processing question with AI model...")
        
#         # Encode image
#         base64_image = self.encode_image(image_path)
#         if not base64_image:
#             return {"error": "Failed to encode image"}
        
#         # Prepare messages
#         messages = [
#             {
#                 "role": "system",
#                 "content": '''You are an expert in document understanding assistant designed to analyze an input document and answer user questions specifically related to that document.

#                 When a user provides a question and a document, carefully check if the question pertains only to the content of the provided document.

#                 - If the question is unrelated to the document, first politely state that the question is outside the document's scope, then answer the question using your general knowledge.
#                 - If the question relates to the document, analyze the relevant parts of the document thoroughly to produce an accurate, well-structured answer.

#                 Additionally, when users submit questions or requests through chat involving documents or files, rewrite and refine their input text queries to improve clarity, correctness, and specificity.

#                 Your output must be structured clearly, highlighting:
#                 - The refined user query (if rewriting was necessary).
#                 - Whether the question is related to the document.
#                 - A detailed, accurate answer (with references to the document where relevant).

#                 Always encourage reasoning steps before providing conclusions. Ensure accuracy and clarity in all responses.

#                 Provide output as a JSON object with the following fields:
#                 - "refined_query": the rewritten, polished user question.
#                 - "is_question_related": boolean indicating if question relates to the document.
#                 - "answer": the finally produced answer to the question, including references to the document if applicable.'''
#             },
#             {
#                 "role": "user",
#                 "content": [
#                     {"type": "text", "text": question},
#                     {"type": "image_url", "image_url": f"data:image/jpeg;base64,{base64_image}"}
#                 ]
#             }
#         ]
        
#         # Initialize Mistral Client and make request
#         client = Mistral(api_key=self.api_key)
#         response = client.chat.complete(model=self.model, messages=messages)
#         model_output_raw = response.choices[0].message.content.strip()
        
#         # Try parsing the output as JSON
#         try:
#             model_output = json.loads(model_output_raw)
#         except json.JSONDecodeError:
#             print("Model response is not JSON, using raw output")
#             return {
#                 "refined_query": question,
#                 "is_question_related": False,
#                 "answer": model_output_raw
#             }
        
#         # Decision logic for fallback
#         is_question_related = model_output.get("is_question_related", False)
#         answer = model_output.get("answer", "")
        
#         fallback_needed = (
#             not is_question_related or
#             not answer.strip() or
#             "not found" in answer.lower() or
#             "unable to" in answer.lower()
#         )
        
#         # Use fallback if needed
#         if fallback_needed:
#             fallback_answer = self.search_fallback_text_file(question)
#             return {
#                 "refined_query": model_output.get("refined_query", question),
#                 "is_question_related": is_question_related,
#                 "answer": fallback_answer,
#                 "used_fallback": True
#             }
#         else:
#             model_output["used_fallback"] = False
#             return model_output
    
#     def process_document_and_answer(self, pdf_path, question, target_page=None):
#         """
#         Complete pipeline: PDF -> Images -> OCR -> AI Q&A
#         """
#         print(f"🚀 Starting complete document processing pipeline...")
#         print(f"📄 PDF: {pdf_path}")
#         print(f"❓ Question: {question}")
        
#         try:
#             # Step 1: Convert PDF to images
#             image_paths = self.convert_pdf_to_images(pdf_path)
            
#             # Step 2: Extract text from images (OCR)
#             self.extract_text_from_images(image_paths)
            
#             # Step 3: Answer question using specific page or first page
#             if target_page and target_page <= len(image_paths):
#                 target_image = image_paths[target_page - 1]
#                 print(f"🎯 Using page {target_page}: {target_image}")
#             else:
#                 target_image = image_paths[0] if image_paths else None
#                 print(f"🎯 Using first page: {target_image}")
            
#             if not target_image:
#                 return {"error": "No images found to process"}
            
#             # Step 4: Get AI answer
#             result = self.answer_question_with_image(question, target_image)
            
#             print(f"✅ Pipeline completed successfully!")
#             return result
            
#         except Exception as e:
#             error_msg = f"Pipeline failed: {str(e)}"
#             print(f"❌ {error_msg}")
#             return {"error": error_msg}

# # Example usage and convenience functions
# def main():
#     """Example usage of the integrated pipeline"""
    
#     # Initialize processor
#     processor = DocumentProcessor()
    
#     # Configuration
#     pdf_path = r"D:\Melbin\Geometra_Chatbot\cortex_mistral\documents\02.pdf"
#     question = "what is the area of room 108?"
#     target_page = 2  # Optional: specify which page to analyze
    
#     # Run complete pipeline
#     result = processor.process_document_and_answer(pdf_path, question, target_page)
    
#     # Display results
#     print("\n" + "="*50)
#     print("FINAL RESULT:")
#     print("="*50)
#     print(json.dumps(result, indent=4, ensure_ascii=False))

# # Alternative: Step-by-step usage
# def step_by_step_example():
#     """Example of using individual steps"""
    
#     processor = DocumentProcessor()
    
#     # Step 1: Convert PDF to images
#     pdf_path = r"D:\Melbin\Geometra_Chatbot\cortex_mistral\documents\02.pdf"
#     image_paths = processor.convert_pdf_to_images(pdf_path)
    
#     # Step 2: Extract text (OCR)
#     processor.extract_text_from_images(image_paths)
    
#     # Step 3: Answer questions about specific pages
#     questions = [
 
#         "What are the dimensions of the building?",

#     ]
    
#     for question in questions:
#         print(f"\n🔍 Question: {question}")
#         result = processor.answer_question_with_image(question, image_paths[1])  # Using page 2
#         print(f"📝 Answer: {result.get('answer', 'No answer found')}")

# if __name__ == "__main__":
#     main()


####=================================================================================================================================

# import base64
# import requests
# import os
# import json
# from mistralai import Mistral
# from dotenv import load_dotenv
# from PIL import Image
# import pytesseract
# from pdf2image import convert_from_path

# # Load environment variables from .env file
# load_dotenv()

# class DocumentProcessor:
#     def __init__(self, base_dir=None):
#         """Initialize the document processor with base directory paths."""
#         self.base_dir = base_dir or r"D:\Melbin\Geometra_Chatbot\cortex_mistral"
#         self.documents_dir = os.path.join(self.base_dir, "documents")
#         self.output_dir = os.path.join(self.documents_dir, "output")
#         self.fallback_text_path = os.path.join(self.base_dir, "sample.txt")
#         self.api_key = os.getenv("MISTRAL_AI_API_KEY")
#         self.model = "mistral-medium-latest"
        
#         # Store processed image paths for reuse
#         self.current_image_paths = []
#         self.current_pdf_path = None
        
#         # Ensure directories exist
#         os.makedirs(self.output_dir, exist_ok=True)
    
#     def convert_pdf_to_images(self, pdf_path, dpi=300, image_format='JPEG'):
#         """Convert PDF to images"""
#         if not os.path.exists(pdf_path):
#             raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
#         images = convert_from_path(pdf_path, dpi=dpi)
#         image_paths = []
        
#         for i, image in enumerate(images):
#             output_path = os.path.join(self.output_dir, f'page_{i+1}.{image_format.lower()}')
#             image.save(output_path, image_format)
#             image_paths.append(output_path)
        
#         return image_paths
    
#     def extract_text_from_images(self, image_paths=None):
#         """Extract text from all images using OCR"""
#         if image_paths is None:
#             image_extensions = ('.jpg', '.jpeg', '.png')
#             image_paths = [
#                 os.path.join(self.output_dir, f) 
#                 for f in os.listdir(self.output_dir) 
#                 if f.lower().endswith(image_extensions)
#             ]
        
#         with open(self.fallback_text_path, "w", encoding="utf-8") as f_out:
#             for image_path in image_paths:
#                 filename = os.path.basename(image_path)
#                 img = Image.open(image_path)
#                 text = pytesseract.image_to_string(img)
#                 f_out.write(f"--- Text from {filename} ---\n")
#                 f_out.write(text + "\n\n")
        
#         return self.fallback_text_path
    
#     def encode_image(self, image_path):
#         """Encode the image to base64."""
#         try:
#             with open(image_path, "rb") as image_file:
#                 return base64.b64encode(image_file.read()).decode('utf-8')
#         except FileNotFoundError:
#             return None
#         except Exception as e:
#             return None
    
#     def search_fallback_text_file(self, question):
#         """Search the fallback text file for a possible answer."""
#         try:
#             with open(self.fallback_text_path, "r", encoding="utf-8") as f:
#                 content = f.read()
#                 if question.lower() in content.lower():
#                     return f"Relevant information from fallback file: {question} appears in:\n\n{content}"
#                 else:
#                     return "The question was not found in the fallback file either."
#         except FileNotFoundError:
#             return f"Fallback file {self.fallback_text_path} not found."
#         except Exception as e:
#             return f"Error reading fallback file: {e}"
    
#     def answer_question_with_image(self, question, image_path):
#         """Use Mistral AI to answer questions based on document image"""
#         base64_image = self.encode_image(image_path)
#         if not base64_image:
#             return {"error": "Failed to encode image"}
        
#         messages = [
#                     {
#                     "role": "system",
#                     "content": '''You are an expert document understanding assistant. When a user submits a question and a document or file:

# First, determine if the question is about the content of the provided document.

# If the question is related to the document, analyze the relevant sections and provide a clear, concise summary in a structured, human-readable format (such as a bulleted or numbered list)—not in JSON.

# If the question is unrelated to the document, answer it directly using your general knowledge, maintaining accuracy, helpfulness, and professionalism.

# Always refine and rewrite the user’s original input to improve clarity, correctness, and specificity.

# Structure all outputs with:

# A refined version of the user query.

# A reasoning statement on whether and how the question is related to the document.

# A final answer in a clear, structured summary (for example, a numbered or bulleted list of rooms and their areas), referencing the document where applicable.

# Encourage structured reasoning before delivering conclusions, and maintain clarity, precision, and a helpful tone throughout.

# Steps:
# Receive the user's question and the input document (if any).

# Determine question relevance to the document.

# Rewrite the user’s query for clarity and specificity.

# If the question is related, analyze the document and provide a concise, structured summary (not JSON).

# If the question is not related, answer it directly using general knowledge.

# Output the result in this format:

# Refined Query:
# <Rewritten and polished version of the user's query>

# Reasoning:
# <Brief explanation on relevance of question to the document>

# Answer:
# <Final concise, structured summary answer (e.g., bulleted or numbered list), with document references if applicable>
# Notes:

# For document-related queries, reference the document specifically.

# For general knowledge queries, provide helpful and accurate answers without noting document irrelevance.

# Maintain professionalism, structured thinking, and a concise yet informative tone.

# For summary answers, focus on the main functions and areas, using a readable list format.'''
#                     }

#                     ,
#             {
#                 "role": "user",
#                 "content": [
#                     {"type": "text", "text": question},
#                     {"type": "image_url", "image_url": f"data:image/jpeg;base64,{base64_image}"}
#                 ]
#             }
#         ]
        
#         client = Mistral(api_key=self.api_key)
#         response = client.chat.complete(model=self.model, messages=messages)
#         model_output_raw = response.choices[0].message.content.strip()
        
#         try:
#             model_output = json.loads(model_output_raw)
#         except json.JSONDecodeError:
#             return {
#                 "refined_query": question,
#                 "is_question_related": False,
#                 "answer": model_output_raw
#             }
        
#         is_question_related = model_output.get("is_question_related", False)
#         answer = model_output.get("answer", "")
        
#         fallback_needed = (
#             not is_question_related or
#             not answer.strip() or
#             "not found" in answer.lower() or
#             "unable to" in answer.lower()
#         )
        
#         if fallback_needed:
#             fallback_answer = self.search_fallback_text_file(question)
#             return {
#                 "refined_query": model_output.get("refined_query", question),
#                 "is_question_related": is_question_related,
#                 "answer": fallback_answer,
#                 "used_fallback": True
#             }
#         else:
#             model_output["used_fallback"] = False
#             return model_output
    
#     def process_document_and_answer(self, pdf_path, question, target_page=None):
#         """Complete pipeline: PDF -> Images -> OCR -> AI Q&A"""
#         try:
#             if self.current_pdf_path != pdf_path or not self.current_image_paths:
#                 self.current_image_paths = self.convert_pdf_to_images(pdf_path)
#                 self.current_pdf_path = pdf_path
#                 self.extract_text_from_images(self.current_image_paths)
            
#             if target_page and target_page <= len(self.current_image_paths):
#                 target_image = self.current_image_paths[target_page - 1]
#             else:
#                 target_image = self.current_image_paths[0] if self.current_image_paths else None
            
#             if not target_image:
#                 return {"error": "No images found to process"}
            
#             result = self.answer_question_with_image(question, target_image)
#             return result
            
#         except Exception as e:
#             return {"error": str(e)}

# def main():
#     """Interactive main function with command-line input."""
#     processor = DocumentProcessor()
#     current_pdf_path = r"D:\Melbin\Geometra_Chatbot\cortex_mistral\documents\02.pdf"
#     current_page = 1
    
#     # Initialize document if it exists
#     if os.path.exists(current_pdf_path):
#         try:
#             processor.process_document_and_answer(current_pdf_path, "Initialize document", 1)
#         except Exception as e:
#             current_pdf_path = None
#     else:
#         current_pdf_path = None
    
#     while True:
#         try:
#             if current_pdf_path:
#                 print(f"Document: {os.path.basename(current_pdf_path)} | Page: {current_page}")
#             else:
#                 print("No document loaded")
            
#             user_input = input("Question: ").strip()
            
#             if not user_input:
#                 continue
            
#             if user_input.lower() in ['quit', 'exit']:
#                 break
            
#             if user_input.lower().startswith('load '):
#                 new_pdf_path = user_input[5:].strip().strip('"\'')
#                 if os.path.exists(new_pdf_path):
#                     current_pdf_path = new_pdf_path
#                     current_page = 1
#                     try:
#                         processor.process_document_and_answer(current_pdf_path, "Initialize document", 1)
#                         print("Document loaded successfully")
#                     except Exception as e:
#                         print(f"Error loading document: {e}")
#                         current_pdf_path = None
#                 else:
#                     print("Document not found")
#                 continue
            
#             if user_input.lower().startswith('page '):
#                 try:
#                     page_num = int(user_input[5:].strip())
#                     if current_pdf_path and len(processor.current_image_paths) > 0:
#                         if 1 <= page_num <= len(processor.current_image_paths):
#                             current_page = page_num
#                             print(f"Switched to page {current_page}")
#                         else:
#                             print(f"Invalid page number. Available pages: 1-{len(processor.current_image_paths)}")
#                     else:
#                         print("No document loaded")
#                 except ValueError:
#                     print("Invalid page number")
#                 continue
            
#             if not current_pdf_path:
#                 print("No document loaded. Use 'load <pdf_path>' to load a document.")
#                 continue
            
#             result = processor.process_document_and_answer(current_pdf_path, user_input, current_page)
            
#             if "error" in result:
#                 print(f"Error: {result['error']}")
#             else:
#                 print(f"Answer: {result.get('answer', 'No answer provided')}")
                
#         except KeyboardInterrupt:
#             break
#         except Exception as e:
#             print(f"Error: {e}")

# if __name__ == "__main__":
#     main()