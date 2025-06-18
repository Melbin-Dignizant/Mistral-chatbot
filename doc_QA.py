import base64
import requests
import os
import json
from mistralai import Mistral
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def encode_image(image_path):
    """Encode the image to base64."""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except FileNotFoundError:
        print(f"Error: The file {image_path} was not found.")
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None

def search_fallback_text_file(question, file_path):
    """Search the fallback text file for a possible answer."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            # Simple keyword-based search logic
            if question.lower() in content.lower():
                return f"Relevant information from fallback file: {question} appears in:\n\n{content}"
            else:
                # You can improve this with fuzzy search or LLM-based matching later
                return "The question was not found in the fallback file either."
    except FileNotFoundError:
        return f"Fallback file {file_path} not found."
    except Exception as e:
        return f"Error reading fallback file: {e}"

# === CONFIGURATION ===
image_path = r"D:\Melbin\Geometra_Chatbot\cortex_mistral\documents\output\page_2.jpeg"
fallback_text_path = r"D:\Melbin\Geometra_Chatbot\cortex_mistral\sample.txt"
model = "mistral-medium-latest"
api_key = os.getenv("MISTRAL_AI_API_KEY")
question = "what is the area of room 207?"

# === Encode Image ===
base64_image = encode_image(image_path)
if not base64_image:
    exit()

# === Prepare Messages ===
messages = [
    {
        "role": "system",
        "content": '''You are  an expert in document understanding assistant designed to analyze an input document and answer user questions specifically related to that document.

                        When a user provides a question and a document, carefully check if the question pertains only to the content of the provided document.

                        - If the question is unrelated to the document, first politely state that the question is outside the document's scope, then answer the question using your general knowledge.
                        - If the question relates to the document, analyze the relevant parts of the document thoroughly to produce an accurate, well-structured answer.

                        Additionally, when users submit questions or requests through chat involving documents or files, rewrite and refine their input text queries to improve clarity, correctness, and specificity. This rewriting step is crucial to enable precise document understanding and high-quality annotation extraction.

                        Your output must be structured clearly, highlighting:

                        - The refined user query (if rewriting was necessary).
                        - Whether the question is related to the document.
                        - A detailed, accurate answer (with references to the document where relevant).

                        Always encourage reasoning steps before providing conclusions. Ensure accuracy and clarity in all responses.

                        # Steps
                        1. Receive the user's question and the input document.
                        2. Determine if the question is about the document content.
                        3. If unrelated, notify the user and answer based on general knowledge.
                        4. If related, analyze the document and generate an accurate answer.
                        5. Rewrite the user's original question/request to improve clarity and precision.
                        6. Present output in a structured format as described.

                        # Output Format
                        Provide output as a JSON object with the following fields:
                        - "refined_query": the rewritten, polished user question.
                        - "is_question_related": boolean indicating if question relates to the document.
                        - "answer": the finally produced answer to the question, including references to the document if applicable.

                        Example:
                        {
                        "refined_query": "What is the main cause of climate change according to the document?",
                        "is_question_related": true,
                        "answer": "According to the document, the main cause of climate change is greenhouse gas emissions due to fossil fuel combustion."
                        }

                        # Notes
                        - Focus on clarity and precision when rewriting user queries.
                        - If multiple documents or files are provided, consider all relevant information.
                        - Always separate the notice regarding question relevance from the answer.
                        - Maintain professional, helpful tone throughout.

                        This prompt guides you to perform refined document understanding, question relevance evaluation, query refinement, and structured output generation for accurate answers and annotations.'''
    },
    {
        "role": "user",
        "content": [
            {"type": "text", "text": question},
            {"type": "image_url", "image_url": f"data:image/jpeg;base64,{base64_image}"}
        ]
    }
]

# === Initialize Mistral Client ===
client = Mistral(api_key=api_key)

# === Make Request to Model ===
response = client.chat.complete(model=model, messages=messages)
model_output_raw = response.choices[0].message.content.strip()

# === Try Parsing the Output as JSON ===
try:
    model_output = json.loads(model_output_raw)
except json.JSONDecodeError:
    print("Model response is not JSON, printing raw output:\n")
    print(model_output_raw)
    exit()

# === Decision Logic ===
is_question_related = model_output.get("is_question_related", False)
answer = model_output.get("answer", "")

# Determine if fallback is needed
fallback_needed = (
    not is_question_related or
    not answer.strip() or
    "not found" in answer.lower() or
    "unable to" in answer.lower()
)

# === Final Output Decision ===
if fallback_needed:
    fallback_answer = search_fallback_text_file(question, fallback_text_path)
    print(json.dumps({
        "refined_query": model_output.get("refined_query", question),
        "is_question_related": is_question_related,
        "answer": fallback_answer
    }, indent=4))
else:
    print(json.dumps(model_output, indent=4))
