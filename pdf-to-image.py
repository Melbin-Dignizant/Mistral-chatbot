import os
from pdf2image import convert_from_path

def convert_pdf_to_images(pdf_path, output_dir, dpi=300, image_format='JPEG'):
    """
    Converts a PDF file to images and saves them in a specified output directory.
    
    Parameters:
        pdf_path (str): The path to the input PDF file.
        output_dir (str): The directory where the output images should be saved.
        dpi (int): Dots per inch for image resolution.
        image_format (str): The format to save images in (e.g., 'PNG', 'JPEG').
    """
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Convert PDF pages to images
    images = convert_from_path(pdf_path, dpi=dpi)
    
    for i, image in enumerate(images):
        output_path = os.path.join(output_dir, f'page_{i+1}.{image_format.lower()}')
        image.save(output_path, image_format)
        print(f"Saved page {i+1} to {output_path}")

# Example usage:
pdf_input_path = r"D:\Melbin\Geometra_Chatbot\cortex_mistral\documents\02.pdf"
output_image_dir = r"D:\Melbin\Geometra_Chatbot\cortex_mistral\documents\output"

convert_pdf_to_images(pdf_input_path, output_image_dir)
