import os
from PIL import Image
import pytesseract

# Directory and output path
image_dir = r"D:\Melbin\Geometra_Chatbot\cortex_mistral\documents\output"
output_file = r"D:\Melbin\Geometra_Chatbot\cortex_mistral\sample.txt"

image_extensions = ('.jpg', '.jpeg', '.png')

with open(output_file, "w", encoding="utf-8") as f_out:
    for filename in os.listdir(image_dir):
        if filename.lower().endswith(image_extensions):
            image_path = os.path.join(image_dir, filename)
            print(f"Processing {filename}...")

            img = Image.open(image_path)
            text = pytesseract.image_to_string(img)

            f_out.write(f"--- Text from {filename} ---\n")
            f_out.write(text + "\n\n")

print(f"\n✅ Text from all images in '{image_dir}' saved to '{output_file}'")
