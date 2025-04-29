"""
sudo apt install openstego
sudo apt install outguess
sudo apt install steghide



"""


import os
import subprocess
from PIL import Image

# Directories and file paths
ORIGINAL_DIR = "ml/clean/"  # Directory with original BMP images
STEGO_STEGHIDE_DIR = "ml/stego_steghide/"  # Directory for Steghide stego images
STEGO_OUTGUESS_DIR = "ml/stego_outguess/"  # Directory for OutGuess stego images
SECRET_MESSAGE_FILE = "temp_message.txt"  # Secret message file
PASSPHRASE = "set"  # Passphrase for Steghide (same as OpenStego)

# Create output directories if they don't exist
os.makedirs(STEGO_STEGHIDE_DIR, exist_ok=True)
os.makedirs(STEGO_OUTGUESS_DIR, exist_ok=True)

# Get list of original images
original_files = sorted([f for f in os.listdir(ORIGINAL_DIR) if f.endswith('.bmp')])
if len(original_files) != 198:
    raise ValueError(f"Expected 198 original BMP images, found {len(original_files)}")


# Function to convert BMP to JPEG
def convert_bmp_to_jpeg(bmp_path, jpeg_path):
    img = Image.open(bmp_path).convert("RGB")
    img.save(jpeg_path, "JPEG", quality=95)


# Generate stego images for each original image
for idx, original_file in enumerate(original_files, 1):
    original_path = os.path.join(ORIGINAL_DIR, original_file)
    base_name = original_file.split('.')[0]

    # Steghide: Embed message in BMP
    stego_steghide_path = os.path.join(STEGO_STEGHIDE_DIR, f"{base_name}_steghide.bmp")
    steghide_cmd = [
        "steghide", "embed",
        "-cf", original_path,  # Cover file
        "-ef", SECRET_MESSAGE_FILE,  # Embed file
        "-sf", stego_steghide_path,  # Stego file output
        "-p", PASSPHRASE,  # Passphrase
        "-f"  # Force overwrite
    ]
    try:
        subprocess.run(steghide_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"[{idx}/198] Generated Steghide stego image: {stego_steghide_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error with Steghide for {original_file}: {e.stderr.decode()}")

    # OutGuess: Convert BMP to JPEG, then embed message
    temp_jpeg_path = os.path.join(STEGO_OUTGUESS_DIR, f"{base_name}_temp.jpg")
    stego_outguess_path = os.path.join(STEGO_OUTGUESS_DIR, f"{base_name}_outguess.jpg")

    # Convert BMP to JPEG
    convert_bmp_to_jpeg(original_path, temp_jpeg_path)

    # OutGuess does not support encryption, so no passphrase is used
    outguess_cmd = [
        "outguess",
        "-d", SECRET_MESSAGE_FILE,  # Data to embed
        temp_jpeg_path,  # Cover image
        stego_outguess_path  # Stego image output
    ]
    try:
        subprocess.run(outguess_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"[{idx}/198] Generated OutGuess stego image: {stego_outguess_path}")
        # Clean up temporary JPEG
        os.remove(temp_jpeg_path)
    except subprocess.CalledProcessError as e:
        print(f"Error with OutGuess for {original_file}: {e.stderr.decode()}")
        if os.path.exists(temp_jpeg_path):
            os.remove(temp_jpeg_path)

print("Finished generating stego images for Steghide and OutGuess.")