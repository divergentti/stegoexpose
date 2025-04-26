"""
Steganography Expose Tool - CLI implementation


Project Goal: StegaExpose – A Modular Framework for Detecting and Decoding Steganography in Media Files

Python-based framework for analyzing, detecting, and ultimately decrypting steganographically hidden content in images
(and potentially audio files). The core idea is to create a system capable of identifying subtle manipulations
indicative of embedded data and, where possible, recovering the hidden message. As an example, uploading an image
to social media platform and downloading image back could reveal subtle hidden messages created by service provider.

The project idea is around three main components (later more detailed):

1. FileTypeAnalyzer
   This class is responsible for extracting and interpreting metadata from media files.
   It will identify the file name, size, timestamps, and determine the file format based on both the extension
   and internal structure. By understanding whether the format is lossy (e.g., JPEG with DCT blocks) or
   lossless (e.g., PNG with DEFLATE compression), the system can make inferences about the potential steganographic
   embedding methods.

2. Guesser
   This component focuses on anomaly detection. It will compare original and stego versions of images to identify
   pixel- or block-level deviations. Examples include LSB alterations, shifts in color channel distributions
   (RGB vs. YCbCr), or artifacts introduced by scaling or compression. A lightweight convolutional neural network
   (CNN) will be trained to highlight regions of interest using binary segmentation, indicating areas likely to
   contain hidden data. The Guesser may also benefit from reinforcement learning techniques to improve accuracy over
   time by learning from both true positives and false alarms.

3. Decrypter
   Utilizing the predictions from the Guesser, this class attempts to decode the hidden content.
   It will support multiple decoding strategies (e.g., LSB extraction, inverse DCT, Reed–Solomon decoding) via a
   plugin-based architecture. If ground truth (original image, stego image, and message) is available, the model can be
   fine-tuned using supervised learning to enhance decryption accuracy.

Planned Features and Workflow:
- The initial dataset will consist of ~100 clean JPEG images, each altered using various steganography tools
(e.g., OpenStego, Steghide, OutGuess).
- The application will be developed in PyCharm, fully in English, including class names, code comments, and documentation.
- Results will be visualized with dynamic anomaly heatmaps and stored in structured JSON format.
- The entire codebase will follow clean code practices with unit tests (pytest), automatic linting (flake8/black),
and GitHub Actions for CI/CD.

This project combines media forensics, machine learning, and cryptographic techniques in a modular and extensible
architecture. The approach is both analytical and creative: learn how steganographic tools operate under the hood,
then systematically reverse-engineer their traces. The ultimate goal is to demystify digital steganography and build an
intelligent system that not only detects but understands and deciphers hidden information.

------ Python Classes based on above idea ------

1. FileTypeAnalyzer

Responsibilities:

- Extracts file metadata (name, size, timestamps)
- Determines file format based on extension and content (jpeg, png, wav, mp3...)
- Identifies compression methods (LOSSY vs. LOSSLESS, e.g., JPEG DCT blocks, PNG DEFLATE)

Technologies:
- `python-magic` or `filetype` for format detection
- `Pillow` for image metadata and format analysis
- `pydub` or `wave` for audio metadata and structure

2. Guesser

Responsibilities:
- Compares original and stego files to find differences
- Detects anomalies: LSB bit variations, color channel distributions (RGB vs. YCbCr), etc.
- Uses a model to predict pixel- or block-level likelihood of modifications

ML Model:
- Since images are 2D data, a CNN-based architecture should work
- Small CNN (2–5 layers) in the beginning, taking in block-level data or entire images depending on dataset size
- Frame as a binary segmentation task (pixel-wise or block-wise anomaly classification)

Feedback (Reinforcement Learning):
- Reward-based learning (e.g., Proximal Policy Optimization via `stable-baselines3`)
- Reward = correct anomaly detections vs. false positives

3. Decrypter

Responsibilities:
- Uses output from `Guesser` (locations and anomaly types such as LSB, DCT, RS parity...)
- Calls specific decryption/extraction methods:
    - LSB extraction (bit-plane slicing)
    - DCT inversion (reversed JPEG block DCT)
    - RS decoding (Reed–Solomon parity recovery)

- If supervised learning is used: feed (original, stego, message) triplets to refine model performance

Technologies:
- Plugin-based architecture: each algorithm "registers" to the `Decrypter` instance
- Libraries like `pycryptodome` for cryptographic and mathematical operations

Class Interfaces
- `FileTypeAnalyzer` returns a metadata object that both `Guesser` and `Decrypter` can consume
- `Guesser.guess(image_pair)` returns a list of anomaly bounding boxes or block indices + probability scores
- `Decrypter.decrypt(stego_image, guess_info)` returns extracted results (bitwise or plaintext)

Test Data:
- User generate stego images using tools like OpenStego, Steghide, OutGuess, Stegatool by Divergentti etc.
- For supervised learning, user may tell which hidden message was embedded to the images / audio.

Logging & Visualization after CLI version works and GUI will be implemented:
- Dynamic anomaly heatmaps (`matplotlib`, `Plotly`)
- Save results in JSON format: `{"filename": "...", "anomalies": [...], "result": ...}`

Language and Documentation
- All code and documentation in English
- Clear docstrings (Google or NumPy style) for all classes and methods

"""
import os
import gc
import magic
from PIL import Image
import numpy as np
import torch
import re
import scipy
from typing import Dict, List, Optional, Tuple

# Clear any potential caches at the start
torch.cuda.empty_cache()
gc.collect()
Image.Image._cache = {}

class JavaRandom:
    def __init__(self, seed: int):
        # Make sure we're using a 48-bit mask as Java does
        self.seed = (seed ^ 0x5DEECE66D) & ((1 << 48) - 1)

    def next(self, bits: int) -> int:
        # Linear congruential generator step
        self.seed = (self.seed * 0x5DEECE66D + 0xB) & ((1 << 48) - 1)

        # Right shift to get the requested number of bits
        # In Java, this is a logical right shift (>>>), not arithmetic (>>)
        return self.seed >> (48 - bits)

    def nextInt(self, n: Optional[int] = None) -> int:
        if n is None:
            return self.next(32)
        if n <= 0:
            raise ValueError("Bound must be positive")

        # Handle power of two case
        if (n & -n) == n:  # n is a power of two
            return (n * self.next(31)) >> 31

        # Handle general case with rejection sampling
        while True:
            bits = self.next(31)
            val = bits % n
            if bits - val + (n - 1) >= 0:
                return val

def test_javarandom():
    """Test the JavaRandom implementation to ensure it matches Java's Random."""
    rand = JavaRandom(0)
    print("Testing JavaRandom with seed 0:")
    print(f"nextInt(1600): {rand.nextInt(1600)}")  # Should be 560
    print(f"nextInt(1200): {rand.nextInt(1200)}")  # Should be 748
    print(f"nextInt(3): {rand.nextInt(3)}")       # Should be 1
    print(f"nextInt(1600): {rand.nextInt(1600)}")  # Should be 1247
    print(f"nextInt(1200): {rand.nextInt(1200)}")  # Should be 1115
    print(f"nextInt(3): {rand.nextInt(3)}")       # Should be 2

class FileTypeAnalyzer:
    """Analyze file metadata and format information."""
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.filename = os.path.basename(filepath)
        self.metadata = {}
        self.format = None

    def suggest_stego_methods(self) -> List[str]:
        if self.format == 'image/jpeg':
            return ['DCT', 'LSB']
        elif self.format == 'image/png':
            return ['LSB']
        return []

    def extract_metadata(self) -> Dict:
        stats = os.stat(self.filepath)
        self.metadata = {
            'filename': self.filename,
            'size_bytes': stats.st_size,
            'created': stats.st_ctime,
            'modified': stats.st_mtime
        }
        print(f"Extracted metadata: {self.metadata}")
        return self.metadata

    def detect_format(self) -> str:
        self.format = magic.from_file(self.filepath, mime=True)
        print(f"Extracted format: {self.format}")
        return self.format

    def analyze(self) -> Dict:
        info = self.extract_metadata()
        info['format'] = self.detect_format()
        if info['format'].startswith('image/'):
            with Image.open(self.filepath) as img:
                info['mode'] = img.mode
                info['size'] = img.size
                info['exif'] = img.getexif() or "No EXIF data"
                if info['format'] == 'image/jpeg':
                    info['compression'] = 'lossy'
                elif info['format'] == 'image/png':
                    info['compression'] = 'lossless'
        print(f"Extracted info: {info}")
        return info

def extract_bits_from_image(image_path: str, num_bits: int = 10000, seed: int = 0) -> List[int]:
    print(f"Loading image for extraction: {image_path}")
    image = Image.open(image_path).convert('RGB')
    print(f"Image size: {image.size}")
    width, height = image.size
    pixels = image.load()
    rand = JavaRandom(seed)
    extracted_bits = []
    used_positions = set()

    print(f"First pixel (0,0): {pixels[0, 0]}")
    print(f"First pixel (1,0): {pixels[1, 0]}")

    # Debug: Print first few positions for seed 0
    if seed == 0:
        print("First 5 positions for seed 0:")
        temp_rand = JavaRandom(seed)
        for i in range(5):
            x = temp_rand.nextInt(width)
            y = temp_rand.nextInt(height)
            channel = temp_rand.nextInt(3)
            print(f"Position {i+1}: (x={x}, y={y}, channel={channel})")

    while len(extracted_bits) < num_bits:
        x = rand.nextInt(width)
        y = rand.nextInt(height)
        channel = rand.nextInt(3)
        key = f"{x}_{y}_{channel}"
        if key in used_positions:
            continue
        used_positions.add(key)
        r, g, b = pixels[x, y]
        color = [r, g, b][channel]
        extracted_bits.append(color & 1)

    image.close()
    return extracted_bits

def extract_bits_sequentially(image_path: str, num_bits: int) -> List[int]:
    print(f"Loading image for sequential extraction: {image_path}")
    image = Image.open(image_path).convert('RGB')
    print(f"Image size: {image.size}")
    width, height = image.size
    pixels = image.load()
    extracted_bits = []

    print(f"First pixel (0,0): {pixels[0, 0]}")
    print(f"First pixel (1,0): {pixels[1, 0]}")

    x, y, channel = 0, 0, 0
    while len(extracted_bits) < num_bits:
        r, g, b = pixels[x, y]
        color = [r, g, b][channel]
        extracted_bits.append(color & 1)
        channel += 1
        if channel == 3:
            channel = 0
            x += 1
            if x == width:
                x = 0
                y += 1
                if y == height:
                    break
    image.close()
    return extracted_bits

def bits_to_bytes(bits: List[int]) -> bytes:
    byte_array = bytearray()
    for i in range(0, len(bits) - 7, 8):
        byte_bits = bits[i:i + 8]
        if len(byte_bits) == 8:
            byte = int(''.join(map(str, byte_bits)), 2)
            byte_array.append(byte)
    return bytes(byte_array)

def debug_bits(bits: List[int], num_bits: int = 80):
    if bits is None:
        print("Error: bits is None")
        return
    bits = bits[:num_bits]
    byte_str = bits_to_bytes(bits)
    print(f"First {num_bits} bits as bytes: {byte_str}")
    print(f"First {num_bits} bits as string: {byte_str.decode('utf-8', errors='ignore')}")

def debug_header(bits: List[int], header_length: int = 328):
    if bits is None:
        print("Error: bits is None in debug_header")
        return
    if len(bits) < header_length:
        print(f"Not enough bits to debug header: {len(bits)} < {header_length}")
        return

    pos = 0
    magic_bits = bits[pos:pos + 72]
    magic_bytes = bits_to_bytes(magic_bits)
    pos += 72
    print(f"Magic number: {magic_bytes.decode('utf-8', errors='ignore')}")

    version_bits = bits[pos:pos + 8]
    version = int(''.join(map(str, version_bits)), 2)
    pos += 8
    print(f"Header version: {version}")

    data_length_bits = bits[pos:pos + 32]
    data_length = int(''.join(map(str, data_length_bits)), 2)
    pos += 32
    print(f"Data length: {data_length} bytes")

    channel_bits_used_bits = bits[pos:pos + 8]
    channel_bits_used = int(''.join(map(str, channel_bits_used_bits)), 2)
    pos += 8
    print(f"Channel bits used: {channel_bits_used}")

    filename_length_bits = bits[pos:pos + 8]
    filename_length = int(''.join(map(str, filename_length_bits)), 2)
    pos += 8
    print(f"Filename length: {filename_length}")

    compression_flag_bits = bits[pos:pos + 8]
    compression_flag = int(''.join(map(str, compression_flag_bits)), 2)
    pos += 8
    print(f"Compression flag: {compression_flag}")

    encryption_flag_bits = bits[pos:pos + 8]
    encryption_flag = int(''.join(map(str, encryption_flag_bits)), 2)
    pos += 8
    print(f"Encryption flag: {encryption_flag}")

    encryption_algo_bits = bits[pos:pos + 64]
    encryption_algo_bytes = bits_to_bytes(encryption_algo_bits)
    pos += 64
    print(f"Encryption algorithm: {encryption_algo_bytes.decode('utf-8', errors='ignore')}")

    filename_bits = bits[pos:pos + filename_length * 8]
    filename_bytes = bits_to_bytes(filename_bits)
    filename = filename_bytes.decode('utf-8', errors='replace')
    pos += filename_length * 8
    print(f"Filename: {filename}")

    if pos < len(bits):
        data_bits = bits[pos:pos + data_length * 8]
        data_bytes = bits_to_bytes(data_bits)
        print(f"Data: {data_bytes.decode('utf-8', errors='ignore')}")

class Guesser:
    def __init__(self, threshold: float = 0.5):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.threshold = threshold
        print(f"The Guesser using {self.device}")

    def guess(self, clean_path: str, stego_path: str) -> Dict:
        print(f"Guesser loading clean image: {clean_path}")
        clean_img = Image.open(clean_path).convert('RGB')
        print(f"Guesser loading stego image: {stego_path}")
        stego_img = Image.open(stego_path).convert('RGB')
        clean_pixels = clean_img.load()
        stego_pixels = stego_img.load()
        print(f"Guesser clean image first pixel (0,0): {clean_pixels[0, 0]}")
        print(f"Guesser stego image first pixel (0,0): {stego_pixels[0, 0]}")

        result = self.lsb_difference_analysis(clean_img, stego_img)
        score = result.get('score', 0.0)
        classification = "Stego" if score > self.threshold else "Clean"

        result['classification'] = classification
        result['score'] = score
        return result

    def calculate_features(self, img_np: np.ndarray) -> np.ndarray:
        # img_np ∈ [H,W,3] uint8
        features = []
        for c in range(3):
            chan = img_np[:, :, c].ravel()
            features += [np.mean(chan), np.std(chan),
                         scipy.stats.skew(chan), scipy.stats.kurtosis(chan)]
        features.append(scipy.stats.entropy(np.histogram(img_np.ravel(), bins=256)[0]))
        return np.array(features, dtype=np.float32)

    def lsb_difference_analysis(self, clean_img: Image.Image, stego_img: Image.Image) -> Dict:
        clean_np = np.array(clean_img, dtype=np.uint8)
        stego_np = np.array(stego_img, dtype=np.uint8)
        clean_lsb = clean_np & 1
        stego_lsb = stego_np & 1
        diff_map = (clean_lsb != stego_lsb).astype(np.uint8)
        embedding_rate = np.mean(diff_map)
        total_pixels = clean_lsb.size
        expected_changes = int(total_pixels * embedding_rate)
        total_changes = np.sum(diff_map)

        height, width, channels = clean_np.shape
        print(f"Image size: {height}x{width}x{channels}, Total pixels: {total_pixels}")
        print(f"Embedding rate: {embedding_rate:.6f}")
        print(f"Expected changes (embedding rate × total pixels): {expected_changes}")
        print(f"Actual changes: {total_changes}")

        block_size = 16
        num_blocks_h = height // block_size
        num_blocks_w = width // block_size
        blocks_with_changes = 0
        total_blocks = num_blocks_h * num_blocks_w * channels

        for i in range(num_blocks_h):
            for j in range(num_blocks_w):
                for c in range(channels):
                    block = diff_map[i * block_size:(i + 1) * block_size, j * block_size:(j + 1) * block_size, c]
                    if np.sum(block) > 0:
                        blocks_with_changes += 1

        fraction_blocks_with_changes = blocks_with_changes / total_blocks if total_blocks > 0 else 0.0
        print(f"Blocks with changes: {blocks_with_changes}/{total_blocks} ({fraction_blocks_with_changes:.4f})")

        prob_stego = min(fraction_blocks_with_changes * 10 + embedding_rate * 10000, 1.0)
        clean_prob_stego = 0.0
        print(f"Prob stego: {prob_stego:.4f}")
        return {
            'clean_prob_stego': clean_prob_stego,
            'stego_prob_stego': float(prob_stego),
            'anomalies': [],
            'score': float(prob_stego)
        }

class Decrypter:
    def __init__(self):
        self.methods = {}

    def register_method(self, name: str, func):
        self.methods[name] = func

    def decrypt(self, stego_path: str, guess_info: Dict, method: str, seed: int = 0) -> Dict:
        if method not in self.methods:
            raise ValueError(f"Method {method} not registered.")
        return self.methods[method](self, stego_path, guess_info, seed)

    def extract_openstego_lsb(self, stego_path: str, guess_info: Dict, seed: int) -> Dict:
        if seed == -1:
            bits = extract_bits_sequentially(stego_path, num_bits=1000)
        else:
            bits = extract_bits_from_image(stego_path, num_bits=1000, seed=seed)

        magic_number = []
        magic_str = "OPENSTEGO"
        for byte in magic_str.encode('utf-8'):
            for i in range(7, -1, -1):
                magic_number.append((byte >> i) & 1)

        magic_length = len(magic_number)
        start_index = -1
        for i in range(len(bits) - magic_length + 1):
            if bits[i:i + magic_length] == magic_number:
                start_index = i
                break

        if start_index == -1:
            print("OpenStego magic number not found.")
            return {"success": False, "message": "Magic number not found", "filename": None, "content": None}

        print(f"OpenStego magic number found at index {start_index}")

        pos = start_index + magic_length
        pos += 8
        data_length_bits = bits[pos:pos + 32]
        data_length = int(''.join(map(str, data_length_bits)), 2)
        pos += 32
        pos += 8
        filename_length_bits = bits[pos:pos + 8]
        filename_length = int(''.join(map(str, filename_length_bits)), 2)
        pos += 8
        pos += 8
        pos += 8
        pos += 64
        filename_bits = bits[pos:pos + filename_length * 8]
        filename_bytes = bits_to_bytes(filename_bits)
        filename = re.sub(r'[\x00-\x1F\x7F-\x9F]', '_', filename_bytes.decode('utf-8', errors='replace'))
        if not filename:
            filename = "extracted_file.txt"
        pos += filename_length * 8

        content_bits = bits[pos:pos + data_length * 8]
        content_bytes = bits_to_bytes(content_bits)

        output_dir = "extracted_files"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, filename)
        try:
            with open(output_path, 'wb') as f:
                f.write(content_bytes)
        except Exception as e:
            print(f"Failed to write file: {e}")
            return {
                "success": False,
                "message": f"Failed to write file: {e}",
                "filename": filename,
                "content": content_bytes
            }

        print(f"Extracted file: {filename}, Size: {data_length} bytes, Saved to: {output_path}")
        print(f"File content: {content_bytes.decode('utf-8', errors='ignore')}")

        return {
            "success": True,
            "message": "File extracted successfully",
            "filename": filename,
            "content": content_bytes
        }

    def extract_openstego_dct(self, stego_path: str, guess_info: Dict, seed: int) -> Dict:
        return {"success": False, "message": "DCT extraction not implemented yet", "filename": None, "content": None}

    def extract_openstego_dwtxie(self, stego_path: str, guess_info: Dict, seed: int) -> Dict:
        return {"success": False, "message": "DWTXie extraction not implemented yet", "filename": None, "content": None}

def get_openstego_seed(password: Optional[str]) -> int:
    """Generate the seed exactly as OpenStego does"""
    if password is None or password == "":
        return 0

    # Java's String.hashCode() implementation
    hash_code = 0
    for char in password:
        hash_code = ((hash_code * 31) + ord(char)) & 0xFFFFFFFF
        if hash_code > 0x7FFFFFFF:
            hash_code -= 0x100000000

    return hash_code

def extract_openstego_bits(image_path: str, seed: int, num_bits: int) -> Tuple[List[int], int]:
    print(f"Extracting bits from {image_path} with seed {seed}")
    image = Image.open(image_path).convert('RGB')
    width, height = image.size
    pixels = image.load()

    rand = JavaRandom(seed)
    extracted_bits = []
    used_positions = set()

    # Debug: Print first 5 positions, pixel values, and bits
    print(f"First 5 positions for seed {seed}:")
    temp_rand = JavaRandom(seed)
    for i in range(5):
        x = temp_rand.nextInt(width)
        y = temp_rand.nextInt(height)
        channel = temp_rand.nextInt(3)
        r, g, b = pixels[x, y]
        color_value = [r, g, b][channel]
        bit = color_value & 1
        print(f"Position {i+1}: (x={x}, y={y}, channel={channel}), Pixel: ({r}, {g}, {b}), Value: {color_value}, Bit: {bit}")

    while len(extracted_bits) < num_bits:
        x = rand.nextInt(width)
        y = rand.nextInt(height)
        channel = rand.nextInt(3)
        bit_pos = 0  # OpenStego uses LSB

        key = f"{x}_{y}_{channel}_{bit_pos}"
        if key in used_positions:
            continue

        used_positions.add(key)
        r, g, b = pixels[x, y]
        color_value = [r, g, b][channel]
        extracted_bit = color_value & 1  # Extract LSB
        extracted_bits.append(extracted_bit)

    # Search for magic number
    magic_str = "OPENSTEGO"
    magic_bits = []
    for byte in magic_str.encode('utf-8'):
        for i in range(7, -1, -1):
            magic_bits.append((byte >> i) & 1)

    start_index = -1
    for i in range(len(extracted_bits) - len(magic_bits) + 1):
        if extracted_bits[i:i + len(magic_bits)] == magic_bits:
            start_index = i
            break

    if start_index == -1:
        print(f"Magic number 'OPENSTEGO' not found in first {num_bits} bits.")
        print(f"First 72 bits: {extracted_bits[:72]}")
        magic_bytes = bits_to_bytes(extracted_bits[:72])
        print(f"First 72 bits as string: {magic_bytes.decode('utf-8', errors='replace')}")
        # Additional debug: Try different bit positions
        for bit_pos in range(1, 8):
            bits_at_pos = []
            temp_rand = JavaRandom(seed)
            temp_used = set()
            for _ in range(72):
                x = temp_rand.nextInt(width)
                y = temp_rand.nextInt(height)
                channel = temp_rand.nextInt(3)
                key = f"{x}_{y}_{channel}_{bit_pos}"
                while key in temp_used:
                    x = temp_rand.nextInt(width)
                    y = temp_rand.nextInt(height)
                    channel = temp_rand.nextInt(3)
                    key = f"{x}_{y}_{channel}_{bit_pos}"
                temp_used.add(key)
                r, g, b = pixels[x, y]
                color_value = [r, g, b][channel]
                bit = (color_value >> bit_pos) & 1
                bits_at_pos.append(bit)
            print(f"First 72 bits at bit position {bit_pos}: {bits_at_pos}")
            magic_bytes = bits_to_bytes(bits_at_pos)
            print(f"First 72 bits at bit position {bit_pos} as string: {magic_bytes.decode('utf-8', errors='replace')}")
    else:
        print(f"Magic number 'OPENSTEGO' found at bit position {start_index}")

    image.close()
    return extracted_bits, start_index

def extract_openstego_bits_sequential(image_path: str, num_bits: int) -> Tuple[List[int], int]:
    print(f"Extracting bits sequentially from {image_path}")
    image = Image.open(image_path).convert('RGB')
    width, height = image.size
    pixels = image.load()

    extracted_bits = []
    # Try channel-first order: all R, then all G, then all B
    for channel in range(3):  # 0=R, 1=G, 2=B
        for y in range(height):
            for x in range(width):
                r, g, b = pixels[x, y]
                color_value = [r, g, b][channel]
                extracted_bits.append(color_value & 1)
                if len(extracted_bits) >= num_bits:
                    break
            if len(extracted_bits) >= num_bits:
                break
        if len(extracted_bits) >= num_bits:
            break

    # Search for magic number
    magic_str = "OPENSTEGO"
    magic_bits = []
    for byte in magic_str.encode('utf-8'):
        for i in range(7, -1, -1):
            magic_bits.append((byte >> i) & 1)

    start_index = -1
    for i in range(len(extracted_bits) - len(magic_bits) + 1):
        if extracted_bits[i:i + len(magic_bits)] == magic_bits:
            start_index = i
            break

    if start_index == -1:
        print(f"Magic number 'OPENSTEGO' not found in first {num_bits} bits (sequential).")
        print(f"First 72 bits: {extracted_bits[:72]}")
        magic_bytes = bits_to_bytes(extracted_bits[:72])
        print(f"First 72 bits as string: {magic_bytes.decode('utf-8', errors='replace')}")
    else:
        print(f"Magic number 'OPENSTEGO' found at bit position {start_index} (sequential)")

    image.close()
    return extracted_bits, start_index

def find_openstego_magic(stego_path: str, password: Optional[str] = None) -> Tuple[int, int]:
    """Find OpenStego's magic number using the correct seed"""
    seed = get_openstego_seed(password)
    print(f"Using seed: {seed} for password: {password}")

    # Get magic number bit pattern
    magic_str = "OPENSTEGO"
    magic_bits = []
    for byte in magic_str.encode('utf-8'):
        for i in range(7, -1, -1):
            magic_bits.append((byte >> i) & 1)

    # Extract more than enough bits for the header
    extracted_bits, start_index = extract_openstego_bits(stego_path, seed, 500)
    return seed, start_index

def extract_openstego_message(stego_path: str, password: str = "") -> Optional[Dict]:
    """Extract hidden message from OpenStego file"""
    seed = get_openstego_seed(password)
    print(f"Extracting with seed: {seed}")

    # Try random extraction with the given seed
    all_bits, magic_start = extract_openstego_bits(stego_path, seed, 20000)
    if magic_start == -1:
        # Try a range of seeds
        print("Trying different seeds...")
        for test_seed in range(0, 10):  # Try seeds 0 to 9
            print(f"Testing seed: {test_seed}")
            all_bits, magic_start = extract_openstego_bits(stego_path, test_seed, 20000)
            if magic_start != -1:
                seed = test_seed
                break

    if magic_start == -1:
        print("Random extraction failed, trying sequential extraction...")
        all_bits, magic_start = extract_openstego_bits_sequential(stego_path, 20000)
        if magic_start == -1:
            return None

    # Parse the header starting from magic number
    pos = magic_start

    # 1. Magic string "OPENSTEGO" (9 bytes = 72 bits)
    pos += 72

    # 2. Header version (1 byte = 8 bits)
    version_bits = all_bits[pos:pos + 8]
    version = int(''.join(map(str, version_bits)), 2)
    pos += 8
    print(f"Header version: {version}")

    # 3. Data length (4 bytes = 32 bits)
    data_length_bits = all_bits[pos:pos + 32]
    data_length = int(''.join(map(str, data_length_bits)), 2)
    pos += 32
    print(f"Data length: {data_length} bytes")

    # 4. Channel bits used (1 byte = 8 bits)
    channel_bits_used_bits = all_bits[pos:pos + 8]
    channel_bits_used = int(''.join(map(str, channel_bits_used_bits)), 2)
    pos += 8
    print(f"Channel bits used: {channel_bits_used}")

    # 5. Filename length (1 byte = 8 bits)
    filename_length_bits = all_bits[pos:pos + 8]
    filename_length = int(''.join(map(str, filename_length_bits)), 2)
    pos += 8
    print(f"Filename length: {filename_length}")

    # 6. Compression flag (1 byte = 8 bits)
    compression_flag_bits = all_bits[pos:pos + 8]
    compression_flag = int(''.join(map(str, compression_flag_bits)), 2)
    pos += 8
    print(f"Compression flag: {compression_flag}")

    # 7. Encryption flag (1 byte = 8 bits)
    encryption_flag_bits = all_bits[pos:pos + 8]
    encryption_flag = int(''.join(map(str, encryption_flag_bits)), 2)
    pos += 8
    print(f"Encryption flag: {encryption_flag}")

    # 8. Encryption algorithm ("AES256" padded to 8 bytes = 64 bits)
    encryption_algo_bits = all_bits[pos:pos + 64]
    encryption_algo_bytes = bits_to_bytes(encryption_algo_bits)
    pos += 64
    print(f"Encryption algorithm: {encryption_algo_bytes.decode('utf-8', errors='ignore')}")

    # 9. Filename
    filename_bits = all_bits[pos:pos + filename_length * 8]
    filename_bytes = bits_to_bytes(filename_bits)
    filename = filename_bytes.decode('utf-8', errors='replace')
    pos += filename_length * 8
    print(f"Filename: {filename}")

    # 10. Data
    data_bits = all_bits[pos:pos + data_length * 8]
    if len(data_bits) < data_length * 8:
        print(f"Warning: Not enough bits for full data. Expected {data_length * 8}, got {len(data_bits)}")

    data_bytes = bits_to_bytes(data_bits)

    # Handle decryption if needed
    if encryption_flag == 1:
        print("Data is encrypted, decryption needed")
        # Add decryption code here if needed

    # Handle decompression if needed
    if compression_flag == 1:
        print("Data is compressed, decompression needed")
        # Add decompression code here if needed

    return {
        "filename": filename,
        "data": data_bytes,
        "data_length": data_length,
        "is_encrypted": encryption_flag == 1,
        "is_compressed": compression_flag == 1
    }

def detect_openstego_lsb(clean_path: str, stego_path: str) -> Dict:
    """Enhanced detector specifically tuned for OpenStego's LSB algorithm"""
    clean_img = Image.open(clean_path).convert('RGB')
    stego_img = Image.open(stego_path).convert('RGB')
    clean_np = np.array(clean_img, dtype=np.uint8)
    stego_np = np.array(stego_img, dtype=np.uint8)

    # 1. LSB difference analysis
    clean_lsb = clean_np & 1
    stego_lsb = stego_np & 1
    lsb_diff = (clean_lsb != stego_lsb).astype(np.int8)

    # 2. Calculate overall embedding rate
    embedding_rate = np.mean(lsb_diff)
    print(f"Overall embedding rate: {embedding_rate:.8f}")

    # 3. Calculate spatial randomness score
    # OpenStego uses Random to select pixels, so changes should be spatially random
    height, width, channels = clean_np.shape
    block_size = 32
    num_blocks_h = height // block_size
    num_blocks_w = width // block_size
    block_changes = np.zeros((num_blocks_h, num_blocks_w, channels))

    for i in range(num_blocks_h):
        for j in range(num_blocks_w):
            for c in range(channels):
                block = lsb_diff[i * block_size:(i + 1) * block_size, j * block_size:(j + 1) * block_size, c]
                block_changes[i, j, c] = np.sum(block) / (block_size * block_size)

    # Calculate standard deviation of change distribution across blocks
    spatial_randomness = np.std(block_changes)
    print(f"Spatial randomness: {spatial_randomness:.8f}")

    # 4. Calculate pair statistics (unique to LSB replacement)
    # In clean images, certain LSB patterns should appear naturally
    # LSB steganography disturbs these patterns
    def calculate_pair_ratios(img: np.ndarray) -> np.ndarray:
        result = np.zeros(4)
        # For each color channel
        for c in range(3):
            channel = img[:, :, c]
            # Count pairs of adjacent pixels where:
            # (even, even), (even, odd), (odd, even), (odd, odd)
            even_pixels = (channel & 1) == 0

            # Horizontal adjacency
            h_pairs = np.zeros(4)
            h_pairs[0] = np.sum((even_pixels[:, :-1] & even_pixels[:, 1:]))  # even,even
            h_pairs[1] = np.sum((even_pixels[:, :-1] & ~even_pixels[:, 1:]))  # even,odd
            h_pairs[2] = np.sum((~even_pixels[:, :-1] & even_pixels[:, 1:]))  # odd,even
            h_pairs[3] = np.sum((~even_pixels[:, :-1] & ~even_pixels[:, 1:]))  # odd,odd

            # Vertical adjacency
            v_pairs = np.zeros(4)
            v_pairs[0] = np.sum((even_pixels[:-1, :] & even_pixels[1:, :]))  # even,even
            v_pairs[1] = np.sum((even_pixels[:-1, :] & ~even_pixels[1:, :]))  # even,odd
            v_pairs[2] = np.sum((~even_pixels[:-1, :] & even_pixels[1:, :]))  # odd,even
            v_pairs[3] = np.sum((~even_pixels[:-1, :] & ~even_pixels[1:, :]))  # odd,odd

            # Average of horizontal and vertical
            result += (h_pairs + v_pairs) / 2

        return result / np.sum(result)  # Normalize

    clean_pair_ratios = calculate_pair_ratios(clean_np)
    stego_pair_ratios = calculate_pair_ratios(stego_np)
    pair_difference = np.sum(np.abs(clean_pair_ratios - stego_pair_ratios))
    print(f"Pair statistics difference: {pair_difference:.8f}")

    # 5. Calculate sample pair analysis metrics (SPA)
    # SPA is specifically designed to detect LSB replacement
    def calculate_spa_metrics(clean: np.ndarray, stego: np.ndarray) -> float:
        metrics = []
        for c in range(3):
            clean_chan = clean[:, :, c]
            stego_chan = stego[:, :, c]

            # Calculate pairs and differences
            clean_diff_h = np.abs(clean_chan[:, :-1] - clean_chan[:, 1:])
            stego_diff_h = np.abs(stego_chan[:, :-1] - stego_chan[:, 1:])

            clean_diff_v = np.abs(clean_chan[:-1, :] - clean_chan[1:, :])
            stego_diff_v = np.abs(stego_chan[:-1, :] - stego_chan[1:, :])

            # Count structural discontinuities
            clean_disc_h = np.sum(clean_diff_h == 1)
            stego_disc_h = np.sum(stego_diff_h == 1)
            clean_disc_v = np.sum(clean_diff_v == 1)
            stego_disc_v = np.sum(stego_diff_v == 1)

            # Calculate ratio differences
            total_h = clean_diff_h.size
            total_v = clean_diff_v.size

            h_ratio_diff = abs((stego_disc_h / total_h) - (clean_disc_h / total_h))
            v_ratio_diff = abs((stego_disc_v / total_v) - (clean_disc_v / total_v))

            metrics.append(h_ratio_diff)
            metrics.append(v_ratio_diff)

        return np.mean(metrics)

    spa_score = calculate_spa_metrics(clean_np, stego_np)
    print(f"SPA score: {spa_score:.8f}")

    # 6. Combined detection score
    score = (
        0.5 * float(min(embedding_rate * 10000, 1.0)) +
        0.2 * float(min(spatial_randomness * 10, 1.0)) +
        0.15 * float(min(pair_difference * 10, 1.0)) +
        0.15 * float(min(spa_score * 20, 1.0))
    )

    # Classification
    classification = "Stego" if score > 0.4 else "Clean"
    confidence = score

    clean_img.close()
    stego_img.close()

    return {
        "classification": classification,
        "confidence": confidence,
        "embedding_rate": embedding_rate,
        "spatial_randomness": spatial_randomness,
        "pair_difference": pair_difference,
        "spa_score": spa_score
    }

def search_magic_number(stego_path: str) -> Tuple[bool, Optional[int], Optional[int]]:
    """Search for OpenStego's magic number through the image using multiple seeds"""
    image = Image.open(stego_path).convert('RGB')
    width, height = image.size
    pixels = image.load()

    # Define OpenStego's magic number
    magic_str = "OPENSTEGO"
    magic_pattern = []
    for byte in magic_str.encode('utf-8'):
        for i in range(7, -1, -1):
            magic_pattern.append((byte >> i) & 1)

    # Try multiple seeds
    for seed in range(0, 1000):  # Try first 1000 seeds
        rand = JavaRandom(seed)
        extracted_bits = []
        used_positions = set()

        # Extract enough bits to find the magic number
        while len(extracted_bits) < 500:  # Get enough bits for header
            x = rand.nextInt(width)
            y = rand.nextInt(height)
            channel = rand.nextInt(3)

            key = f"{x}_{y}_{channel}"
            if key in used_positions:
                continue

            used_positions.add(key)
            r, g, b = pixels[x, y]
            color = [r, g, b][channel]
            extracted_bits.append(color & 1)

        # Look for magic number
        for i in range(len(extracted_bits) - len(magic_pattern) + 1):
            if extracted_bits[i:i + len(magic_pattern)] == magic_pattern:
                print(f"Found magic number with seed {seed} at position {i}")
                return True, seed, i

    return False, None, None

def try_common_passwords(stego_path: str) -> Tuple[bool, Optional[str], Optional[Dict]]:
    """Try common passwords for OpenStego extraction"""
    common_passwords = ["", "password", "123456", "admin", "welcome", "secret",
                        "stego", "hidden", "steganography", "openstego"]

    for password in common_passwords:
        print(f"Trying password: '{password}'")
        seed = get_openstego_seed(password)
        result = extract_openstego_message(stego_path, password)
        if result:
            return True, password, result

    return False, None, None

def detect_sequential_lsb(clean_path: str, stego_path: str) -> Dict:
    """Detect sequential LSB embedding (used by many tools)"""
    clean_img = Image.open(clean_path).convert('RGB')
    stego_img = Image.open(stego_path).convert('RGB')

    # Get the first 1000 LSBs sequentially
    clean_bits = []
    stego_bits = []

    width, height = clean_img.size
    for y in range(height):
        for x in range(width):
            if len(clean_bits) >= 1000:
                break

            r1, g1, b1 = clean_img.getpixel((x, y))
            r2, g2, b2 = stego_img.getpixel((x, y))

            clean_bits.extend([r1 & 1, g1 & 1, b1 & 1])
            stego_bits.extend([r2 & 1, g2 & 1, b2 & 1])

    # Compare LSBs
    diff_count = sum(1 for a, b in zip(clean_bits, stego_bits) if a != b)
    diff_rate = diff_count / len(clean_bits)

    # Check for patterns in the differences
    return {
        "sequential_diff_rate": diff_rate,
        "classification": "Stego" if diff_rate > 0.0001 else "Clean"
    }


decrypter = Decrypter()
decrypter.register_method("openstego_lsb", Decrypter.extract_openstego_lsb)
decrypter.register_method("openstego_dct", Decrypter.extract_openstego_dct)
decrypter.register_method("openstego_dwtxie", Decrypter.extract_openstego_dwtxie)

def fingerprint_openstego_lsb(clean_path: str, stego_path: str) -> Dict:
    detection = detect_openstego_lsb(clean_path, stego_path)
    return {
        "tool": "OpenStego",
        "method": "LSB",
        "spatial_randomness": detection["spatial_randomness"],
        "pair_difference": detection["pair_difference"],
        "spa_score": detection["spa_score"],
        "embedding_rate": detection["embedding_rate"],
        "magic_number": "OPENSTEGO"  # If found during extraction
    }

def fingerprint_steghide(clean_path: str, stego_path: str) -> Dict:
    detection = detect_sequential_lsb(clean_path, stego_path)
    # Extract bits to look for Steghide's header (if known)
    bits = extract_bits_sequentially(stego_path, num_bits=1000)
    header = bits_to_bytes(bits[:80]).decode('utf-8', errors='ignore')
    return {
        "tool": "Steghide",
        "method": "Sequential LSB",
        "sequential_diff_rate": detection["sequential_diff_rate"],
        "header_pattern": header if "steghide" in header.lower() else "Not found"
    }


if __name__ == '__main__':
    torch.cuda.empty_cache()
    gc.collect()

    # Run the JavaRandom test
    test_javarandom()

    print("\nOpenStego Extraction Test")
    print("------------------------")

    # Extract from test_stego_new.bmp
    stego_path1 = "test_stego_new.bmp"
    if not os.path.exists(stego_path1):
        print(f"Error: Stego image {stego_path1} does not exist")
        exit(1)

    result1 = extract_openstego_message(stego_path1, password="")
    if result1:
        print("\nExtraction successful for test_stego_new.bmp!")
        print(f"Filename: {result1['filename']}")
        print(f"Data size: {len(result1['data'])} bytes")
        print(f"Content (as text): {result1['data'].decode('utf-8', errors='ignore')}")
    else:
        print("Failed to extract message from test_stego_new.bmp.")

    # Extract from openstego/encrypted-from-bmp/image1.bmp
    stego_path2 = "openstego/encrypted-from-bmp/image1.bmp"
    if not os.path.exists(stego_path2):
        print(f"Error: Stego image {stego_path2} does not exist")
        exit(1)

    result2 = extract_openstego_message(stego_path2, password="")
    if result2:
        print("\nExtraction successful for image1.bmp!")
        print(f"Filename: {result2['filename']}")
        print(f"Data size: {len(result2['data'])} bytes")
        print(f"Content (as text): {result2['data'].decode('utf-8', errors='ignore')}")
    else:
        print("Failed to extract message from image1.bmp.")
