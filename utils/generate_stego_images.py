import os
import subprocess
from PIL import Image
import pandas as pd
import logging
import time
import json
from utils import settings



# Setup logging
logging.basicConfig(filename="../stego_generation.log", level=logging.INFO, filemode="w")

os.makedirs(settings.OPENSTEGO_DIR, exist_ok=True)
os.makedirs(settings.STEGHIDE_DIR, exist_ok=True)
os.makedirs(settings.OUTGUESS_DIR, exist_ok=True)
os.makedirs(settings.VERIFY_MESSAGE_PATH, exist_ok=True)
os.makedirs(settings.MESSAGE_PATH, exist_ok=True)

def clean_paths():
    # Tyhjennetään kaikki stego-hakemistot aluksi
    for d in [settings.OPENSTEGO_DIR, settings.STEGHIDE_DIR, settings.OUTGUESS_DIR]:
        for f in os.listdir(d):
            try:
                os.remove(os.path.join(d, f))
            except Exception as e:
                logging.warning(f"[Init Cleanup Error] Couldn't delete {f}: {e}")

clean_paths()

def create_message_files():
    messages = {
        "small": "Small message",
        "medium": "Medium message",
        "large": "Large message repeated to increase size"
    }
    with open(settings.MESSAGE_FILE_SMALL, "w") as f:
        f.write(messages["small"])
    with open(settings.MESSAGE_FILE_MEDIUM, "w") as f:
        f.write(messages["medium"] * 3 + messages["medium"][:8])
    with open(settings.MESSAGE_FILE_LARGE, "w") as f:
        f.write(messages["large"] * 2 + messages["large"][:20])


create_message_files()

# Rename files in ORIGINAL_DIR to replace spaces with XXX
for filename in os.listdir(settings.ORIGINALS_DIR):
    if " " in filename:
        old_path = os.path.join(settings.ORIGINALS_DIR, filename)
        new_filename = filename.replace(" ", "XXX")
        new_path = os.path.join(settings.ORIGINALS_DIR, new_filename)
        if not os.path.exists(new_path):
            os.rename(old_path, new_path)

filetype_matrix = {
    "openstego": {
        "input": ["bmp", "png", "jpg"],
        "output": {"bmp": "bmp", "png": "png", "jpg": "bmp"}
    },
    "steghide": {
        "input": ["bmp", "jpg", "wav", "au"],
        "output": {"bmp": "bmp", "jpg": "jpg", "wav": "wav", "au": "au"}
    },
    "outguess": {
        "input": ["jpg", "ppm", "pnm", "pgm", "pbm"],
        "output": {"jpg": "jpg", "ppm": "ppm", "pnm": "pnm", "pgm": "pgm", "pbm": "pbm"}
    }
}

def convert_to_jpeg(input_path, output_path):
    try:
        img = Image.open(input_path)
        img.convert("RGB").save(output_path, "JPEG", quality=90, subsampling=0)
        return True
    except Exception as e:
        logging.warning(f"[Conversion Error] {input_path}: {e}")
        return False

def verify_extraction(tool, stego_path, expected_file):
    try:
        if tool == "steghide":
            cmd = ["steghide", "extract", "-sf", stego_path, "-xf", settings.VERIFY_MESSAGE_PATH, "-p", settings.PASSPHRASE, "-f"]
        elif tool == "openstego":
            cmd = ["openstego", "extract", "-sf", stego_path, "-xf", settings.VERIFY_MESSAGE_PATH, "-p", settings.PASSPHRASE]
        elif tool == "outguess":
            cmd = ["outguess", "-k", settings.PASSPHRASE, "-r", stego_path, settings.VERIFY_MESSAGE_PATH]
        else:
            return False

        start = time.time()
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"    [TIME] Embed {tool} {rate} took {time.time() - start:.2f}s")

        with open(settings.VERIFY_MESSAGE_PATH, "rb") as f:
            extracted = f.read()
        with open(expected_file, "rb") as f:
            expected = f.read()

        return extracted == expected
    except Exception as e:
        logging.warning(f"[Verify Extraction Error] {tool} {stego_path}: {e}")
        return False

metadata = []
original_files = sorted(os.listdir(settings.ORIGINALS_DIR))

for idx, filename in enumerate(original_files, 1):
    failed = False
    original_path = os.path.join(settings.ORIGINALS_DIR, filename)
    if not os.path.isfile(original_path):
        continue
    base, ext = os.path.splitext(filename)
    ext = ext[1:].lower()

    temp_outputs = []
    temp_metadata = []
    success = True

    MESSAGE_FILES = {
        "small": settings.MESSAGE_FILE_SMALL,
        "medium": settings.MESSAGE_FILE_MEDIUM,
        "large": settings.MESSAGE_FILE_LARGE
    }

    for rate, message_file in MESSAGE_FILES.items():
        for tool in ["steghide", "outguess", "openstego"]:
            out_path = None
            cmd = None

            if tool == "steghide":
                if ext == "png":
                    temp_jpeg = os.path.join(settings.STEGHIDE_DIR, f"{base}_temp.jpg")
                    if not convert_to_jpeg(original_path, temp_jpeg):
                        continue
                    cover = temp_jpeg
                elif ext in filetype_matrix[tool]["input"]:
                    cover = original_path
                else:
                    continue
                suffix = "jpg" if ext == "png" else ext
                out_path = os.path.join(settings.STEGHIDE_DIR, f"{base}_steghide_{rate}.{suffix}")
                cmd = ["steghide", "embed", "-cf", cover, "-ef", message_file, "-sf", out_path, "-p", settings.PASSPHRASE, "-f"]

            elif tool == "outguess":
                if ext in filetype_matrix[tool]["input"]:
                    cover = original_path
                elif ext in ["png", "bmp"]:
                    temp_jpeg = os.path.join(settings.OUTGUESS_DIR, f"{base}_temp.jpg")
                    if not convert_to_jpeg(original_path, temp_jpeg):
                        continue
                    cover = temp_jpeg
                else:
                    continue
                out_path = os.path.join(settings.OUTGUESS_DIR, f"{base}_outguess_{rate}.jpg")
                cmd = ["outguess", "-k", settings.PASSPHRASE, "-d", message_file, cover, out_path]

            elif tool == "openstego":
                if ext in filetype_matrix[tool]["input"]:
                    out_ext = filetype_matrix[tool]["output"][ext]
                    out_path = os.path.join(settings.OPENSTEGO_DIR, f"{base}_openstego_{rate}.{out_ext}")
                    cmd = ["openstego", "embed", "-mf", message_file, "-cf", original_path, "-sf", out_path, "-p", settings.PASSPHRASE]
                else:
                    continue

            if not out_path or not cmd:
                continue

            temp_outputs.append(out_path)
            try:
                start = time.time()
                subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                print(f"    [TIME] Embed {tool} {rate} took {time.time() - start:.2f}s")

                if not os.path.exists(out_path):
                    logging.warning(f"[Reject Reason] {filename}: {tool} {rate} did not create output file.")
                    success = False
                    failed = True
                    continue

                start = time.time()
                verified = verify_extraction(tool, out_path, message_file)
                print(f"    [TIME] Verify {tool} {rate} took {time.time() - start:.2f}s")
                if verified:
                    temp_metadata.append({"original": filename, "tool": tool, "outfile": os.path.basename(out_path), "rate": rate})
                else:
                    logging.warning(f"[Reject Reason] {filename}: {tool} {rate} failed verification.")
                    success = False
                    failed = True
            except subprocess.CalledProcessError as e:
                logging.warning(f"[Reject Reason] {filename}: {tool} {rate} embed error: {e}")
                success = False
                failed = True

    if success and not failed and len(temp_metadata) == 9:
        metadata.extend(temp_metadata)
        print(f"[{idx}] ✅ Accepted: {filename}")

    else:
        # Poista kaikki onnistuneet ja epäonnistuneet stego-tiedostot kyseiselle kuvalle
        cleanup_dirs = [settings.OPENSTEGO_DIR, settings.STEGHIDE_DIR, settings.OUTGUESS_DIR]
        base_prefix = base + "_"
        for d in cleanup_dirs:
            for f in os.listdir(d):
                if f.startswith(base_prefix):
                    f_path = os.path.join(d, f)
                    try:
                        os.remove(f_path)
                    except Exception as e:
                        logging.warning(f"[Cleanup Error] Couldn't delete {f_path}: {e}")
        print(f"[{idx}] ❌ Rejected: {filename} (Incomplete stego set)")


if metadata:
    df = pd.DataFrame(metadata)
    df.to_csv(settings.METADATA_PATH, index=False)
    print(f"\n📁 Metadata saved to {settings.METADATA_PATH}")
    print(f"\n📊 Dataset summary:\n  Accepted originals : {len(metadata) // 9}\n  Rejected originals : {len(original_files) - len(metadata) // 9}\n  Total stego images : {len(metadata)}")

STEGO_DIRS = {settings.OPENSTEGO_DIR, settings.STEGHIDE_DIR, settings.OUTGUESS_DIR}
f = pd.read_csv(settings.METADATA_PATH)
valid_files = set(df["outfile"].values)

# Poista ylimääräiset tiedostot
for tool, path in STEGO_DIRS.items():
    for fname in os.listdir(path):
        full_path = os.path.join(path, fname)
        if fname not in valid_files:
            try:
                os.remove(full_path)
                print(f"[CLEANED] {fname}")
            except Exception as e:
                print(f"[ERROR] Could not remove {fname}: {e}")

# Poista temp-tiedostot
for temp_dir in [settings.STEGHIDE_DIR, settings.OUTGUESS_DIR]:
    for fname in os.listdir(temp_dir):
        if fname.endswith("_temp.jpg"):
            try:
                os.remove(os.path.join(temp_dir, fname))
                print(f"[TEMP CLEANED] {fname}")
            except Exception as e:
                print(f"[ERROR] Could not remove temp {fname}: {e}")

print("\n✅ All stego image generation complete.")