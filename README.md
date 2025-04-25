# 🕵️ StegaExpose – A Modular Framework for Detecting and Decoding Steganography in Media Files

**StegaExpose** is a Python-based CLI tool and evolving framework for **analyzing**, **detecting**, and **decrypting** steganographic content in digital media. Currently focused on image-based steganography (e.g., OpenStego), the system uses machine learning and classical forensics to expose the invisible.

---

## 🎯 Project Objectives

- Detect signs of **hidden messages** in media files using visual fingerprints, LSB alterations, or compression artifacts.
- Train a lightweight **CNN model** for binary classification: clean vs. stego.
- Reverse-engineer and extract embedded content, with a focus on **OpenStego's format** and header structure.
- Use **metadata analysis**, **heuristics**, and **cryptographic insight** to understand and decrypt stego content.
- Build toward a plugin-based architecture capable of supporting multiple stego tools.

---

## 🧠 Key Components

### `FileTypeAnalyzer`
- Extracts media metadata: name, size, timestamps, compression type.
- Identifies whether the file is lossy or lossless (e.g., JPEG vs PNG).
- Suggests steganography methods used, depending on format.

### `Guesser`
- Compares original vs. stego image to detect **pixel-level anomalies**.
- Calculates LSB deviation maps, spatial randomness, and SPA (Sample Pair Analysis).
- Uses a **trained CNN** model (`model.safetensors`) to classify files.
- Implements both heuristic and learned techniques for **fingerprinting tools** like OpenStego.

### `Decrypter`
- Uses structured bit-level extraction to recover data.
- Decodes OpenStego headers and payloads.
- Implements both **random-seeded LSB extraction** and **sequential fallback** when needed.
- Contains a **Java-style Random** class to replicate OpenStego’s PRNG.

---

## 🧪 Model Training

A lightweight **CNN with attention blocks** has been trained to differentiate between clean and stego images using **residual images** computed from the difference between original and modified files.

Example from training output:
```
Epoch 4/30, Train Loss: 0.0000, Train Acc: 1.0000
Epoch 4/30, Val Loss: 0.0000, Val Acc: 1.0000
```

The model converges fast due to the synthetic, labeled image pairs. The training data includes images manipulated by OpenStego.

Trained model is saved as: `model.safetensors`

---

## 🔍 Fingerprinting Example

From the `Guesser` and `Decrypter`, the framework can currently:
- Detect OpenStego's "OPENSTEGO" **magic number** in the image’s LSBs
- Decode the **header fields**: version, compression, encryption flags, filename, and actual message
- Attempt to recover payload even with incorrect seeds, by **scanning multiple PRNG outputs**

---

## 📉 Architecture Overview

```
          ┌────────────────────────┐
          │ Media Input  │
          └────────────────────────┘
                 ▼
       ┌──────────────────────────┐
       │ FileTypeAnalyzer   │───> Format + Metadata
       └──────────────────────────┘
                 ▼
       ┌──────────────────────────┐
       │     Guesser        │───> Prob(Stego), Regions, Features
       └──────────────────────────┘
                 ▼
       ┌──────────────────────────┐
       │     Decrypter      │───> Message, Filename, Metadata
       └──────────────────────────┘
```

---

## 🛠 Usage (CLI Example)

```bash
# Run analysis on suspected image
python stega-day3-1.py

# Training and evaluating the model
python model_training.py
```

---

## 🧬 Dataset & Structure

Expected folder structure:

```
ml/
├── clean/
│   ├── image1.bmp
│   └── image2.bmp
├── stego/
│   ├── image1.bmp
│   └── image2.bmp

openstego/
├── original-bmp/
│   └── image1.bmp
├── encrypted-from-bmp/
│   └── image1.bmp

test_stego_new.bmp
```

---

## 📆 Model Features

- Input: residual maps (difference between clean and stego images)
- Attention blocks for enhanced feature capture
- Automatic training with `autocast` and `GradScaler` for mixed precision on CUDA
- Trained using BCEWithLogits for binary classification

---

## 🧠 Intelligence Layer (Under Development)

Planned reinforcement learning for Guesser:
- Adjust thresholds based on feedback from Decrypter
- Learn to reduce false positives across tools (OpenStego, Steghide, etc.)

---

## 📊 Output Format

Example prediction:
```
Prediction for test_stego_new.bmp: Prob=1.0000, Class=Stego
Prediction for image1.bmp: Prob=1.0000, Class=Stego
```

Future output in JSON:
```json
{
  "filename": "image1.bmp",
  "classification": "Stego",
  "confidence": 0.9987,
  "tool_guess": "OpenStego",
  "decrypted": {
    "filename": "secret.txt",
    "content": "The eagle has landed."
  }
}
```

---

## ✅ Code Quality

- Python 3.10+
- Linting via `flake8` / `black`
- Unit testing with `pytest`
- Continuous integration via GitHub Actions (planned)

---

## 🧪 Future Ideas

- Support for audio files (MP3, WAV) using phase coding and echo hiding.
- Add GUI interface with anomaly heatmaps and drag-and-drop analysis.
- Brute-force OpenStego seeds based on entropy matching.
- Integration with `stable-baselines3` for RL-enhanced guessing.

---

## 📜 License

MIT License (or specify your own)
