# 🕵️ StegaExpose – A Modular Framework for Detecting and Decoding Steganography in Media Files

**StegaExpose** is a Python-based CLI tool and evolving framework for **analyzing**, **detecting**, and **classifying** steganographic content in digital media. It currently focuses on image-based steganography (e.g., OpenStego, Steghide, OutGuess) and supports both heuristic analysis and machine learning.

---

## 🌟 Update Summary (2025-05-07)

Originally, the project explored a CNN-based binary classification approach (**clean vs stego**), but training was negatively affected by **class imbalance** and **biases in the training set**. The current version introduces a **feature-based analysis pipeline**, where structured traces are extracted from images and used in classification models.

---

## 🧪 Key Improvements

- Clean/stego classification is now handled via extracted **absolute and relative features**, such as:
  - Histogram KL divergence
  - DCT energy profiles
  - LSB change ratios
  - Entropy and variance
- Fingerprint models are trained per tool (**OpenStego, Steghide, OutGuess**) using robust trace vectors
- A `DecisionTree`-style logic governs conditional feature selection based on image type, compression format, or mismatch
- A dual-pipeline allows for:
  - **Comparison-based classification** (with original image)
  - **Single-image classification** (without original)

---

## 🧠 Refactored & Modular Python Scripts

Below is a summary of the main Python files and their roles:

| File | Purpose |
|------|---------|
| `parse2database.py` | Ingests original + stego image paths, computes trace features, stores all results in SQLite database |
| `predictor.py` | Contains `predict_from_images` and `predict_single_clean_or_stego` functions using RandomForest models |
| `run_dual_prediction_test.py` | Helper script to test prediction with and without reference images |
| `stegofeatureextractor.py` | Extracts entropy, variance, DCT, and LSB-based features from a single image |
| `train_clean_stego_singleimage.py` | Trains a model to detect clean vs stego based on single image analysis |
| `train_stego_vs_clean.py` | Trains a model using trace features comparing original and stego images |
| `train_model.py` | Trains a tool classifier to predict which stego tool was used |
| `process_clean_images.py` | Adds clean images to the database and analyzes them using the same trace pipeline |
| `generate_stego_images.py` | Automates embedding using OpenStego, Steghide, and OutGuess + generates metadata.csv |

---

## 🤖 Current Models & Outputs

| Model Path | Description |
|------------|-------------|
| `rf_clean_vs_stego.pkl` | Relative trace model trained with image pairs |
| `rf_single_clean_vs_stego.pkl` | Absolute feature model for single-image stego detection |
| `rf_stegomodel.pkl` | Tool classifier predicting which stego software was used |

Each model outputs both class prediction and class-wise probabilities.

---

## 🔨 How to Use

```bash
# Extract trace features and store to database
python parse2database.py

# Train models
python train_clean_stego_singleimage.py
python train_stego_vs_clean.py
python train_model.py

# Run test predictions
python run_dual_prediction_test.py
```

---

## 🔍 Future Steps

- Improve feature normalization and balance for clean/stego classification
- Integrate GUI to visualize traces, differences, and classification confidence
- Add support for custom tools (e.g., Jari's encoder)
- Bundle as CLI and GUI hybrid app for forensic analysts

---

## 📄 License

MIT License

