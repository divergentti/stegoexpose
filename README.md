
# 🕵️ StegaExpose – A Modular Framework for Detecting and Decoding Steganography in Media Files

**StegaExpose** is a Python-based CLI tool and evolving framework for **analyzing**, **detecting**, and **classifying** steganographic content in digital media. It currently focuses on image-based steganography (e.g., OpenStego, Steghide, OutGuess) and supports both heuristic analysis and machine learning.

Videos about development steps:

![StegoExpose draft at YouTube](https://youtu.be/Vkgzyo-4qbU)


---

## 🌟 Update Summary (2025-05-07) (code now at oldcodes path)

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

## 🧠 Refactored & Modular Python Scripts (old)

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

## 🤖 Old Models & Outputs

| Model Path | Description |
|------------|-------------|
| `rf_clean_vs_stego.pkl` | Relative trace model trained with image pairs |
| `rf_single_clean_vs_stego.pkl` | Absolute feature model for single-image stego detection |
| `rf_stegomodel.pkl` | Tool classifier predicting which stego software was used |

Each model outputs both class prediction and class-wise probabilities.

---

## 🔨 Old code How to Use

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

## 📸 Dataset + GUI Integration Update (2025-05-09)

Over the last 24 hours, a structured database of stego images was generated. The final SQLite database is 6.9 MB and contains data for 14,868 stego images (4956 base images × 3 tools × 3 payload sizes), created using OpenStego, Steghide, and OutGuess derived from 2,076 original images. These originals span various resolutions, formats, devices, and even AI-generated content.

To balance the dataset, images for which all stego tools could not embed messages of all sizes were excluded. Additionally, 139 clean images were included to support single-image analysis. These supplement the reference-based comparison approach.

All code was refactored to use a unified `settings.py` that dynamically resolves paths using `runtimeconfig.json`, supporting both absolute and relative modes. Paths are now reflecting change, the GUI at ./gui and utils at ./utils. If you like to test, execute main_window.py at ./gui

### 🖼 Analyze GUI

A new PyQt6-based analysis window has been implemented, enabling users to:
- Select an original image, stego tool, and message size
- View original and stego image side-by-side
- Inspect metadata and feature traces including:
  - Shape mismatch
  - KL divergence
  - LSB avg change %
  - SRM and wavelet metrics

This interface sets the foundation for future visualizations, including heatmaps and unsupervised feature inspection.

![GUI screenshot](https://github.com/divergentti/stegoexpose/blob/main/images/gui090525.png) 


## 🔍 Future Steps

- Add support for custom tools (e.g., Jari's [encoder](https://github.com/divergentti/steganography))

---

## 📄 License

MIT License

