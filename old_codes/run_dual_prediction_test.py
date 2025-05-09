from predictor import predict_from_images, predict_single_clean_or_stego

def run_dual_prediction(original_path: str, test_path: str):
    """
    Run two prediction methods on the same test image:
    1. predict_from_images: Compares the test image against a reference (original) image.
    2. predict_single_clean_or_stego: Predicts stego vs. clean without reference.

    Args:
        original_path (str): Path to the clean original image.
        test_path (str): Path to the image to be tested (possibly stego).

    Returns:
        Tuple containing the results from both prediction methods.
    """
    print(f"\n[TEST] Original: {original_path}")
    print(f"[TEST] Test Image: {test_path}")

    print("\n🧠 [1] predict_from_images() - With reference image")
    try:
        result_pair = predict_from_images(original_path, test_path)
        print(f"Prediction (tool): {result_pair['prediction']}")
        print(f"Confidence: {result_pair['confidence']:.2%}")
        print(f"Probabilities: {result_pair['probabilities']}")
    except Exception as e:
        print(f"[ERROR] predict_from_images() failed: {e}")
        result_pair = None

    print("\n🧠 [2] predict_single_clean_or_stego() - Without reference image")
    try:
        result_single = predict_single_clean_or_stego(test_path)
        print(f"Prediction (clean/stego): {result_single['prediction']}")
        print(f"Confidence: {result_single['confidence']:.2%}")
        print(f"Probabilities: {result_single['probabilities']}")
    except Exception as e:
        print(f"[ERROR] predict_single_clean_or_stego() failed: {e}")
        result_single = None

    return result_pair, result_single

if __name__ == "__main__":
    # Replace these paths with test files of your choice
    orig = "./training/testimages/original_clean_reference.jpg"
    test = "./training/testimages/DSCN9224.JPG"

    run_dual_prediction(orig, test)
