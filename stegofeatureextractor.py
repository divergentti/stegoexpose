import cv2
import numpy as np
from scipy.fftpack import dct
from scipy.stats import entropy

class StegoFeatureExtractor:
    """
    Extracts absolute statistical features from a single image for use in
    clean vs. stego classification.

    Features include entropy, variance, DCT energy distribution,
    and least significant bit (LSB) ratios.
    """
    def __init__(self, image_path: str):
        """
        Initialize the extractor with an image path.

        Args:
            image_path (str): Path to the image to analyze.

        Raises:
            ValueError: If the image cannot be loaded.
        """
        self.image_path = image_path
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise ValueError(f"Image could not be loaded: {image_path}")
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)

    def calculate_entropy(self):
        """Calculate average entropy across RGB channels."""
        entropies = []
        for i in range(3):
            hist = cv2.calcHist([self.image], [i], None, [256], [0, 256]).flatten()
            hist_prob = hist / np.sum(hist)
            entropies.append(entropy(hist_prob + 1e-10))  # add epsilon to avoid log(0)
        return np.mean(entropies)

    def calculate_variance(self):
        """Calculate average pixel intensity variance across RGB channels."""
        return float(np.mean([np.var(self.image[:, :, i]) for i in range(3)]))

    def calculate_dct_energy(self):
        """
        Calculate DCT energy statistics on 8x8 grayscale blocks.

        Returns:
            Tuple[float, float]: Mean and variance of energy across blocks.
        """
        gray = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)
        h, w = gray.shape
        h -= h % 8
        w -= w % 8
        blocks = [
            gray[i:i+8, j:j+8]
            for i in range(0, h, 8)
            for j in range(0, w, 8)
        ]
        energies = [
            np.sum(dct(dct(block.T, norm='ortho').T, norm='ortho')**2)
            for block in blocks
        ]
        return float(np.mean(energies)), float(np.var(energies))

    def calculate_lsb_ratios(self):
        """
        Calculate the average and variance of 1-bit ratios in LSB plane of RGB channels.

        Returns:
            Tuple[float, float]: Mean and variance of LSB-1s ratios.
        """
        ratios = []
        for i in range(3):
            channel = self.image[:, :, i]
            lsb = channel & 1
            ones_ratio = np.sum(lsb) / lsb.size
            ratios.append(ones_ratio)
        return float(np.mean(ratios)), float(np.var(ratios))

    def extract_features(self) -> dict:
        """
        Run all feature extraction routines and return a flat dictionary.

        Returns:
            dict: Extracted feature names and values.
        """
        entropy_val = self.calculate_entropy()
        variance_val = self.calculate_variance()
        dct_mean, dct_var = self.calculate_dct_energy()
        lsb_mean, lsb_var = self.calculate_lsb_ratios()

        return {
            'entropy_rgb': entropy_val,
            'variance_rgb': variance_val,
            'dct_energy_mean': dct_mean,
            'dct_energy_var': dct_var,
            'lsb_ratio_1s_mean': lsb_mean,
            'lsb_ratio_var': lsb_var
        }
