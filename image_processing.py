import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

class SonarImageProcessor:
    def __init__(self):
        self.config = {
            # Horizontal line removal
            "apply_fft_filter": True,
            "fft_band_width": 15,            # Try 5-15
            "fft_band_strength": 0.0,       # 0.0 = complete removal, 0.5 = partial
            
            "apply_remove_hlines": True,    # Morphological horizontal line removal
            "hline_kernel_width": 40,       # Try 15-40
            
            "apply_bilateral": True,        # Edge-preserving smoothing
            "bilateral_d": 9,
            "bilateral_sigma_color": 75,
            "bilateral_sigma_space": 75,
            
            # Basic filters
            "apply_denoise": True,
            "apply_gaussian": True,
            "gaussian_ksize": (3, 3),
            "gaussian_sigma": 3,
            "apply_median": False,
            "median_ksize": 3,
            
            # Enhancement
            "apply_otsu": False,
            "apply_fuzzy": True,
            "gamma_value": 1.2,
            
            # Morphology
            "apply_opening": True,
            "morph_kernel": cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
            
            # Final
            "final_normalization": True
        }

    # ========== HORIZONTAL LINE REMOVAL METHODS ==========
    
    def remove_horizontal_lines_fft(self, img: np.ndarray) -> np.ndarray:
        """
        Remove horizontal lines using FFT filtering.
        Blocks horizontal frequency components in Fourier domain.
        """
        if not self.config.get("apply_fft_filter", False):
            return img
        
        try:
            # Perform FFT
            dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
            dft_shift = np.fft.fftshift(dft)
            
            # Get dimensions
            rows, cols = img.shape
            crow, ccol = rows // 2, cols // 2
            
            # Create mask to remove horizontal frequencies
            mask = np.ones((rows, cols, 2), np.float32)
            
            # Use configurable band width and strength
            band_width = self.config.get("fft_band_width", 2)
            band_strength = self.config.get("fft_band_strength", 0.0)
            
            # Block horizontal frequencies (center horizontal band)
            mask[crow-band_width:crow+band_width, :] = band_strength
            
            # Apply mask
            fshift = dft_shift * mask
            
            # Inverse FFT
            f_ishift = np.fft.ifftshift(fshift)
            img_back = cv2.idft(f_ishift)
            img_back = cv2.magnitude(img_back[:,:,0], img_back[:,:,1])
            
            # Normalize
            return cv2.normalize(img_back, None, 0, 1, cv2.NORM_MINMAX).astype(np.float32)
        
        except Exception as e:
            return img
    
    def remove_horizontal_lines_morph(self, img: np.ndarray) -> np.ndarray:
        """
        Remove horizontal line artifacts using morphological operations.
        Detects and subtracts horizontal structures.
        """
        if not self.config.get("apply_remove_hlines", False):
            return img
        
        try:
            img_u8 = (img * 255).astype(np.uint8)
            
            # Create horizontal kernel
            kernel_width = self.config.get("hline_kernel_width", 25)
            horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_width, 1))
            
            # Detect horizontal lines
            detected_lines = cv2.morphologyEx(img_u8, cv2.MORPH_OPEN, horizontal_kernel)
            
            # Subtract detected lines (weighted to avoid over-subtraction)
            result = cv2.subtract(img_u8, detected_lines // 2)
            
            return result.astype(np.float32) / 255.0
        
        except Exception as e:
            return img
    
    def apply_bilateral_filter(self, img: np.ndarray) -> np.ndarray:
        """
        Bilateral filter - edge-preserving smoothing.
        Smooths uniform regions while preserving edges.
        Great for reducing horizontal banding without blurring features.
        """
        if not self.config.get("apply_bilateral", False):
            return img
        
        try:
            img_u8 = (img * 255).astype(np.uint8)
            filtered = cv2.bilateralFilter(
                img_u8, 
                self.config["bilateral_d"],
                self.config["bilateral_sigma_color"],
                self.config["bilateral_sigma_space"]
            )
            return filtered.astype(np.float32) / 255.0
        except Exception as e:
            return img

    # ========== BASIC PROCESSING METHODS ==========
    
    def apply_denoising(self, img: np.ndarray) -> np.ndarray:
        """
        Apply row-wise background subtraction denoising.
        Removes systematic background bias from sonar images.
        """
        if not self.config.get("apply_denoise", False):
            return img
        
        try:
            if img.dtype != np.float32:
                img = img.astype(np.float32) / 255.0 if img.max() > 1 else img.astype(np.float32)
            
            # Estimate background using quantile per row and coloumn 
            background = np.quantile(img, 0.4, axis=1, keepdims=True)
            img_denoised = np.clip(img - background, 0.0, None)
            background = np.quantile(img_denoised, 0.4, axis=0, keepdims=True)
            img_denoised = np.clip(img_denoised - background, 0.0, None)

            # Renormalize if image has content
            if img_denoised.max() > 0:
                img_denoised = img_denoised / (img_denoised.max() + 1e-6)
            
            return img_denoised
        
        except Exception as e:
            return img
    
    def apply_smoothing(self, img: np.ndarray) -> np.ndarray:
        """
        Apply Gaussian and median smoothing filters.
        Reduces random noise and speckle.
        """
        # Gaussian blur
        if self.config.get("apply_gaussian", True):
            img = cv2.GaussianBlur(
                img, 
                self.config["gaussian_ksize"], 
                self.config["gaussian_sigma"]
            )
        
        # Median filter
        if self.config.get("apply_median", True):
            img_u8 = (img * 255).astype(np.uint8)
            img_median = cv2.medianBlur(img_u8, self.config["median_ksize"])
            img = img_median.astype(np.float32) / 255.0
        
        return img
    
    def apply_thresholding(self, img: np.ndarray) -> np.ndarray:
        """
        Apply Otsu thresholding for binary masking.
        Automatically separates foreground from background.
        """
        if not self.config.get("apply_otsu", False):
            return img
        
        try:
            img_u8 = (img * 255).astype(np.uint8)
            ret, binary_mask = cv2.threshold(img_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            binary_mask = binary_mask.astype(np.float32) / 255.0
            return img * binary_mask
        
        except Exception as e:
            return img
    
    def apply_enhancement(self, img: np.ndarray) -> np.ndarray:
        """
        Apply fuzzy gamma enhancement.
        Enhances contrast using fuzzy set theory.
        """
        if not self.config.get("apply_fuzzy", False):
            return img
        
        try:
            epsilon = 1e-20
            img = np.clip(img, 0.0, 1.0)
            gamma = float(self.config.get("gamma_value", 2.0))
            
            # Fuzzy gamma enhancement
            img_enhanced = (img ** gamma) / (img ** gamma + (1.0 - img + epsilon) ** gamma)
            return img_enhanced
        
        except Exception as e:
            return img
    
    def apply_morphological_operations(self, img: np.ndarray) -> np.ndarray:
        """
        Apply morphological opening for noise removal.
        Removes small bright spots while preserving larger features.
        """
        if not self.config.get("apply_opening", False):
            return img
        
        try:
            img_u8 = (img * 255).astype(np.uint8)
            img_opened = cv2.morphologyEx(
                img_u8, 
                cv2.MORPH_OPEN, 
                self.config["morph_kernel"]
            )
            return img_opened.astype(np.float32) / 255.0
        
        except Exception as e:
            return img
    
    def apply_final_normalization(self, img: np.ndarray) -> np.ndarray:
        """
        Apply final normalization to ensure [0,1] range.
        Standardizes output intensity range.
        """
        if not self.config.get("final_normalization", True):
            return img
        
        return cv2.normalize(img, None, 0, 1.0, cv2.NORM_MINMAX)
    
    def size_image(self, img: np.ndarray, top_rows: int = 20, bottom_rows: int = 20) -> np.ndarray:
        """
        Set pixels to zero for top and bottom rows.
        Removes noisy edge regions common in sonar images.
        """
        height = img.shape[0]
        new_image = img.copy()
        
        new_image[0:top_rows, :] = 0
        new_image[height-bottom_rows:height, :] = 0
        
        return new_image
    

    def process_image_for_matching(self, img: np.ndarray) -> np.ndarray:
        """Minimal processing to preserve features"""
        # Only remove noisy edges
        img = self.size_image(img, top_rows=20, bottom_rows=20)
        
        # Very light denoising only
        img = self.apply_denoising(img)
        
        # Light Gaussian blur to reduce speckle noise
        img = cv2.GaussianBlur(img, (3, 3), 1.0)
        
        # Normalize
        return cv2.normalize(img, None, 0, 1.0, cv2.NORM_MINMAX)



    # ========== MAIN PROCESSING PIPELINE ==========
    
    def process_image(self, img: np.ndarray) -> np.ndarray:
        """
        Complete processing pipeline optimized for horizontal line removal.
        
        Pipeline order:
        1. Denoise - Remove background bias
        2. FFT Filter - Remove horizontal frequencies
        3. Morphological Line Removal - Remove horizontal structures
        4. Bilateral Filter - Edge-preserving smoothing
        5. General Smoothing - Reduce remaining noise
        6. Enhancement - Boost contrast
        7. Thresholding - Separate targets
        8. Morphology - Clean up
        9. Normalize - Standardize output
        """
        # Phase 1: Initial denoising
        #img = self.size_image(img)
        #img = self.apply_denoising(img)
       
        
        # Phase 2: Horizontal line removal (multi-method approach)
        #img = self.remove_horizontal_lines_fft(img)         # FFT filtering
        #img = self.remove_horizontal_lines_morph(img)       # Morphological removal
        #img = self.apply_bilateral_filter(img)              # Edge-preserving smoothing
        
        # Phase 3: General smoothing
        #img = self.apply_smoothing(img)

        
        # Phase 4: Enhancement and thresholding
        #img = self.apply_enhancement(img)
        #img = self.apply_thresholding(img)
        
        # Phase 5: Final cleanup
        #img = self.apply_morphological_operations(img)
        #img = self.apply_final_normalization(img)
        #img = self.process_image_for_matching(img)
        return img

    
# ========== TUNING GUIDE ==========
"""
HORIZONTAL LINE REMOVAL TUNING:

1. FFT Filter (remove_horizontal_lines_fft):
   - fft_band_width: 5-15 (higher = more aggressive)
   - fft_band_strength: 0.0-0.5 (0.0 = complete removal)
   
   Try first: fft_band_width=8, fft_band_strength=0.0
   If too aggressive: fft_band_width=5, fft_band_strength=0.2
   If not enough: fft_band_width=12, fft_band_strength=0.0

2. Morphological Removal (remove_horizontal_lines_morph):
   - hline_kernel_width: 15-40 (width of lines to detect)
   
   Try first: hline_kernel_width=25
   For thin lines: hline_kernel_width=15
   For thick lines: hline_kernel_width=35

3. Bilateral Filter (apply_bilateral_filter):
   - bilateral_d: 7-15 (neighborhood size)
   - bilateral_sigma_color/space: 50-100
   
   Try first: d=9, sigma=75
   More smoothing: d=15, sigma=100
   Less smoothing: d=7, sigma=50

QUICK TEST CONFIGURATIONS:

Mild (preserve detail):
    "fft_band_width": 5
    "apply_remove_hlines": False
    "apply_bilateral": True
    "median_ksize": 5

Moderate (balanced):
    "fft_band_width": 8
    "apply_remove_hlines": True
    "apply_bilateral": True
    "median_ksize": 7

Aggressive (maximum line removal):
    "fft_band_width": 12
    "apply_remove_hlines": True
    "hline_kernel_width": 35
    "apply_bilateral": True
    "median_ksize": 9
    "gaussian_ksize": (9, 9)
"""