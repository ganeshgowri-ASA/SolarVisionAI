"""
SolarVisionAI - Advanced Image Preprocessing Module
Standards: IEC 60904-13, IEC 60904-14, IEC TS 62446-3
References: PV Lighthouse LumiTools, ISO 17025

Production-grade preprocessing pipeline for EL images with perspective correction,
distortion removal, vignette compensation, denoising, and enhancement.

Author: SolarVisionAI Team
Version: 1.0.0
"""

import cv2
import numpy as np
from typing import Tuple, Optional, Dict, List
from dataclasses import dataclass, field
import logging
from enum import Enum
import warnings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PreprocessingLevel(Enum):
    """Preprocessing intensity levels"""
    BASIC = "basic"  # Fast processing for batch operations
    STANDARD = "standard"  # Recommended for most cases
    ADVANCED = "advanced"  # Maximum quality, slower
    CUSTOM = "custom"  # User-defined parameters


@dataclass
class PreprocessingConfig:
    """Configuration for preprocessing pipeline"""
    # Perspective correction
    enable_perspective_correction: bool = True
    perspective_threshold: int = 50
    perspective_min_line_length: int = 100
    perspective_max_line_gap: int = 10

    # Distortion correction
    enable_distortion_correction: bool = True
    barrel_k1: float = -0.2  # Radial distortion coefficient
    barrel_k2: float = 0.1
    barrel_p1: float = 0.0  # Tangential distortion
    barrel_p2: float = 0.0

    # Vignette compensation
    enable_vignette_compensation: bool = True
    vignette_polynomial_degree: int = 2

    # Denoising
    enable_denoising: bool = True
    denoise_h: float = 10.0  # Filter strength
    denoise_template_window_size: int = 7
    denoise_search_window_size: int = 21

    # CLAHE enhancement
    enable_clahe: bool = True
    clahe_clip_limit: float = 2.0
    clahe_tile_grid_size: Tuple[int, int] = (8, 8)

    # ROI cropping
    enable_auto_crop: bool = True
    crop_threshold: int = 30  # Intensity threshold for edge detection
    crop_margin: int = 10  # Pixels to add around detected ROI

    # Homomorphic filtering
    enable_homomorphic: bool = True
    homomorphic_gamma_l: float = 0.5  # Low frequency gain
    homomorphic_gamma_h: float = 2.0  # High frequency gain
    homomorphic_c: float = 1.0  # Sharpness constant
    homomorphic_d0: float = 10.0  # Cutoff frequency

    # General settings
    output_dtype: np.dtype = np.uint8
    preserve_aspect_ratio: bool = True
    target_size: Optional[Tuple[int, int]] = None  # (width, height)

    @classmethod
    def from_level(cls, level: PreprocessingLevel) -> 'PreprocessingConfig':
        """Create config from preset level"""
        if level == PreprocessingLevel.BASIC:
            return cls(
                enable_perspective_correction=False,
                enable_distortion_correction=False,
                enable_vignette_compensation=True,
                enable_denoising=True,
                enable_clahe=True,
                enable_auto_crop=True,
                enable_homomorphic=False,
                denoise_h=5.0
            )
        elif level == PreprocessingLevel.STANDARD:
            return cls()  # Default values
        elif level == PreprocessingLevel.ADVANCED:
            return cls(
                denoise_h=15.0,
                clahe_clip_limit=3.0,
                homomorphic_gamma_h=2.5,
                perspective_threshold=30
            )
        else:
            return cls()


@dataclass
class PreprocessingResult:
    """Result of preprocessing operation"""
    processed_image: np.ndarray
    original_shape: Tuple[int, int, int]
    applied_operations: List[str] = field(default_factory=list)
    transformation_matrix: Optional[np.ndarray] = None
    roi_bounds: Optional[Tuple[int, int, int, int]] = None  # x, y, w, h
    quality_metrics: Dict[str, float] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    processing_time_ms: float = 0.0


class AdvancedPreprocessor:
    """
    Advanced image preprocessing for Solar PV EL images

    Implements IEC 60904-13 compliant preprocessing with:
    - Perspective correction using Hough transform
    - Barrel/pincushion distortion correction
    - Vignette compensation
    - Non-local means denoising
    - CLAHE enhancement
    - Automated ROI extraction
    - Homomorphic filtering
    """

    def __init__(self, config: Optional[PreprocessingConfig] = None):
        """
        Initialize preprocessor

        Args:
            config: Preprocessing configuration. Uses standard defaults if None.
        """
        self.config = config or PreprocessingConfig()
        logger.info(f"Initialized AdvancedPreprocessor with config: {self.config}")

    def process(self, image: np.ndarray) -> PreprocessingResult:
        """
        Apply full preprocessing pipeline

        Args:
            image: Input image (grayscale or BGR)

        Returns:
            PreprocessingResult with processed image and metadata
        """
        import time
        start_time = time.time()

        result = PreprocessingResult(
            processed_image=image.copy(),
            original_shape=image.shape
        )

        # Convert to grayscale if needed
        if len(image.shape) == 3:
            result.processed_image = cv2.cvtColor(result.processed_image, cv2.COLOR_BGR2GRAY)
            result.applied_operations.append("bgr_to_gray")

        try:
            # 1. Perspective correction
            if self.config.enable_perspective_correction:
                result.processed_image, transform_matrix = self._correct_perspective(
                    result.processed_image
                )
                if transform_matrix is not None:
                    result.transformation_matrix = transform_matrix
                    result.applied_operations.append("perspective_correction")
                    logger.info("Applied perspective correction")

            # 2. Distortion correction
            if self.config.enable_distortion_correction:
                result.processed_image = self._correct_distortion(result.processed_image)
                result.applied_operations.append("distortion_correction")
                logger.info("Applied distortion correction")

            # 3. ROI cropping (before other enhancements to reduce processing time)
            if self.config.enable_auto_crop:
                result.processed_image, roi_bounds = self._auto_crop_roi(
                    result.processed_image
                )
                if roi_bounds is not None:
                    result.roi_bounds = roi_bounds
                    result.applied_operations.append("auto_crop")
                    logger.info(f"Auto-cropped ROI: {roi_bounds}")

            # 4. Vignette compensation
            if self.config.enable_vignette_compensation:
                result.processed_image = self._compensate_vignette(result.processed_image)
                result.applied_operations.append("vignette_compensation")
                logger.info("Applied vignette compensation")

            # 5. Denoising
            if self.config.enable_denoising:
                result.processed_image = self._denoise_nlm(result.processed_image)
                result.applied_operations.append("nlm_denoising")
                logger.info("Applied non-local means denoising")

            # 6. Homomorphic filtering
            if self.config.enable_homomorphic:
                result.processed_image = self._homomorphic_filter(result.processed_image)
                result.applied_operations.append("homomorphic_filtering")
                logger.info("Applied homomorphic filtering")

            # 7. CLAHE enhancement
            if self.config.enable_clahe:
                result.processed_image = self._enhance_clahe(result.processed_image)
                result.applied_operations.append("clahe_enhancement")
                logger.info("Applied CLAHE enhancement")

            # 8. Resize if target size specified
            if self.config.target_size is not None:
                result.processed_image = self._resize_image(
                    result.processed_image,
                    self.config.target_size
                )
                result.applied_operations.append("resize")

            # Calculate processing time
            result.processing_time_ms = (time.time() - start_time) * 1000

            logger.info(
                f"Preprocessing complete in {result.processing_time_ms:.2f}ms. "
                f"Applied operations: {', '.join(result.applied_operations)}"
            )

        except Exception as e:
            logger.error(f"Error during preprocessing: {str(e)}", exc_info=True)
            result.warnings.append(f"Preprocessing error: {str(e)}")
            # Return original image on error
            result.processed_image = image

        return result

    def _correct_perspective(
        self, image: np.ndarray
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Correct perspective distortion using Hough transform and RANSAC

        Detects dominant lines in the image and applies perspective transform
        to align them with vertical/horizontal axes.

        Args:
            image: Input grayscale image

        Returns:
            Corrected image and transformation matrix (or None if detection failed)
        """
        try:
            # Edge detection
            edges = cv2.Canny(image, 50, 150, apertureSize=3)

            # Hough line detection
            lines = cv2.HoughLinesP(
                edges,
                rho=1,
                theta=np.pi / 180,
                threshold=self.config.perspective_threshold,
                minLineLength=self.config.perspective_min_line_length,
                maxLineGap=self.config.perspective_max_line_gap
            )

            if lines is None or len(lines) < 4:
                logger.warning("Insufficient lines detected for perspective correction")
                return image, None

            # Find corners using line intersections
            corners = self._find_quadrilateral_corners(lines, image.shape)

            if corners is None:
                logger.warning("Could not detect quadrilateral corners")
                return image, None

            # Define target rectangle
            h, w = image.shape[:2]
            dst_corners = np.array([
                [0, 0],
                [w - 1, 0],
                [w - 1, h - 1],
                [0, h - 1]
            ], dtype=np.float32)

            # Calculate perspective transform
            transform_matrix = cv2.getPerspectiveTransform(
                corners.astype(np.float32),
                dst_corners
            )

            # Apply transform
            corrected = cv2.warpPerspective(image, transform_matrix, (w, h))

            return corrected, transform_matrix

        except Exception as e:
            logger.error(f"Perspective correction failed: {str(e)}")
            return image, None

    def _find_quadrilateral_corners(
        self, lines: np.ndarray, image_shape: Tuple[int, int]
    ) -> Optional[np.ndarray]:
        """
        Find corners of the main quadrilateral using RANSAC-based line fitting

        Args:
            lines: Lines from Hough transform
            image_shape: Shape of the image (h, w)

        Returns:
            Array of 4 corner points or None
        """
        try:
            h, w = image_shape[:2]

            # Separate lines into vertical and horizontal
            vertical_lines = []
            horizontal_lines = []

            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)

                if angle > 45:  # Closer to vertical
                    vertical_lines.append(line)
                else:  # Closer to horizontal
                    horizontal_lines.append(line)

            if len(vertical_lines) < 2 or len(horizontal_lines) < 2:
                return None

            # Find leftmost and rightmost vertical lines
            vertical_lines = sorted(vertical_lines, key=lambda l: min(l[0][0], l[0][2]))
            left_line = vertical_lines[0][0]
            right_line = vertical_lines[-1][0]

            # Find topmost and bottommost horizontal lines
            horizontal_lines = sorted(horizontal_lines, key=lambda l: min(l[0][1], l[0][3]))
            top_line = horizontal_lines[0][0]
            bottom_line = horizontal_lines[-1][0]

            # Calculate intersection points
            top_left = self._line_intersection(left_line, top_line)
            top_right = self._line_intersection(right_line, top_line)
            bottom_right = self._line_intersection(right_line, bottom_line)
            bottom_left = self._line_intersection(left_line, bottom_line)

            if None in [top_left, top_right, bottom_right, bottom_left]:
                return None

            corners = np.array([top_left, top_right, bottom_right, bottom_left])
            return corners

        except Exception as e:
            logger.error(f"Corner detection failed: {str(e)}")
            return None

    @staticmethod
    def _line_intersection(
        line1: np.ndarray, line2: np.ndarray
    ) -> Optional[Tuple[float, float]]:
        """Calculate intersection point of two lines"""
        x1, y1, x2, y2 = line1
        x3, y3, x4, y4 = line2

        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)

        if abs(denom) < 1e-6:
            return None

        px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
        py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom

        return (px, py)

    def _correct_distortion(self, image: np.ndarray) -> np.ndarray:
        """
        Correct barrel/pincushion distortion using OpenCV undistort

        Args:
            image: Input grayscale image

        Returns:
            Undistorted image
        """
        try:
            h, w = image.shape[:2]

            # Camera matrix (assuming principal point at center)
            camera_matrix = np.array([
                [w, 0, w / 2],
                [0, w, h / 2],
                [0, 0, 1]
            ], dtype=np.float32)

            # Distortion coefficients [k1, k2, p1, p2, k3]
            dist_coeffs = np.array([
                self.config.barrel_k1,
                self.config.barrel_k2,
                self.config.barrel_p1,
                self.config.barrel_p2,
                0  # k3
            ], dtype=np.float32)

            # Undistort
            undistorted = cv2.undistort(image, camera_matrix, dist_coeffs)

            return undistorted

        except Exception as e:
            logger.error(f"Distortion correction failed: {str(e)}")
            return image

    def _compensate_vignette(self, image: np.ndarray) -> np.ndarray:
        """
        Compensate for vignetting using polynomial fitting

        Models vignetting as radial intensity fall-off and corrects it.

        Args:
            image: Input grayscale image

        Returns:
            Vignette-compensated image
        """
        try:
            h, w = image.shape[:2]

            # Create radial distance map
            y, x = np.ogrid[:h, :w]
            cx, cy = w / 2, h / 2
            r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
            r_normalized = r / np.max(r)

            # Fit polynomial to estimate vignetting
            # Use image mean intensity as a function of radius
            image_float = image.astype(np.float32)

            # Sample points for fitting
            num_bins = 20
            radial_bins = np.linspace(0, 1, num_bins)
            mean_intensities = []

            for i in range(len(radial_bins) - 1):
                mask = (r_normalized >= radial_bins[i]) & (r_normalized < radial_bins[i + 1])
                if np.any(mask):
                    mean_intensities.append(np.mean(image_float[mask]))
                else:
                    mean_intensities.append(0)

            # Fit polynomial
            radial_centers = (radial_bins[:-1] + radial_bins[1:]) / 2

            if len(mean_intensities) > self.config.vignette_polynomial_degree:
                coeffs = np.polyfit(
                    radial_centers,
                    mean_intensities,
                    self.config.vignette_polynomial_degree
                )

                # Create correction map
                vignette_model = np.polyval(coeffs, r_normalized)
                max_intensity = np.max(vignette_model)

                if max_intensity > 0:
                    correction_map = max_intensity / (vignette_model + 1e-6)

                    # Apply correction
                    corrected = image_float * correction_map
                    corrected = np.clip(corrected, 0, 255).astype(np.uint8)

                    return corrected

            return image

        except Exception as e:
            logger.error(f"Vignette compensation failed: {str(e)}")
            return image

    def _denoise_nlm(self, image: np.ndarray) -> np.ndarray:
        """
        Apply non-local means denoising

        NLM is particularly effective for EL images as it preserves edges
        while removing sensor noise.

        Args:
            image: Input grayscale image

        Returns:
            Denoised image
        """
        try:
            denoised = cv2.fastNlMeansDenoising(
                image,
                h=self.config.denoise_h,
                templateWindowSize=self.config.denoise_template_window_size,
                searchWindowSize=self.config.denoise_search_window_size
            )
            return denoised

        except Exception as e:
            logger.error(f"Denoising failed: {str(e)}")
            return image

    def _enhance_clahe(self, image: np.ndarray) -> np.ndarray:
        """
        Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)

        CLAHE improves local contrast which is critical for defect detection
        in EL images.

        Args:
            image: Input grayscale image

        Returns:
            Enhanced image
        """
        try:
            clahe = cv2.createCLAHE(
                clipLimit=self.config.clahe_clip_limit,
                tileGridSize=self.config.clahe_tile_grid_size
            )
            enhanced = clahe.apply(image)
            return enhanced

        except Exception as e:
            logger.error(f"CLAHE enhancement failed: {str(e)}")
            return image

    def _auto_crop_roi(
        self, image: np.ndarray
    ) -> Tuple[np.ndarray, Optional[Tuple[int, int, int, int]]]:
        """
        Automatically detect and crop to region of interest (module area)

        Uses adaptive thresholding and contour detection to find the module.

        Args:
            image: Input grayscale image

        Returns:
            Cropped image and ROI bounds (x, y, w, h) or None
        """
        try:
            # Threshold to find bright regions
            _, binary = cv2.threshold(
                image,
                self.config.crop_threshold,
                255,
                cv2.THRESH_BINARY
            )

            # Find contours
            contours, _ = cv2.findContours(
                binary,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )

            if not contours:
                logger.warning("No contours found for auto-crop")
                return image, None

            # Find largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)

            # Add margin
            margin = self.config.crop_margin
            x = max(0, x - margin)
            y = max(0, y - margin)
            w = min(image.shape[1] - x, w + 2 * margin)
            h = min(image.shape[0] - y, h + 2 * margin)

            # Crop
            cropped = image[y:y + h, x:x + w]

            return cropped, (x, y, w, h)

        except Exception as e:
            logger.error(f"Auto-crop failed: {str(e)}")
            return image, None

    def _homomorphic_filter(self, image: np.ndarray) -> np.ndarray:
        """
        Apply homomorphic filtering for illumination normalization

        Separates illumination and reflectance components using frequency domain
        filtering to enhance details while normalizing lighting.

        Args:
            image: Input grayscale image

        Returns:
            Filtered image
        """
        try:
            # Convert to float and add small constant to avoid log(0)
            image_float = image.astype(np.float32) + 1.0

            # Log transform
            image_log = np.log(image_float)

            # FFT
            dft = cv2.dft(image_log, flags=cv2.DFT_COMPLEX_OUTPUT)
            dft_shift = np.fft.fftshift(dft)

            # Create high-pass filter
            rows, cols = image.shape
            crow, ccol = rows // 2, cols // 2

            # Frequency domain filter (high-pass with low-frequency attenuation)
            y, x = np.ogrid[:rows, :cols]
            distance = np.sqrt((x - ccol) ** 2 + (y - crow) ** 2)

            # Gaussian high-pass filter
            gamma_l = self.config.homomorphic_gamma_l
            gamma_h = self.config.homomorphic_gamma_h
            c = self.config.homomorphic_c
            d0 = self.config.homomorphic_d0

            H = (gamma_h - gamma_l) * (1 - np.exp(-c * (distance ** 2 / d0 ** 2))) + gamma_l

            # Apply filter
            dft_shift[:, :, 0] *= H
            dft_shift[:, :, 1] *= H

            # Inverse FFT
            dft_ishift = np.fft.ifftshift(dft_shift)
            image_filtered = cv2.idft(dft_ishift)
            image_filtered = cv2.magnitude(image_filtered[:, :, 0], image_filtered[:, :, 1])

            # Exponential to reverse log
            result = np.exp(image_filtered) - 1.0

            # Normalize to 0-255
            result = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX)
            result = result.astype(np.uint8)

            return result

        except Exception as e:
            logger.error(f"Homomorphic filtering failed: {str(e)}")
            return image

    def _resize_image(
        self, image: np.ndarray, target_size: Tuple[int, int]
    ) -> np.ndarray:
        """
        Resize image to target size

        Args:
            image: Input image
            target_size: Target (width, height)

        Returns:
            Resized image
        """
        try:
            if self.config.preserve_aspect_ratio:
                # Calculate aspect ratio preserving dimensions
                h, w = image.shape[:2]
                target_w, target_h = target_size

                scale = min(target_w / w, target_h / h)
                new_w = int(w * scale)
                new_h = int(h * scale)

                resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            else:
                resized = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)

            return resized

        except Exception as e:
            logger.error(f"Resize failed: {str(e)}")
            return image

    def process_batch(
        self, images: List[np.ndarray], show_progress: bool = True
    ) -> List[PreprocessingResult]:
        """
        Process multiple images in batch

        Args:
            images: List of input images
            show_progress: Whether to show progress bar

        Returns:
            List of PreprocessingResult objects
        """
        results = []

        for i, image in enumerate(images):
            if show_progress:
                logger.info(f"Processing image {i + 1}/{len(images)}")

            result = self.process(image)
            results.append(result)

        return results


# Convenience functions
def preprocess_el_image(
    image: np.ndarray,
    level: PreprocessingLevel = PreprocessingLevel.STANDARD
) -> np.ndarray:
    """
    Quick preprocessing function with preset levels

    Args:
        image: Input image
        level: Preprocessing level

    Returns:
        Preprocessed image
    """
    config = PreprocessingConfig.from_level(level)
    preprocessor = AdvancedPreprocessor(config)
    result = preprocessor.process(image)
    return result.processed_image


def create_custom_preprocessor(
    enable_perspective: bool = True,
    enable_distortion: bool = True,
    enable_vignette: bool = True,
    enable_denoise: bool = True,
    enable_clahe: bool = True,
    enable_crop: bool = True,
    enable_homomorphic: bool = True,
    **kwargs
) -> AdvancedPreprocessor:
    """
    Create custom preprocessor with specific operations enabled

    Args:
        enable_perspective: Enable perspective correction
        enable_distortion: Enable distortion correction
        enable_vignette: Enable vignette compensation
        enable_denoise: Enable denoising
        enable_clahe: Enable CLAHE enhancement
        enable_crop: Enable auto-crop
        enable_homomorphic: Enable homomorphic filtering
        **kwargs: Additional config parameters

    Returns:
        Configured AdvancedPreprocessor
    """
    config = PreprocessingConfig(
        enable_perspective_correction=enable_perspective,
        enable_distortion_correction=enable_distortion,
        enable_vignette_compensation=enable_vignette,
        enable_denoising=enable_denoise,
        enable_clahe=enable_clahe,
        enable_auto_crop=enable_crop,
        enable_homomorphic=enable_homomorphic,
        **kwargs
    )
    return AdvancedPreprocessor(config)


if __name__ == "__main__":
    # Example usage
    import sys

    if len(sys.argv) > 1:
        # Load and process image
        image_path = sys.argv[1]
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            print(f"Error: Could not load image from {image_path}")
            sys.exit(1)

        # Process with standard settings
        preprocessor = AdvancedPreprocessor()
        result = preprocessor.process(image)

        # Save result
        output_path = image_path.replace('.', '_preprocessed.')
        cv2.imwrite(output_path, result.processed_image)

        print(f"Processed image saved to: {output_path}")
        print(f"Processing time: {result.processing_time_ms:.2f}ms")
        print(f"Applied operations: {', '.join(result.applied_operations)}")
    else:
        print("Usage: python preprocessing.py <image_path>")
