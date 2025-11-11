"""
SolarVisionAI - Image Quality Analytics Engine
Standards: IEC 60904-13 (Sharpness Class), ISO/IEC 12233, ISO 17025

Comprehensive image quality assessment for EL images including:
- SNR (Signal-to-Noise Ratio)
- Sharpness metrics
- Statistical measures
- Shannon entropy
- IEC 60904-13 compliance validation

Author: SolarVisionAI Team
Version: 1.0.0
"""

import cv2
import numpy as np
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass, field, asdict
from enum import Enum
import logging
from scipy import ndimage, stats as scipy_stats
import warnings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class IECSharpnessClass(Enum):
    """IEC 60904-13 Sharpness Classification"""
    CLASS_A = "A"  # Excellent sharpness (MTF50 > 0.6)
    CLASS_B = "B"  # Good sharpness (0.4 < MTF50 <= 0.6)
    CLASS_C = "C"  # Acceptable sharpness (0.2 < MTF50 <= 0.4)
    CLASS_D = "D"  # Poor sharpness (MTF50 <= 0.2)
    INVALID = "Invalid"  # Cannot determine


class QualityLevel(Enum):
    """Overall quality classification"""
    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    POOR = "poor"
    UNUSABLE = "unusable"


@dataclass
class ImageQualityMetrics:
    """Comprehensive image quality metrics"""
    # Signal-to-Noise Ratio
    snr_db: float = 0.0
    psnr_db: float = 0.0  # Peak SNR

    # Sharpness metrics
    sharpness_laplacian: float = 0.0
    sharpness_gradient: float = 0.0
    sharpness_fft: float = 0.0
    mtf50: float = 0.0  # Modulation Transfer Function at 50%
    edge_contrast: float = 0.0

    # Statistical measures
    mean_intensity: float = 0.0
    std_deviation: float = 0.0
    variance: float = 0.0
    kurtosis: float = 0.0
    skewness: float = 0.0
    cv_coefficient: float = 0.0  # Coefficient of variation

    # Entropy and information content
    shannon_entropy: float = 0.0
    normalized_entropy: float = 0.0

    # Histogram metrics
    dynamic_range: int = 0
    histogram_uniformity: float = 0.0
    contrast_ratio: float = 0.0

    # IEC 60904-13 compliance
    iec_sharpness_class: str = IECSharpnessClass.INVALID.value
    iec_compliant: bool = False
    iec_min_resolution_met: bool = False
    iec_min_contrast_met: bool = False

    # Overall quality
    quality_score: float = 0.0  # 0-100 composite score
    quality_level: str = QualityLevel.POOR.value

    # Image properties
    width: int = 0
    height: int = 0
    total_pixels: int = 0
    bit_depth: int = 8

    # Warnings and issues
    issues: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return asdict(self)

    def get_summary(self) -> str:
        """Get human-readable summary"""
        return f"""
Image Quality Analysis Summary
{'=' * 50}
Dimensions: {self.width}x{self.height} ({self.total_pixels:,} pixels)
Quality Level: {self.quality_level.upper()} (Score: {self.quality_score:.1f}/100)

Signal Quality:
  - SNR: {self.snr_db:.2f} dB
  - PSNR: {self.psnr_db:.2f} dB

Sharpness:
  - IEC 60904-13 Class: {self.iec_sharpness_class}
  - MTF50: {self.mtf50:.3f}
  - Edge Contrast: {self.edge_contrast:.2f}
  - Laplacian Variance: {self.sharpness_laplacian:.2f}

Statistics:
  - Mean: {self.mean_intensity:.2f}
  - Std Dev: {self.std_deviation:.2f}
  - Kurtosis: {self.kurtosis:.2f}
  - Skewness: {self.skewness:.2f}

Entropy:
  - Shannon Entropy: {self.shannon_entropy:.3f} bits
  - Normalized: {self.normalized_entropy:.3f}

IEC Compliance:
  - Overall: {'✓ PASS' if self.iec_compliant else '✗ FAIL'}
  - Resolution: {'✓' if self.iec_min_resolution_met else '✗'}
  - Contrast: {'✓' if self.iec_min_contrast_met else '✗'}

Issues: {len(self.issues)}
{chr(10).join(f'  - {issue}' for issue in self.issues) if self.issues else '  None'}
{'=' * 50}
"""


class ImageAnalyticsEngine:
    """
    Advanced image quality analytics for EL images

    Implements comprehensive quality assessment following:
    - IEC 60904-13: EL measurement of photovoltaic modules
    - ISO/IEC 12233: Photography - Resolution and spatial frequency responses
    - ISO 17025: General requirements for competence of testing labs
    """

    # IEC 60904-13 minimum requirements
    IEC_MIN_RESOLUTION = 640  # pixels (minimum dimension)
    IEC_MIN_CONTRAST = 20  # minimum dynamic range
    IEC_MIN_SHARPNESS = 0.2  # minimum MTF50

    def __init__(
        self,
        reference_image: Optional[np.ndarray] = None,
        enable_advanced_metrics: bool = True
    ):
        """
        Initialize analytics engine

        Args:
            reference_image: Optional reference/noise-free image for SNR calculation
            enable_advanced_metrics: Enable computationally expensive metrics
        """
        self.reference_image = reference_image
        self.enable_advanced_metrics = enable_advanced_metrics
        logger.info("Initialized ImageAnalyticsEngine")

    def analyze(self, image: np.ndarray) -> ImageQualityMetrics:
        """
        Perform comprehensive quality analysis

        Args:
            image: Input image (grayscale or BGR)

        Returns:
            ImageQualityMetrics with all computed metrics
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray_image = image.copy()

        metrics = ImageQualityMetrics()

        # Basic properties
        metrics.height, metrics.width = gray_image.shape
        metrics.total_pixels = metrics.width * metrics.height
        metrics.bit_depth = 8 if gray_image.dtype == np.uint8 else 16

        try:
            # 1. Statistical measures
            self._calculate_statistics(gray_image, metrics)

            # 2. Signal-to-Noise Ratio
            self._calculate_snr(gray_image, metrics)

            # 3. Sharpness metrics
            self._calculate_sharpness(gray_image, metrics)

            # 4. Entropy measures
            self._calculate_entropy(gray_image, metrics)

            # 5. Histogram analysis
            self._calculate_histogram_metrics(gray_image, metrics)

            # 6. IEC 60904-13 compliance
            self._assess_iec_compliance(metrics)

            # 7. Overall quality score
            self._calculate_quality_score(metrics)

            logger.info(
                f"Analysis complete: Quality={metrics.quality_level}, "
                f"Score={metrics.quality_score:.1f}, IEC Class={metrics.iec_sharpness_class}"
            )

        except Exception as e:
            logger.error(f"Error during analysis: {str(e)}", exc_info=True)
            metrics.issues.append(f"Analysis error: {str(e)}")

        return metrics

    def _calculate_statistics(
        self, image: np.ndarray, metrics: ImageQualityMetrics
    ) -> None:
        """Calculate basic statistical measures"""
        try:
            # Mean and standard deviation
            metrics.mean_intensity = float(np.mean(image))
            metrics.std_deviation = float(np.std(image))
            metrics.variance = float(np.var(image))

            # Coefficient of variation (normalized std dev)
            if metrics.mean_intensity > 0:
                metrics.cv_coefficient = metrics.std_deviation / metrics.mean_intensity
            else:
                metrics.cv_coefficient = 0.0

            # Skewness and kurtosis
            flat_image = image.flatten()
            metrics.skewness = float(scipy_stats.skew(flat_image))
            metrics.kurtosis = float(scipy_stats.kurtosis(flat_image))

            logger.debug(
                f"Statistics: mean={metrics.mean_intensity:.2f}, "
                f"std={metrics.std_deviation:.2f}, "
                f"skew={metrics.skewness:.2f}, kurt={metrics.kurtosis:.2f}"
            )

        except Exception as e:
            logger.error(f"Statistics calculation failed: {str(e)}")
            metrics.issues.append("Failed to calculate statistics")

    def _calculate_snr(
        self, image: np.ndarray, metrics: ImageQualityMetrics
    ) -> None:
        """
        Calculate Signal-to-Noise Ratio

        If reference image is available, uses it for accurate SNR.
        Otherwise estimates noise from image homogeneous regions.
        """
        try:
            if self.reference_image is not None:
                # True SNR with reference
                signal = np.mean(self.reference_image)
                noise = np.std(image - self.reference_image)
            else:
                # Estimate noise using local standard deviation
                # Assumption: noise is present in all areas equally
                signal = metrics.mean_intensity

                # Use Laplacian to estimate noise (MAD - Median Absolute Deviation)
                laplacian = cv2.Laplacian(image, cv2.CV_64F)
                noise = np.median(np.abs(laplacian)) / 0.6745

            if noise > 0:
                metrics.snr_db = 20 * np.log10(signal / noise)
            else:
                metrics.snr_db = float('inf')

            # PSNR (Peak SNR)
            max_pixel = 255 if image.dtype == np.uint8 else 65535
            if noise > 0:
                metrics.psnr_db = 20 * np.log10(max_pixel / noise)
            else:
                metrics.psnr_db = float('inf')

            # Cap infinite values for reporting
            if np.isinf(metrics.snr_db):
                metrics.snr_db = 100.0
            if np.isinf(metrics.psnr_db):
                metrics.psnr_db = 100.0

            logger.debug(f"SNR: {metrics.snr_db:.2f} dB, PSNR: {metrics.psnr_db:.2f} dB")

            # Quality check
            if metrics.snr_db < 20:
                metrics.issues.append(f"Low SNR ({metrics.snr_db:.1f} dB < 20 dB)")

        except Exception as e:
            logger.error(f"SNR calculation failed: {str(e)}")
            metrics.issues.append("Failed to calculate SNR")

    def _calculate_sharpness(
        self, image: np.ndarray, metrics: ImageQualityMetrics
    ) -> None:
        """
        Calculate multiple sharpness metrics

        Implements various sharpness measures for robustness:
        1. Laplacian variance (most common)
        2. Gradient magnitude
        3. FFT-based frequency content
        4. MTF50 (IEC standard)
        5. Edge contrast
        """
        try:
            # 1. Laplacian variance method
            laplacian = cv2.Laplacian(image, cv2.CV_64F)
            metrics.sharpness_laplacian = float(laplacian.var())

            # 2. Gradient magnitude (Sobel)
            sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
            gradient_mag = np.sqrt(sobelx ** 2 + sobely ** 2)
            metrics.sharpness_gradient = float(np.mean(gradient_mag))

            # 3. FFT-based sharpness (high-frequency content)
            metrics.sharpness_fft = self._calculate_fft_sharpness(image)

            # 4. MTF50 (Modulation Transfer Function at 50%)
            if self.enable_advanced_metrics:
                metrics.mtf50 = self._calculate_mtf50(image)
            else:
                # Approximate MTF50 from Laplacian
                metrics.mtf50 = min(1.0, metrics.sharpness_laplacian / 1000.0)

            # 5. Edge contrast
            metrics.edge_contrast = self._calculate_edge_contrast(image)

            logger.debug(
                f"Sharpness: Laplacian={metrics.sharpness_laplacian:.2f}, "
                f"MTF50={metrics.mtf50:.3f}, Edge Contrast={metrics.edge_contrast:.2f}"
            )

            # Quality checks
            if metrics.sharpness_laplacian < 100:
                metrics.issues.append(
                    f"Low sharpness (Laplacian variance {metrics.sharpness_laplacian:.1f} < 100)"
                )

        except Exception as e:
            logger.error(f"Sharpness calculation failed: {str(e)}")
            metrics.issues.append("Failed to calculate sharpness")

    def _calculate_fft_sharpness(self, image: np.ndarray) -> float:
        """
        Calculate sharpness using FFT (high frequency content)

        Args:
            image: Input grayscale image

        Returns:
            Sharpness score (higher = sharper)
        """
        try:
            # Compute FFT
            f_transform = np.fft.fft2(image)
            f_shift = np.fft.fftshift(f_transform)
            magnitude_spectrum = np.abs(f_shift)

            # Calculate total power
            total_power = np.sum(magnitude_spectrum ** 2)

            # Calculate high-frequency power (outer 50% of spectrum)
            h, w = image.shape
            cy, cx = h // 2, w // 2
            radius = min(cy, cx)

            y, x = np.ogrid[:h, :w]
            mask = (x - cx) ** 2 + (y - cy) ** 2 > (radius * 0.5) ** 2

            high_freq_power = np.sum((magnitude_spectrum[mask]) ** 2)

            # Ratio of high-freq to total power
            if total_power > 0:
                sharpness = high_freq_power / total_power
            else:
                sharpness = 0.0

            return float(sharpness)

        except Exception as e:
            logger.error(f"FFT sharpness calculation failed: {str(e)}")
            return 0.0

    def _calculate_mtf50(self, image: np.ndarray) -> float:
        """
        Calculate MTF50 (Modulation Transfer Function at 50%)

        MTF50 is the spatial frequency at which MTF drops to 50%.
        This is the IEC 60904-13 standard sharpness metric.

        Simplified implementation using slanted-edge method.

        Args:
            image: Input grayscale image

        Returns:
            MTF50 value (0-1, higher = sharper)
        """
        try:
            # Find edges using Canny
            edges = cv2.Canny(image, 50, 150)

            # Find edge lines
            lines = cv2.HoughLinesP(
                edges,
                rho=1,
                theta=np.pi / 180,
                threshold=50,
                minLineLength=100,
                maxLineGap=10
            )

            if lines is None or len(lines) == 0:
                # Fallback: estimate from gradient
                sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
                sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
                gradient = np.sqrt(sobelx ** 2 + sobely ** 2)
                mtf_estimate = np.percentile(gradient, 95) / 255.0
                return min(1.0, mtf_estimate)

            # Use strongest edge
            line = lines[0][0]
            x1, y1, x2, y2 = line

            # Extract edge profile perpendicular to edge
            length = int(np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2))
            if length < 20:
                return 0.5  # Default for short edges

            # Sample perpendicular to edge
            angle = np.arctan2(y2 - y1, x2 - x1)
            perp_angle = angle + np.pi / 2

            # Extract profile
            profile_length = 50
            profiles = []

            for i in range(0, length, 5):
                t = i / length
                px = int(x1 + t * (x2 - x1))
                py = int(y1 + t * (y2 - y1))

                profile = []
                for d in range(-profile_length // 2, profile_length // 2):
                    sx = int(px + d * np.cos(perp_angle))
                    sy = int(py + d * np.sin(perp_angle))

                    if 0 <= sx < image.shape[1] and 0 <= sy < image.shape[0]:
                        profile.append(image[sy, sx])

                if len(profile) == profile_length:
                    profiles.append(profile)

            if not profiles:
                return 0.5

            # Average profile
            avg_profile = np.mean(profiles, axis=0)

            # Calculate edge spread function (ESF)
            esf = avg_profile

            # Differentiate to get line spread function (LSF)
            lsf = np.diff(esf)

            # FFT of LSF gives MTF
            mtf = np.abs(np.fft.fft(lsf))
            mtf = mtf / mtf[0]  # Normalize

            # Find frequency where MTF = 0.5
            mtf_half = len(mtf) // 2
            mtf = mtf[:mtf_half]

            try:
                idx_50 = np.where(mtf <= 0.5)[0][0]
                mtf50 = idx_50 / len(mtf)
            except IndexError:
                # MTF never reaches 0.5 (very sharp)
                mtf50 = 1.0

            return float(mtf50)

        except Exception as e:
            logger.error(f"MTF50 calculation failed: {str(e)}")
            # Fallback to gradient-based estimate
            try:
                sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0)
                sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1)
                gradient = np.sqrt(sobelx ** 2 + sobely ** 2)
                return min(1.0, float(np.mean(gradient)) / 100.0)
            except:
                return 0.5

    def _calculate_edge_contrast(self, image: np.ndarray) -> float:
        """
        Calculate edge contrast using Michelson contrast

        Args:
            image: Input grayscale image

        Returns:
            Edge contrast (0-1)
        """
        try:
            # Detect edges
            edges = cv2.Canny(image, 50, 150)

            # Dilate to get edge regions
            kernel = np.ones((5, 5), np.uint8)
            edge_regions = cv2.dilate(edges, kernel, iterations=1)

            # Calculate contrast at edges
            edge_pixels = image[edge_regions > 0]

            if len(edge_pixels) < 10:
                return 0.0

            # Michelson contrast: (max - min) / (max + min)
            max_val = np.percentile(edge_pixels, 95)
            min_val = np.percentile(edge_pixels, 5)

            if max_val + min_val > 0:
                contrast = (max_val - min_val) / (max_val + min_val)
            else:
                contrast = 0.0

            return float(contrast)

        except Exception as e:
            logger.error(f"Edge contrast calculation failed: {str(e)}")
            return 0.0

    def _calculate_entropy(
        self, image: np.ndarray, metrics: ImageQualityMetrics
    ) -> None:
        """
        Calculate Shannon entropy and information content

        Args:
            image: Input grayscale image
            metrics: Metrics object to update
        """
        try:
            # Calculate histogram
            hist, _ = np.histogram(image.flatten(), bins=256, range=(0, 256))

            # Normalize to get probability distribution
            hist = hist / hist.sum()

            # Remove zeros to avoid log(0)
            hist = hist[hist > 0]

            # Shannon entropy: -sum(p * log2(p))
            metrics.shannon_entropy = float(-np.sum(hist * np.log2(hist)))

            # Normalized entropy (0-1)
            max_entropy = 8.0  # log2(256) for 8-bit images
            metrics.normalized_entropy = metrics.shannon_entropy / max_entropy

            logger.debug(
                f"Entropy: {metrics.shannon_entropy:.3f} bits "
                f"(normalized: {metrics.normalized_entropy:.3f})"
            )

            # Quality check
            if metrics.normalized_entropy < 0.3:
                metrics.issues.append(
                    f"Low entropy ({metrics.normalized_entropy:.2f} < 0.3) - "
                    "image may be overexposed or underexposed"
                )

        except Exception as e:
            logger.error(f"Entropy calculation failed: {str(e)}")
            metrics.issues.append("Failed to calculate entropy")

    def _calculate_histogram_metrics(
        self, image: np.ndarray, metrics: ImageQualityMetrics
    ) -> None:
        """Calculate histogram-based metrics"""
        try:
            # Calculate histogram
            hist, bins = np.histogram(image.flatten(), bins=256, range=(0, 256))

            # Dynamic range
            non_zero = np.where(hist > 0)[0]
            if len(non_zero) > 0:
                metrics.dynamic_range = int(non_zero[-1] - non_zero[0])
            else:
                metrics.dynamic_range = 0

            # Histogram uniformity (chi-square test)
            expected = np.full_like(hist, fill_value=hist.sum() / 256, dtype=float)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                chi_square = np.sum((hist - expected) ** 2 / (expected + 1e-10))
            metrics.histogram_uniformity = 1.0 / (1.0 + chi_square / 100000)

            # Contrast ratio (95th percentile / 5th percentile)
            p95 = np.percentile(image, 95)
            p5 = np.percentile(image, 5)
            if p5 > 0:
                metrics.contrast_ratio = float(p95 / p5)
            else:
                metrics.contrast_ratio = float('inf')
            if np.isinf(metrics.contrast_ratio):
                metrics.contrast_ratio = 100.0

            logger.debug(
                f"Histogram: dynamic_range={metrics.dynamic_range}, "
                f"contrast_ratio={metrics.contrast_ratio:.2f}"
            )

            # Quality checks
            if metrics.dynamic_range < self.IEC_MIN_CONTRAST:
                metrics.issues.append(
                    f"Low dynamic range ({metrics.dynamic_range} < {self.IEC_MIN_CONTRAST})"
                )

        except Exception as e:
            logger.error(f"Histogram metrics calculation failed: {str(e)}")
            metrics.issues.append("Failed to calculate histogram metrics")

    def _assess_iec_compliance(self, metrics: ImageQualityMetrics) -> None:
        """
        Assess IEC 60904-13 compliance

        Requirements:
        - Minimum resolution: 640 pixels (smallest dimension)
        - Minimum contrast/dynamic range
        - Sharpness class determination
        """
        # Resolution check
        min_dimension = min(metrics.width, metrics.height)
        metrics.iec_min_resolution_met = min_dimension >= self.IEC_MIN_RESOLUTION

        # Contrast check
        metrics.iec_min_contrast_met = metrics.dynamic_range >= self.IEC_MIN_CONTRAST

        # Sharpness class (based on MTF50)
        if metrics.mtf50 > 0.6:
            metrics.iec_sharpness_class = IECSharpnessClass.CLASS_A.value
        elif metrics.mtf50 > 0.4:
            metrics.iec_sharpness_class = IECSharpnessClass.CLASS_B.value
        elif metrics.mtf50 > 0.2:
            metrics.iec_sharpness_class = IECSharpnessClass.CLASS_C.value
        else:
            metrics.iec_sharpness_class = IECSharpnessClass.CLASS_D.value

        # Overall compliance
        metrics.iec_compliant = (
            metrics.iec_min_resolution_met
            and metrics.iec_min_contrast_met
            and metrics.mtf50 >= self.IEC_MIN_SHARPNESS
        )

        logger.info(
            f"IEC 60904-13 Compliance: {metrics.iec_compliant}, "
            f"Sharpness Class: {metrics.iec_sharpness_class}"
        )

        if not metrics.iec_compliant:
            if not metrics.iec_min_resolution_met:
                metrics.issues.append(
                    f"IEC resolution requirement not met "
                    f"({min_dimension} < {self.IEC_MIN_RESOLUTION})"
                )
            if not metrics.iec_min_contrast_met:
                metrics.issues.append("IEC contrast requirement not met")
            if metrics.mtf50 < self.IEC_MIN_SHARPNESS:
                metrics.issues.append(
                    f"IEC sharpness requirement not met (MTF50 {metrics.mtf50:.2f} < 0.2)"
                )

    def _calculate_quality_score(self, metrics: ImageQualityMetrics) -> None:
        """
        Calculate overall quality score (0-100)

        Weighted combination of multiple metrics
        """
        try:
            # Weights for different aspects
            weights = {
                'sharpness': 0.30,
                'snr': 0.25,
                'contrast': 0.20,
                'entropy': 0.15,
                'iec_compliance': 0.10
            }

            # Normalize individual scores to 0-100
            # Sharpness score (based on MTF50)
            sharpness_score = min(100, metrics.mtf50 * 150)

            # SNR score (30 dB = 100, 10 dB = 0)
            snr_score = max(0, min(100, (metrics.snr_db - 10) * 5))

            # Contrast score (dynamic range)
            contrast_score = min(100, metrics.dynamic_range / 2.5)

            # Entropy score
            entropy_score = metrics.normalized_entropy * 100

            # IEC compliance score
            iec_score = 100 if metrics.iec_compliant else 0

            # Weighted average
            metrics.quality_score = (
                weights['sharpness'] * sharpness_score
                + weights['snr'] * snr_score
                + weights['contrast'] * contrast_score
                + weights['entropy'] * entropy_score
                + weights['iec_compliance'] * iec_score
            )

            # Determine quality level
            if metrics.quality_score >= 80:
                metrics.quality_level = QualityLevel.EXCELLENT.value
            elif metrics.quality_score >= 60:
                metrics.quality_level = QualityLevel.GOOD.value
            elif metrics.quality_score >= 40:
                metrics.quality_level = QualityLevel.ACCEPTABLE.value
            elif metrics.quality_score >= 20:
                metrics.quality_level = QualityLevel.POOR.value
            else:
                metrics.quality_level = QualityLevel.UNUSABLE.value

            logger.info(
                f"Quality Score: {metrics.quality_score:.1f}/100 ({metrics.quality_level})"
            )

        except Exception as e:
            logger.error(f"Quality score calculation failed: {str(e)}")
            metrics.quality_score = 0.0
            metrics.quality_level = QualityLevel.UNUSABLE.value

    def compare_images(
        self, image1: np.ndarray, image2: np.ndarray
    ) -> Dict[str, float]:
        """
        Compare two images and return difference metrics

        Args:
            image1: First image
            image2: Second image

        Returns:
            Dictionary with comparison metrics
        """
        # Convert to grayscale if needed
        if len(image1.shape) == 3:
            image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        if len(image2.shape) == 3:
            image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

        # Resize if dimensions don't match
        if image1.shape != image2.shape:
            image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]))

        comparison = {}

        try:
            # Mean Squared Error
            mse = np.mean((image1.astype(float) - image2.astype(float)) ** 2)
            comparison['mse'] = float(mse)

            # Peak Signal-to-Noise Ratio
            if mse > 0:
                comparison['psnr'] = 20 * np.log10(255.0 / np.sqrt(mse))
            else:
                comparison['psnr'] = float('inf')

            # Structural Similarity Index (SSIM) - simplified version
            mean1, mean2 = np.mean(image1), np.mean(image2)
            var1, var2 = np.var(image1), np.var(image2)
            covar = np.mean((image1 - mean1) * (image2 - mean2))

            c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
            ssim = ((2 * mean1 * mean2 + c1) * (2 * covar + c2)) / \
                   ((mean1 ** 2 + mean2 ** 2 + c1) * (var1 + var2 + c2))
            comparison['ssim'] = float(ssim)

            # Normalized Cross-Correlation
            image1_norm = (image1 - np.mean(image1)) / (np.std(image1) + 1e-10)
            image2_norm = (image2 - np.mean(image2)) / (np.std(image2) + 1e-10)
            ncc = np.mean(image1_norm * image2_norm)
            comparison['ncc'] = float(ncc)

        except Exception as e:
            logger.error(f"Image comparison failed: {str(e)}")

        return comparison


# Convenience functions
def analyze_el_image(image: np.ndarray) -> ImageQualityMetrics:
    """
    Quick analysis function

    Args:
        image: Input image

    Returns:
        ImageQualityMetrics
    """
    engine = ImageAnalyticsEngine()
    return engine.analyze(image)


def batch_analyze(images: List[np.ndarray]) -> List[ImageQualityMetrics]:
    """
    Analyze multiple images

    Args:
        images: List of input images

    Returns:
        List of ImageQualityMetrics
    """
    engine = ImageAnalyticsEngine()
    return [engine.analyze(img) for img in images]


if __name__ == "__main__":
    # Example usage
    import sys

    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            print(f"Error: Could not load image from {image_path}")
            sys.exit(1)

        # Analyze
        engine = ImageAnalyticsEngine(enable_advanced_metrics=True)
        metrics = engine.analyze(image)

        # Print summary
        print(metrics.get_summary())

        # Save detailed report
        import json
        report_path = image_path.replace('.', '_quality_report.')
        report_path = report_path.rsplit('.', 1)[0] + '.json'

        with open(report_path, 'w') as f:
            json.dump(metrics.to_dict(), f, indent=2)

        print(f"\nDetailed report saved to: {report_path}")
    else:
        print("Usage: python analytics_engine.py <image_path>")
