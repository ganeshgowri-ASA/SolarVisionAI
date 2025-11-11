"""
SolarVisionAI - Image Quality Validator
Standards: IEC 60904-13, IEC 60904-14, ISO 17025

Automated quality filtering and validation for EL images with:
- Resolution compliance checking
- Exposure validation (over/underexposure detection)
- Blur detection
- Perspective/angle validation
- Metadata completeness verification
- User-friendly rejection messages

Author: SolarVisionAI Team
Version: 1.0.0
"""

import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
from datetime import datetime

from analytics_engine import ImageAnalyticsEngine, ImageQualityMetrics

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ValidationStatus(Enum):
    """Validation result status"""
    PASSED = "passed"
    WARNING = "warning"
    FAILED = "failed"


class RejectionReason(Enum):
    """Standardized rejection reasons"""
    # Resolution issues
    RESOLUTION_TOO_LOW = "Resolution below minimum requirement"
    ASPECT_RATIO_INVALID = "Invalid aspect ratio for PV module"

    # Exposure issues
    OVEREXPOSED = "Image overexposed (clipping in highlights)"
    UNDEREXPOSED = "Image underexposed (insufficient detail in shadows)"
    LOW_DYNAMIC_RANGE = "Insufficient dynamic range"

    # Focus/sharpness issues
    SEVERE_BLUR = "Image severely out of focus or motion blur"
    LOW_SHARPNESS = "Sharpness below acceptable threshold"

    # Geometric issues
    EXCESSIVE_PERSPECTIVE = "Excessive perspective distortion"
    INVALID_ORIENTATION = "Invalid module orientation"

    # Technical issues
    CORRUPTED_FILE = "File corrupted or unreadable"
    WRONG_FORMAT = "Unsupported image format"
    EXCESSIVE_NOISE = "Excessive noise (low SNR)"

    # Metadata issues
    MISSING_METADATA = "Required metadata missing"
    INVALID_METADATA = "Metadata validation failed"

    # Other
    MULTIPLE_ISSUES = "Multiple quality issues detected"
    UNKNOWN = "Unknown validation error"


@dataclass
class ValidationRule:
    """Individual validation rule configuration"""
    name: str
    enabled: bool = True
    severity: str = "error"  # "error", "warning", "info"
    auto_reject: bool = True  # Automatically reject if failed
    threshold: Optional[float] = None
    message: str = ""


@dataclass
class ValidationResult:
    """Result of image validation"""
    status: str = ValidationStatus.PASSED.value
    is_valid: bool = True
    quality_score: float = 0.0

    # Detailed checks
    resolution_check: bool = True
    exposure_check: bool = True
    blur_check: bool = True
    perspective_check: bool = True
    metadata_check: bool = True
    noise_check: bool = True

    # Issues and warnings
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    info: List[str] = field(default_factory=list)

    # Rejection info
    rejection_reason: Optional[str] = None
    rejection_details: str = ""

    # Recommendations
    recommendations: List[str] = field(default_factory=list)

    # Metrics
    metrics: Optional[ImageQualityMetrics] = None

    # Metadata
    validation_timestamp: datetime = field(default_factory=datetime.now)
    validator_version: str = "1.0.0"

    def get_user_message(self) -> str:
        """Get user-friendly validation message"""
        if self.is_valid:
            return f"✓ Image PASSED validation (Quality Score: {self.quality_score:.1f}/100)"
        else:
            msg = f"✗ Image REJECTED\n"
            msg += f"Reason: {self.rejection_reason}\n"
            if self.rejection_details:
                msg += f"Details: {self.rejection_details}\n"

            if self.errors:
                msg += f"\nErrors ({len(self.errors)}):\n"
                for error in self.errors:
                    msg += f"  • {error}\n"

            if self.warnings:
                msg += f"\nWarnings ({len(self.warnings)}):\n"
                for warning in self.warnings:
                    msg += f"  • {warning}\n"

            if self.recommendations:
                msg += f"\nRecommendations:\n"
                for rec in self.recommendations:
                    msg += f"  → {rec}\n"

            return msg

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'status': self.status,
            'is_valid': self.is_valid,
            'quality_score': self.quality_score,
            'resolution_check': self.resolution_check,
            'exposure_check': self.exposure_check,
            'blur_check': self.blur_check,
            'perspective_check': self.perspective_check,
            'metadata_check': self.metadata_check,
            'noise_check': self.noise_check,
            'errors': self.errors,
            'warnings': self.warnings,
            'info': self.info,
            'rejection_reason': self.rejection_reason,
            'rejection_details': self.rejection_details,
            'recommendations': self.recommendations,
            'validation_timestamp': self.validation_timestamp.isoformat(),
            'validator_version': self.validator_version
        }


@dataclass
class ValidationConfig:
    """Configuration for quality validation"""
    # Resolution requirements (IEC 60904-13)
    min_width: int = 640
    min_height: int = 640
    max_width: int = 10000
    max_height: int = 10000
    min_megapixels: float = 0.4  # 640x640
    max_megapixels: float = 100.0

    # Aspect ratio (typical PV modules are 1.5:1 to 2:1)
    min_aspect_ratio: float = 1.0
    max_aspect_ratio: float = 3.0

    # Exposure thresholds
    overexposure_threshold: float = 0.05  # Max 5% of pixels can be saturated
    underexposure_threshold: float = 0.05  # Max 5% of pixels can be completely dark
    min_mean_intensity: float = 30.0
    max_mean_intensity: float = 225.0

    # Dynamic range
    min_dynamic_range: int = 20  # IEC requirement

    # Sharpness thresholds
    min_laplacian_variance: float = 50.0  # Below this = severe blur
    min_mtf50: float = 0.15  # Minimum acceptable sharpness
    warning_mtf50: float = 0.3  # Below this = warning

    # Noise thresholds
    min_snr_db: float = 15.0  # Minimum SNR
    warning_snr_db: float = 25.0  # Below this = warning

    # Perspective distortion (maximum angle in degrees)
    max_perspective_angle: float = 50.0

    # Metadata requirements
    require_timestamp: bool = False
    require_camera_info: bool = False
    require_test_conditions: bool = False

    # Overall quality
    min_quality_score: float = 30.0  # Minimum composite quality score
    warning_quality_score: float = 50.0

    # Validation behavior
    strict_mode: bool = False  # Fail on warnings
    enable_recommendations: bool = True


class ImageQualityValidator:
    """
    Automated image quality validator for EL images

    Performs comprehensive validation against IEC standards and best practices.
    Provides detailed rejection reasons and recommendations for improvement.
    """

    def __init__(self, config: Optional[ValidationConfig] = None):
        """
        Initialize validator

        Args:
            config: Validation configuration. Uses defaults if None.
        """
        self.config = config or ValidationConfig()
        self.analytics_engine = ImageAnalyticsEngine(enable_advanced_metrics=True)
        logger.info("Initialized ImageQualityValidator")

    def validate(
        self,
        image: np.ndarray,
        metadata: Optional[Dict] = None
    ) -> ValidationResult:
        """
        Perform comprehensive validation

        Args:
            image: Input image (grayscale or BGR)
            metadata: Optional image metadata dictionary

        Returns:
            ValidationResult with detailed analysis
        """
        result = ValidationResult()

        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray_image = image.copy()

            # Get quality metrics
            logger.info("Analyzing image quality metrics...")
            result.metrics = self.analytics_engine.analyze(gray_image)
            result.quality_score = result.metrics.quality_score

            # Run validation checks
            self._validate_resolution(gray_image, result)
            self._validate_exposure(gray_image, result)
            self._validate_blur(gray_image, result)
            self._validate_perspective(gray_image, result)
            self._validate_noise(gray_image, result)

            if metadata is not None:
                self._validate_metadata(metadata, result)

            # Overall quality check
            self._validate_overall_quality(result)

            # Determine final status
            self._determine_final_status(result)

            # Generate recommendations
            if self.config.enable_recommendations:
                self._generate_recommendations(result)

            logger.info(
                f"Validation complete: Status={result.status}, "
                f"Valid={result.is_valid}, Score={result.quality_score:.1f}"
            )

        except Exception as e:
            logger.error(f"Validation error: {str(e)}", exc_info=True)
            result.is_valid = False
            result.status = ValidationStatus.FAILED.value
            result.rejection_reason = RejectionReason.UNKNOWN.value
            result.rejection_details = str(e)
            result.errors.append(f"Validation failed: {str(e)}")

        return result

    def _validate_resolution(
        self, image: np.ndarray, result: ValidationResult
    ) -> None:
        """Validate image resolution"""
        try:
            h, w = image.shape[:2]
            megapixels = (w * h) / 1_000_000

            # Check minimum dimensions
            if w < self.config.min_width or h < self.config.min_height:
                result.resolution_check = False
                result.errors.append(
                    f"Resolution {w}x{h} below minimum {self.config.min_width}x"
                    f"{self.config.min_height} (IEC 60904-13 requirement)"
                )
                if result.rejection_reason is None:
                    result.rejection_reason = RejectionReason.RESOLUTION_TOO_LOW.value
                    result.rejection_details = (
                        f"Minimum resolution: {self.config.min_width}x"
                        f"{self.config.min_height}, Got: {w}x{h}"
                    )

            # Check maximum dimensions
            if w > self.config.max_width or h > self.config.max_height:
                result.resolution_check = False
                result.warnings.append(
                    f"Resolution {w}x{h} exceeds recommended maximum "
                    f"{self.config.max_width}x{self.config.max_height}"
                )

            # Check megapixels
            if megapixels < self.config.min_megapixels:
                result.resolution_check = False
                result.errors.append(
                    f"Image size {megapixels:.2f}MP below minimum "
                    f"{self.config.min_megapixels}MP"
                )

            # Check aspect ratio
            aspect_ratio = max(w, h) / min(w, h)
            if aspect_ratio < self.config.min_aspect_ratio or \
               aspect_ratio > self.config.max_aspect_ratio:
                result.warnings.append(
                    f"Unusual aspect ratio {aspect_ratio:.2f}. "
                    f"Expected range: {self.config.min_aspect_ratio:.1f}-"
                    f"{self.config.max_aspect_ratio:.1f}"
                )

            if result.resolution_check:
                result.info.append(
                    f"✓ Resolution {w}x{h} ({megapixels:.2f}MP) meets requirements"
                )
            else:
                logger.warning(f"Resolution check failed: {w}x{h}")

        except Exception as e:
            logger.error(f"Resolution validation failed: {str(e)}")
            result.errors.append(f"Resolution check error: {str(e)}")

    def _validate_exposure(
        self, image: np.ndarray, result: ValidationResult
    ) -> None:
        """Validate image exposure"""
        try:
            # Check for overexposure (clipped highlights)
            saturated_pixels = np.sum(image >= 250)
            total_pixels = image.size
            overexposure_ratio = saturated_pixels / total_pixels

            if overexposure_ratio > self.config.overexposure_threshold:
                result.exposure_check = False
                result.errors.append(
                    f"Image overexposed: {overexposure_ratio * 100:.1f}% of pixels "
                    f"saturated (threshold: {self.config.overexposure_threshold * 100:.1f}%)"
                )
                if result.rejection_reason is None:
                    result.rejection_reason = RejectionReason.OVEREXPOSED.value
                    result.rejection_details = (
                        f"Reduce exposure time or current. "
                        f"{overexposure_ratio * 100:.1f}% pixels clipped"
                    )

            # Check for underexposure
            dark_pixels = np.sum(image <= 5)
            underexposure_ratio = dark_pixels / total_pixels

            if underexposure_ratio > self.config.underexposure_threshold:
                result.exposure_check = False
                result.errors.append(
                    f"Image underexposed: {underexposure_ratio * 100:.1f}% of pixels "
                    f"nearly black (threshold: {self.config.underexposure_threshold * 100:.1f}%)"
                )
                if result.rejection_reason is None:
                    result.rejection_reason = RejectionReason.UNDEREXPOSED.value
                    result.rejection_details = (
                        f"Increase exposure time or current. "
                        f"{underexposure_ratio * 100:.1f}% pixels too dark"
                    )

            # Check mean intensity
            mean_intensity = result.metrics.mean_intensity
            if mean_intensity < self.config.min_mean_intensity:
                result.exposure_check = False
                result.errors.append(
                    f"Mean intensity {mean_intensity:.1f} below minimum "
                    f"{self.config.min_mean_intensity}"
                )
            elif mean_intensity > self.config.max_mean_intensity:
                result.exposure_check = False
                result.errors.append(
                    f"Mean intensity {mean_intensity:.1f} above maximum "
                    f"{self.config.max_mean_intensity}"
                )

            # Check dynamic range
            if result.metrics.dynamic_range < self.config.min_dynamic_range:
                result.exposure_check = False
                result.errors.append(
                    f"Dynamic range {result.metrics.dynamic_range} below IEC minimum "
                    f"{self.config.min_dynamic_range}"
                )
                if result.rejection_reason is None:
                    result.rejection_reason = RejectionReason.LOW_DYNAMIC_RANGE.value
                    result.rejection_details = (
                        f"Adjust camera settings to increase contrast. "
                        f"Got {result.metrics.dynamic_range}, need {self.config.min_dynamic_range}"
                    )

            if result.exposure_check:
                result.info.append(
                    f"✓ Exposure within acceptable range "
                    f"(mean: {mean_intensity:.1f}, dynamic range: {result.metrics.dynamic_range})"
                )
            else:
                logger.warning("Exposure check failed")

        except Exception as e:
            logger.error(f"Exposure validation failed: {str(e)}")
            result.errors.append(f"Exposure check error: {str(e)}")

    def _validate_blur(
        self, image: np.ndarray, result: ValidationResult
    ) -> None:
        """Validate image sharpness (detect blur)"""
        try:
            laplacian_var = result.metrics.sharpness_laplacian
            mtf50 = result.metrics.mtf50

            # Severe blur check (Laplacian variance)
            if laplacian_var < self.config.min_laplacian_variance:
                result.blur_check = False
                result.errors.append(
                    f"Severe blur detected: Laplacian variance {laplacian_var:.1f} "
                    f"below threshold {self.config.min_laplacian_variance}"
                )
                if result.rejection_reason is None:
                    result.rejection_reason = RejectionReason.SEVERE_BLUR.value
                    result.rejection_details = (
                        f"Image out of focus or motion blur. "
                        f"Sharpness score: {laplacian_var:.1f} (need >{self.config.min_laplacian_variance})"
                    )

            # MTF50 check
            if mtf50 < self.config.min_mtf50:
                result.blur_check = False
                result.errors.append(
                    f"MTF50 {mtf50:.3f} below minimum {self.config.min_mtf50}"
                )
                if result.rejection_reason is None:
                    result.rejection_reason = RejectionReason.LOW_SHARPNESS.value
            elif mtf50 < self.config.warning_mtf50:
                result.warnings.append(
                    f"MTF50 {mtf50:.3f} below recommended {self.config.warning_mtf50}"
                )

            # IEC sharpness class check
            if result.metrics.iec_sharpness_class == 'D':
                result.blur_check = False
                result.errors.append(
                    "IEC 60904-13 Sharpness Class D (poor quality)"
                )
            elif result.metrics.iec_sharpness_class == 'C':
                result.warnings.append(
                    "IEC 60904-13 Sharpness Class C (acceptable but not optimal)"
                )

            if result.blur_check:
                result.info.append(
                    f"✓ Sharpness acceptable "
                    f"(IEC Class: {result.metrics.iec_sharpness_class}, MTF50: {mtf50:.3f})"
                )
            else:
                logger.warning(f"Blur check failed: Laplacian={laplacian_var:.1f}, MTF50={mtf50:.3f}")

        except Exception as e:
            logger.error(f"Blur validation failed: {str(e)}")
            result.errors.append(f"Blur check error: {str(e)}")

    def _validate_perspective(
        self, image: np.ndarray, result: ValidationResult
    ) -> None:
        """Validate perspective distortion"""
        try:
            # Detect edges
            edges = cv2.Canny(image, 50, 150)

            # Detect lines
            lines = cv2.HoughLinesP(
                edges,
                rho=1,
                theta=np.pi / 180,
                threshold=50,
                minLineLength=100,
                maxLineGap=10
            )

            if lines is not None and len(lines) > 0:
                # Calculate angles of detected lines
                angles = []
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
                    angles.append(angle)

                # Check if lines are reasonably vertical/horizontal
                # Lines should be close to 0° or 90°
                deviations = []
                for angle in angles:
                    # Find deviation from nearest 90° increment
                    nearest_90 = round(angle / 90) * 90
                    deviation = abs(angle - nearest_90)
                    deviations.append(deviation)

                max_deviation = max(deviations) if deviations else 0

                if max_deviation > self.config.max_perspective_angle:
                    result.perspective_check = False
                    result.errors.append(
                        f"Excessive perspective distortion: {max_deviation:.1f}° "
                        f"(max allowed: {self.config.max_perspective_angle}°)"
                    )
                    if result.rejection_reason is None:
                        result.rejection_reason = RejectionReason.EXCESSIVE_PERSPECTIVE.value
                        result.rejection_details = (
                            f"Camera not perpendicular to module. "
                            f"Angle deviation: {max_deviation:.1f}° (max: {self.config.max_perspective_angle}°)"
                        )
                elif max_deviation > 30:
                    result.warnings.append(
                        f"Moderate perspective distortion detected: {max_deviation:.1f}°"
                    )
                else:
                    result.info.append(
                        f"✓ Perspective acceptable (max deviation: {max_deviation:.1f}°)"
                    )
            else:
                # No lines detected - may indicate very blurry or uniform image
                result.warnings.append("Could not detect module edges for perspective check")

        except Exception as e:
            logger.error(f"Perspective validation failed: {str(e)}")
            result.warnings.append(f"Perspective check could not be completed: {str(e)}")

    def _validate_noise(
        self, image: np.ndarray, result: ValidationResult
    ) -> None:
        """Validate noise levels"""
        try:
            snr_db = result.metrics.snr_db

            if snr_db < self.config.min_snr_db:
                result.noise_check = False
                result.errors.append(
                    f"Excessive noise: SNR {snr_db:.1f} dB below minimum "
                    f"{self.config.min_snr_db} dB"
                )
                if result.rejection_reason is None:
                    result.rejection_reason = RejectionReason.EXCESSIVE_NOISE.value
                    result.rejection_details = (
                        f"High noise level (SNR: {snr_db:.1f} dB). "
                        f"Use lower ISO or longer exposure"
                    )
            elif snr_db < self.config.warning_snr_db:
                result.warnings.append(
                    f"SNR {snr_db:.1f} dB below recommended {self.config.warning_snr_db} dB"
                )
            else:
                result.info.append(f"✓ Noise level acceptable (SNR: {snr_db:.1f} dB)")

        except Exception as e:
            logger.error(f"Noise validation failed: {str(e)}")
            result.errors.append(f"Noise check error: {str(e)}")

    def _validate_metadata(
        self, metadata: Dict, result: ValidationResult
    ) -> None:
        """Validate image metadata completeness"""
        try:
            missing_fields = []

            if self.config.require_timestamp:
                if 'timestamp' not in metadata and 'capture_time' not in metadata:
                    missing_fields.append('timestamp')

            if self.config.require_camera_info:
                required_camera_fields = ['camera_make', 'camera_model', 'exposure_time', 'iso']
                for field in required_camera_fields:
                    if field not in metadata or not metadata[field]:
                        missing_fields.append(field)

            if self.config.require_test_conditions:
                required_test_fields = ['test_current', 'module_temp', 'ambient_temp']
                for field in required_test_fields:
                    if field not in metadata or metadata[field] is None:
                        missing_fields.append(field)

            if missing_fields:
                result.metadata_check = False
                result.warnings.append(
                    f"Missing metadata fields: {', '.join(missing_fields)}"
                )
                if self.config.strict_mode:
                    result.errors.append(
                        f"Required metadata missing: {', '.join(missing_fields)}"
                    )
            else:
                result.info.append("✓ Metadata complete")

        except Exception as e:
            logger.error(f"Metadata validation failed: {str(e)}")
            result.warnings.append(f"Metadata check error: {str(e)}")

    def _validate_overall_quality(self, result: ValidationResult) -> None:
        """Validate overall quality score"""
        try:
            if result.quality_score < self.config.min_quality_score:
                result.errors.append(
                    f"Overall quality score {result.quality_score:.1f} below minimum "
                    f"{self.config.min_quality_score}"
                )
                if result.rejection_reason is None:
                    result.rejection_reason = RejectionReason.MULTIPLE_ISSUES.value
                    result.rejection_details = (
                        f"Multiple quality issues detected. Quality score: "
                        f"{result.quality_score:.1f} (need >{self.config.min_quality_score})"
                    )
            elif result.quality_score < self.config.warning_quality_score:
                result.warnings.append(
                    f"Quality score {result.quality_score:.1f} below recommended "
                    f"{self.config.warning_quality_score}"
                )

        except Exception as e:
            logger.error(f"Overall quality validation failed: {str(e)}")

    def _determine_final_status(self, result: ValidationResult) -> None:
        """Determine final validation status"""
        # Check if any critical errors
        has_errors = len(result.errors) > 0
        has_warnings = len(result.warnings) > 0

        if has_errors:
            result.is_valid = False
            result.status = ValidationStatus.FAILED.value
        elif has_warnings and self.config.strict_mode:
            result.is_valid = False
            result.status = ValidationStatus.FAILED.value
            result.rejection_reason = "Failed in strict mode due to warnings"
        elif has_warnings:
            result.is_valid = True
            result.status = ValidationStatus.WARNING.value
        else:
            result.is_valid = True
            result.status = ValidationStatus.PASSED.value

    def _generate_recommendations(self, result: ValidationResult) -> None:
        """Generate improvement recommendations"""
        try:
            # Resolution recommendations
            if not result.resolution_check:
                result.recommendations.append(
                    "Use a higher resolution camera or move closer to the module"
                )

            # Exposure recommendations
            if not result.exposure_check:
                if "overexposed" in ' '.join(result.errors).lower():
                    result.recommendations.append(
                        "Reduce exposure time, lower ISO, or decrease test current"
                    )
                elif "underexposed" in ' '.join(result.errors).lower():
                    result.recommendations.append(
                        "Increase exposure time, raise ISO, or increase test current"
                    )
                if "dynamic range" in ' '.join(result.errors).lower():
                    result.recommendations.append(
                        "Adjust camera settings to maximize dynamic range (avoid auto-exposure)"
                    )

            # Blur recommendations
            if not result.blur_check:
                result.recommendations.append(
                    "Ensure proper focus, use tripod to avoid motion blur, check lens quality"
                )
                result.recommendations.append(
                    "Consider using auto-focus or manual focus with live view"
                )

            # Perspective recommendations
            if not result.perspective_check:
                result.recommendations.append(
                    "Position camera perpendicular to module surface"
                )
                result.recommendations.append(
                    "Use a leveling tool or tripod with bubble level"
                )

            # Noise recommendations
            if not result.noise_check:
                result.recommendations.append(
                    "Use lower ISO setting (100-400 recommended for EL imaging)"
                )
                result.recommendations.append(
                    "Increase exposure time instead of raising ISO"
                )
                result.recommendations.append(
                    "Ensure camera sensor is not overheating during long exposures"
                )

            # IEC compliance recommendations
            if result.metrics and not result.metrics.iec_compliant:
                result.recommendations.append(
                    "Review IEC 60904-13 standard for EL imaging best practices"
                )

        except Exception as e:
            logger.error(f"Recommendation generation failed: {str(e)}")

    def validate_batch(
        self,
        images: List[np.ndarray],
        metadata_list: Optional[List[Dict]] = None
    ) -> List[ValidationResult]:
        """
        Validate multiple images

        Args:
            images: List of images
            metadata_list: Optional list of metadata dictionaries

        Returns:
            List of ValidationResult objects
        """
        results = []

        for i, image in enumerate(images):
            logger.info(f"Validating image {i + 1}/{len(images)}")

            metadata = None
            if metadata_list and i < len(metadata_list):
                metadata = metadata_list[i]

            result = self.validate(image, metadata)
            results.append(result)

        # Summary statistics
        passed = sum(1 for r in results if r.is_valid)
        failed = len(results) - passed

        logger.info(
            f"Batch validation complete: {passed} passed, {failed} failed "
            f"out of {len(results)} images"
        )

        return results


# Convenience functions
def quick_validate(
    image: np.ndarray,
    strict: bool = False
) -> bool:
    """
    Quick validation (returns True/False only)

    Args:
        image: Input image
        strict: Enable strict mode

    Returns:
        True if valid, False otherwise
    """
    config = ValidationConfig(strict_mode=strict)
    validator = ImageQualityValidator(config)
    result = validator.validate(image)
    return result.is_valid


def validate_with_report(
    image: np.ndarray,
    metadata: Optional[Dict] = None
) -> Tuple[bool, str]:
    """
    Validate and get user-friendly report

    Args:
        image: Input image
        metadata: Optional metadata

    Returns:
        Tuple of (is_valid, message)
    """
    validator = ImageQualityValidator()
    result = validator.validate(image, metadata)
    return result.is_valid, result.get_user_message()


if __name__ == "__main__":
    # Example usage
    import sys

    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            print(f"Error: Could not load image from {image_path}")
            sys.exit(1)

        # Validate
        validator = ImageQualityValidator()
        result = validator.validate(image)

        # Print result
        print(result.get_user_message())

        # Save detailed report
        import json
        report_path = image_path.replace('.', '_validation.')
        report_path = report_path.rsplit('.', 1)[0] + '.json'

        with open(report_path, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)

        print(f"\nDetailed report saved to: {report_path}")

        # Exit with appropriate code
        sys.exit(0 if result.is_valid else 1)
    else:
        print("Usage: python quality_validator.py <image_path>")
