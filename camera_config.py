"""
SolarVisionAI - Camera Configuration and Calibration Module
Standards: IEC 60904-13, IEC 60904-14

Camera calibration, metadata capture, and test conditions management for EL imaging:
- Current recipes (Isc, 0.1*Isc, Impp)
- Exposure time, ISO, aperture settings
- Focal length, resolution tracking
- Timestamp and environmental data
- Emissivity adjustments
- Calibration matrix storage and loading

Author: SolarVisionAI Team
Version: 1.0.0
"""

import json
import pickle
from typing import Dict, Optional, List, Tuple, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from enum import Enum
import logging

import numpy as np
import cv2

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CurrentRecipe(Enum):
    """Standard current recipes for EL imaging (IEC 60904-13)"""
    ISC = "Isc"  # Short-circuit current
    ISC_10_PERCENT = "0.1*Isc"  # 10% of short-circuit current
    IMPP = "Impp"  # Maximum power point current
    CUSTOM = "Custom"  # User-defined


class CameraMode(Enum):
    """Camera operating modes"""
    MANUAL = "manual"  # Full manual control
    SEMI_AUTO = "semi_auto"  # Aperture or shutter priority
    AUTO = "auto"  # Automatic exposure


class LensType(Enum):
    """Common lens types for EL imaging"""
    WIDE_ANGLE = "wide_angle"  # < 35mm
    NORMAL = "normal"  # 35-70mm
    TELEPHOTO = "telephoto"  # > 70mm
    MACRO = "macro"  # Macro lens


@dataclass
class CameraSettings:
    """Camera capture settings"""
    # Camera identification
    camera_make: str = ""
    camera_model: str = ""
    camera_serial: str = ""
    sensor_type: str = ""  # e.g., "CMOS", "CCD"
    sensor_size_mm: Tuple[float, float] = (36.0, 24.0)  # Full frame default

    # Lens
    lens_model: str = ""
    lens_type: str = LensType.NORMAL.value
    focal_length_mm: float = 50.0
    aperture_fstop: float = 5.6  # f-number
    min_focus_distance_m: float = 0.5

    # Exposure settings
    exposure_time_ms: float = 1000.0  # 1 second default for EL
    iso: int = 400
    exposure_mode: str = CameraMode.MANUAL.value

    # Image settings
    resolution_width: int = 4000
    resolution_height: int = 3000
    bit_depth: int = 14  # 14-bit raw common for EL
    image_format: str = "RAW"  # RAW, JPEG, TIFF

    # White balance (for color cameras)
    white_balance_k: int = 5500  # Color temperature in Kelvin
    white_balance_mode: str = "manual"

    # Advanced
    gain_db: float = 0.0
    black_level: int = 0
    enable_noise_reduction: bool = False

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> 'CameraSettings':
        """Create from dictionary"""
        return cls(**data)


@dataclass
class TestConditions:
    """IEC 60904-14 compliant test conditions"""
    # Test identification
    test_id: str = ""
    test_date: datetime = field(default_factory=datetime.now)
    operator_name: str = ""
    lab_name: str = ""

    # Module information
    module_serial: str = ""
    module_manufacturer: str = ""
    module_model: str = ""
    module_rated_power_w: float = 300.0
    module_technology: str = "Monocrystalline"  # Mono, Poly, Thin-film

    # Electrical test conditions
    current_recipe: str = CurrentRecipe.ISC_10_PERCENT.value
    test_current_a: float = 0.0
    isc_current_a: float = 9.0  # Typical for 300W module
    voc_voltage_v: float = 40.0
    impp_current_a: float = 8.5
    vmpp_voltage_v: float = 35.0

    # Environmental conditions
    ambient_temp_c: float = 25.0
    module_temp_c: float = 25.0
    humidity_percent: float = 50.0
    air_pressure_mbar: float = 1013.25

    # Equipment
    power_supply_make: str = ""
    power_supply_model: str = ""
    power_supply_accuracy_percent: float = 0.1
    cable_length_m: float = 2.0
    cable_resistance_ohm: float = 0.01

    # Dark environment
    darkroom_certified: bool = False
    background_light_lux: float = 0.0

    # Emissivity and corrections
    emissivity_factor: float = 0.85  # Typical for solar cells
    distance_to_module_m: float = 1.0

    # Standards compliance
    standard_reference: str = "IEC 60904-13:2018"
    accreditation: str = ""  # e.g., "ISO 17025"

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        data = asdict(self)
        data['test_date'] = self.test_date.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict) -> 'TestConditions':
        """Create from dictionary"""
        if 'test_date' in data and isinstance(data['test_date'], str):
            data['test_date'] = datetime.fromisoformat(data['test_date'])
        return cls(**data)

    def calculate_test_current(self) -> float:
        """
        Calculate test current based on recipe

        Returns:
            Test current in Amperes
        """
        if self.current_recipe == CurrentRecipe.ISC.value:
            return self.isc_current_a
        elif self.current_recipe == CurrentRecipe.ISC_10_PERCENT.value:
            return self.isc_current_a * 0.1
        elif self.current_recipe == CurrentRecipe.IMPP.value:
            return self.impp_current_a
        else:
            return self.test_current_a

    def validate(self) -> Tuple[bool, List[str]]:
        """
        Validate test conditions

        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []

        # Check required fields
        if not self.module_serial:
            errors.append("Module serial number required")

        # Check electrical parameters
        if self.isc_current_a <= 0:
            errors.append("Isc must be positive")
        if self.voc_voltage_v <= 0:
            errors.append("Voc must be positive")

        # Check environmental ranges
        if not (-40 <= self.ambient_temp_c <= 85):
            errors.append("Ambient temperature out of range (-40 to 85°C)")
        if not (0 <= self.humidity_percent <= 100):
            errors.append("Humidity must be 0-100%")

        # Check dark environment
        if self.background_light_lux > 1.0:
            errors.append("Background light too high for EL imaging (> 1 lux)")

        return len(errors) == 0, errors


@dataclass
class CalibrationData:
    """Camera calibration data"""
    # Calibration identification
    calibration_id: str = ""
    calibration_date: datetime = field(default_factory=datetime.now)
    camera_serial: str = ""
    lens_serial: str = ""

    # Intrinsic parameters
    camera_matrix: Optional[np.ndarray] = None  # 3x3 matrix
    distortion_coeffs: Optional[np.ndarray] = None  # (k1, k2, p1, p2, k3)

    # Extrinsic parameters (if applicable)
    rotation_matrix: Optional[np.ndarray] = None  # 3x3
    translation_vector: Optional[np.ndarray] = None  # 3x1

    # Calibration quality
    reprojection_error_px: float = 0.0
    calibration_confidence: float = 0.0  # 0-1

    # Calibration settings
    checkerboard_size: Tuple[int, int] = (9, 6)  # Internal corners
    square_size_mm: float = 25.0
    num_calibration_images: int = 0

    # Validation
    is_valid: bool = False
    notes: str = ""

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        data = asdict(self)
        data['calibration_date'] = self.calibration_date.isoformat()

        # Convert numpy arrays to lists
        if self.camera_matrix is not None:
            data['camera_matrix'] = self.camera_matrix.tolist()
        if self.distortion_coeffs is not None:
            data['distortion_coeffs'] = self.distortion_coeffs.tolist()
        if self.rotation_matrix is not None:
            data['rotation_matrix'] = self.rotation_matrix.tolist()
        if self.translation_vector is not None:
            data['translation_vector'] = self.translation_vector.tolist()

        return data

    @classmethod
    def from_dict(cls, data: Dict) -> 'CalibrationData':
        """Create from dictionary"""
        # Parse date
        if 'calibration_date' in data and isinstance(data['calibration_date'], str):
            data['calibration_date'] = datetime.fromisoformat(data['calibration_date'])

        # Convert lists to numpy arrays
        if 'camera_matrix' in data and data['camera_matrix'] is not None:
            data['camera_matrix'] = np.array(data['camera_matrix'])
        if 'distortion_coeffs' in data and data['distortion_coeffs'] is not None:
            data['distortion_coeffs'] = np.array(data['distortion_coeffs'])
        if 'rotation_matrix' in data and data['rotation_matrix'] is not None:
            data['rotation_matrix'] = np.array(data['rotation_matrix'])
        if 'translation_vector' in data and data['translation_vector'] is not None:
            data['translation_vector'] = np.array(data['translation_vector'])

        return cls(**data)

    def save(self, filepath: Path) -> None:
        """Save calibration data to file"""
        filepath = Path(filepath)

        if filepath.suffix == '.json':
            # Save as JSON
            with open(filepath, 'w') as f:
                json.dump(self.to_dict(), f, indent=2)
        else:
            # Save as pickle (preserves numpy arrays better)
            with open(filepath, 'wb') as f:
                pickle.dump(self, f)

        logger.info(f"Saved calibration data to {filepath}")

    @classmethod
    def load(cls, filepath: Path) -> 'CalibrationData':
        """Load calibration data from file"""
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"Calibration file not found: {filepath}")

        if filepath.suffix == '.json':
            with open(filepath, 'r') as f:
                data = json.load(f)
            return cls.from_dict(data)
        else:
            with open(filepath, 'rb') as f:
                return pickle.load(f)


class CameraCalibrator:
    """
    Camera calibration utility

    Performs camera calibration using checkerboard patterns
    to determine intrinsic and distortion parameters.
    """

    def __init__(
        self,
        checkerboard_size: Tuple[int, int] = (9, 6),
        square_size_mm: float = 25.0
    ):
        """
        Initialize calibrator

        Args:
            checkerboard_size: Number of internal corners (width, height)
            square_size_mm: Size of checkerboard squares in mm
        """
        self.checkerboard_size = checkerboard_size
        self.square_size_mm = square_size_mm

        # Calibration data
        self.object_points: List[np.ndarray] = []
        self.image_points: List[np.ndarray] = []
        self.image_size: Optional[Tuple[int, int]] = None

        logger.info(
            f"Initialized CameraCalibrator: "
            f"checkerboard={checkerboard_size}, square_size={square_size_mm}mm"
        )

    def add_calibration_image(self, image: np.ndarray) -> bool:
        """
        Add calibration image

        Args:
            image: Grayscale or color image with checkerboard

        Returns:
            True if checkerboard was detected
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Find checkerboard corners
        ret, corners = cv2.findChessboardCorners(
            gray,
            self.checkerboard_size,
            cv2.CALIB_CB_ADAPTIVE_THRESH +
            cv2.CALIB_CB_NORMALIZE_IMAGE +
            cv2.CALIB_CB_FAST_CHECK
        )

        if ret:
            # Refine corner locations
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

            # Store points
            self.image_points.append(corners_refined)

            # Create object points (3D coordinates of checkerboard)
            objp = np.zeros((self.checkerboard_size[0] * self.checkerboard_size[1], 3), np.float32)
            objp[:, :2] = np.mgrid[
                0:self.checkerboard_size[0],
                0:self.checkerboard_size[1]
            ].T.reshape(-1, 2)
            objp *= self.square_size_mm
            self.object_points.append(objp)

            # Store image size
            self.image_size = gray.shape[::-1]

            logger.info(f"Added calibration image {len(self.image_points)}")
            return True
        else:
            logger.warning("Checkerboard not detected in image")
            return False

    def calibrate(self) -> CalibrationData:
        """
        Perform camera calibration

        Returns:
            CalibrationData with calibration parameters
        """
        if len(self.object_points) < 3:
            raise ValueError("Need at least 3 calibration images with detected checkerboards")

        logger.info(f"Calibrating with {len(self.object_points)} images...")

        # Calibrate camera
        ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            self.object_points,
            self.image_points,
            self.image_size,
            None,
            None
        )

        # Calculate reprojection error
        total_error = 0
        for i in range(len(self.object_points)):
            imgpoints2, _ = cv2.projectPoints(
                self.object_points[i],
                rvecs[i],
                tvecs[i],
                camera_matrix,
                dist_coeffs
            )
            error = cv2.norm(self.image_points[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
            total_error += error

        mean_error = total_error / len(self.object_points)

        # Create calibration data
        calibration = CalibrationData(
            calibration_id=f"calib_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            camera_matrix=camera_matrix,
            distortion_coeffs=dist_coeffs,
            reprojection_error_px=mean_error,
            calibration_confidence=1.0 / (1.0 + mean_error),
            checkerboard_size=self.checkerboard_size,
            square_size_mm=self.square_size_mm,
            num_calibration_images=len(self.object_points),
            is_valid=mean_error < 1.0  # Good calibration if error < 1 pixel
        )

        logger.info(
            f"Calibration complete: "
            f"Reprojection error = {mean_error:.3f} px, "
            f"Valid = {calibration.is_valid}"
        )

        return calibration

    def undistort_image(
        self,
        image: np.ndarray,
        calibration: CalibrationData
    ) -> np.ndarray:
        """
        Undistort image using calibration data

        Args:
            image: Input image
            calibration: CalibrationData with camera matrix and distortion coeffs

        Returns:
            Undistorted image
        """
        if calibration.camera_matrix is None or calibration.distortion_coeffs is None:
            logger.warning("Calibration data incomplete, returning original image")
            return image

        h, w = image.shape[:2]

        # Get optimal new camera matrix
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
            calibration.camera_matrix,
            calibration.distortion_coeffs,
            (w, h),
            1,
            (w, h)
        )

        # Undistort
        undistorted = cv2.undistort(
            image,
            calibration.camera_matrix,
            calibration.distortion_coeffs,
            None,
            new_camera_matrix
        )

        # Crop to ROI
        x, y, w, h = roi
        if w > 0 and h > 0:
            undistorted = undistorted[y:y + h, x:x + w]

        return undistorted


class CameraConfigManager:
    """
    Camera configuration and metadata manager

    Handles storage and retrieval of camera settings, test conditions,
    and calibration data.
    """

    def __init__(self, config_dir: Optional[Path] = None):
        """
        Initialize config manager

        Args:
            config_dir: Directory for storing configurations
        """
        self.config_dir = config_dir or Path.home() / '.solarvisionai' / 'camera_configs'
        self.config_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Camera config directory: {self.config_dir}")

    def save_camera_settings(self, settings: CameraSettings, name: str) -> None:
        """Save camera settings to file"""
        filepath = self.config_dir / f"camera_{name}.json"

        with open(filepath, 'w') as f:
            json.dump(settings.to_dict(), f, indent=2)

        logger.info(f"Saved camera settings: {name}")

    def load_camera_settings(self, name: str) -> CameraSettings:
        """Load camera settings from file"""
        filepath = self.config_dir / f"camera_{name}.json"

        if not filepath.exists():
            raise FileNotFoundError(f"Camera settings not found: {name}")

        with open(filepath, 'r') as f:
            data = json.load(f)

        return CameraSettings.from_dict(data)

    def list_camera_settings(self) -> List[str]:
        """List available camera settings"""
        return [
            f.stem.replace('camera_', '')
            for f in self.config_dir.glob('camera_*.json')
        ]

    def save_test_conditions(self, conditions: TestConditions, name: str) -> None:
        """Save test conditions to file"""
        filepath = self.config_dir / f"test_{name}.json"

        with open(filepath, 'w') as f:
            json.dump(conditions.to_dict(), f, indent=2)

        logger.info(f"Saved test conditions: {name}")

    def load_test_conditions(self, name: str) -> TestConditions:
        """Load test conditions from file"""
        filepath = self.config_dir / f"test_{name}.json"

        if not filepath.exists():
            raise FileNotFoundError(f"Test conditions not found: {name}")

        with open(filepath, 'r') as f:
            data = json.load(f)

        return TestConditions.from_dict(data)

    def create_metadata_package(
        self,
        camera_settings: CameraSettings,
        test_conditions: TestConditions,
        calibration: Optional[CalibrationData] = None
    ) -> Dict[str, Any]:
        """
        Create complete metadata package

        Args:
            camera_settings: Camera settings
            test_conditions: Test conditions
            calibration: Optional calibration data

        Returns:
            Complete metadata dictionary
        """
        metadata = {
            'camera_settings': camera_settings.to_dict(),
            'test_conditions': test_conditions.to_dict(),
            'capture_timestamp': datetime.now().isoformat(),
            'system_info': {
                'software': 'SolarVisionAI',
                'version': '1.0.0'
            }
        }

        if calibration is not None:
            metadata['calibration'] = calibration.to_dict()

        return metadata


# Convenience functions
def create_default_el_settings() -> CameraSettings:
    """Create default camera settings for EL imaging"""
    return CameraSettings(
        exposure_time_ms=1000.0,  # 1 second
        iso=400,
        aperture_fstop=5.6,
        exposure_mode=CameraMode.MANUAL.value,
        bit_depth=14,
        image_format="RAW",
        enable_noise_reduction=False
    )


def create_iec_test_conditions(
    module_serial: str,
    isc_current_a: float,
    voc_voltage_v: float
) -> TestConditions:
    """Create IEC-compliant test conditions"""
    conditions = TestConditions(
        module_serial=module_serial,
        isc_current_a=isc_current_a,
        voc_voltage_v=voc_voltage_v,
        current_recipe=CurrentRecipe.ISC_10_PERCENT.value,
        standard_reference="IEC 60904-13:2018"
    )

    # Calculate test current
    conditions.test_current_a = conditions.calculate_test_current()

    return conditions


if __name__ == "__main__":
    # Example usage
    import sys

    # Create example camera settings
    settings = create_default_el_settings()
    settings.camera_make = "Canon"
    settings.camera_model = "EOS R5"

    print("Camera Settings:")
    print(json.dumps(settings.to_dict(), indent=2))

    # Create example test conditions
    conditions = create_iec_test_conditions(
        module_serial="MOD-2024-001",
        isc_current_a=9.5,
        voc_voltage_v=40.2
    )

    print("\nTest Conditions:")
    print(json.dumps(conditions.to_dict(), indent=2))

    # Validate
    is_valid, errors = conditions.validate()
    print(f"\nValidation: {'✓ PASS' if is_valid else '✗ FAIL'}")
    if errors:
        for error in errors:
            print(f"  - {error}")

    # Save configurations
    manager = CameraConfigManager()
    manager.save_camera_settings(settings, "canon_eos_r5")
    manager.save_test_conditions(conditions, "example_test")

    print(f"\nConfigurations saved to: {manager.config_dir}")
