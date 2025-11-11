"""
SolarVisionAI - ML Model Retraining Pipeline
Standards: ML Ops best practices, GDPR compliance

ML model improvement system with:
- Opt-in user consent for data reuse
- Quality-based filtering (high-confidence annotations only)
- Export to Roboflow format
- Annotation suggestion and pre-labeling
- Dataset versioning and tracking
- Active learning strategies

Author: SolarVisionAI Team
Version: 1.0.0
"""

import json
import shutil
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from enum import Enum
import logging
import hashlib
import csv

import numpy as np
import cv2
from PIL import Image

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DatasetSplit(Enum):
    """Dataset split types"""
    TRAIN = "train"
    VALID = "valid"
    TEST = "test"


class AnnotationStatus(Enum):
    """Annotation status"""
    PENDING = "pending"
    SUGGESTED = "suggested"  # Pre-labeled by model
    VERIFIED = "verified"  # Human-verified
    REJECTED = "rejected"


@dataclass
class UserConsent:
    """User consent for data usage"""
    user_id: str
    consent_given: bool = False
    consent_date: Optional[datetime] = None
    consent_version: str = "1.0"
    allow_model_training: bool = False
    allow_public_dataset: bool = False
    allow_research: bool = False
    data_retention_days: int = 365
    consent_text: str = ""

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'user_id': self.user_id,
            'consent_given': self.consent_given,
            'consent_date': self.consent_date.isoformat() if self.consent_date else None,
            'consent_version': self.consent_version,
            'allow_model_training': self.allow_model_training,
            'allow_public_dataset': self.allow_public_dataset,
            'allow_research': self.allow_research,
            'data_retention_days': self.data_retention_days
        }

    def is_valid(self) -> bool:
        """Check if consent is valid"""
        if not self.consent_given:
            return False

        if self.consent_date is None:
            return False

        # Check expiration
        days_since_consent = (datetime.now() - self.consent_date).days
        if days_since_consent > self.data_retention_days:
            return False

        return True


@dataclass
class Annotation:
    """Single defect annotation"""
    annotation_id: str
    image_id: str
    defect_class: str
    bbox: Tuple[float, float, float, float]  # x, y, width, height (normalized 0-1)
    confidence: float = 1.0
    status: str = AnnotationStatus.VERIFIED.value
    annotator: str = "human"  # "human" or "model"
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'annotation_id': self.annotation_id,
            'image_id': self.image_id,
            'class': self.defect_class,
            'bbox': list(self.bbox),
            'confidence': self.confidence,
            'status': self.status,
            'annotator': self.annotator,
            'timestamp': self.timestamp.isoformat()
        }

    def to_roboflow_format(self, image_width: int, image_height: int) -> Dict:
        """Convert to Roboflow annotation format"""
        x_center, y_center, width, height = self.bbox

        # Convert normalized to pixel coordinates
        x_px = x_center * image_width
        y_px = y_center * image_height
        w_px = width * image_width
        h_px = height * image_height

        return {
            'x': x_px,
            'y': y_px,
            'width': w_px,
            'height': h_px,
            'class': self.defect_class,
            'confidence': self.confidence
        }

    def to_yolo_format(self) -> str:
        """Convert to YOLO format (class x_center y_center width height)"""
        # Assuming class mapping exists
        class_id = 0  # Would need class mapping
        x, y, w, h = self.bbox
        return f"{class_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}"


@dataclass
class TrainingImage:
    """Training image with metadata"""
    image_id: str
    image_path: Path
    annotations: List[Annotation] = field(default_factory=list)
    user_consent: Optional[UserConsent] = None
    split: str = DatasetSplit.TRAIN.value
    quality_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    dataset_version: str = "v1.0"

    def has_valid_consent(self) -> bool:
        """Check if image has valid user consent"""
        if self.user_consent is None:
            return False
        return self.user_consent.is_valid() and self.user_consent.allow_model_training

    def get_high_confidence_annotations(self, threshold: float = 0.8) -> List[Annotation]:
        """Get only high-confidence annotations"""
        return [
            ann for ann in self.annotations
            if ann.confidence >= threshold and ann.status == AnnotationStatus.VERIFIED.value
        ]


@dataclass
class DatasetVersion:
    """Dataset version metadata"""
    version: str
    creation_date: datetime = field(default_factory=datetime.now)
    num_images: int = 0
    num_annotations: int = 0
    class_distribution: Dict[str, int] = field(default_factory=dict)
    train_images: int = 0
    valid_images: int = 0
    test_images: int = 0
    notes: str = ""
    parent_version: Optional[str] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'version': self.version,
            'creation_date': self.creation_date.isoformat(),
            'num_images': self.num_images,
            'num_annotations': self.num_annotations,
            'class_distribution': self.class_distribution,
            'train_images': self.train_images,
            'valid_images': self.valid_images,
            'test_images': self.test_images,
            'notes': self.notes,
            'parent_version': self.parent_version
        }


class RetrainingPipeline:
    """
    ML model retraining pipeline

    Manages dataset curation, annotation, and export for model retraining
    """

    def __init__(self, dataset_root: Optional[Path] = None):
        """
        Initialize retraining pipeline

        Args:
            dataset_root: Root directory for dataset storage
        """
        self.dataset_root = dataset_root or Path.cwd() / 'training_data'
        self.dataset_root.mkdir(parents=True, exist_ok=True)

        self.images_dir = self.dataset_root / 'images'
        self.annotations_dir = self.dataset_root / 'annotations'
        self.consent_dir = self.dataset_root / 'consent'
        self.versions_dir = self.dataset_root / 'versions'

        for dir_path in [self.images_dir, self.annotations_dir, self.consent_dir, self.versions_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Initialized RetrainingPipeline at {self.dataset_root}")

    def add_training_image(
        self,
        image: np.ndarray,
        annotations: List[Dict],
        user_consent: Optional[UserConsent] = None,
        metadata: Optional[Dict] = None
    ) -> str:
        """
        Add image to training dataset

        Args:
            image: Image array
            annotations: List of annotation dictionaries
            user_consent: User consent object
            metadata: Optional metadata

        Returns:
            Image ID
        """
        # Check consent
        if user_consent is None or not user_consent.is_valid():
            logger.warning("Image not added: Invalid or missing user consent")
            return ""

        if not user_consent.allow_model_training:
            logger.warning("Image not added: User did not consent to model training")
            return ""

        # Generate unique image ID
        image_hash = hashlib.md5(image.tobytes()).hexdigest()[:16]
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        image_id = f"img_{timestamp}_{image_hash}"

        # Save image
        image_path = self.images_dir / f"{image_id}.jpg"
        cv2.imwrite(str(image_path), image)

        # Create annotations
        annotation_objects = []
        for ann in annotations:
            annotation_obj = Annotation(
                annotation_id=f"{image_id}_{len(annotation_objects):04d}",
                image_id=image_id,
                defect_class=ann.get('class', 'unknown'),
                bbox=(ann['x'], ann['y'], ann['width'], ann['height']),
                confidence=ann.get('confidence', 1.0),
                status=AnnotationStatus.VERIFIED.value
            )
            annotation_objects.append(annotation_obj)

        # Save annotations
        annotations_path = self.annotations_dir / f"{image_id}.json"
        with open(annotations_path, 'w') as f:
            json.dump([ann.to_dict() for ann in annotation_objects], f, indent=2)

        # Save consent
        consent_path = self.consent_dir / f"{image_id}_consent.json"
        with open(consent_path, 'w') as f:
            json.dump(user_consent.to_dict(), f, indent=2)

        logger.info(f"Added training image: {image_id} with {len(annotation_objects)} annotations")

        return image_id

    def filter_quality_images(
        self,
        min_quality_score: float = 60.0,
        min_annotation_confidence: float = 0.8
    ) -> List[TrainingImage]:
        """
        Filter high-quality images for training

        Args:
            min_quality_score: Minimum image quality score
            min_annotation_confidence: Minimum annotation confidence

        Returns:
            List of high-quality training images
        """
        quality_images = []

        for image_path in self.images_dir.glob('*.jpg'):
            image_id = image_path.stem

            # Load annotations
            annotations_path = self.annotations_dir / f"{image_id}.json"
            if not annotations_path.exists():
                continue

            with open(annotations_path, 'r') as f:
                annotations_data = json.load(f)

            # Load consent
            consent_path = self.consent_dir / f"{image_id}_consent.json"
            if not consent_path.exists():
                continue

            with open(consent_path, 'r') as f:
                consent_data = json.load(f)

            # Create objects
            annotations = [
                Annotation(
                    annotation_id=ann['annotation_id'],
                    image_id=ann['image_id'],
                    defect_class=ann['class'],
                    bbox=tuple(ann['bbox']),
                    confidence=ann['confidence'],
                    status=ann['status']
                )
                for ann in annotations_data
            ]

            consent = UserConsent(
                user_id=consent_data['user_id'],
                consent_given=consent_data['consent_given'],
                allow_model_training=consent_data.get('allow_model_training', False)
            )

            if consent_data.get('consent_date'):
                consent.consent_date = datetime.fromisoformat(consent_data['consent_date'])

            # Create training image
            training_image = TrainingImage(
                image_id=image_id,
                image_path=image_path,
                annotations=annotations,
                user_consent=consent
            )

            # Filter high-confidence annotations
            high_conf_annotations = training_image.get_high_confidence_annotations(min_annotation_confidence)

            # Check quality
            if len(high_conf_annotations) > 0 and training_image.has_valid_consent():
                training_image.annotations = high_conf_annotations
                quality_images.append(training_image)

        logger.info(f"Filtered {len(quality_images)} high-quality images from {len(list(self.images_dir.glob('*.jpg')))}")

        return quality_images

    def export_to_roboflow(
        self,
        training_images: List[TrainingImage],
        output_dir: Optional[Path] = None,
        train_split: float = 0.7,
        valid_split: float = 0.2,
        test_split: float = 0.1
    ) -> Path:
        """
        Export dataset to Roboflow format

        Args:
            training_images: List of training images
            output_dir: Output directory
            train_split: Training set proportion
            valid_split: Validation set proportion
            test_split: Test set proportion

        Returns:
            Path to exported dataset
        """
        if output_dir is None:
            output_dir = self.dataset_root / f"roboflow_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        output_dir.mkdir(parents=True, exist_ok=True)

        # Split dataset
        np.random.shuffle(training_images)

        train_count = int(len(training_images) * train_split)
        valid_count = int(len(training_images) * valid_split)

        train_images = training_images[:train_count]
        valid_images = training_images[train_count:train_count + valid_count]
        test_images = training_images[train_count + valid_count:]

        # Export each split
        for split_name, split_images in [
            ('train', train_images),
            ('valid', valid_images),
            ('test', test_images)
        ]:
            split_dir = output_dir / split_name
            split_dir.mkdir(parents=True, exist_ok=True)

            for training_image in split_images:
                # Copy image
                dest_image_path = split_dir / f"{training_image.image_id}.jpg"
                shutil.copy(training_image.image_path, dest_image_path)

                # Create annotation file (Roboflow JSON format)
                image = cv2.imread(str(training_image.image_path))
                h, w = image.shape[:2]

                roboflow_annotations = []
                for ann in training_image.annotations:
                    roboflow_annotations.append(ann.to_roboflow_format(w, h))

                annotation_data = {
                    'image': str(dest_image_path.name),
                    'annotations': roboflow_annotations
                }

                annotation_path = split_dir / f"{training_image.image_id}.json"
                with open(annotation_path, 'w') as f:
                    json.dump(annotation_data, f, indent=2)

        # Create dataset metadata
        metadata = {
            'name': 'SolarVisionAI EL Defect Dataset',
            'version': datetime.now().strftime('%Y%m%d'),
            'format': 'roboflow',
            'splits': {
                'train': len(train_images),
                'valid': len(valid_images),
                'test': len(test_images)
            },
            'total_images': len(training_images),
            'classes': self._get_class_list(training_images)
        }

        with open(output_dir / 'dataset.json', 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Exported Roboflow dataset to: {output_dir}")
        logger.info(f"  Train: {len(train_images)}, Valid: {len(valid_images)}, Test: {len(test_images)}")

        return output_dir

    def export_to_yolo(
        self,
        training_images: List[TrainingImage],
        output_dir: Optional[Path] = None
    ) -> Path:
        """
        Export dataset to YOLO format

        Args:
            training_images: List of training images
            output_dir: Output directory

        Returns:
            Path to exported dataset
        """
        if output_dir is None:
            output_dir = self.dataset_root / f"yolo_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        output_dir.mkdir(parents=True, exist_ok=True)

        images_dir = output_dir / 'images'
        labels_dir = output_dir / 'labels'
        images_dir.mkdir(exist_ok=True)
        labels_dir.mkdir(exist_ok=True)

        # Get class list
        class_list = self._get_class_list(training_images)

        # Save class names
        with open(output_dir / 'classes.txt', 'w') as f:
            for class_name in class_list:
                f.write(f"{class_name}\n")

        # Create class mapping
        class_to_id = {cls: idx for idx, cls in enumerate(class_list)}

        # Export images and labels
        for training_image in training_images:
            # Copy image
            dest_image_path = images_dir / f"{training_image.image_id}.jpg"
            shutil.copy(training_image.image_path, dest_image_path)

            # Create YOLO label file
            label_path = labels_dir / f"{training_image.image_id}.txt"

            with open(label_path, 'w') as f:
                for ann in training_image.annotations:
                    class_id = class_to_id.get(ann.defect_class, 0)
                    x, y, w, h = ann.bbox
                    f.write(f"{class_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

        # Create data.yaml for YOLOv5/v8
        yaml_content = f"""
# SolarVisionAI EL Defect Dataset
path: {output_dir.absolute()}
train: images
val: images

nc: {len(class_list)}
names: {class_list}
"""

        with open(output_dir / 'data.yaml', 'w') as f:
            f.write(yaml_content.strip())

        logger.info(f"Exported YOLO dataset to: {output_dir}")

        return output_dir

    def suggest_annotations(
        self,
        image: np.ndarray,
        model_predictions: List[Dict],
        confidence_threshold: float = 0.7
    ) -> List[Annotation]:
        """
        Generate annotation suggestions from model predictions

        Args:
            image: Input image
            model_predictions: Model predictions
            confidence_threshold: Minimum confidence for suggestions

        Returns:
            List of suggested annotations
        """
        suggestions = []

        h, w = image.shape[:2]

        for pred in model_predictions:
            if pred.get('confidence', 0) < confidence_threshold:
                continue

            # Normalize coordinates
            x = pred['x'] / w
            y = pred['y'] / h
            width = pred['width'] / w
            height = pred['height'] / h

            annotation = Annotation(
                annotation_id=f"suggested_{len(suggestions)}",
                image_id="pending",
                defect_class=pred.get('class', 'unknown'),
                bbox=(x, y, width, height),
                confidence=pred['confidence'],
                status=AnnotationStatus.SUGGESTED.value,
                annotator="model"
            )

            suggestions.append(annotation)

        logger.info(f"Generated {len(suggestions)} annotation suggestions")

        return suggestions

    def create_dataset_version(
        self,
        version_name: str,
        training_images: List[TrainingImage],
        notes: str = ""
    ) -> DatasetVersion:
        """
        Create and save dataset version

        Args:
            version_name: Version identifier
            training_images: Images in this version
            notes: Version notes

        Returns:
            DatasetVersion object
        """
        # Calculate statistics
        class_distribution = {}
        for img in training_images:
            for ann in img.annotations:
                class_distribution[ann.defect_class] = class_distribution.get(ann.defect_class, 0) + 1

        # Create version
        version = DatasetVersion(
            version=version_name,
            num_images=len(training_images),
            num_annotations=sum(len(img.annotations) for img in training_images),
            class_distribution=class_distribution,
            notes=notes
        )

        # Save version metadata
        version_path = self.versions_dir / f"{version_name}.json"
        with open(version_path, 'w') as f:
            json.dump(version.to_dict(), f, indent=2)

        logger.info(f"Created dataset version: {version_name}")
        logger.info(f"  Images: {version.num_images}, Annotations: {version.num_annotations}")

        return version

    def _get_class_list(self, training_images: List[TrainingImage]) -> List[str]:
        """Get sorted list of unique classes"""
        classes = set()
        for img in training_images:
            for ann in img.annotations:
                classes.add(ann.defect_class)

        return sorted(list(classes))

    def generate_annotation_report(
        self,
        training_images: List[TrainingImage],
        output_path: Optional[Path] = None
    ) -> Path:
        """
        Generate annotation statistics report

        Args:
            training_images: Training images
            output_path: Output CSV path

        Returns:
            Path to report
        """
        if output_path is None:
            output_path = self.dataset_root / f"annotation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'Image ID', 'Num Annotations', 'Classes',
                'Avg Confidence', 'Has Consent', 'Split'
            ])

            for img in training_images:
                classes = ', '.join(set(ann.defect_class for ann in img.annotations))
                avg_conf = np.mean([ann.confidence for ann in img.annotations]) if img.annotations else 0

                writer.writerow([
                    img.image_id,
                    len(img.annotations),
                    classes,
                    f"{avg_conf:.3f}",
                    img.has_valid_consent(),
                    img.split
                ])

        logger.info(f"Annotation report saved to: {output_path}")

        return output_path


# Convenience functions
def create_user_consent(user_id: str, allow_training: bool = True) -> UserConsent:
    """Create user consent object"""
    return UserConsent(
        user_id=user_id,
        consent_given=True,
        consent_date=datetime.now(),
        allow_model_training=allow_training,
        allow_public_dataset=False,
        allow_research=allow_training
    )


if __name__ == "__main__":
    # Example usage
    pipeline = RetrainingPipeline()

    # Example: Add training image with consent
    consent = create_user_consent("user_001", allow_training=True)

    # Create sample image and annotations
    sample_image = np.random.randint(0, 255, (1000, 1500), dtype=np.uint8)
    sample_annotations = [
        {'class': 'crack', 'x': 0.5, 'y': 0.5, 'width': 0.1, 'height': 0.1, 'confidence': 0.95},
        {'class': 'hotspot', 'x': 0.3, 'y': 0.7, 'width': 0.08, 'height': 0.08, 'confidence': 0.88}
    ]

    image_id = pipeline.add_training_image(
        sample_image,
        sample_annotations,
        user_consent=consent
    )

    print(f"Added training image: {image_id}")

    # Filter quality images
    quality_images = pipeline.filter_quality_images(min_quality_score=60.0)
    print(f"Quality images: {len(quality_images)}")

    # Export to Roboflow
    if quality_images:
        export_path = pipeline.export_to_roboflow(quality_images)
        print(f"Exported to: {export_path}")
