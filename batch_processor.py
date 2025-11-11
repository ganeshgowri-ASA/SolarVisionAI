"""
SolarVisionAI - Multi-threaded Batch Processor
Standards: Thread-safe processing with subscription tier limits

High-performance batch processing system with:
- Queue-based producer-consumer pattern
- Multi-threaded parallel processing
- Subscription tier limits (Basic/Pro/Advanced/Enterprise)
- Progress tracking with callbacks
- Error handling and recovery
- Graceful shutdown and resource cleanup

Author: SolarVisionAI Team
Version: 1.0.0
"""

import threading
import queue
import time
import logging
from typing import List, Callable, Optional, Dict, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import traceback

import numpy as np
import cv2

from preprocessing import AdvancedPreprocessor, PreprocessingResult, PreprocessingConfig
from analytics_engine import ImageAnalyticsEngine, ImageQualityMetrics
from quality_validator import ImageQualityValidator, ValidationResult

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(threadName)s] %(message)s'
)
logger = logging.getLogger(__name__)


class SubscriptionTier(Enum):
    """User subscription tiers with limits"""
    BASIC = {
        'name': 'Basic',
        'single_batch_limit': 5,
        'total_limit': 5,
        'concurrent_threads': 1,
        'priority': 1
    }
    PRO = {
        'name': 'Pro',
        'single_batch_limit': 50,
        'total_limit': 50,
        'concurrent_threads': 2,
        'priority': 2
    }
    ADVANCED = {
        'name': 'Advanced',
        'single_batch_limit': 500,
        'total_limit': 500,
        'concurrent_threads': 4,
        'priority': 3
    }
    ENTERPRISE = {
        'name': 'Enterprise',
        'single_batch_limit': -1,  # Unlimited
        'total_limit': -1,  # Unlimited
        'concurrent_threads': 8,
        'priority': 4
    }


class ProcessingStatus(Enum):
    """Processing status for individual items"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class BatchItem:
    """Individual item in batch processing queue"""
    item_id: str
    image: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)
    status: str = ProcessingStatus.PENDING.value

    # Results
    preprocessing_result: Optional[PreprocessingResult] = None
    analytics_result: Optional[ImageQualityMetrics] = None
    validation_result: Optional[ValidationResult] = None

    # Timing
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    processing_time_ms: float = 0.0

    # Error handling
    error: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 2


@dataclass
class BatchProcessingConfig:
    """Configuration for batch processing"""
    # Subscription tier
    subscription_tier: SubscriptionTier = SubscriptionTier.BASIC

    # Processing options
    enable_preprocessing: bool = True
    enable_analytics: bool = True
    enable_validation: bool = True

    # Quality filtering
    auto_reject_invalid: bool = True
    save_rejected_images: bool = False

    # Performance
    max_workers: Optional[int] = None  # None = auto-detect from tier
    queue_size: int = 100

    # Progress tracking
    enable_progress_callbacks: bool = True
    progress_update_interval_ms: int = 500

    # Error handling
    stop_on_error: bool = False
    max_retries: int = 2

    # Resource management
    max_memory_mb: Optional[int] = None
    enable_memory_monitoring: bool = True


@dataclass
class BatchProcessingResult:
    """Result of batch processing operation"""
    total_items: int = 0
    completed: int = 0
    failed: int = 0
    skipped: int = 0

    items: List[BatchItem] = field(default_factory=list)

    # Timing
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    total_time_ms: float = 0.0
    avg_time_per_item_ms: float = 0.0

    # Resource usage
    peak_memory_mb: float = 0.0

    # Errors
    errors: List[str] = field(default_factory=list)

    def get_summary(self) -> str:
        """Get human-readable summary"""
        success_rate = (self.completed / self.total_items * 100) if self.total_items > 0 else 0

        return f"""
Batch Processing Summary
{'=' * 60}
Total Items: {self.total_items}
Completed: {self.completed} ({success_rate:.1f}%)
Failed: {self.failed}
Skipped: {self.skipped}

Timing:
  Total Time: {self.total_time_ms / 1000:.2f}s
  Avg per Item: {self.avg_time_per_item_ms:.0f}ms
  Peak Memory: {self.peak_memory_mb:.1f} MB

Status: {'✓ SUCCESS' if self.failed == 0 else f'⚠ {self.failed} FAILURES'}
{'=' * 60}
"""


class BatchProcessor:
    """
    Multi-threaded batch processor for EL images

    Implements producer-consumer pattern with:
    - Thread-safe queue management
    - Subscription tier enforcement
    - Progress tracking
    - Error recovery
    - Resource monitoring
    """

    def __init__(self, config: Optional[BatchProcessingConfig] = None):
        """
        Initialize batch processor

        Args:
            config: Batch processing configuration
        """
        self.config = config or BatchProcessingConfig()

        # Determine worker count from subscription tier
        if self.config.max_workers is None:
            tier_info = self.config.subscription_tier.value
            self.config.max_workers = tier_info['concurrent_threads']

        # Initialize components
        self.preprocessor = AdvancedPreprocessor() if self.config.enable_preprocessing else None
        self.analytics = ImageAnalyticsEngine() if self.config.enable_analytics else None
        self.validator = ImageQualityValidator() if self.config.enable_validation else None

        # Threading
        self.processing_queue: queue.Queue = queue.Queue(maxsize=self.config.queue_size)
        self.stop_event = threading.Event()
        self.pause_event = threading.Event()

        # Progress tracking
        self.progress_lock = threading.Lock()
        self.progress_callbacks: List[Callable] = []

        # State
        self.is_running = False
        self.current_result: Optional[BatchProcessingResult] = None

        logger.info(
            f"Initialized BatchProcessor: Tier={self.config.subscription_tier.value['name']}, "
            f"Workers={self.config.max_workers}"
        )

    def add_progress_callback(self, callback: Callable[[int, int, str], None]) -> None:
        """
        Add progress callback

        Args:
            callback: Function(current, total, status_message)
        """
        self.progress_callbacks.append(callback)

    def process_batch(
        self,
        images: List[np.ndarray],
        metadata_list: Optional[List[Dict]] = None
    ) -> BatchProcessingResult:
        """
        Process batch of images

        Args:
            images: List of input images
            metadata_list: Optional list of metadata dictionaries

        Returns:
            BatchProcessingResult with detailed results
        """
        # Enforce subscription limits
        tier_info = self.config.subscription_tier.value
        batch_limit = tier_info['single_batch_limit']

        if batch_limit > 0 and len(images) > batch_limit:
            raise ValueError(
                f"Batch size {len(images)} exceeds {tier_info['name']} tier limit "
                f"of {batch_limit} images. Please upgrade your subscription."
            )

        # Initialize result
        result = BatchProcessingResult()
        result.total_items = len(images)
        result.start_time = datetime.now()
        self.current_result = result

        # Create batch items
        batch_items = []
        for i, image in enumerate(images):
            metadata = metadata_list[i] if metadata_list and i < len(metadata_list) else {}

            item = BatchItem(
                item_id=f"item_{i:04d}",
                image=image,
                metadata=metadata,
                max_retries=self.config.max_retries
            )
            batch_items.append(item)

        logger.info(f"Starting batch processing: {len(batch_items)} items")

        try:
            # Process using thread pool
            self._process_with_threadpool(batch_items, result)

        except Exception as e:
            logger.error(f"Batch processing error: {str(e)}", exc_info=True)
            result.errors.append(f"Batch processing failed: {str(e)}")

        finally:
            # Finalize result
            result.end_time = datetime.now()
            result.total_time_ms = (result.end_time - result.start_time).total_seconds() * 1000

            if result.completed > 0:
                result.avg_time_per_item_ms = result.total_time_ms / result.completed

            result.items = batch_items

            # Update counters
            result.completed = sum(1 for item in batch_items if item.status == ProcessingStatus.COMPLETED.value)
            result.failed = sum(1 for item in batch_items if item.status == ProcessingStatus.FAILED.value)
            result.skipped = sum(1 for item in batch_items if item.status == ProcessingStatus.SKIPPED.value)

            logger.info(result.get_summary())

        return result

    def _process_with_threadpool(
        self,
        batch_items: List[BatchItem],
        result: BatchProcessingResult
    ) -> None:
        """Process items using ThreadPoolExecutor"""
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            # Submit all tasks
            future_to_item = {
                executor.submit(self._process_single_item, item): item
                for item in batch_items
            }

            # Process completed tasks
            completed_count = 0

            for future in as_completed(future_to_item):
                item = future_to_item[future]

                try:
                    # Get result (already updated in-place)
                    future.result()

                except Exception as e:
                    logger.error(f"Item {item.item_id} processing failed: {str(e)}")
                    item.status = ProcessingStatus.FAILED.value
                    item.error = str(e)
                    result.errors.append(f"{item.item_id}: {str(e)}")

                finally:
                    completed_count += 1

                    # Progress callback
                    if self.config.enable_progress_callbacks:
                        self._notify_progress(completed_count, len(batch_items), item.item_id)

    def _process_single_item(self, item: BatchItem) -> None:
        """
        Process a single batch item

        Args:
            item: BatchItem to process (updated in-place)
        """
        item.start_time = datetime.now()
        item.status = ProcessingStatus.PROCESSING.value

        try:
            # 1. Preprocessing
            if self.config.enable_preprocessing and self.preprocessor:
                logger.debug(f"Preprocessing {item.item_id}")
                item.preprocessing_result = self.preprocessor.process(item.image)
                processed_image = item.preprocessing_result.processed_image
            else:
                processed_image = item.image

            # 2. Analytics
            if self.config.enable_analytics and self.analytics:
                logger.debug(f"Analyzing {item.item_id}")
                item.analytics_result = self.analytics.analyze(processed_image)

            # 3. Validation
            if self.config.enable_validation and self.validator:
                logger.debug(f"Validating {item.item_id}")
                item.validation_result = self.validator.validate(
                    processed_image,
                    item.metadata
                )

                # Auto-reject if enabled
                if self.config.auto_reject_invalid and not item.validation_result.is_valid:
                    item.status = ProcessingStatus.SKIPPED.value
                    logger.warning(f"Item {item.item_id} rejected: {item.validation_result.rejection_reason}")
                    return

            # Success
            item.status = ProcessingStatus.COMPLETED.value
            logger.debug(f"Completed {item.item_id}")

        except Exception as e:
            logger.error(f"Error processing {item.item_id}: {str(e)}", exc_info=True)
            item.error = str(e)

            # Retry logic
            if item.retry_count < item.max_retries:
                item.retry_count += 1
                logger.info(f"Retrying {item.item_id} (attempt {item.retry_count}/{item.max_retries})")
                time.sleep(0.5 * item.retry_count)  # Exponential backoff
                self._process_single_item(item)  # Recursive retry
            else:
                item.status = ProcessingStatus.FAILED.value

                if self.config.stop_on_error:
                    raise

        finally:
            item.end_time = datetime.now()
            if item.start_time:
                item.processing_time_ms = (item.end_time - item.start_time).total_seconds() * 1000

    def _notify_progress(self, current: int, total: int, item_id: str) -> None:
        """Notify progress callbacks"""
        with self.progress_lock:
            status_msg = f"Processing {item_id} ({current}/{total})"

            for callback in self.progress_callbacks:
                try:
                    callback(current, total, status_msg)
                except Exception as e:
                    logger.error(f"Progress callback error: {str(e)}")

    def get_valid_items(self, result: BatchProcessingResult) -> List[BatchItem]:
        """Get only valid items from result"""
        return [
            item for item in result.items
            if item.status == ProcessingStatus.COMPLETED.value
            and (not item.validation_result or item.validation_result.is_valid)
        ]

    def get_failed_items(self, result: BatchProcessingResult) -> List[BatchItem]:
        """Get failed items from result"""
        return [
            item for item in result.items
            if item.status == ProcessingStatus.FAILED.value
        ]

    def get_rejected_items(self, result: BatchProcessingResult) -> List[BatchItem]:
        """Get rejected items from result"""
        return [
            item for item in result.items
            if item.status == ProcessingStatus.SKIPPED.value
        ]


class StreamingBatchProcessor:
    """
    Streaming batch processor for large datasets

    Processes images as they arrive without loading all into memory.
    Implements true producer-consumer pattern with queue.
    """

    def __init__(self, config: Optional[BatchProcessingConfig] = None):
        """Initialize streaming processor"""
        self.config = config or BatchProcessingConfig()
        self.batch_processor = BatchProcessor(config)

        # Queue for streaming
        self.input_queue: queue.Queue = queue.Queue(maxsize=self.config.queue_size)
        self.output_queue: queue.Queue = queue.Queue(maxsize=self.config.queue_size)

        # Control
        self.stop_event = threading.Event()
        self.worker_threads: List[threading.Thread] = []

        logger.info("Initialized StreamingBatchProcessor")

    def start_workers(self) -> None:
        """Start worker threads"""
        num_workers = self.config.max_workers or self.config.subscription_tier.value['concurrent_threads']

        for i in range(num_workers):
            worker = threading.Thread(
                target=self._worker_loop,
                name=f"Worker-{i}",
                daemon=True
            )
            worker.start()
            self.worker_threads.append(worker)

        logger.info(f"Started {num_workers} worker threads")

    def stop_workers(self) -> None:
        """Stop worker threads"""
        self.stop_event.set()

        # Wake up all workers
        for _ in self.worker_threads:
            try:
                self.input_queue.put(None, timeout=1)
            except queue.Full:
                pass

        # Wait for completion
        for worker in self.worker_threads:
            worker.join(timeout=5)

        logger.info("Stopped all worker threads")

    def _worker_loop(self) -> None:
        """Worker thread main loop"""
        while not self.stop_event.is_set():
            try:
                # Get item from queue
                item = self.input_queue.get(timeout=1)

                if item is None:  # Poison pill
                    break

                # Process item
                self.batch_processor._process_single_item(item)

                # Put result in output queue
                self.output_queue.put(item)

                # Mark task as done
                self.input_queue.task_done()

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Worker error: {str(e)}", exc_info=True)

    def submit_item(self, image: np.ndarray, metadata: Optional[Dict] = None) -> None:
        """Submit item for processing"""
        item = BatchItem(
            item_id=f"item_{time.time()}",
            image=image,
            metadata=metadata or {}
        )
        self.input_queue.put(item)

    def get_result(self, timeout: float = 1.0) -> Optional[BatchItem]:
        """Get processed result"""
        try:
            return self.output_queue.get(timeout=timeout)
        except queue.Empty:
            return None


# Convenience functions
def quick_batch_process(
    images: List[np.ndarray],
    tier: SubscriptionTier = SubscriptionTier.BASIC
) -> List[np.ndarray]:
    """
    Quick batch preprocessing

    Args:
        images: List of input images
        tier: Subscription tier

    Returns:
        List of preprocessed images
    """
    config = BatchProcessingConfig(
        subscription_tier=tier,
        enable_preprocessing=True,
        enable_analytics=False,
        enable_validation=False
    )

    processor = BatchProcessor(config)
    result = processor.process_batch(images)

    return [
        item.preprocessing_result.processed_image
        for item in result.items
        if item.preprocessing_result is not None
    ]


def batch_validate(
    images: List[np.ndarray],
    tier: SubscriptionTier = SubscriptionTier.BASIC
) -> Tuple[List[np.ndarray], List[ValidationResult]]:
    """
    Batch validation

    Args:
        images: List of input images
        tier: Subscription tier

    Returns:
        Tuple of (valid_images, validation_results)
    """
    config = BatchProcessingConfig(
        subscription_tier=tier,
        enable_preprocessing=False,
        enable_analytics=True,
        enable_validation=True,
        auto_reject_invalid=False
    )

    processor = BatchProcessor(config)
    result = processor.process_batch(images)

    valid_images = []
    validation_results = []

    for item in result.items:
        if item.validation_result:
            validation_results.append(item.validation_result)
            if item.validation_result.is_valid:
                valid_images.append(item.image)

    return valid_images, validation_results


if __name__ == "__main__":
    # Example usage
    import sys

    if len(sys.argv) > 1:
        # Load multiple images
        image_paths = sys.argv[1:]
        images = []

        for path in image_paths:
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                images.append(img)

        if not images:
            print("Error: No valid images loaded")
            sys.exit(1)

        print(f"Loaded {len(images)} images")

        # Progress callback
        def progress_callback(current: int, total: int, status: str) -> None:
            print(f"Progress: {current}/{total} - {status}")

        # Configure and process
        config = BatchProcessingConfig(
            subscription_tier=SubscriptionTier.PRO,
            enable_preprocessing=True,
            enable_analytics=True,
            enable_validation=True
        )

        processor = BatchProcessor(config)
        processor.add_progress_callback(progress_callback)

        result = processor.process_batch(images)

        # Print summary
        print(result.get_summary())

        # Show valid items
        valid_items = processor.get_valid_items(result)
        print(f"\nValid items: {len(valid_items)}")

        for item in valid_items:
            if item.analytics_result:
                print(f"  {item.item_id}: Quality Score = {item.analytics_result.quality_score:.1f}")

    else:
        print("Usage: python batch_processor.py <image1> <image2> ...")
