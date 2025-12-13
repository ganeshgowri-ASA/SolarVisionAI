"""
SolarVisionAI Database Models

This module provides data models for managing sample allocations,
inspection records, and related entities for the SolarVisionAI platform.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict, Any
from enum import Enum
import uuid


class AllocationStatus(Enum):
    """Status of a sample allocation"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class SampleType(Enum):
    """Type of sample for inspection"""
    SOLAR_MODULE = "solar_module"
    SOLAR_CELL = "solar_cell"
    SOLAR_STRING = "solar_string"
    SOLAR_ARRAY = "solar_array"


@dataclass
class SampleAllocation:
    """
    Represents a sample allocation for EL inspection.

    A sample allocation tracks the assignment of solar PV samples
    (modules, cells, strings) to inspection batches with associated
    metadata and test conditions.

    Attributes:
        allocation_id: Unique identifier for the allocation
        sample_id: Identifier of the sample being allocated
        sample_type: Type of sample (module, cell, string, array)
        batch_id: Optional batch identifier for grouped processing
        status: Current status of the allocation
        created_at: Timestamp when allocation was created
        updated_at: Timestamp of last update
        assigned_to: User or system assigned to process this allocation
        priority: Processing priority (1-10, higher = more urgent)
        test_conditions: Dictionary of IEC-compliant test conditions
        notes: Additional notes or comments
        metadata: Additional metadata dictionary
    """

    sample_id: str
    sample_type: SampleType = SampleType.SOLAR_MODULE
    allocation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    batch_id: Optional[str] = None
    status: AllocationStatus = AllocationStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    assigned_to: Optional[str] = None
    priority: int = 5
    test_conditions: Dict[str, Any] = field(default_factory=dict)
    notes: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate and normalize fields after initialization"""
        if isinstance(self.sample_type, str):
            self.sample_type = SampleType(self.sample_type)
        if isinstance(self.status, str):
            self.status = AllocationStatus(self.status)
        if not 1 <= self.priority <= 10:
            self.priority = max(1, min(10, self.priority))

    def update_status(self, new_status: AllocationStatus) -> None:
        """Update the allocation status and timestamp"""
        self.status = new_status
        self.updated_at = datetime.now()

    def is_active(self) -> bool:
        """Check if the allocation is in an active state"""
        return self.status in (AllocationStatus.PENDING, AllocationStatus.IN_PROGRESS)

    def is_complete(self) -> bool:
        """Check if the allocation has been completed"""
        return self.status == AllocationStatus.COMPLETED

    def to_dict(self) -> Dict[str, Any]:
        """Convert allocation to dictionary representation"""
        return {
            "allocation_id": self.allocation_id,
            "sample_id": self.sample_id,
            "sample_type": self.sample_type.value,
            "batch_id": self.batch_id,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "assigned_to": self.assigned_to,
            "priority": self.priority,
            "test_conditions": self.test_conditions,
            "notes": self.notes,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SampleAllocation":
        """Create a SampleAllocation from dictionary representation"""
        # Parse datetime strings
        if isinstance(data.get("created_at"), str):
            data["created_at"] = datetime.fromisoformat(data["created_at"])
        if isinstance(data.get("updated_at"), str):
            data["updated_at"] = datetime.fromisoformat(data["updated_at"])

        return cls(**data)


@dataclass
class InspectionRecord:
    """
    Represents an inspection record for a sample.

    Links a sample allocation to its inspection results including
    quality metrics, defect data, and compliance status.
    """

    record_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    allocation_id: str = ""
    inspection_date: datetime = field(default_factory=datetime.now)
    inspector_id: Optional[str] = None
    quality_score: float = 0.0
    iec_compliant: bool = False
    defects_found: int = 0
    defect_summary: Dict[str, Any] = field(default_factory=dict)
    images_processed: int = 0
    processing_time_ms: float = 0.0
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert record to dictionary representation"""
        return {
            "record_id": self.record_id,
            "allocation_id": self.allocation_id,
            "inspection_date": self.inspection_date.isoformat(),
            "inspector_id": self.inspector_id,
            "quality_score": self.quality_score,
            "iec_compliant": self.iec_compliant,
            "defects_found": self.defects_found,
            "defect_summary": self.defect_summary,
            "images_processed": self.images_processed,
            "processing_time_ms": self.processing_time_ms,
            "notes": self.notes
        }


# Export all public classes and enums
__all__ = [
    "SampleAllocation",
    "InspectionRecord",
    "AllocationStatus",
    "SampleType"
]
