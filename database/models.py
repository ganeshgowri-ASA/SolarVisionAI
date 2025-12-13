"""
Database models for SolarVisionAI.

This module defines the SQLAlchemy ORM models for managing solar panel
samples, test protocols, and sample allocations.
"""

from datetime import datetime
from enum import Enum as PyEnum
from typing import Optional

from sqlalchemy import (
    Column,
    DateTime,
    Enum,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker

Base = declarative_base()


class SampleStatus(PyEnum):
    """Status enumeration for samples."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class AllocationStatus(PyEnum):
    """Status enumeration for sample allocations."""
    PENDING = "pending"
    ALLOCATED = "allocated"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


class Sample(Base):
    """
    Model representing a solar panel sample for analysis.

    Attributes:
        id: Primary key identifier.
        sample_name: Human-readable name for the sample.
        sample_type: Type/category of the sample.
        source: Origin or source of the sample.
        created_at: Timestamp when the sample was created.
        updated_at: Timestamp when the sample was last updated.
        status: Current status of the sample.
        description: Optional detailed description of the sample.
    """
    __tablename__ = "samples"

    id = Column(Integer, primary_key=True, autoincrement=True)
    sample_name = Column(String(255), nullable=False)
    sample_type = Column(String(100), nullable=True)
    source = Column(String(255), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    status = Column(Enum(SampleStatus), default=SampleStatus.PENDING)
    description = Column(Text, nullable=True)

    # Relationships
    allocations = relationship("SampleAllocation", back_populates="sample")

    def __repr__(self) -> str:
        return f"<Sample(id={self.id}, name='{self.sample_name}', status={self.status})>"


class TestProtocol(Base):
    """
    Model representing a test protocol for sample analysis.

    Attributes:
        id: Primary key identifier.
        protocol_name: Name of the test protocol.
        protocol_version: Version string of the protocol.
        description: Detailed description of the protocol.
        created_at: Timestamp when the protocol was created.
        is_active: Whether the protocol is currently active.
    """
    __tablename__ = "test_protocols"

    id = Column(Integer, primary_key=True, autoincrement=True)
    protocol_name = Column(String(255), nullable=False)
    protocol_version = Column(String(50), nullable=True)
    description = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    is_active = Column(Integer, default=1)  # 1 = active, 0 = inactive

    # Relationships
    allocations = relationship("SampleAllocation", back_populates="protocol")

    def __repr__(self) -> str:
        return f"<TestProtocol(id={self.id}, name='{self.protocol_name}', version='{self.protocol_version}')>"


class SampleAllocation(Base):
    """
    Model representing the allocation of a sample to a test protocol.

    This model tracks when and how samples are allocated for testing,
    including the trigger for allocation and current status.

    Attributes:
        id: Primary key identifier.
        sample_id: Foreign key reference to the Sample.
        protocol_id: Foreign key reference to the TestProtocol.
        allocated_sample_id: Unique identifier for this specific allocation.
        allocation_trigger: What triggered this allocation (e.g., 'manual', 'automatic', 'scheduled').
        allocated_at: Timestamp when the allocation was made.
        status: Current status of the allocation.
        notes: Optional notes or comments about the allocation.
    """
    __tablename__ = "sample_allocations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    sample_id = Column(Integer, ForeignKey("samples.id"), nullable=False)
    protocol_id = Column(Integer, ForeignKey("test_protocols.id"), nullable=False)
    allocated_sample_id = Column(String(100), unique=True, nullable=False)
    allocation_trigger = Column(String(100), nullable=False)
    allocated_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    status = Column(Enum(AllocationStatus), default=AllocationStatus.PENDING)
    notes = Column(Text, nullable=True)

    # Relationships
    sample = relationship("Sample", back_populates="allocations")
    protocol = relationship("TestProtocol", back_populates="allocations")

    def __repr__(self) -> str:
        return (
            f"<SampleAllocation(id={self.id}, "
            f"allocated_sample_id='{self.allocated_sample_id}', "
            f"status={self.status})>"
        )


def get_engine(database_url: str = "sqlite:///solarvision.db"):
    """Create and return a database engine."""
    return create_engine(database_url)


def get_session(engine=None, database_url: str = "sqlite:///solarvision.db"):
    """Create and return a database session."""
    if engine is None:
        engine = get_engine(database_url)
    Session = sessionmaker(bind=engine)
    return Session()


def init_db(database_url: str = "sqlite:///solarvision.db"):
    """Initialize the database by creating all tables."""
    engine = get_engine(database_url)
    Base.metadata.create_all(engine)
    return engine
