"""
Quick validation test for modules 1-3
"""

import sys
import numpy as np

def test_imports():
    """Test that all modules can be imported"""
    print("Testing module imports...")

    try:
        import preprocessing
        print("✓ preprocessing.py imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import preprocessing: {e}")
        return False

    try:
        import analytics_engine
        print("✓ analytics_engine.py imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import analytics_engine: {e}")
        return False

    try:
        import quality_validator
        print("✓ quality_validator.py imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import quality_validator: {e}")
        return False

    return True

def test_basic_functionality():
    """Test basic functionality with synthetic image"""
    print("\nTesting basic functionality...")

    # Create a synthetic test image
    test_image = np.random.randint(0, 255, (1000, 1500), dtype=np.uint8)
    print(f"Created test image: {test_image.shape}")

    try:
        # Test preprocessing
        from preprocessing import AdvancedPreprocessor, PreprocessingLevel
        preprocessor = AdvancedPreprocessor()
        print("✓ AdvancedPreprocessor instantiated")

        # Test analytics
        from analytics_engine import ImageAnalyticsEngine
        analytics = ImageAnalyticsEngine()
        print("✓ ImageAnalyticsEngine instantiated")

        # Test validator
        from quality_validator import ImageQualityValidator
        validator = ImageQualityValidator()
        print("✓ ImageQualityValidator instantiated")

        return True

    except Exception as e:
        print(f"✗ Functionality test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("SolarVisionAI - Module Validation Test")
    print("=" * 60)

    # Test imports
    if not test_imports():
        print("\n❌ Import tests FAILED")
        return 1

    # Test basic functionality
    if not test_basic_functionality():
        print("\n❌ Functionality tests FAILED")
        return 1

    print("\n" + "=" * 60)
    print("✅ All validation tests PASSED")
    print("=" * 60)
    return 0

if __name__ == "__main__":
    sys.exit(main())
