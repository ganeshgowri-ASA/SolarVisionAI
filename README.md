# ☀️ SolarVisionAI - Enterprise Solar PV Defect Detection System

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![IEC 60904-13](https://img.shields.io/badge/Standard-IEC%2060904--13-green.svg)](https://webstore.iec.ch/)
[![ISO 17025](https://img.shields.io/badge/Standard-ISO%2017025-green.svg)](https://www.iso.org/)

**World-class, production-grade Electroluminescence (EL) defect detection system** for solar photovoltaic modules. Implements IEC 60904-13/14 standards with advanced image processing, ML-powered defect detection, and comprehensive reporting.

---

## 🎯 Overview

SolarVisionAI is an enterprise-ready platform for automated solar panel quality inspection using electroluminescence imaging. The system combines:

- ✅ **IEC/ISO Standards Compliance**: Full IEC 60904-13/14 and ISO 17025 compliance
- ✅ **Advanced Image Processing**: Perspective correction, distortion removal, vignette compensation
- ✅ **ML-Powered Detection**: Roboflow API integration with continuous model improvement
- ✅ **Multi-Format Reporting**: Professional Excel, Word, and PDF reports
- ✅ **Production Architecture**: Multi-threaded processing, subscription tiers, quality validation

---

## 🏗️ Architecture

### Core Modules

#### 1. **preprocessing.py** - Advanced Image Preprocessing
Production-grade preprocessing pipeline implementing IEC 60904-13 standards:

**Features:**
- Perspective correction using Hough transform + RANSAC
- Barrel/pincushion distortion correction with OpenCV undistort
- Vignette compensation using polynomial fitting
- Non-local means denoising (edge-preserving)
- CLAHE (Contrast Limited Adaptive Histogram Equalization)
- Automated ROI (Region of Interest) cropping
- Homomorphic filtering for illumination normalization

**Usage:**
```python
from preprocessing import AdvancedPreprocessor, PreprocessingLevel

preprocessor = AdvancedPreprocessor()
result = preprocessor.process(image)
print(f"Applied: {result.applied_operations}")
print(f"Processing time: {result.processing_time_ms:.2f}ms")
```

**Configuration Levels:**
- `BASIC`: Fast processing (batch operations)
- `STANDARD`: Recommended for most cases (default)
- `ADVANCED`: Maximum quality, slower

---

#### 2. **analytics_engine.py** - Image Quality Metrics
Comprehensive quality assessment following IEC 60904-13 and ISO/IEC 12233:

**Metrics Calculated:**
- **SNR & PSNR**: Signal-to-Noise Ratio with MAD estimation
- **Sharpness**: Laplacian variance, gradient, FFT, MTF50, edge contrast
- **Statistics**: Mean, std dev, variance, kurtosis, skewness, CV coefficient
- **Entropy**: Shannon entropy and normalized information content
- **Histogram**: Dynamic range, uniformity, contrast ratio
- **IEC Compliance**: Sharpness class (A/B/C/D), resolution, contrast checks
- **Quality Score**: 0-100 composite score with quality level classification

**Usage:**
```python
from analytics_engine import ImageAnalyticsEngine

engine = ImageAnalyticsEngine(enable_advanced_metrics=True)
metrics = engine.analyze(image)

print(metrics.get_summary())
print(f"IEC Sharpness Class: {metrics.iec_sharpness_class}")
print(f"Quality Score: {metrics.quality_score}/100")
print(f"Compliant: {metrics.iec_compliant}")
```

---

#### 3. **quality_validator.py** - Automated Quality Filtering
Auto-reject unsuitable images with detailed feedback:

**Validation Checks:**
- ✅ **Resolution**: Min 640px (IEC requirement), megapixels, aspect ratio
- ✅ **Exposure**: Over/underexposure detection, dynamic range >20
- ✅ **Blur**: Severe blur detection, MTF50 thresholds, IEC sharpness class
- ✅ **Perspective**: Angle validation (max 50° distortion)
- ✅ **Noise**: SNR thresholds (min 15dB with warnings at 25dB)
- ✅ **Metadata**: Completeness verification (optional)

**Rejection Reasons:**
- User-friendly error messages
- Actionable recommendations
- Detailed technical diagnostics

**Usage:**
```python
from quality_validator import ImageQualityValidator

validator = ImageQualityValidator()
result = validator.validate(image, metadata)

if result.is_valid:
    print(f"✓ PASSED (Score: {result.quality_score}/100)")
else:
    print(result.get_user_message())  # Includes errors, warnings, recommendations
```

---

#### 4. **batch_processor.py** - Multi-threaded Batch Processing
High-performance parallel processing with subscription tier limits:

**Features:**
- Queue-based producer-consumer pattern
- Thread-safe operation with ThreadPoolExecutor
- Subscription tier enforcement:
  - **Basic**: 5 images, 1 thread
  - **Pro**: 50 images, 2 threads
  - **Advanced**: 500 images, 4 threads
  - **Enterprise**: Unlimited, 8 threads
- Progress tracking with callbacks
- Error handling and retry logic
- Resource monitoring

**Usage:**
```python
from batch_processor import BatchProcessor, SubscriptionTier

config = BatchProcessingConfig(
    subscription_tier=SubscriptionTier.PRO,
    enable_preprocessing=True,
    enable_analytics=True,
    enable_validation=True
)

processor = BatchProcessor(config)
result = processor.process_batch(images)

print(result.get_summary())
valid_items = processor.get_valid_items(result)
```

---

#### 5. **camera_config.py** - Metadata & Calibration
Camera configuration and test conditions management:

**Components:**
- **CameraSettings**: Exposure, ISO, aperture, focal length, white balance
- **TestConditions**: IEC 60904-14 compliant test parameters
  - Current recipes: Isc, 0.1*Isc, Impp
  - Electrical parameters: Voc, Isc, Vmpp, Impp
  - Environmental: Temperature, humidity, pressure
  - Equipment: Power supply, cable specs
- **CalibrationData**: Camera matrix, distortion coefficients, extrinsic parameters
- **CameraCalibrator**: Checkerboard-based calibration with reprojection error

**Usage:**
```python
from camera_config import create_iec_test_conditions, CameraCalibrator

# Create IEC-compliant test conditions
conditions = create_iec_test_conditions(
    module_serial="MOD-2024-001",
    isc_current_a=9.5,
    voc_voltage_v=40.2
)

# Calibrate camera
calibrator = CameraCalibrator(checkerboard_size=(9, 6))
for image in calibration_images:
    calibrator.add_calibration_image(image)

calibration = calibrator.calibrate()
calibration.save('camera_calibration.json')
```

---

#### 6. **report_generator.py** - Multi-Format Reports
Professional report generation in Excel, Word, and PDF:

**Excel Reports** (openpyxl):
- Multiple worksheets (Summary, Defects, Statistics, Quality, Test Conditions)
- Conditional formatting (severity-based colors)
- Charts (pie charts for distribution, bar charts for severity)
- Auto-fitted columns

**Word Reports** (python-docx):
- IEC/ISO 17025/9001 compliant templates
- Title page with company branding
- Executive summary
- Test conditions table
- Images (raw, processed, annotated)
- Defect analysis tables
- Compliance statement with signature section

**PDF Reports** (reportlab):
- Multi-page professional layout
- Tables with styling
- Summary statistics
- Professional formatting with headers/footers

**Usage:**
```python
from report_generator import generate_all_reports

report_paths = generate_all_reports(
    defect_data=defect_df,
    images={'original': img, 'annotated': annotated_img},
    quality_metrics=metrics,
    test_conditions=conditions
)

print(f"Excel: {report_paths['excel']}")
print(f"Word: {report_paths['word']}")
print(f"PDF: {report_paths['pdf']}")
```

---

#### 7. **retraining_pipeline.py** - ML Model Improvement
Continuous model improvement with data curation:

**Features:**
- **User Consent Management**: GDPR-compliant opt-in system
- **Quality Filtering**: High-confidence annotations only (threshold: 0.8)
- **Export Formats**:
  - Roboflow JSON format
  - YOLO format (with data.yaml)
- **Annotation Suggestions**: Pre-labeling with model predictions
- **Dataset Versioning**: Track versions with metadata
- **Active Learning**: Quality-based selection for retraining

**Usage:**
```python
from retraining_pipeline import RetrainingPipeline, create_user_consent

pipeline = RetrainingPipeline()

# Add training image with consent
consent = create_user_consent("user_001", allow_training=True)
image_id = pipeline.add_training_image(
    image, annotations, user_consent=consent
)

# Filter quality images
quality_images = pipeline.filter_quality_images(
    min_quality_score=60.0,
    min_annotation_confidence=0.8
)

# Export to Roboflow
export_path = pipeline.export_to_roboflow(quality_images)
```

---

#### 8. **genspark_deploy.py** - AI Agent Integration
GenSpark AI platform integration for intelligent assistance:

**Capabilities:**
- Defect analysis interpretation
- Quality assessment guidance
- Compliance recommendations
- Interactive troubleshooting
- Camera/test setup recommendations

**Usage:**
```python
from genspark_deploy import create_defect_analysis_agent

agent = create_defect_analysis_agent(api_key)

# Analyze defects
analysis = agent.analyze_defects(defect_data)

# Check compliance
compliance = agent.check_compliance(quality_metrics)

# Get recommendations
recommendations = agent.get_recommendations(
    camera_settings, test_conditions
)
```

---

## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- (Optional) CUDA-enabled GPU for acceleration

### Quick Install

```bash
# Clone repository
git clone https://github.com/ganeshgowri-ASA/SolarVisionAI.git
cd SolarVisionAI

# Install dependencies
pip install -r requirements.txt

# Run Streamlit app
streamlit run streamlit_app.py
```

### Dependencies

**Core:**
- numpy, scipy, pandas
- opencv-python, scikit-image
- Pillow

**Image Processing:**
- opencv-contrib-python (advanced features)

**Visualization:**
- plotly, matplotlib, seaborn

**Document Generation:**
- openpyxl (Excel)
- python-docx (Word)
- reportlab, fpdf2 (PDF)

**ML & AI:**
- scikit-learn
- torch, torchvision (optional, for future models)

**Web & API:**
- streamlit
- requests
- fastapi, uvicorn (for future API)

**Testing:**
- pytest, pytest-cov
- black, flake8

---

## 🚀 Quick Start

### Basic Usage

```python
import cv2
from preprocessing import preprocess_el_image, PreprocessingLevel
from analytics_engine import analyze_el_image
from quality_validator import quick_validate

# Load image
image = cv2.imread('el_image.jpg', cv2.IMREAD_GRAYSCALE)

# Validate
is_valid = quick_validate(image, strict=False)
print(f"Valid: {is_valid}")

# Preprocess
processed = preprocess_el_image(image, level=PreprocessingLevel.STANDARD)

# Analyze
metrics = analyze_el_image(processed)
print(f"Quality: {metrics.quality_level}, Score: {metrics.quality_score}/100")
```

### Streamlit App

```bash
# Run web interface
streamlit run streamlit_app.py

# Access at http://localhost:8501
```

---

## 📊 Standards Compliance

### IEC 60904-13:2018
Electroluminescence measurement of photovoltaic modules:
- ✅ Minimum resolution: 640px
- ✅ Sharpness classification: A/B/C/D (MTF50)
- ✅ Image quality requirements
- ✅ Test condition documentation

### IEC 60904-14:2020
Guidelines for EL imaging:
- ✅ Current recipe specifications (Isc, 0.1*Isc, Impp)
- ✅ Environmental condition recording
- ✅ Equipment specifications

### ISO/IEC 17025:2017
Testing laboratory competence:
- ✅ Traceability of measurements
- ✅ Calibration data management
- ✅ Quality control procedures
- ✅ Documentation and reporting

### ISO 9001
Quality management:
- ✅ Process documentation
- ✅ Continuous improvement (retraining pipeline)
- ✅ Customer satisfaction focus

---

## 🎯 Use Cases

### 1. Manufacturing Quality Control
- Inline inspection during production
- Batch processing of multiple modules
- Automated pass/fail decisions
- Traceability with serial numbers

### 2. Field Inspection & Maintenance
- On-site defect detection
- Historical trend analysis
- Maintenance scheduling
- Warranty claim documentation

### 3. Research & Development
- Material defect studies
- Process optimization
- Dataset curation for ML
- Comparative analysis

### 4. Certification & Compliance
- IEC standard compliance verification
- Test laboratory operations
- Third-party inspection services
- Regulatory reporting

---

## 🔧 Configuration

### Environment Variables

```bash
# API Keys
export ROBOFLOW_API_KEY="your_key_here"
export GENSPARK_API_KEY="your_key_here"

# Storage
export TRAINING_DATA_PATH="/path/to/training/data"
export CALIBRATION_DATA_PATH="/path/to/calibration"

# Performance
export MAX_WORKERS=4
export BATCH_SIZE=50
```

### Custom Configuration

```python
# preprocessing_config.py
from preprocessing import PreprocessingConfig

custom_config = PreprocessingConfig(
    enable_perspective_correction=True,
    enable_distortion_correction=True,
    denoise_h=12.0,
    clahe_clip_limit=3.0
)
```

---

## 📈 Performance

### Benchmark Results

| Operation | Time (ms) | Throughput |
|-----------|-----------|------------|
| Preprocessing (Standard) | ~500ms | 120 img/min |
| Analytics (Full) | ~300ms | 200 img/min |
| Validation | ~200ms | 300 img/min |
| Batch (10 images, 4 threads) | ~1.5s | 400 img/min |

*Tested on Intel i7-10700K, 32GB RAM, no GPU acceleration*

### Optimization Tips

1. **Use Basic preprocessing for batch operations**
2. **Enable GPU acceleration (cupy)**
3. **Adjust thread count based on CPU cores**
4. **Pre-filter images with quick_validate()**
5. **Cache calibration data**

---

## 🧪 Testing

```bash
# Run all tests
pytest

# With coverage
pytest --cov=. --cov-report=html

# Specific module
pytest test_preprocessing.py -v

# Quick syntax check
python -m py_compile *.py
```

---

## 📚 Documentation

### Module Documentation

Each module includes comprehensive docstrings:

```python
help(AdvancedPreprocessor)
help(ImageAnalyticsEngine)
help(ImageQualityValidator)
```

### Example Notebooks

See `/examples` directory for Jupyter notebooks:
- `01_preprocessing_demo.ipynb`
- `02_quality_analysis.ipynb`
- `03_batch_processing.ipynb`
- `04_report_generation.ipynb`

---

## 🤝 Contributing

Contributions welcome! Please follow these guidelines:

1. **Fork** the repository
2. **Create** feature branch (`git checkout -b feature/AmazingFeature`)
3. **Follow** PEP 8 style guide
4. **Add** tests for new features
5. **Document** with docstrings
6. **Commit** with clear messages
7. **Push** to branch
8. **Open** Pull Request

### Development Setup

```bash
# Install dev dependencies
pip install -r requirements.txt
pip install pytest black flake8

# Run linter
black .
flake8 .

# Run tests
pytest
```

---

## 📝 License

This project is licensed under the MIT License - see LICENSE file for details.

---

## 🙏 Acknowledgments

- **IEC Standards**: IEC 60904-13/14 technical committees
- **PV Lighthouse**: LumiTools standards and best practices
- **Research**: hackingmaterials/pv-vision, ucf-photovoltaics/UCF-EL-Defect
- **Roboflow**: Computer vision API platform
- **OpenCV**: Open source computer vision library
- **Streamlit**: Web app framework

---

## 📧 Support

- **Issues**: [GitHub Issues](https://github.com/ganeshgowri-ASA/SolarVisionAI/issues)
- **Email**: support@solarvisionai.com
- **Documentation**: [Wiki](https://github.com/ganeshgowri-ASA/SolarVisionAI/wiki)

---

## 🌟 Citation

If you use this software in your research, please cite:

```bibtex
@software{solarvisionai2024,
  title = {SolarVisionAI: Enterprise Solar PV Defect Detection System},
  author = {SolarVisionAI Team},
  year = {2024},
  url = {https://github.com/ganeshgowri-ASA/SolarVisionAI}
}
```

---

**Built with ❤️ for the renewable energy industry**

*Empowering solar quality assurance through AI and standards compliance*
