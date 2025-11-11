"""
SolarVisionAI - Production-Ready Streamlit Application v4
Complete Integration: Preprocessing, Analytics, Validation, Batch Processing, Reports

Features:
- Multi-file batch upload with subscription tier limits
- Advanced preprocessing with before/after previews
- Camera calibration and metadata input
- Real-time analytics dashboard
- Quality validation with pass/fail status
- Parallel batch processing with progress tracking
- Multi-format report generation (Excel, Word, PDF)
- Modern multi-tab UI layout

Author: SolarVisionAI Team
Version: 4.0.0
"""

import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import io
import time
from datetime import datetime
from pathlib import Path
import logging
from typing import List, Dict, Optional, Tuple
import traceback

# Import all SolarVisionAI modules
from preprocessing import (
    AdvancedPreprocessor,
    PreprocessingConfig,
    PreprocessingLevel,
    PreprocessingResult
)
from analytics_engine import (
    ImageAnalyticsEngine,
    ImageQualityMetrics
)
from quality_validator import (
    ImageQualityValidator,
    ValidationResult,
    ValidationConfig
)
from batch_processor import (
    BatchProcessor,
    SubscriptionTier,
    BatchProcessingConfig,
    BatchProcessingResult,
    BatchItem
)
from camera_config import (
    CameraSettings,
    TestConditions,
    CurrentRecipe,
    CameraMode,
    create_default_el_settings,
    create_iec_test_conditions
)
from report_generator import (
    ExcelReportGenerator,
    WordReportGenerator,
    PDFReportGenerator,
    ReportConfig,
    generate_all_reports
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="SolarVisionAI - EL Image Analysis Platform",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1F4E78;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1F4E78;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 0.25rem;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        color: #856404;
        padding: 1rem;
        border-radius: 0.25rem;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        padding: 1rem;
        border-radius: 0.25rem;
        margin: 1rem 0;
    }
    .stProgress > div > div > div > div {
        background-color: #1F4E78;
    }
</style>
""", unsafe_allow_html=True)


# Session state initialization
def init_session_state():
    """Initialize session state variables"""
    if 'uploaded_images' not in st.session_state:
        st.session_state.uploaded_images = []
    if 'processed_results' not in st.session_state:
        st.session_state.processed_results = None
    if 'batch_result' not in st.session_state:
        st.session_state.batch_result = None
    if 'camera_settings' not in st.session_state:
        st.session_state.camera_settings = create_default_el_settings()
    if 'test_conditions' not in st.session_state:
        st.session_state.test_conditions = TestConditions()
    if 'subscription_tier' not in st.session_state:
        st.session_state.subscription_tier = SubscriptionTier.BASIC
    if 'preprocessing_config' not in st.session_state:
        st.session_state.preprocessing_config = PreprocessingConfig.from_level(PreprocessingLevel.STANDARD)


def get_tier_limits(tier: SubscriptionTier) -> Dict:
    """Get subscription tier limits"""
    return tier.value


def load_image_from_upload(uploaded_file) -> np.ndarray:
    """Load image from uploaded file"""
    try:
        # Read file bytes
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)

        # Decode image
        image = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

        if image is None:
            # Try color
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        return image
    except Exception as e:
        logger.error(f"Error loading image: {str(e)}")
        return None


def display_image_comparison(original: np.ndarray, processed: np.ndarray, title: str = "Comparison"):
    """Display before/after image comparison"""
    col1, col2 = st.columns(2)

    with col1:
        st.write("**Original**")
        st.image(original, use_column_width=True, clamp=True)

    with col2:
        st.write("**Processed**")
        st.image(processed, use_column_width=True, clamp=True)


def create_defect_dataframe(batch_result: BatchProcessingResult) -> pd.DataFrame:
    """Create defect dataframe from batch results (placeholder for actual defect detection)"""
    # This is a placeholder - in production, you'd have actual defect detection
    defects = []

    for item in batch_result.items:
        if item.validation_result and not item.validation_result.is_valid:
            defects.append({
                'Defect_ID': item.item_id,
                'Defect_Type': 'Quality Issue',
                'Cell_Location': 'N/A',
                'Severity': 'High' if len(item.validation_result.errors) > 2 else 'Medium',
                'Confidence_%': 85.0,
                'Power_Loss_W': 5.0,
                'MBJ_Classification': 'Non-Standard'
            })

    return pd.DataFrame(defects)


# ==================== SIDEBAR ====================
def render_sidebar():
    """Render sidebar with settings"""
    st.sidebar.markdown("# ⚙️ Settings")

    # Subscription tier selection
    st.sidebar.markdown("### Subscription Tier")
    tier_options = {
        "Basic (5 images)": SubscriptionTier.BASIC,
        "Pro (50 images)": SubscriptionTier.PRO,
        "Advanced (500 images)": SubscriptionTier.ADVANCED,
        "Enterprise (Unlimited)": SubscriptionTier.ENTERPRISE
    }

    selected_tier = st.sidebar.selectbox(
        "Select your tier:",
        options=list(tier_options.keys()),
        index=0
    )
    st.session_state.subscription_tier = tier_options[selected_tier]

    # Display tier info
    tier_info = get_tier_limits(st.session_state.subscription_tier)
    st.sidebar.info(
        f"**{tier_info['name']} Tier**\n\n"
        f"📸 Max images: {tier_info['single_batch_limit'] if tier_info['single_batch_limit'] > 0 else '∞'}\n\n"
        f"⚡ Parallel threads: {tier_info['concurrent_threads']}"
    )

    st.sidebar.markdown("---")

    # Preprocessing level
    st.sidebar.markdown("### Preprocessing")
    preproc_level = st.sidebar.selectbox(
        "Processing level:",
        options=["BASIC", "STANDARD", "ADVANCED"],
        index=1
    )

    st.session_state.preprocessing_config = PreprocessingConfig.from_level(
        PreprocessingLevel[preproc_level]
    )

    st.sidebar.markdown("---")

    # About section
    st.sidebar.markdown("### About")
    st.sidebar.info(
        "**SolarVisionAI v4.0**\n\n"
        "Production-grade EL image analysis platform\n\n"
        "Standards: IEC 60904-13, IEC 60904-14, ISO 17025"
    )


# ==================== TAB 1: UPLOAD ====================
def render_upload_tab():
    """Render upload tab"""
    st.markdown('<p class="main-header">📤 Image Upload</p>', unsafe_allow_html=True)

    tier_info = get_tier_limits(st.session_state.subscription_tier)
    max_files = tier_info['single_batch_limit']

    if max_files > 0:
        st.info(f"Your {tier_info['name']} tier allows up to **{max_files} images** per batch")
    else:
        st.success(f"Your {tier_info['name']} tier allows **unlimited images**")

    # File uploader
    uploaded_files = st.file_uploader(
        "Upload EL images (PNG, JPG, TIFF)",
        type=['png', 'jpg', 'jpeg', 'tiff', 'tif'],
        accept_multiple_files=True,
        key="image_uploader"
    )

    if uploaded_files:
        # Check tier limits
        if max_files > 0 and len(uploaded_files) > max_files:
            st.error(
                f"❌ You've uploaded {len(uploaded_files)} images, but your "
                f"{tier_info['name']} tier limit is {max_files}. "
                f"Please upgrade your subscription or reduce the number of images."
            )
            return

        st.success(f"✅ {len(uploaded_files)} image(s) uploaded successfully")

        # Load images
        with st.spinner("Loading images..."):
            images = []
            valid_files = []

            for uploaded_file in uploaded_files:
                image = load_image_from_upload(uploaded_file)
                if image is not None:
                    images.append(image)
                    valid_files.append(uploaded_file)
                else:
                    st.warning(f"⚠️ Could not load: {uploaded_file.name}")

            st.session_state.uploaded_images = images
            st.session_state.uploaded_filenames = [f.name for f in valid_files]

        if len(images) > 0:
            st.write(f"**{len(images)} valid image(s) loaded**")

            # Display thumbnails
            st.markdown("### Preview")
            cols = st.columns(min(5, len(images)))
            for idx, (img, col) in enumerate(zip(images[:5], cols)):
                with col:
                    st.image(img, caption=f"Image {idx+1}", use_column_width=True, clamp=True)

            if len(images) > 5:
                st.info(f"Showing first 5 of {len(images)} images")

    else:
        st.info("👆 Upload EL images to begin analysis")


# ==================== TAB 2: PREPROCESSING ====================
def render_preprocessing_tab():
    """Render preprocessing tab"""
    st.markdown('<p class="main-header">🔧 Preprocessing</p>', unsafe_allow_html=True)

    if not st.session_state.uploaded_images:
        st.warning("⚠️ Please upload images in the Upload tab first")
        return

    st.write(f"**{len(st.session_state.uploaded_images)} image(s) ready for preprocessing**")

    # Preprocessing options
    st.markdown("### Preprocessing Options")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Geometric Corrections")
        enable_perspective = st.checkbox("Perspective Correction", value=True)
        enable_distortion = st.checkbox("Barrel Distortion Correction", value=True)

        st.markdown("#### Enhancement")
        enable_clahe = st.checkbox("CLAHE Enhancement", value=True)
        enable_vignette = st.checkbox("Vignette Compensation", value=True)

    with col2:
        st.markdown("#### Noise Reduction")
        enable_denoise = st.checkbox("Non-Local Means Denoising", value=True)
        enable_homomorphic = st.checkbox("Homomorphic Filtering", value=True)

        st.markdown("#### ROI")
        enable_crop = st.checkbox("Auto-Crop ROI", value=True)

    # Update config
    config = st.session_state.preprocessing_config
    config.enable_perspective_correction = enable_perspective
    config.enable_distortion_correction = enable_distortion
    config.enable_clahe = enable_clahe
    config.enable_vignette_compensation = enable_vignette
    config.enable_denoising = enable_denoise
    config.enable_homomorphic = enable_homomorphic
    config.enable_auto_crop = enable_crop

    # Preview single image
    st.markdown("---")
    st.markdown("### Preview (First Image)")

    if st.button("🔍 Generate Preview", type="primary"):
        with st.spinner("Processing preview..."):
            try:
                preprocessor = AdvancedPreprocessor(config)
                result = preprocessor.process(st.session_state.uploaded_images[0])

                # Display comparison
                display_image_comparison(
                    st.session_state.uploaded_images[0],
                    result.processed_image,
                    "Preprocessing Preview"
                )

                # Show processing info
                st.markdown("#### Processing Details")
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("Processing Time", f"{result.processing_time_ms:.0f}ms")
                with col2:
                    st.metric("Operations Applied", len(result.applied_operations))
                with col3:
                    if result.roi_bounds:
                        st.metric("ROI Detected", "Yes")
                    else:
                        st.metric("ROI Detected", "No")

                st.write("**Applied Operations:**")
                st.write(", ".join(result.applied_operations))

            except Exception as e:
                st.error(f"❌ Preprocessing failed: {str(e)}")
                logger.error(traceback.format_exc())


# ==================== TAB 3: CAMERA CALIBRATION ====================
def render_camera_calibration_tab():
    """Render camera calibration tab"""
    st.markdown('<p class="main-header">📷 Camera Configuration</p>', unsafe_allow_html=True)

    st.markdown("### Camera Settings")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Camera Info")
        camera_make = st.text_input("Camera Make", value=st.session_state.camera_settings.camera_make)
        camera_model = st.text_input("Camera Model", value=st.session_state.camera_settings.camera_model)

        st.markdown("#### Exposure Settings")
        exposure_time = st.number_input(
            "Exposure Time (ms)",
            min_value=1.0,
            max_value=30000.0,
            value=float(st.session_state.camera_settings.exposure_time_ms),
            step=100.0
        )
        iso = st.number_input(
            "ISO",
            min_value=100,
            max_value=12800,
            value=st.session_state.camera_settings.iso,
            step=100
        )

        aperture = st.number_input(
            "Aperture (f-number)",
            min_value=1.0,
            max_value=32.0,
            value=float(st.session_state.camera_settings.aperture_fstop),
            step=0.1
        )

    with col2:
        st.markdown("#### Test Conditions (IEC 60904-14)")

        current_recipe = st.selectbox(
            "Current Recipe",
            options=[e.value for e in CurrentRecipe],
            index=1  # Default to 0.1*Isc
        )

        isc_current = st.number_input(
            "Isc Current (A)",
            min_value=0.1,
            max_value=20.0,
            value=9.0,
            step=0.1
        )

        voc_voltage = st.number_input(
            "Voc Voltage (V)",
            min_value=1.0,
            max_value=100.0,
            value=40.0,
            step=0.1
        )

        module_temp = st.number_input(
            "Module Temperature (°C)",
            min_value=-40.0,
            max_value=85.0,
            value=25.0,
            step=0.1
        )

        ambient_temp = st.number_input(
            "Ambient Temperature (°C)",
            min_value=-40.0,
            max_value=85.0,
            value=25.0,
            step=0.1
        )

    # Update session state
    st.session_state.camera_settings.camera_make = camera_make
    st.session_state.camera_settings.camera_model = camera_model
    st.session_state.camera_settings.exposure_time_ms = exposure_time
    st.session_state.camera_settings.iso = iso
    st.session_state.camera_settings.aperture_fstop = aperture

    st.session_state.test_conditions.current_recipe = current_recipe
    st.session_state.test_conditions.isc_current_a = isc_current
    st.session_state.test_conditions.voc_voltage_v = voc_voltage
    st.session_state.test_conditions.module_temp_c = module_temp
    st.session_state.test_conditions.ambient_temp_c = ambient_temp
    st.session_state.test_conditions.test_current_a = st.session_state.test_conditions.calculate_test_current()

    # Display calculated test current
    st.markdown("---")
    st.success(f"**Calculated Test Current: {st.session_state.test_conditions.test_current_a:.3f} A**")

    # Validation
    is_valid, errors = st.session_state.test_conditions.validate()
    if is_valid:
        st.markdown('<div class="success-box">✅ Test conditions are valid and IEC compliant</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="error-box">❌ Test condition errors:\n\n' + '\n'.join([f"- {e}" for e in errors]) + '</div>', unsafe_allow_html=True)


# ==================== TAB 4: BATCH PROCESSING ====================
def render_batch_processing_tab():
    """Render batch processing tab"""
    st.markdown('<p class="main-header">⚡ Batch Processing</p>', unsafe_allow_html=True)

    if not st.session_state.uploaded_images:
        st.warning("⚠️ Please upload images in the Upload tab first")
        return

    st.write(f"**Ready to process {len(st.session_state.uploaded_images)} image(s)**")

    # Processing options
    col1, col2, col3 = st.columns(3)

    with col1:
        enable_preprocessing = st.checkbox("Enable Preprocessing", value=True)
    with col2:
        enable_analytics = st.checkbox("Enable Analytics", value=True)
    with col3:
        enable_validation = st.checkbox("Enable Validation", value=True)

    auto_reject = st.checkbox("Auto-reject invalid images", value=True)

    st.markdown("---")

    # Process button
    if st.button("🚀 Start Batch Processing", type="primary", use_container_width=True):

        # Create batch processing config
        batch_config = BatchProcessingConfig(
            subscription_tier=st.session_state.subscription_tier,
            enable_preprocessing=enable_preprocessing,
            enable_analytics=enable_analytics,
            enable_validation=enable_validation,
            auto_reject_invalid=auto_reject,
            enable_progress_callbacks=True
        )

        # Initialize processor
        processor = BatchProcessor(batch_config)

        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()

        def progress_callback(current: int, total: int, message: str):
            progress = current / total
            progress_bar.progress(progress)
            status_text.text(f"{message} - {current}/{total} ({progress*100:.1f}%)")

        processor.add_progress_callback(progress_callback)

        # Process batch
        try:
            status_text.text("Initializing batch processing...")
            start_time = time.time()

            result = processor.process_batch(st.session_state.uploaded_images)

            elapsed_time = time.time() - start_time

            # Store result
            st.session_state.batch_result = result

            # Clear progress
            progress_bar.empty()
            status_text.empty()

            # Display results
            st.success(f"✅ Batch processing complete in {elapsed_time:.2f}s")

            # Summary metrics
            st.markdown("### Processing Summary")
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Total Processed", result.total_items)
            with col2:
                st.metric("Completed", result.completed, delta=None)
            with col3:
                st.metric("Failed", result.failed, delta=None)
            with col4:
                st.metric("Rejected", result.skipped, delta=None)

            # Performance metrics
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Total Time", f"{result.total_time_ms/1000:.2f}s")
            with col2:
                st.metric("Avg per Image", f"{result.avg_time_per_item_ms:.0f}ms")
            with col3:
                success_rate = (result.completed / result.total_items * 100) if result.total_items > 0 else 0
                st.metric("Success Rate", f"{success_rate:.1f}%")

            # Show errors if any
            if result.errors:
                st.markdown("### Errors")
                for error in result.errors:
                    st.error(error)

        except Exception as e:
            progress_bar.empty()
            status_text.empty()
            st.error(f"❌ Batch processing failed: {str(e)}")
            logger.error(traceback.format_exc())

    # Display previous results
    if st.session_state.batch_result:
        st.markdown("---")
        st.markdown("### Previous Results Available")
        st.info("Navigate to Analytics or Reports tab to view detailed results")


# ==================== TAB 5: ANALYTICS ====================
def render_analytics_tab():
    """Render analytics dashboard tab"""
    st.markdown('<p class="main-header">📊 Analytics Dashboard</p>', unsafe_allow_html=True)

    if not st.session_state.batch_result:
        st.warning("⚠️ Please run batch processing first")
        return

    result = st.session_state.batch_result

    # Filter valid items
    valid_items = [item for item in result.items if item.status == "completed"]

    if not valid_items:
        st.error("No successfully processed images to analyze")
        return

    st.write(f"**Analyzing {len(valid_items)} successfully processed image(s)**")

    # Collect metrics
    metrics_data = []
    for item in valid_items:
        if item.analytics_result:
            metrics_data.append({
                'Image': item.item_id,
                'Quality Score': item.analytics_result.quality_score,
                'Quality Level': item.analytics_result.quality_level,
                'SNR (dB)': item.analytics_result.snr_db,
                'Sharpness (Laplacian)': item.analytics_result.sharpness_laplacian,
                'MTF50': item.analytics_result.mtf50,
                'Variance': item.analytics_result.variance,
                'Kurtosis': item.analytics_result.kurtosis,
                'Skewness': item.analytics_result.skewness,
                'IEC Class': item.analytics_result.iec_sharpness_class,
                'IEC Compliant': item.analytics_result.iec_compliant
            })

    if not metrics_data:
        st.error("No analytics data available")
        return

    df = pd.DataFrame(metrics_data)

    # Summary statistics
    st.markdown("### Summary Statistics")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        avg_quality = df['Quality Score'].mean()
        st.metric("Avg Quality Score", f"{avg_quality:.1f}/100")

    with col2:
        avg_snr = df['SNR (dB)'].mean()
        st.metric("Avg SNR", f"{avg_snr:.1f} dB")

    with col3:
        avg_sharpness = df['Sharpness (Laplacian)'].mean()
        st.metric("Avg Sharpness", f"{avg_sharpness:.1f}")

    with col4:
        iec_pass_rate = (df['IEC Compliant'].sum() / len(df) * 100)
        st.metric("IEC Pass Rate", f"{iec_pass_rate:.1f}%")

    # Quality distribution
    st.markdown("### Quality Distribution")
    quality_counts = df['Quality Level'].value_counts()
    st.bar_chart(quality_counts)

    # IEC Sharpness Class Distribution
    st.markdown("### IEC Sharpness Classification")
    iec_counts = df['IEC Class'].value_counts()
    st.bar_chart(iec_counts)

    # Detailed metrics table
    st.markdown("### Detailed Metrics")
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True
    )

    # Image selector for detailed view
    st.markdown("---")
    st.markdown("### Detailed Image Analysis")

    selected_image = st.selectbox(
        "Select image for detailed analysis:",
        options=range(len(valid_items)),
        format_func=lambda x: f"{valid_items[x].item_id}"
    )

    if selected_image is not None:
        item = valid_items[selected_image]

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Image")
            if item.preprocessing_result:
                st.image(item.preprocessing_result.processed_image, use_column_width=True, clamp=True)
            else:
                st.image(item.image, use_column_width=True, clamp=True)

        with col2:
            if item.analytics_result:
                metrics = item.analytics_result
                st.markdown("#### Quality Metrics")
                st.write(f"**Quality Score:** {metrics.quality_score:.1f}/100")
                st.write(f"**Quality Level:** {metrics.quality_level.upper()}")
                st.write(f"**IEC Compliant:** {'✅ Yes' if metrics.iec_compliant else '❌ No'}")
                st.write(f"**IEC Sharpness Class:** {metrics.iec_sharpness_class}")

                st.markdown("#### Signal Quality")
                st.write(f"**SNR:** {metrics.snr_db:.2f} dB")
                st.write(f"**PSNR:** {metrics.psnr_db:.2f} dB")

                st.markdown("#### Sharpness")
                st.write(f"**Laplacian Variance:** {metrics.sharpness_laplacian:.2f}")
                st.write(f"**MTF50:** {metrics.mtf50:.3f}")
                st.write(f"**Edge Contrast:** {metrics.edge_contrast:.2f}")

                st.markdown("#### Statistics")
                st.write(f"**Mean Intensity:** {metrics.mean_intensity:.2f}")
                st.write(f"**Std Deviation:** {metrics.std_deviation:.2f}")
                st.write(f"**Variance:** {metrics.variance:.2f}")
                st.write(f"**Kurtosis:** {metrics.kurtosis:.2f}")
                st.write(f"**Skewness:** {metrics.skewness:.2f}")

            if item.validation_result:
                st.markdown("---")
                st.markdown("#### Validation")
                val = item.validation_result

                if val.is_valid:
                    st.success(f"✅ PASSED - Score: {val.quality_score:.1f}/100")
                else:
                    st.error(f"❌ FAILED - {val.rejection_reason}")

                # Show checks
                checks = [
                    ("Resolution", val.resolution_check),
                    ("Exposure", val.exposure_check),
                    ("Blur", val.blur_check),
                    ("Perspective", val.perspective_check),
                    ("Noise", val.noise_check)
                ]

                for check_name, passed in checks:
                    status = "✅" if passed else "❌"
                    st.write(f"{status} {check_name}")


# ==================== TAB 6: QUALITY VALIDATION ====================
def render_validation_tab():
    """Render quality validation tab"""
    st.markdown('<p class="main-header">✓ Quality Validation</p>', unsafe_allow_html=True)

    if not st.session_state.batch_result:
        st.warning("⚠️ Please run batch processing first")
        return

    result = st.session_state.batch_result

    # Summary statistics
    passed = sum(1 for item in result.items if item.validation_result and item.validation_result.is_valid)
    failed = sum(1 for item in result.items if item.validation_result and not item.validation_result.is_valid)
    not_validated = sum(1 for item in result.items if not item.validation_result)

    st.markdown("### Validation Summary")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Images", len(result.items))
    with col2:
        st.metric("Passed", passed, delta=None, delta_color="normal")
    with col3:
        st.metric("Failed", failed, delta=None, delta_color="inverse")
    with col4:
        pass_rate = (passed / len(result.items) * 100) if len(result.items) > 0 else 0
        st.metric("Pass Rate", f"{pass_rate:.1f}%")

    # Pass/Fail breakdown
    st.markdown("### Validation Results")

    # Create tabs for passed/failed
    tab1, tab2 = st.tabs(["✅ Passed Images", "❌ Failed Images"])

    with tab1:
        passed_items = [item for item in result.items if item.validation_result and item.validation_result.is_valid]

        if passed_items:
            st.success(f"{len(passed_items)} image(s) passed validation")

            for item in passed_items:
                with st.expander(f"📸 {item.item_id}"):
                    col1, col2 = st.columns([1, 2])

                    with col1:
                        if item.preprocessing_result:
                            st.image(item.preprocessing_result.processed_image, use_column_width=True, clamp=True)
                        else:
                            st.image(item.image, use_column_width=True, clamp=True)

                    with col2:
                        val = item.validation_result
                        st.write(f"**Status:** {val.status.upper()}")
                        st.write(f"**Quality Score:** {val.quality_score:.1f}/100")

                        if val.warnings:
                            st.warning("**Warnings:**")
                            for warning in val.warnings:
                                st.write(f"- {warning}")

                        if val.info:
                            st.info("**Info:**")
                            for info in val.info:
                                st.write(f"- {info}")
        else:
            st.info("No images passed validation")

    with tab2:
        failed_items = [item for item in result.items if item.validation_result and not item.validation_result.is_valid]

        if failed_items:
            st.error(f"{len(failed_items)} image(s) failed validation")

            for item in failed_items:
                with st.expander(f"📸 {item.item_id}"):
                    col1, col2 = st.columns([1, 2])

                    with col1:
                        if item.preprocessing_result:
                            st.image(item.preprocessing_result.processed_image, use_column_width=True, clamp=True)
                        else:
                            st.image(item.image, use_column_width=True, clamp=True)

                    with col2:
                        val = item.validation_result
                        st.write(f"**Status:** {val.status.upper()}")
                        st.write(f"**Quality Score:** {val.quality_score:.1f}/100")
                        st.write(f"**Rejection Reason:** {val.rejection_reason}")

                        if val.rejection_details:
                            st.write(f"**Details:** {val.rejection_details}")

                        if val.errors:
                            st.error("**Errors:**")
                            for error in val.errors:
                                st.write(f"- {error}")

                        if val.recommendations:
                            st.info("**Recommendations:**")
                            for rec in val.recommendations:
                                st.write(f"→ {rec}")
        else:
            st.success("All validated images passed!")


# ==================== TAB 7: REPORTS ====================
def render_reports_tab():
    """Render reports generation tab"""
    st.markdown('<p class="main-header">📄 Report Generation</p>', unsafe_allow_html=True)

    if not st.session_state.batch_result:
        st.warning("⚠️ Please run batch processing first")
        return

    st.markdown("### Generate Professional Reports")
    st.info("Generate comprehensive reports in Excel, Word, and PDF formats compliant with IEC and ISO standards")

    # Report configuration
    col1, col2 = st.columns(2)

    with col1:
        report_title = st.text_input(
            "Report Title",
            value="Electroluminescence Inspection Report"
        )
        company_name = st.text_input(
            "Company Name",
            value="SolarVisionAI"
        )

    with col2:
        include_raw = st.checkbox("Include Raw Images", value=True)
        include_processed = st.checkbox("Include Processed Images", value=True)
        include_stats = st.checkbox("Include Statistics", value=True)

    # Create report config
    report_config = ReportConfig(
        report_title=report_title,
        company_name=company_name,
        include_raw_images=include_raw,
        include_processed_images=include_processed,
        include_statistics=include_stats
    )

    st.markdown("---")
    st.markdown("### Generate Reports")

    col1, col2, col3 = st.columns(3)

    # Prepare data
    defect_data = create_defect_dataframe(st.session_state.batch_result)

    # Get first valid item for metrics
    valid_items = [item for item in st.session_state.batch_result.items if item.analytics_result]
    quality_metrics = valid_items[0].analytics_result if valid_items else None

    with col1:
        if st.button("📊 Generate Excel", type="primary", use_container_width=True):
            with st.spinner("Generating Excel report..."):
                try:
                    excel_gen = ExcelReportGenerator(report_config)
                    output_path = Path("EL_Report.xlsx")

                    excel_gen.generate(
                        defect_data=defect_data,
                        quality_metrics=quality_metrics,
                        test_conditions=st.session_state.test_conditions,
                        output_path=output_path
                    )

                    # Provide download
                    with open(output_path, 'rb') as f:
                        st.download_button(
                            label="⬇️ Download Excel",
                            data=f,
                            file_name=f"EL_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )

                    st.success("✅ Excel report generated successfully")

                except Exception as e:
                    st.error(f"❌ Excel generation failed: {str(e)}")
                    logger.error(traceback.format_exc())

    with col2:
        if st.button("📝 Generate Word", type="primary", use_container_width=True):
            with st.spinner("Generating Word report..."):
                try:
                    word_gen = WordReportGenerator(report_config)
                    output_path = Path("EL_Report.docx")

                    word_gen.generate(
                        defect_data=defect_data,
                        quality_metrics=quality_metrics,
                        test_conditions=st.session_state.test_conditions,
                        output_path=output_path
                    )

                    # Provide download
                    with open(output_path, 'rb') as f:
                        st.download_button(
                            label="⬇️ Download Word",
                            data=f,
                            file_name=f"EL_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.docx",
                            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                        )

                    st.success("✅ Word report generated successfully")

                except Exception as e:
                    st.error(f"❌ Word generation failed: {str(e)}")
                    logger.error(traceback.format_exc())

    with col3:
        if st.button("📄 Generate PDF", type="primary", use_container_width=True):
            with st.spinner("Generating PDF report..."):
                try:
                    pdf_gen = PDFReportGenerator(report_config)
                    output_path = Path("EL_Report.pdf")

                    pdf_gen.generate(
                        defect_data=defect_data,
                        quality_metrics=quality_metrics,
                        test_conditions=st.session_state.test_conditions,
                        output_path=output_path
                    )

                    # Provide download
                    with open(output_path, 'rb') as f:
                        st.download_button(
                            label="⬇️ Download PDF",
                            data=f,
                            file_name=f"EL_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                            mime="application/pdf"
                        )

                    st.success("✅ PDF report generated successfully")

                except Exception as e:
                    st.error(f"❌ PDF generation failed: {str(e)}")
                    logger.error(traceback.format_exc())

    st.markdown("---")

    # Generate all reports at once
    if st.button("📦 Generate All Reports", type="secondary", use_container_width=True):
        with st.spinner("Generating all reports..."):
            try:
                report_paths = generate_all_reports(
                    defect_data=defect_data,
                    quality_metrics=quality_metrics,
                    test_conditions=st.session_state.test_conditions,
                    output_dir=Path(".")
                )

                st.success(f"✅ Generated {len(report_paths)} report(s) successfully")

                for format_type, path in report_paths.items():
                    st.write(f"- {format_type.upper()}: `{path.name}`")

            except Exception as e:
                st.error(f"❌ Report generation failed: {str(e)}")
                logger.error(traceback.format_exc())


# ==================== MAIN APP ====================
def main():
    """Main application"""

    # Initialize session state
    init_session_state()

    # Render sidebar
    render_sidebar()

    # Header
    st.markdown('<p class="main-header">☀️ SolarVisionAI</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Production-Grade Electroluminescence Image Analysis Platform</p>', unsafe_allow_html=True)

    # Main tabs
    tabs = st.tabs([
        "📤 Upload",
        "🔧 Preprocessing",
        "📷 Camera Config",
        "⚡ Batch Processing",
        "📊 Analytics",
        "✓ Validation",
        "📄 Reports"
    ])

    with tabs[0]:
        render_upload_tab()

    with tabs[1]:
        render_preprocessing_tab()

    with tabs[2]:
        render_camera_calibration_tab()

    with tabs[3]:
        render_batch_processing_tab()

    with tabs[4]:
        render_analytics_tab()

    with tabs[5]:
        render_validation_tab()

    with tabs[6]:
        render_reports_tab()

    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666; font-size: 0.9rem;'>"
        "SolarVisionAI v4.0 | Standards: IEC 60904-13, IEC 60904-14, ISO/IEC 17025 | "
        f"© {datetime.now().year}"
        "</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
