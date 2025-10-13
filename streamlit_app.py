# At the top of your script
from fpdf import FPDF
import requests
import datetime
import streamlit as st
import requests
import pandas as pd
from PIL import Image
import io
import base64
from datetime import datetime
import json
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
import tempfile
import os
def generate_pdf_report(company_name, logo_url, standards, defect_list, orig_img_url, proc_img_url, obs, file_name="Defect_Report.pdf"):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    # Header
    if logo_url:
        logo_data = requests.get(logo_url).content
        with open("company_logo.png", 'wb') as f:
            f.write(logo_data)
        pdf.image("company_logo.png", x=10, y=8, w=30)
    pdf.set_font("Arial", "B", 18)
    pdf.cell(0, 10, company_name, ln=True, align='R')
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, f"Standard: {standards}", ln=True, align='R')
    pdf.ln(10)
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Solar Vision AI Defect Report", ln=True, align='C')
    pdf.set_font("Arial", size=10)
    pdf.cell(0, 10, f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
    pdf.ln(7)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "Detected Defects Index:", ln=True)
    pdf.set_font("Arial", size=10)
    for idx, d in enumerate(defect_list, 1):
        pdf.cell(0, 8, f"{idx}. {d['class']} | Confidence: {d['confidence']:.2f}", ln=True)
    pdf.ln(7)
    # Images
    try:
        orig_data = requests.get(orig_img_url).content
        with open("orig_img.png", 'wb') as f:
            f.write(orig_data)
        proc_data = requests.get(proc_img_url).content
        with open("proc_img.png", 'wb') as f:
            f.write(proc_data)
    except Exception:
        pass
    pdf.cell(0, 10, "Original EL Image:", ln=True)
    pdf.image("orig_img.png", x=30, w=150)
    pdf.ln(2)
    pdf.cell(0, 10, "Processed (Annotated) Image:", ln=True)
    pdf.image("proc_img.png", x=30, w=150)
    pdf.ln(7)
    summary = {}
    for d in defect_list:
        k = d['class']
        summary[k] = summary.get(k,0) + 1
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "Defect Count Summary:", ln=True)
    pdf.set_font("Arial", size=10)
    for k, v in summary.items():
        pdf.cell(0, 8, f"{k}: {v}", ln=True)
    pdf.ln(7)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "Observations:", ln=True)
    pdf.set_font("Arial", size=10)
    pdf.multi_cell(0, 8, obs)
    pdf.ln(10)
    pdf.set_y(-28)
    pdf.set_font("Arial", "I", 10)
    pdf.cell(0, 10, f"{company_name} | {standards}", 0, 0, 'L')
    pdf.cell(0, 10, f"Powered by SolarVisionAI ©", 0, 0, 'R')
    pdf.output(file_name)

# Page Configuration
st.set_page_config(
    page_title="SolarVisionAI - Solar Panel Defect Detection",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for elegant UI
st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        background-color: #FF4B4B;
        color: white;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        border: none;
        font-weight: 600;
    }
    .stButton>button:hover {
        background-color: #FF6B6B;
    }
    .upload-section {
        border: 2px dashed #FF4B4B;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        background-color: #F8F9FA;
    }
    .defect-box {
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 4px solid;
    }
    .high-severity { border-color: #FF4444; background-color: #FFE6E6; }
    .medium-severity { border-color: #FFA500; background-color: #FFF4E6; }
    .low-severity { border-color: #4CAF50; background-color: #E8F5E9; }
    .metric-card {
        background-color: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'results' not in st.session_state:
    st.session_state.results = None
if 'uploaded_image' not in st.session_state:
    st.session_state.uploaded_image = None
if 'company_logo' not in st.session_state:
    st.session_state.company_logo = None

# Sidebar Configuration
with st.sidebar:
    st.image("https://raw.githubusercontent.com/streamlit/streamlit/develop/examples/data/logo.png", width=100)
    st.title("⚙️ Configuration")
    
    # Company Branding
    st.subheader("🏢 Company Branding")
    company_name = st.text_input("Company Name", value="SolarVisionAI")
    company_logo_file = st.file_uploader("Upload Company Logo", type=['png', 'jpg', 'jpeg'])
    if company_logo_file:
        st.session_state.company_logo = Image.open(company_logo_file)
    
    st.divider()
    
    # API Configuration
    st.subheader("🔑 API Configuration")
    api_key = st.text_input("Roboflow API Key", type="password", help="Enter your Roboflow API key")
    workspace_id = st.text_input("Workspace ID", value="solar-panel-detection")
    project_id = st.text_input("Project ID", value="solar-defect-detection")
    model_version = st.number_input("Model Version", min_value=1, value=1)
    confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)
    
    st.divider()
    
    # Report Configuration
    st.subheader("📄 Report Settings")
    report_format = st.selectbox("Export Format", ["PDF", "Excel", "Both"])
    include_metadata = st.checkbox("Include Image Metadata", value=True)
    include_timestamp = st.checkbox("Include Timestamp", value=True)
    
    st.divider()
    
    # Admin Panel
    st.subheader("👤 Admin Panel")
    admin_mode = st.checkbox("Enable Admin Mode", value=False)
    if admin_mode:
        st.info("Admin mode enabled. Advanced features available.")
        if st.button("🔄 Reset All Settings"):
            st.session_state.clear()
            st.success("Settings reset successfully!")
        if st.button("📊 View Analytics"):
            st.info("Analytics feature coming soon!")

# Main Application
st.title(f"☀️ {company_name}")
st.markdown("### AI-Powered Solar Panel Defect Detection System")

# Defect Legend
with st.expander("📊 Defect Classification Legend", expanded=False):
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("<div class='defect-box high-severity'><b>🔴 High Severity</b><br>Immediate attention required</div>", unsafe_allow_html=True)
    with col2:
        st.markdown("<div class='defect-box medium-severity'><b>🟡 Medium Severity</b><br>Schedule maintenance</div>", unsafe_allow_html=True)
    with col3:
        st.markdown("<div class='defect-box low-severity'><b>🟢 Low Severity</b><br>Monitor regularly</div>", unsafe_allow_html=True)

st.divider()

# Image Upload Section
st.subheader("📤 Upload Solar Panel Image")
col_upload, col_info = st.columns([2, 1])

with col_upload:
    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=['jpg', 'jpeg', 'png'],
        help="Upload a clear image of the solar panel for defect detection"
    )
    
    if uploaded_file is not None:
        st.session_state.uploaded_image = Image.open(uploaded_file)
        
        # Image Quality Pre-check
        img_width, img_height = st.session_state.uploaded_image.size
        img_size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)
        
        quality_pass = True
        quality_messages = []
        
        if img_width < 640 or img_height < 640:
            quality_messages.append("⚠️ Image resolution is low. Recommend at least 640x640 pixels.")
            quality_pass = False
        
        if img_size_mb > 10:
            quality_messages.append("⚠️ Image file size is large. Processing may be slower.")
        
        if quality_pass:
            st.success("✅ Image quality check passed!")
        else:
            for msg in quality_messages:
                st.warning(msg)

with col_info:
    if st.session_state.uploaded_image:
        st.info(f"📏 Dimensions: {st.session_state.uploaded_image.size[0]}x{st.session_state.uploaded_image.size[1]}")
        st.info(f"📦 Size: {img_size_mb:.2f} MB")
        st.info(f"🎨 Mode: {st.session_state.uploaded_image.mode}")

# Display uploaded image
if st.session_state.uploaded_image:
    st.image(st.session_state.uploaded_image, caption="Uploaded Solar Panel Image", use_container_width=True)

# Analysis Button
if st.session_state.uploaded_image and api_key:
    if st.button("🔍 Analyze Defects", use_container_width=True):
        with st.spinner("🤖 AI is analyzing the image..."):
            try:
                # Convert image to base64
                buffered = io.BytesIO()
                st.session_state.uploaded_image.save(buffered, format="JPEG")
                img_str = base64.b64encode(buffered.getvalue()).decode()
                
                # Roboflow API call
                api_url = f"https://detect.roboflow.com/{project_id}/{model_version}"
                params = {
                    "api_key": api_key,
                    "confidence": confidence_threshold
                }
                
                response = requests.post(
                    api_url,
                    params=params,
                    data=img_str,
                    headers={"Content-Type": "application/x-www-form-urlencoded"}
                )
                
                if response.status_code == 200:
                    st.session_state.results = response.json()
                    st.success("✅ Analysis completed successfully!")
                    if st.session_state.results:
                        st.divider()
                        st.subheader("🎯 Detection Results")
                        predictions = st.session_state.results.get('predictions', [])
                        # ...metrics and defect analysis code

                else:
                    st.error(f"❌ API Error: {response.status_code} - {response.text}")
                    
            except Exception as e:
                st.error(f"❌ Error during analysis: {str(e)}")

# Display Results
if st.session_state.results:
    st.divider()
    st.subheader("🎯 Detection Results")
# NEW: show bounding/annotated image if available
annotated_image_url = st.session_state.results.get("image", None)
if annotated_image_url:
    st.image(annotated_image_url, caption="Defect Boundaries (Roboflow Overlay)", use_container_width=True)
    col1, col2, col3, col4 = st.columns(4)
    # ... rest of metrics code
    predictions = st.session_state.results.get('predictions', [])
        # Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Defects", len(predictions))
    with col2:
        high_severity = sum(1 for p in predictions if p.get('confidence', 0) > 0.8)
        st.metric("High Severity", high_severity)
    with col3:
        medium_severity = sum(1 for p in predictions if 0.5 < p.get('confidence', 0) <= 0.8)
        st.metric("Medium Severity", medium_severity)
    with col4:
        avg_confidence = sum(p.get('confidence', 0) for p in predictions) / len(predictions) if predictions else 0
        st.metric("Avg Confidence", f"{avg_confidence:.2%}")
    
    st.divider()
    
    # Detailed Results
    if predictions:
        st.subheader("📋 Detailed Defect Analysis")
        
        for idx, pred in enumerate(predictions, 1):
            confidence = pred.get('confidence', 0)
            defect_class = pred.get('class', 'Unknown')
            
            severity_class = "high-severity" if confidence > 0.8 else "medium-severity" if confidence > 0.5 else "low-severity"
            severity_label = "High" if confidence > 0.8 else "Medium" if confidence > 0.5 else "Low"
            
            st.markdown(f"""
            <div class='defect-box {severity_class}'>
                <b>Defect #{idx}: {defect_class}</b><br>
                Confidence: {confidence:.2%} | Severity: {severity_label}<br>
                Location: X={pred.get('x', 0):.0f}, Y={pred.get('y', 0):.0f}, W={pred.get('width', 0):.0f}, H={pred.get('height', 0):.0f}
            </div>
            """, unsafe_allow_html=True)
        
        # Export Options
        st.divider()
        st.subheader("📥 Export Report")
        
        col_export1, col_export2 = st.columns(2)
        
        with col_export1:
            if report_format in ["Excel", "Both"]:
                # Create DataFrame
                df = pd.DataFrame(predictions)
                
                # Add metadata
                if include_metadata:
                    df['Analysis_Time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    df['Company'] = company_name
                
                # Convert to Excel
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    df.to_excel(writer, index=False, sheet_name='Defect Analysis')
                excel_data = output.getvalue()
                
                st.download_button(
                    label="📊 Download Excel Report",
                    data=excel_data,
                    file_name=f"solar_defect_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
        
        with col_export2:
            if report_format in ["PDF", "Both"]:
                st.info("📄 PDF report generation available. Click to generate.")
                if st.button("Generate PDF Report"):
                    # Add this block inside the button handler:
                    defects = st.session_state.results.get("predictions", [])
                    annotated = st.session_state.results.get("image", "")
                    original = uploaded_image_url  # Change this to your raw uploaded image variable
                    obs = "Auto-analysis complete. Please see defect summary above."
                    generate_pdf_report(
                    company_name="Anantah Energies Pvt Ltd",
                    logo_url="https://static.yoursite.com/logo.png",  # Change to your logo URL or file path
                    standards="IEC 60891, IS 14286",
                    defect_list=defects,
                    orig_img_url=original,
                    proc_img_url=annotated,
                    obs=obs,
                    file_name="SolarVisionAI-Defect-Report.pdf"
                 )
                with open("SolarVisionAI-Defect-Report.pdf", "rb") as f:
                    st.download_button("📄 Download PDF Report", f.read(), file_name="SolarVisionAI-Defect-Report.pdf")
                st.success("✅ PDF professional report generated with branding!")

                    st.info("PDF generation in progress... (Feature ready for extension)")
    else:
        st.info("No defects detected in the image. The solar panel appears to be in good condition.")

else:
    if not api_key:
        st.warning("⚠️ Please enter your Roboflow API key in the sidebar to begin analysis.")
    else:
        st.info("👆 Upload an image to start detecting defects.")

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem;'>
    <p>🌟 Powered by SolarVisionAI | Built with Streamlit & Roboflow</p>
    <p>For support and feedback: <a href='mailto:support@solarvisionai.com'>support@solarvisionai.com</a></p>
</div>
""", unsafe_allow_html=True)
