"""
SolarVisionAI Professional - Enterprise EL Inspection Platform v3.0
Standards: IEC TS 60904-13, IEC 60904-14, MBJ PV-Module Criteria Rev 5.0
"""

from fpdf import FPDF
import streamlit as st
import pandas as pd
import requests
from PIL import Image
import io
import base64
from datetime import datetime
import json
import os
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

# Plotly imports with fallback
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# ═══════════════════════════════════════════════════════════════════
# CONFIGURATION & DATA MODELS
# ═══════════════════════════════════════════════════════════════════

class UserTier(Enum):
    """User subscription tiers"""
    BASIC = {"name": "Basic", "single_limit": 1, "total_limit": 5, "batch": False}
    PRO = {"name": "Pro", "single_limit": 10, "total_limit": 50, "batch": True}
    ADVANCED = {"name": "Advanced", "single_limit": 100, "total_limit": 500, "batch": True}
    ENTERPRISE = {"name": "Enterprise", "single_limit": -1, "total_limit": -1, "batch": True}


class DefectClassification(Enum):
    """MBJ-based defect classification"""
    STANDARD = "Standard"
    NON_STANDARD = "Non-Standard"


@dataclass
class ModuleConfig:
    """Enhanced solar module configuration"""
    rows: int = 6
    cols: int = 10
    cell_width_mm: float = 156.0
    cell_height_mm: float = 156.0
    rated_power_w: float = 300.0
    cell_type: str = "Monocrystalline"
    busbar_count: int = 6
    
    def get_cell_position(self, x: float, y: float, img_width: int, img_height: int) -> str:
        """Convert pixel coordinates to cell position"""
        cell_width_px = img_width / self.cols
        cell_height_px = img_height / self.rows
        
        col_idx = min(int(x / cell_width_px), self.cols - 1)
        row_idx = min(int(y / cell_height_px), self.rows - 1)
        
        col_letter = chr(65 + col_idx) if col_idx < 26 else f"A{chr(65 + col_idx - 26)}"
        row_number = row_idx + 1
        
        return f"{col_letter}{row_number}"


@dataclass
class TestConditions:
    """IEC 60904-14 Test Conditions"""
    test_date: datetime = field(default_factory=datetime.now)
    isc_current_a: float = 0.0
    voc_voltage_v: float = 0.0
    test_current_a: float = 0.0
    ambient_temp_c: float = 25.0
    module_temp_c: float = 25.0
    power_supply_make: str = ""
    power_supply_model: str = ""
    cable_length_m: float = 0.0
    camera_make: str = ""
    camera_model: str = ""
    exposure_time_ms: float = 0.0
    iso: int = 0
    aperture: str = ""
    focal_length_mm: float = 0.0
    resolution: str = ""
    
    def to_dict(self) -> Dict:
        return {
            "Test Date": self.test_date.strftime("%Y-%m-%d %H:%M:%S"),
            "Isc (A)": self.isc_current_a,
            "Voc (V)": self.voc_voltage_v,
            "Test Current (A)": self.test_current_a,
            "Ambient Temp (C)": self.ambient_temp_c,
            "Module Temp (C)": self.module_temp_c,
            "Power Supply": f"{self.power_supply_make} {self.power_supply_model}",
            "Cable Length (m)": self.cable_length_m,
            "Camera": f"{self.camera_make} {self.camera_model}",
            "Exposure (ms)": self.exposure_time_ms,
            "ISO": self.iso,
            "Resolution": self.resolution
        }


@dataclass
class DefectImpactAnalysis:
    """Comprehensive defect impact"""
    area_loss_percent: float
    power_loss_w: float
    power_loss_percent: float
    performance_impact: str
    safety_risk: str
    reliability_impact: str
    financial_impact_usd: float
    mbj_classification: DefectClassification
    defect_grade: str


# ═══════════════════════════════════════════════════════════════════
# SESSION STATE
# ═══════════════════════════════════════════════════════════════════

def initialize_session_state():
    """Initialize session state"""
    defaults = {
        'results': None,
        'uploaded_images': [],
        'test_conditions': TestConditions(),
        'user_tier': UserTier.BASIC,
        'upload_count': 0,
        'api_diagnostics': None
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# ═══════════════════════════════════════════════════════════════════
# DEFECT ANALYSIS ENGINE
# ═══════════════════════════════════════════════════════════════════

class EnhancedDefectAnalyzer:
    """MBJ-compliant defect analyzer"""
    
    STANDARD_DEFECTS = [
        'crystal_dislocations', 'edge_wafer', 'striation_rings',
        'belt_marks', 'reduced_lifetime_cast_silicon'
    ]
    
    NON_STANDARD_DEFECTS = [
        'micro_crack', 'crack', 'branch_crack', 'dendritic_crack',
        'hotspot', 'isolated_area', 'dead_cell', 'finger_interruption',
        'shunted_cell', 'pid_effect', 'busbar_crack'
    ]
    
    def __init__(self, module_config: ModuleConfig):
        self.module_config = module_config
        
        self.severity_thresholds = {
            'micro_crack': {'low': 0.6, 'medium': 0.75, 'high': 0.85},
            'crack': {'low': 0.7, 'medium': 0.8, 'high': 0.9},
            'branch_crack': {'low': 0.7, 'medium': 0.82, 'high': 0.92},
            'hotspot': {'low': 0.65, 'medium': 0.78, 'high': 0.88},
            'isolated_area': {'low': 0.6, 'medium': 0.75, 'high': 0.85},
        }
    
    def analyze_defects(self, predictions: List[Dict], img_width: int, img_height: int) -> pd.DataFrame:
        """Analyze defects with MBJ classification"""
        if not predictions:
            return pd.DataFrame()
        
        analysis_data = []
        
        for idx, pred in enumerate(predictions, 1):
            defect_type = pred.get('class', 'Unknown').lower().replace(' ', '_')
            confidence = pred.get('confidence', 0)
            x, y = pred.get('x', 0), pred.get('y', 0)
            width, height = pred.get('width', 0), pred.get('height', 0)
            
            cell_position = self.module_config.get_cell_position(x, y, img_width, img_height)
            
            mbj_class = (DefectClassification.STANDARD if defect_type in self.STANDARD_DEFECTS 
                        else DefectClassification.NON_STANDARD)
            
            severity = self._calculate_severity(defect_type, confidence)
            impact = self._calculate_impact(defect_type, width, height, img_width, img_height, confidence)
            grade = self._mbj_grading(impact.power_loss_percent, mbj_class)
            
            analysis_data.append({
                'Defect_ID': f"D{idx:03d}",
                'Defect_Type': defect_type.replace('_', ' ').title(),
                'Cell_Location': cell_position,
                'Confidence_%': round(confidence * 100, 2),
                'Severity': severity,
                'MBJ_Classification': mbj_class.value,
                'MBJ_Grade': grade,
                'Area_Loss_%': round(impact.area_loss_percent, 2),
                'Power_Loss_W': round(impact.power_loss_w, 2),
                'Power_Loss_%': round(impact.power_loss_percent, 2),
                'Performance_Impact': impact.performance_impact,
                'Safety_Risk': impact.safety_risk,
                'Reliability_Impact': impact.reliability_impact,
                'Financial_Impact_USD': round(impact.financial_impact_usd, 2),
                'X': int(x), 'Y': int(y), 'Width': int(width), 'Height': int(height)
            })
        
        return pd.DataFrame(analysis_data)
    
    def _calculate_severity(self, defect_type: str, confidence: float) -> str:
        """Calculate severity"""
        thresholds = self.severity_thresholds.get(
            defect_type, {'low': 0.6, 'medium': 0.75, 'high': 0.85}
        )
        
        if confidence >= thresholds['high']:
            return 'Critical'
        elif confidence >= thresholds['medium']:
            return 'High'
        elif confidence >= thresholds['low']:
            return 'Medium'
        return 'Low'
    
    def _calculate_impact(self, defect_type: str, width: float, height: float,
                         img_width: int, img_height: int, confidence: float) -> DefectImpactAnalysis:
        """Calculate impact"""
        defect_area = width * height
        total_area = img_width * img_height
        area_loss_percent = (defect_area / total_area) * 100
        
        severity_multipliers = {
            'micro_crack': 0.5, 'crack': 1.2, 'branch_crack': 2.0,
            'hotspot': 2.5, 'isolated_area': 1.8, 'dead_cell': 5.0
        }
        
        multiplier = severity_multipliers.get(defect_type, 1.0)
        bb_factor = 1.0 if self.module_config.busbar_count <= 6 else 0.6
        
        base_power_loss = area_loss_percent * multiplier * confidence * bb_factor
        power_loss_w = (base_power_loss / 100) * self.module_config.rated_power_w
        power_loss_percent = (power_loss_w / self.module_config.rated_power_w) * 100
        
        if power_loss_percent >= 15:
            performance = 'Critical'
        elif power_loss_percent >= 8:
            performance = 'High'
        elif power_loss_percent >= 3:
            performance = 'Medium'
        else:
            performance = 'Low'
        
        if defect_type in ['hotspot'] and confidence > 0.8:
            safety = 'High - Fire Risk'
        elif defect_type in ['crack', 'branch_crack'] and confidence > 0.85:
            safety = 'Medium - Structural'
        else:
            safety = 'Low - Monitor'
        
        if power_loss_percent >= 10:
            reliability = 'High - Replace'
        elif power_loss_percent >= 5:
            reliability = 'Medium - Repair'
        else:
            reliability = 'Low - Monitor'
        
        annual_kwh_loss = (power_loss_w * 4 * 365) / 1000
        financial = annual_kwh_loss * 25 * 0.10
        
        mbj_class = (DefectClassification.STANDARD if defect_type in self.STANDARD_DEFECTS 
                    else DefectClassification.NON_STANDARD)
        
        grade = self._mbj_grading(power_loss_percent, mbj_class)
        
        return DefectImpactAnalysis(
            area_loss_percent=area_loss_percent,
            power_loss_w=power_loss_w,
            power_loss_percent=power_loss_percent,
            performance_impact=performance,
            safety_risk=safety,
            reliability_impact=reliability,
            financial_impact_usd=financial,
            mbj_classification=mbj_class,
            defect_grade=grade
        )
    
    def _mbj_grading(self, power_loss_percent: float, classification: DefectClassification) -> str:
        """MBJ grading"""
        if classification == DefectClassification.STANDARD:
            return 'A'
        
        if power_loss_percent >= 20:
            return 'D'
        elif power_loss_percent >= 10:
            return 'C'
        elif power_loss_percent >= 5:
            return 'B'
        return 'A'
    
    def create_cell_heatmap(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create cell heatmap"""
        matrix = pd.DataFrame(
            0,
            index=range(1, self.module_config.rows + 1),
            columns=[chr(65 + i) for i in range(min(self.module_config.cols, 26))]
        )
        
        if not df.empty:
            for _, row in df.iterrows():
                cell = row['Cell_Location']
                if len(cell) >= 2:
                    col = cell[0]
                    try:
                        row_num = int(cell[1:])
                        if col in matrix.columns and row_num in matrix.index:
                            matrix.at[row_num, col] += 1
                    except (ValueError, KeyError):
                        continue
        
        return matrix


# ═══════════════════════════════════════════════════════════════════
# ROBOFLOW API
# ═══════════════════════════════════════════════════════════════════

class EnhancedRoboflowAPI:
    """Roboflow API with error diagnostics"""
    
    @staticmethod
    def analyze_image(image: Image.Image, api_key: str, workspace: str,
                     project: str, version: int, confidence: float = 0.5) -> Optional[Dict]:
        """Call Roboflow with error handling"""
        
        # Build correct API URL
        # Roboflow format: https://detect.roboflow.com/PROJECT_ID/VERSION
        # PROJECT_ID can be either "project-name" or "workspace/project-name"
        
        if "/" in project:
            # User provided full path like "workspace/project"
            project_id = project
        else:
            # User provided just project name
            project_id = f"{workspace}/{project}" if workspace else project
        
        api_url = f"https://detect.roboflow.com/{project_id}/{version}"
        
        diagnostics = {
            'api_key_valid': bool(api_key and len(api_key) > 10),
            'workspace': workspace,
            'project': project,
            'project_id_used': project_id,
            'version': version,
            'image_size': image.size,
            'request_url': api_url,
            'error': None,
            'status_code': None
        }
        
        try:
            buffered = io.BytesIO()
            image.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            
            params = {
                "api_key": api_key,
                "confidence": int(confidence * 100),
                "overlap": 30
            }
            
            response = requests.post(
                api_url,
                params=params,
                data=img_str,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                timeout=30
            )
            
            diagnostics['status_code'] = response.status_code
            
            if response.status_code == 403:
                diagnostics['error'] = "403 Forbidden"
                diagnostics['response_text'] = response.text
                st.session_state.api_diagnostics = diagnostics
                
                st.error("🚫 403 Forbidden - Check API key permissions")
                with st.expander("🔍 Diagnostics & Solutions"):
                    st.markdown("""
                    **Possible Issues:**
                    1. ❌ Invalid API Key
                    2. ❌ Wrong Project ID format
                    3. ❌ API access disabled
                    
                    **Common Fixes:**
                    - Use FULL project ID (e.g., `el-images-trbib-gcqce` not just `1`)
                    - Check your Roboflow project URL
                    - Verify API key has not expired
                    """)
                    st.json(diagnostics)
                
                return None
            
            elif response.status_code == 200:
                result = response.json()
                
                # Get annotated image
                viz_params = {
                    "api_key": api_key,
                    "confidence": int(confidence * 100),
                    "labels": "on",
                    "stroke": 3
                }
                
                viz_response = requests.post(
                    api_url,
                    params=viz_params,
                    data=img_str,
                    headers={"Content-Type": "application/x-www-form-urlencoded"},
                    timeout=30
                )
                
                if viz_response.status_code == 200:
                    result['annotated_image_base64'] = base64.b64encode(viz_response.content).decode()
                
                diagnostics['success'] = True
                st.session_state.api_diagnostics = diagnostics
                return result
            
            else:
                diagnostics['error'] = f"HTTP {response.status_code}"
                st.session_state.api_diagnostics = diagnostics
                st.error(f"❌ Error: {response.status_code} - {response.text}")
                return None
                
        except requests.exceptions.Timeout:
            st.error("⏱️ Request timed out")
            return None
        except Exception as e:
            diagnostics['error'] = str(e)
            st.session_state.api_diagnostics = diagnostics
            st.error(f"❌ Error: {str(e)}")
            return None
# ═══════════════════════════════════════════════════════════════════
# PDF REPORT
# ═══════════════════════════════════════════════════════════════════

class ELTemplatePDFReport:
    """EL template-based PDF report"""
    
    @staticmethod
    def generate_report(company_name: str, df: pd.DataFrame,
                       original_image: Image.Image, annotated_base64: Optional[str],
                       module_config: ModuleConfig, test_conditions: TestConditions,
                       analysis_level: str) -> str:
        """Generate PDF report"""
        
        pdf = FPDF()
        pdf.add_page()
        pdf.set_auto_page_break(auto=True, margin=15)
        
        # Header
        pdf.set_font("Arial", "B", 18)
        pdf.cell(0, 12, company_name, ln=True, align='C')
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, "ELECTROLUMINESCENCE INSPECTION REPORT", ln=True, align='C')
        pdf.set_font("Arial", "I", 9)
        pdf.cell(0, 6, "IEC TS 60904-13 | IEC 60904-14 | MBJ Rev 5.0", ln=True, align='C')
        pdf.ln(8)
        
        # General Info
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 8, "GENERAL INFORMATION", ln=True)
        pdf.set_font("Arial", "", 9)
        
        info_data = [
            ["Job No.", f"EL-{datetime.now().strftime('%Y%m%d-%H%M')}"],
            ["Test Date", test_conditions.test_date.strftime("%Y-%m-%d %H:%M:%S")],
            ["Module Type", module_config.cell_type],
            ["Rated Power", f"{module_config.rated_power_w}W"],
            ["Cell Config", f"{module_config.rows}x{module_config.cols}"],
            ["Busbar Count", f"{module_config.busbar_count}BB"]
        ]
        
        for row in info_data:
            pdf.set_font("Arial", "B", 9)
            pdf.cell(60, 6, row[0], 1)
            pdf.set_font("Arial", "", 9)
            pdf.cell(130, 6, str(row[1]), 1)
            pdf.ln()
        
        pdf.ln(5)
        
        # Executive Summary
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 8, "EXECUTIVE SUMMARY", ln=True)
        pdf.set_font("Arial", "", 9)
        
        if not df.empty:
            total_defects = len(df)
            non_standard = len(df[df['MBJ_Classification'] == 'Non-Standard'])
            critical = len(df[df['Severity'] == 'Critical'])
            total_power_loss = df['Power_Loss_W'].sum()
            
            summary_text = (
                f"Total Defects: {total_defects}\n"
                f"Non-Standard: {non_standard}\n"
                f"Critical: {critical}\n"
                f"Power Loss: {total_power_loss:.2f}W\n"
            )
            
            pdf.multi_cell(0, 5, summary_text)
        else:
            pdf.cell(0, 6, "No defects detected", ln=True)
        
        # Images
        pdf.add_page()
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 8, "VISUAL ANALYSIS", ln=True)
        
        try:
            orig_path = "temp_orig.png"
            original_image.save(orig_path)
            pdf.set_font("Arial", "B", 9)
            pdf.cell(0, 6, "Original EL Image:", ln=True)
            pdf.image(orig_path, x=10, w=190)
            
            if annotated_base64:
                ann_path = "temp_ann.png"
                with open(ann_path, 'wb') as f:
                    f.write(base64.b64decode(annotated_base64))
                pdf.cell(0, 6, "Annotated Image:", ln=True)
                pdf.image(ann_path, x=10, w=190)
                if os.path.exists(ann_path):
                    os.remove(ann_path)
            
            if os.path.exists(orig_path):
                os.remove(orig_path)
        except Exception as e:
            pdf.cell(0, 6, f"Error: {str(e)}", ln=True)
        
        filename = f"EL_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        pdf.output(filename)
        return filename


# ═══════════════════════════════════════════════════════════════════
# USER INTERFACE
# ═══════════════════════════════════════════════════════════════════

def render_ui():
    """Main UI"""
    
    initialize_session_state()
    
    st.set_page_config(
        page_title="SolarVisionAI - EL Inspection",
        page_icon="☀️",
        layout="wide"
    )
    
    # Sidebar
    with st.sidebar:
        st.markdown("# Configuration")
        
        # User Tier
        tier_names = [t.value['name'] for t in UserTier]
        selected = st.selectbox("Plan", tier_names)
        st.session_state.user_tier = next(t for t in UserTier if t.value['name'] == selected)
        
        st.divider()
        
        # Company
        company_name = st.text_input("Company", "SolarVisionAI")
        
        st.divider()
        
        # API
        api_key = st.text_input("Roboflow API Key", type="password")
        workspace = st.text_input("Workspace", "el-images-trbib-gcqce")
        project = st.text_input("Project", "1")
        version = st.number_input("Version", 1, 10, 1)
        confidence = st.slider("Confidence", 0.0, 1.0, 0.5)
        
        st.divider()
        
        # Module
        module_rows = st.number_input("Rows", 1, 26, 6)
        module_cols = st.number_input("Columns", 1, 8, 6)
        rated_power = st.number_input("Power (W)", 50, 600, 500)
        
        module_config = ModuleConfig(
            rows=module_rows,
            cols=module_cols,
            rated_power_w=float(rated_power)
        )
    
    # Main
    st.title(f"☀️ {company_name}")
    st.markdown("### Professional EL Inspection Platform")
    
    # Upload
    st.markdown("## Upload Images")
    
    uploaded_file = st.file_uploader("Drop image", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file:
        st.session_state.uploaded_images = [Image.open(uploaded_file)]
        st.image(st.session_state.uploaded_images[0], width=400)
    
    # Analyze
    if st.session_state.uploaded_images and api_key:
        if st.button("Analyze Defects", type="primary"):
            with st.spinner("Analyzing..."):
                image = st.session_state.uploaded_images[0]
                result = EnhancedRoboflowAPI.analyze_image(
                    image, api_key, workspace, project, version, confidence
                )
                
                if result:
                    st.session_state.results = [result]
                    st.session_state.results[0]['image_obj'] = image
                    st.success("Analysis complete!")
                    st.rerun()
    
    # Results
    if st.session_state.results:
        st.divider()
        st.markdown("# Results")
        
        result = st.session_state.results[0]
        predictions = result.get('predictions', [])
        image = result['image_obj']
        annotated = result.get('annotated_image_base64')
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Original", use_container_width=True)
        with col2:
            if annotated:
                ann_img = Image.open(io.BytesIO(base64.b64decode(annotated)))
                st.image(ann_img, caption="Annotated", use_container_width=True)
        
        if predictions:
            analyzer = EnhancedDefectAnalyzer(module_config)
            df = analyzer.analyze_defects(predictions, image.size[0], image.size[1])
            
            st.markdown("### Metrics")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Defects", len(df))
            with col2:
                critical = len(df[df['Severity'] == 'Critical'])
                st.metric("Critical", critical)
            with col3:
                loss = df['Power_Loss_W'].sum()
                st.metric("Power Loss", f"{loss:.1f}W")
            
            st.markdown("### Defect Table")
            st.dataframe(df, use_container_width=True)
            
            # Export
            st.markdown("### Export")
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df.to_excel(writer, index=False)
            
            st.download_button(
                "Download Excel",
                output.getvalue(),
                f"analysis_{datetime.now().strftime('%Y%m%d')}.xlsx"
            )


if __name__ == "__main__":
    render_ui()
