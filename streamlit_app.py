"""
SolarVisionAI - Enterprise Solar Panel Defect Detection & Analysis Platform
Version: 2.0.0
Standards: IEC TS 60904-13, IEA PVPS, MBJ Criteria
Author: Experienced PV  Expert (11+ years)
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
import tempfile
import os
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

# ═══════════════════════════════════════════════════════════════════
# CONFIGURATION & DATA MODELS
# ═══════════════════════════════════════════════════════════════════

class UserTier(Enum):
    """User subscription tiers with upload limits"""
    BASIC = {"name": "Basic", "single_limit": 1, "total_limit": 5, "batch": False}
    PRO = {"name": "Pro", "single_limit": 10, "total_limit": 50, "batch": True}
    ADVANCED = {"name": "Advanced", "single_limit": 100, "total_limit": 500, "batch": True}
    ENTERPRISE = {"name": "Enterprise", "single_limit": -1, "total_limit": -1, "batch": True}


@dataclass
class ModuleConfig:
    """Solar module configuration"""
    rows: int = 6  # Default 6x10 = 60 cells
    cols: int = 10
    cell_width_mm: float = 156.0  # Standard cell size
    cell_height_mm: float = 156.0
    rated_power_w: float = 300.0
    
    def get_cell_position(self, x: float, y: float, img_width: int, img_height: int) -> str:
        """Convert pixel coordinates to cell position (e.g., 'A1', 'B5')"""
        # Calculate cell dimensions in pixels
        cell_width_px = img_width / self.cols
        cell_height_px = img_height / self.rows
        
        # Get cell indices
        col_idx = int(x / cell_width_px)
        row_idx = int(y / cell_height_px)
        
        # Convert to Excel-style notation
        col_letter = chr(65 + min(col_idx, 25))  # A-Z
        row_number = row_idx + 1
        
        return f"{col_letter}{row_number}"


@dataclass
class DefectImpactAnalysis:
    """Defect impact assessment"""
    area_loss_percent: float
    power_loss_w: float
    power_loss_percent: float
    performance_impact: str  # Low, Medium, High, Critical
    safety_risk: str
    reliability_impact: str
    financial_impact_usd: float


# ═══════════════════════════════════════════════════════════════════
# SESSION STATE INITIALIZATION
# ═══════════════════════════════════════════════════════════════════

def initialize_session_state():
    """Initialize all session state variables"""
    if 'results' not in st.session_state:
        st.session_state.results = None
    if 'uploaded_images' not in st.session_state:
        st.session_state.uploaded_images = []
    if 'analysis_history' not in st.session_state:
        st.session_state.analysis_history = []
    if 'company_logo' not in st.session_state:
        st.session_state.company_logo = None
    if 'user_tier' not in st.session_state:
        st.session_state.user_tier = UserTier.BASIC
    if 'upload_count' not in st.session_state:
        st.session_state.upload_count = 0


# ═══════════════════════════════════════════════════════════════════
# MODULE 1: DEFECT ANALYSIS ENGINE
# ═══════════════════════════════════════════════════════════════════

class DefectAnalyzer:
    """Core defect analysis and classification engine"""
    
    def __init__(self, module_config: ModuleConfig):
        self.module_config = module_config
        
        # IEC TS 60904-13 defect classification
        self.defect_severity_thresholds = {
            'micro_crack': {'low': 0.6, 'medium': 0.75, 'high': 0.85},
            'crack': {'low': 0.7, 'medium': 0.8, 'high': 0.9},
            'hotspot': {'low': 0.65, 'medium': 0.78, 'high': 0.88},
            'isolated_area': {'low': 0.6, 'medium': 0.75, 'high': 0.85},
            'branch_crack': {'low': 0.7, 'medium': 0.82, 'high': 0.92},
        }
    
    def analyze_defects(self, predictions: List[Dict], img_width: int, img_height: int) -> pd.DataFrame:
        """
        Analyze defects and create structured DataFrame
        
        Returns DataFrame with columns:
        - Defect ID, Type, Cell Location, Confidence, Severity
        - Area Loss %, Power Loss (W), Performance Impact
        - Safety Risk, Reliability Impact, Financial Impact ($)
        """
        if not predictions:
            return pd.DataFrame()
        
        analysis_data = []
        
        for idx, pred in enumerate(predictions, 1):
            defect_type = pred.get('class', 'Unknown').lower().replace(' ', '_')
            confidence = pred.get('confidence', 0)
            x, y = pred.get('x', 0), pred.get('y', 0)
            width, height = pred.get('width', 0), pred.get('height', 0)
            
            # Get cell location
            cell_position = self.module_config.get_cell_position(x, y, img_width, img_height)
            
            # Calculate severity
            severity = self._calculate_severity(defect_type, confidence)
            
            # Calculate impact
            impact = self._calculate_impact(defect_type, width, height, img_width, img_height, confidence)
            
            analysis_data.append({
                'Defect_ID': f"D{idx:03d}",
                'Defect_Type': defect_type.replace('_', ' ').title(),
                'Cell_Location': cell_position,
                'Confidence_%': round(confidence * 100, 2),
                'Severity': severity,
                'Area_Loss_%': round(impact.area_loss_percent, 2),
                'Power_Loss_W': round(impact.power_loss_w, 2),
                'Power_Loss_%': round(impact.power_loss_percent, 2),
                'Performance_Impact': impact.performance_impact,
                'Safety_Risk': impact.safety_risk,
                'Reliability_Impact': impact.reliability_impact,
                'Financial_Impact_USD': round(impact.financial_impact_usd, 2),
                'X': int(x),
                'Y': int(y),
                'Width': int(width),
                'Height': int(height),
            })
        
        return pd.DataFrame(analysis_data)
    
    def _calculate_severity(self, defect_type: str, confidence: float) -> str:
        """Calculate severity based on IEC standards"""
        thresholds = self.defect_severity_thresholds.get(defect_type, 
                                                         {'low': 0.6, 'medium': 0.75, 'high': 0.85})
        
        if confidence >= thresholds['high']:
            return 'Critical'
        elif confidence >= thresholds['medium']:
            return 'High'
        elif confidence >= thresholds['low']:
            return 'Medium'
        else:
            return 'Low'
    
    def _calculate_impact(self, defect_type: str, width: float, height: float, 
                         img_width: int, img_height: int, confidence: float) -> DefectImpactAnalysis:
        """
        Calculate defect impact based on IEA PVPS and MBJ criteria
        References: IEC TS 60904-13, IEA PVPS Task 13
        """
        # Calculate area loss
        defect_area = width * height
        total_area = img_width * img_height
        area_loss_percent = (defect_area / total_area) * 100
        
        # Power loss estimation (based on literature)
        # Crack severity multipliers from IEA PVPS studies
        severity_multipliers = {
            'micro_crack': 0.5,
            'crack': 1.2,
            'hotspot': 2.5,
            'isolated_area': 1.8,
            'branch_crack': 2.0,
        }
        
        multiplier = severity_multipliers.get(defect_type, 1.0)
        base_power_loss = area_loss_percent * multiplier * confidence
        power_loss_w = (base_power_loss / 100) * self.module_config.rated_power_w
        power_loss_percent = (power_loss_w / self.module_config.rated_power_w) * 100
        
        # Performance impact classification
        if power_loss_percent >= 15:
            performance = 'Critical'
        elif power_loss_percent >= 8:
            performance = 'High'
        elif power_loss_percent >= 3:
            performance = 'Medium'
        else:
            performance = 'Low'
        
        # Safety risk (based on defect type and severity)
        if defect_type in ['hotspot'] and confidence > 0.8:
            safety = 'High - Fire Risk'
        elif defect_type in ['crack', 'branch_crack'] and confidence > 0.85:
            safety = 'Medium - Structural'
        else:
            safety = 'Low - Monitor'
        
        # Reliability impact
        if power_loss_percent >= 10:
            reliability = 'High - Replace'
        elif power_loss_percent >= 5:
            reliability = 'Medium - Repair'
        else:
            reliability = 'Low - Monitor'
        
        # Financial impact (25 year lifecycle, $0.10/kWh)
        annual_kwh_loss = (power_loss_w * 4 * 365) / 1000  # Assuming 4 peak sun hours
        lifecycle_kwh_loss = annual_kwh_loss * 25
        financial_impact = lifecycle_kwh_loss * 0.10  # $0.10 per kWh
        
        return DefectImpactAnalysis(
            area_loss_percent=area_loss_percent,
            power_loss_w=power_loss_w,
            power_loss_percent=power_loss_percent,
            performance_impact=performance,
            safety_risk=safety,
            reliability_impact=reliability,
            financial_impact_usd=financial_impact
        )
    
    def create_cell_heatmap(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create cell-level defect heatmap matrix"""
        # Create empty matrix
        matrix = pd.DataFrame(0, 
                            index=range(1, self.module_config.rows + 1),
                            columns=[chr(65 + i) for i in range(self.module_config.cols)])
        
        # Count defects per cell
        if not df.empty:
            for _, row in df.iterrows():
                cell = row['Cell_Location']
                col = cell[0]
                row_num = int(cell[1:])
                if col in matrix.columns and row_num in matrix.index:
                    matrix.at[row_num, col] += 1
        
        return matrix


# ═══════════════════════════════════════════════════════════════════
# MODULE 2: VISUALIZATION ENGINE
# ═══════════════════════════════════════════════════════════════════

class VisualizationEngine:
    """Advanced visualization for different user tiers"""
    
    @staticmethod
    def create_defect_distribution_chart(df: pd.DataFrame, user_tier: UserTier) -> go.Figure:
        """Create defect distribution visualization based on user tier"""
        if df.empty:
            return None
        
        if user_tier == UserTier.BASIC:
            # Simple bar chart for basic users
            defect_counts = df['Defect_Type'].value_counts()
            fig = px.bar(x=defect_counts.index, y=defect_counts.values,
                        labels={'x': 'Defect Type', 'y': 'Count'},
                        title='Defect Distribution',
                        color=defect_counts.values,
                        color_continuous_scale='Reds')
        else:
            # Advanced visualizations for Pro/Advanced/Enterprise
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Defect Distribution', 'Severity Analysis', 
                              'Power Loss Impact', 'Cell Location Heatmap'),
                specs=[[{'type': 'bar'}, {'type': 'pie'}],
                       [{'type': 'scatter'}, {'type': 'heatmap'}]]
            )
            
            # Defect distribution
            defect_counts = df['Defect_Type'].value_counts()
            fig.add_trace(go.Bar(x=defect_counts.index, y=defect_counts.values,
                               name='Defects', marker_color='indianred'),
                         row=1, col=1)
            
            # Severity distribution
            severity_counts = df['Severity'].value_counts()
            fig.add_trace(go.Pie(labels=severity_counts.index, values=severity_counts.values,
                               name='Severity'),
                         row=1, col=2)
            
            # Power loss scatter
            fig.add_trace(go.Scatter(x=df['Confidence_%'], y=df['Power_Loss_W'],
                                   mode='markers', name='Power Loss',
                                   marker=dict(size=10, color=df['Power_Loss_W'],
                                             colorscale='Reds', showscale=True)),
                         row=2, col=1)
            
            fig.update_layout(height=800, showlegend=False,
                            title_text="Comprehensive Defect Analysis Dashboard")
        
        return fig
    
    @staticmethod
    def create_cell_heatmap_viz(cell_matrix: pd.DataFrame) -> go.Figure:
        """Create interactive cell heatmap"""
        fig = go.Figure(data=go.Heatmap(
            z=cell_matrix.values,
            x=cell_matrix.columns,
            y=cell_matrix.index,
            colorscale='Reds',
            text=cell_matrix.values,
            texttemplate='%{text}',
            textfont={"size": 10},
            colorbar=dict(title="Defect Count")
        ))
        
        fig.update_layout(
            title='Module Cell Defect Heatmap',
            xaxis_title='Column',
            yaxis_title='Row',
            height=400
        )
        
        return fig
    
    @staticmethod
    def create_fishbone_diagram(df: pd.DataFrame) -> go.Figure:
        """
        Create Ishikawa (Fishbone) diagram for root cause analysis
        Categories: Material, Method, Machine, Measurement, Environment, Management
        """
        if df.empty:
            return None
        
        # Root cause categories based on defect types
        root_causes = {
            'Material': ['Micro Crack', 'Isolated Area'],
            'Manufacturing': ['Branch Crack', 'Crack'],
            'Installation': ['Crack', 'Branch Crack'],
            'Environmental': ['Hotspot', 'Degradation'],
            'Electrical': ['Hotspot', 'Isolated Area'],
            'Mechanical': ['Crack', 'Micro Crack']
        }
        
        # Count defects by category
        category_counts = {}
        for category, defect_types in root_causes.items():
            count = sum(df['Defect_Type'].isin(defect_types))
            category_counts[category] = count
        
        fig = go.Figure()
        
        # Main spine
        fig.add_trace(go.Scatter(x=[0, 10], y=[5, 5], mode='lines',
                               line=dict(color='black', width=3),
                               name='Effect: Module Degradation'))
        
        # Add branches
        angles = [30, 60, 120, 150, 210, 240, 300, 330]
        categories = list(category_counts.keys())
        
        for i, (category, count) in enumerate(category_counts.items()):
            if i < len(angles):
                angle = np.radians(angles[i])
                x_end = 5 + 3 * np.cos(angle)
                y_end = 5 + 3 * np.sin(angle)
                
                fig.add_trace(go.Scatter(
                    x=[5, x_end], y=[5, y_end],
                    mode='lines+text',
                    line=dict(color='red' if count > 0 else 'gray', width=2),
                    text=[None, f"{category}<br>({count})"],
                    textposition="top center",
                    name=category
                ))
        
        fig.update_layout(
            title='Root Cause Analysis - Fishbone Diagram',
            showlegend=False,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=500,
            annotations=[
                dict(x=10, y=5, text="<b>Module<br>Degradation</b>",
                    showarrow=False, font=dict(size=14, color='red'))
            ]
        )
        
        return fig
    
    @staticmethod
    def create_financial_impact_chart(df: pd.DataFrame) -> go.Figure:
        """Create financial impact visualization"""
        if df.empty:
            return None
        
        # Aggregate by defect type
        financial_by_type = df.groupby('Defect_Type')['Financial_Impact_USD'].sum().sort_values(ascending=False)
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=financial_by_type.index,
            y=financial_by_type.values,
            marker_color='crimson',
            text=[f'${val:,.0f}' for val in financial_by_type.values],
            textposition='outside'
        ))
        
        fig.update_layout(
            title='25-Year Financial Impact by Defect Type',
            xaxis_title='Defect Type',
            yaxis_title='Financial Loss (USD)',
            height=400
        )
        
        return fig


# ═══════════════════════════════════════════════════════════════════
# MODULE 3: ROBOFLOW API INTEGRATION
# ═══════════════════════════════════════════════════════════════════

class RoboflowAPI:
    """Enhanced Roboflow API integration with proper image handling"""
    
    @staticmethod
    def analyze_image(image: Image.Image, api_key: str, workspace: str, 
                     project: str, version: int, confidence: float = 0.5) -> Dict:
        """
        Call Roboflow API and get predictions with annotated image
        
        Returns:
            Dict with 'predictions' and 'image' (annotated base64 or URL)
        """
        try:
            # Convert image to base64
            buffered = io.BytesIO()
            image.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            
            # API endpoint - Use correct format
            api_url = f"https://detect.roboflow.com/{project}/{version}"
            
            params = {
                "api_key": api_key,
                "confidence": confidence,
                "overlap": 30,
                "format": "json",  # Get JSON response
                "labels": "on",    # Include labels on image
                "stroke": 3        # Bounding box thickness
            }
            
            response = requests.post(
                api_url,
                params=params,
                data=img_str,
                headers={"Content-Type": "application/x-www-form-urlencoded"}
            )
            
            if response.status_code == 200:
                result = response.json()
                
                # Get annotated image URL if available
                # Roboflow returns visualization_url or we need to request it separately
                annotated_params = params.copy()
                annotated_params['format'] = 'image'
                
                viz_response = requests.post(
                    api_url,
                    params=annotated_params,
                    data=img_str,
                    headers={"Content-Type": "application/x-www-form-urlencoded"}
                )
                
                if viz_response.status_code == 200:
                    # Convert response to base64 for display
                    annotated_base64 = base64.b64encode(viz_response.content).decode()
                    result['annotated_image_base64'] = annotated_base64
                
                return result
            else:
                st.error(f"Roboflow API Error: {response.status_code}")
                return None
                
        except Exception as e:
            st.error(f"Error calling Roboflow API: {str(e)}")
            return None


# ═══════════════════════════════════════════════════════════════════
# MODULE 4: PDF REPORT GENERATOR (ENHANCED)
# ═══════════════════════════════════════════════════════════════════

class EnhancedPDFReport:
    """Standards-compliant PDF report generator"""
    
    @staticmethod
    def generate_report(company_name: str, df: pd.DataFrame, 
                       original_image: Image.Image, annotated_image_base64: str,
                       module_config: ModuleConfig, analysis_level: str,
                       standards: List[str] = None) -> str:
        """
        Generate comprehensive PDF report
        
        Standards referenced:
        - IEC TS 60904-13: PV module electroluminescence imaging
        - IEA PVPS Task 13: Performance and reliability
        - MBJ Criteria: Module quality standards
        """
        if standards is None:
            standards = ["IEC TS 60904-13", "IEA PVPS Task 13", "MBJ Criteria"]
        
        pdf = FPDF()
        pdf.add_page()
        pdf.set_auto_page_break(auto=True, margin=15)
        
        # Header
        pdf.set_font("Arial", "B", 20)
        pdf.cell(0, 15, company_name, ln=True, align='C')
        pdf.set_font("Arial", "B", 16)
        pdf.cell(0, 10, "Solar Module EL Inspection Report", ln=True, align='C')
        pdf.set_font("Arial", "I", 10)
        pdf.cell(0, 8, f"Standards: {', '.join(standards)}", ln=True, align='C')
        pdf.ln(5)
        
        # Report metadata
        pdf.set_font("Arial", "", 10)
        pdf.cell(0, 6, f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
        pdf.cell(0, 6, f"Analysis Level: {analysis_level.upper()}", ln=True)
        pdf.cell(0, 6, f"Module Configuration: {module_config.rows}x{module_config.cols} cells", ln=True)
        pdf.cell(0, 6, f"Rated Power: {module_config.rated_power_w}W", ln=True)
        pdf.ln(8)
        
        # Executive Summary
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, "Executive Summary", ln=True)
        pdf.set_font("Arial", "", 10)
        
        if not df.empty:
            total_defects = len(df)
            critical_defects = len(df[df['Severity'] == 'Critical'])
            total_power_loss = df['Power_Loss_W'].sum()
            total_financial_impact = df['Financial_Impact_USD'].sum()
            
            pdf.multi_cell(0, 6, 
                f"Total Defects Detected: {total_defects}\n"
                f"Critical Defects: {critical_defects}\n"
                f"Total Power Loss: {total_power_loss:.2f}W ({(total_power_loss/module_config.rated_power_w)*100:.1f}%)\n"
                f"25-Year Financial Impact: ${total_financial_impact:,.2f}\n"
                f"\nRecommendation: {'IMMEDIATE ACTION REQUIRED' if critical_defects > 0 else 'Schedule maintenance'}"
            )
        else:
            pdf.cell(0, 6, "No defects detected. Module appears to be in good condition.", ln=True)
        
        pdf.ln(10)
        
        # Defect Summary Table
        if not df.empty:
            pdf.set_font("Arial", "B", 14)
            pdf.cell(0, 10, "Defect Analysis Summary", ln=True)
            pdf.set_font("Arial", "B", 9)
            
            # Table header
            col_widths = [20, 30, 20, 25, 25, 30, 30]
            headers = ['Defect ID', 'Type', 'Cell Loc.', 'Confidence', 'Severity', 'Power Loss', 'Impact']
            
            for i, header in enumerate(headers):
                pdf.cell(col_widths[i], 7, header, 1, 0, 'C')
            pdf.ln()
            
            # Table rows (limit to 20 for space)
            pdf.set_font("Arial", "", 8)
            for idx, row in df.head(20).iterrows():
                pdf.cell(col_widths[0], 6, row['Defect_ID'], 1)
                pdf.cell(col_widths[1], 6, row['Defect_Type'][:15], 1)
                pdf.cell(col_widths[2], 6, row['Cell_Location'], 1)
                pdf.cell(col_widths[3], 6, f"{row['Confidence_%']:.1f}%", 1)
                pdf.cell(col_widths[4], 6, row['Severity'], 1)
                pdf.cell(col_widths[5], 6, f"{row['Power_Loss_W']:.1f}W", 1)
                pdf.cell(col_widths[6], 6, row['Performance_Impact'], 1)
                pdf.ln()
            
            if len(df) > 20:
                pdf.set_font("Arial", "I", 8)
                pdf.cell(0, 6, f"... and {len(df) - 20} more defects. See full data export.", ln=True)
        
        # Images
        pdf.add_page()
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, "Visual Analysis", ln=True)
        
        try:
            # Save original image
            orig_path = "temp_orig.png"
            original_image.save(orig_path)
            pdf.set_font("Arial", "B", 10)
            pdf.cell(0, 8, "Original EL Image:", ln=True)
            pdf.image(orig_path, x=10, w=190)
            pdf.ln(5)
            
            # Annotated image
            if annotated_image_base64:
                ann_path = "temp_annotated.png"
                with open(ann_path, 'wb') as f:
                    f.write(base64.b64decode(annotated_image_base64))
                pdf.cell(0, 8, "Annotated Image with Defect Boundaries:", ln=True)
                pdf.image(ann_path, x=10, w=190)
                
                # Cleanup
                if os.path.exists(ann_path):
                    os.remove(ann_path)
            
            if os.path.exists(orig_path):
                os.remove(orig_path)
        except Exception as e:
            pdf.cell(0, 6, f"Error adding images: {str(e)}", ln=True)
        
        # Standards compliance note
        pdf.add_page()
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, "Standards Compliance & Methodology", ln=True)
        pdf.set_font("Arial", "", 9)
        pdf.multi_cell(0, 5,
            "This report follows international standards for PV module inspection:\n\n"
            "IEC TS 60904-13: Guidelines for electroluminescence imaging of PV modules. "
            "This standard defines quality metrics, defect classification, and imaging protocols.\n\n"
            "IEA PVPS Task 13: International guidelines for PV system performance, reliability, "
            "and lifetime prediction. Used for power loss calculations.\n\n"
            "MBJ Criteria: Module quality and defect severity classification framework.\n\n"
            "Analysis Methodology:\n"
            "- Defect detection via computer vision (Roboflow API)\n"
            "- Severity classification based on confidence scores\n"
            "- Power loss estimation using published degradation models\n"
            "- Financial impact calculated over 25-year module lifetime\n"
            "- Safety and reliability assessments per industry best practices"
        )
        
        # Footer
        pdf.set_y(-20)
        pdf.set_font("Arial", "I", 8)
        pdf.cell(0, 5, f"{company_name} | SolarVisionAI Platform", 0, 0, 'L')
        pdf.cell(0, 5, f"Page {pdf.page_no()}", 0, 0, 'R')
        
        # Save PDF
        filename = f"Solar_Inspection_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        pdf.output(filename)
        return filename


# ═══════════════════════════════════════════════════════════════════
# MODULE 5: USER INTERFACE
# ═══════════════════════════════════════════════════════════════════

def render_ui():
    """Main UI rendering function"""
    
    initialize_session_state()
    
    # Page config
    st.set_page_config(
        page_title="SolarVisionAI - Professional Defect Detection",
        page_icon="☀️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
        .main { padding: 1rem; }
        .stButton>button {
            background-color: #FF4B4B;
            color: white;
            border-radius: 8px;
            padding: 0.5rem 2rem;
            font-weight: 600;
        }
        .metric-card {
            background: white;
            padding: 1.5rem;
            border-radius: 10px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        .defect-critical { border-left: 4px solid #FF4444; background: #FFE6E6; padding: 1rem; margin: 0.5rem 0; }
        .defect-high { border-left: 4px solid #FFA500; background: #FFF4E6; padding: 1rem; margin: 0.5rem 0; }
        .defect-medium { border-left: 4px solid #FFD700; background: #FFFACD; padding: 1rem; margin: 0.5rem 0; }
        .defect-low { border-left: 4px solid #4CAF50; background: #E8F5E9; padding: 1rem; margin: 0.5rem 0; }
    </style>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://via.placeholder.com/150x50/FF4B4B/FFFFFF?text=SolarVisionAI", width=150)
        st.title("⚙️ Configuration")
        
        # User Tier Selection
        st.subheader("👤 Subscription Tier")
        tier_names = [tier.value['name'] for tier in UserTier]
        selected_tier_name = st.selectbox("Select Your Plan", tier_names)
        st.session_state.user_tier = next(tier for tier in UserTier if tier.value['name'] == selected_tier_name)
        
        tier_info = st.session_state.user_tier.value
        st.info(f"📊 **{tier_info['name']} Plan**\n\n"
               f"Single Upload Limit: {tier_info['single_limit'] if tier_info['single_limit'] > 0 else '∞'}\n\n"
               f"Total Uploads: {st.session_state.upload_count}/{tier_info['total_limit'] if tier_info['total_limit'] > 0 else '∞'}\n\n"
               f"Batch Processing: {'✅' if tier_info['batch'] else '❌'}")
        
        st.divider()
        
        # Company Branding
        st.subheader("🏢 Company Branding")
        company_name = st.text_input("Company Name", value="SolarVisionAI")
        
        st.divider()
        
        # API Configuration
        st.subheader("🔑 Roboflow API")
        api_key = st.text_input("API Key", type="password", help="Get from Roboflow dashboard")
        workspace = st.text_input("Workspace", value="el-images-trbib-gcqce")
        project = st.text_input("Project", value="1")
        version = st.number_input("Version", min_value=1, value=1)
        confidence = st.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)
        
        st.divider()
        
        # Module Configuration
        st.subheader("🔧 Module Configuration")
        module_rows = st.number_input("Rows", 1, 20, 6)
        module_cols = st.number_input("Columns", 1, 20, 10)
        rated_power = st.number_input("Rated Power (W)", 50, 500, 300)
        
        module_config = ModuleConfig(
            rows=module_rows,
            cols=module_cols,
            rated_power_w=float(rated_power)
        )
        
        st.divider()
        
        # Analysis Settings
        st.subheader("📊 Analysis Settings")
        analysis_level = st.selectbox("Analysis Level", 
                                     ["Basic", "Intermediate", "Advanced"],
                                     help="Basic: Tables | Intermediate: + Graphs | Advanced: + AI Insights")
        
        report_elements = st.multiselect(
            "Report Elements",
            ["Executive Summary", "Defect Table", "Cell Heatmap", "Visualizations", 
             "Financial Impact", "Root Cause Analysis", "Standards Compliance"],
            default=["Executive Summary", "Defect Table", "Visualizations"]
        )
    
    # Main content
    st.title(f"☀️ {company_name}")
    st.markdown("### AI-Powered Solar Panel Defect Detection & Analysis Platform")
    st.caption("Standards: IEC TS 60904-13 | IEA PVPS Task 13 | MBJ Criteria")
    
    # Upload Section
    st.subheader("📤 Upload Solar Panel Image")
    
    # Check upload limits
    tier_info = st.session_state.user_tier.value
    can_upload = True
    
    if tier_info['total_limit'] > 0 and st.session_state.upload_count >= tier_info['total_limit']:
        can_upload = False
        st.error(f"❌ Upload limit reached ({tier_info['total_limit']}). Please upgrade your plan.")
    
    if can_upload:
        if tier_info['batch']:
            uploaded_files = st.file_uploader(
                "Choose images...",
                type=['jpg', 'jpeg', 'png', 'tiff'],
                accept_multiple_files=True,
                help=f"Upload up to {tier_info['single_limit']} images"
            )
        else:
            uploaded_files = [st.file_uploader(
                "Choose an image...",
                type=['jpg', 'jpeg', 'png', 'tiff'],
                help="Single image upload"
            )]
        
        # Process uploads
        if uploaded_files and uploaded_files[0] is not None:
            # Check limits
            if tier_info['single_limit'] > 0 and len(uploaded_files) > tier_info['single_limit']:
                st.error(f"❌ Too many files. Your plan allows {tier_info['single_limit']} files.")
            else:
                st.session_state.uploaded_images = [Image.open(f) for f in uploaded_files if f is not None]
                
                # Display uploaded images
                cols = st.columns(min(len(st.session_state.uploaded_images), 3))
                for idx, img in enumerate(st.session_state.uploaded_images):
                    with cols[idx % 3]:
                        st.image(img, caption=f"Image {idx+1}", use_container_width=True)
                        st.caption(f"Size: {img.size[0]}x{img.size[1]}")
    
    # Analysis Button
    if st.session_state.uploaded_images and api_key:
        if st.button("🔍 Analyze Defects", type="primary", use_container_width=True):
            with st.spinner("🤖 AI analyzing images..."):
                all_results = []
                
                for idx, image in enumerate(st.session_state.uploaded_images):
                    # Call Roboflow API
                    result = RoboflowAPI.analyze_image(
                        image, api_key, workspace, project, version, confidence
                    )
                    
                    if result:
                        result['image_obj'] = image
                        result['image_idx'] = idx
                        all_results.append(result)
                        st.session_state.upload_count += 1
                
                if all_results:
                    st.session_state.results = all_results
                    st.success(f"✅ Analysis completed! Processed {len(all_results)} image(s).")
                    st.rerun()
    
    # Display Results
    if st.session_state.results:
        st.divider()
        st.header("🎯 Analysis Results")
        
        # Process each result
        for result_idx, result in enumerate(st.session_state.results):
            st.subheader(f"Image {result_idx + 1} Analysis")
            
            predictions = result.get('predictions', [])
            image = result['image_obj']
            annotated_base64 = result.get('annotated_image_base64')
            
            # Show images side by side
            col1, col2 = st.columns(2)
            with col1:
                st.image(image, caption="Original Image", use_container_width=True)
            with col2:
                if annotated_base64:
                    annotated_img = Image.open(io.BytesIO(base64.b64decode(annotated_base64)))
                    st.image(annotated_img, caption="Annotated with Defect Boundaries", use_container_width=True)
                else:
                    st.info("No annotated image available")
            
            # Analyze defects
            if predictions:
                analyzer = DefectAnalyzer(module_config)
                df = analyzer.analyze_defects(predictions, image.size[0], image.size[1])
                
                # Metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Defects", len(df))
                with col2:
                    critical = len(df[df['Severity'] == 'Critical'])
                    st.metric("Critical", critical, delta="⚠️" if critical > 0 else None)
                with col3:
                    total_loss = df['Power_Loss_W'].sum()
                    st.metric("Power Loss", f"{total_loss:.1f}W")
                with col4:
                    financial = df['Financial_Impact_USD'].sum()
                    st.metric("25Y Impact", f"${financial:,.0f}")
                
                # Defect Table (All tiers)
                st.subheader("📋 Defect Analysis Table")
                
                # Display options
                display_cols = st.multiselect(
                    "Select columns to display",
                    df.columns.tolist(),
                    default=['Defect_ID', 'Defect_Type', 'Cell_Location', 'Confidence_%', 
                            'Severity', 'Power_Loss_W', 'Performance_Impact']
                )
                
                st.dataframe(df[display_cols], use_container_width=True, height=300)
                
                # Cell Heatmap
                if 'Cell Heatmap' in report_elements:
                    st.subheader("🗺️ Module Cell Defect Heatmap")
                    cell_matrix = analyzer.create_cell_heatmap(df)
                    fig_heatmap = VisualizationEngine.create_cell_heatmap_viz(cell_matrix)
                    st.plotly_chart(fig_heatmap, use_container_width=True)
                
                # Visualizations (Pro and above)
                if analysis_level in ["Intermediate", "Advanced"] and 'Visualizations' in report_elements:
                    st.subheader("📊 Visual Analytics")
                    viz_engine = VisualizationEngine()
                    
                    fig_dist = viz_engine.create_defect_distribution_chart(df, st.session_state.user_tier)
                    if fig_dist:
                        st.plotly_chart(fig_dist, use_container_width=True)
                
                # Financial Impact (Advanced)
                if analysis_level == "Advanced" and 'Financial Impact' in report_elements:
                    st.subheader("💰 Financial Impact Analysis")
                    fig_financial = VisualizationEngine.create_financial_impact_chart(df)
                    if fig_financial:
                        st.plotly_chart(fig_financial, use_container_width=True)
                
                # Root Cause Analysis (Advanced)
                if analysis_level == "Advanced" and 'Root Cause Analysis' in report_elements:
                    st.subheader("🔍 Root Cause Analysis - Ishikawa Diagram")
                    fig_fishbone = VisualizationEngine.create_fishbone_diagram(df)
                    if fig_fishbone:
                        st.plotly_chart(fig_fishbone, use_container_width=True)
                
                # Export Options
                st.divider()
                st.subheader("📥 Export Report")
                
                col_ex1, col_ex2 = st.columns(2)
                
                with col_ex1:
                    # Excel Export
                    output = io.BytesIO()
                    with pd.ExcelWriter(output, engine='openpyxl') as writer:
                        df.to_excel(writer, index=False, sheet_name='Defect Analysis')
                        
                        # Add summary sheet
                        summary_df = pd.DataFrame({
                            'Metric': ['Total Defects', 'Critical', 'High', 'Medium', 'Low',
                                      'Total Power Loss (W)', 'Financial Impact ($)'],
                            'Value': [
                                len(df),
                                len(df[df['Severity'] == 'Critical']),
                                len(df[df['Severity'] == 'High']),
                                len(df[df['Severity'] == 'Medium']),
                                len(df[df['Severity'] == 'Low']),
                                df['Power_Loss_W'].sum(),
                                df['Financial_Impact_USD'].sum()
                            ]
                        })
                        summary_df.to_excel(writer, index=False, sheet_name='Summary')
                    
                    excel_data = output.getvalue()
                    st.download_button(
                        "📊 Download Excel Report",
                        excel_data,
                        f"defect_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                
                with col_ex2:
                    # PDF Export
                    if st.button("📄 Generate PDF Report"):
                        with st.spinner("Generating PDF..."):
                            try:
                                pdf_gen = EnhancedPDFReport()
                                pdf_file = pdf_gen.generate_report(
                                    company_name=company_name,
                                    df=df,
                                    original_image=image,
                                    annotated_image_base64=annotated_base64,
                                    module_config=module_config,
                                    analysis_level=analysis_level
                                )
                                
                                with open(pdf_file, 'rb') as f:
                                    st.download_button(
                                        "📄 Download PDF Report",
                                        f.read(),
                                        pdf_file,
                                        "application/pdf"
                                    )
                                
                                st.success("✅ PDF generated successfully!")
                            except Exception as e:
                                st.error(f"Error generating PDF: {str(e)}")
            else:
                st.success("✅ No defects detected. Module appears healthy.")
            
            st.divider()
    
    elif not api_key:
        st.warning("⚠️ Please configure Roboflow API credentials in the sidebar.")
    else:
        st.info("👆 Upload solar panel images to begin analysis.")
    
    # Footer
    st.divider()
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 2rem;'>
        <p>🌟 <b>SolarVisionAI Professional Platform</b></p>
        <p>Standards-Compliant PV Module Defect Detection | Built with Streamlit & Roboflow</p>
        <p style='font-size: 0.9em;'>IEC TS 60904-13 | IEA PVPS Task 13 | MBJ Criteria Compliant</p>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    render_ui()
