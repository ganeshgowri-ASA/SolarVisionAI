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
    st.warning("⚠️ Install Plotly for advanced visualizations: `pip install plotly`")

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
    STANDARD = "Standard"  # Expected manufacturing variations
    NON_STANDARD = "Non-Standard"  # Quality issues requiring attention


@dataclass
class ModuleConfig:
    """Enhanced solar module configuration for half-cut and full cells"""
    rows: int = 6  # Up to 26 for half-cut modules
    cols: int = 10  # Up to 8 for various configurations
    cell_width_mm: float = 156.0
    cell_height_mm: float = 156.0
    rated_power_w: float = 300.0
    cell_type: str = "Monocrystalline"  # Mono/Poly
    busbar_count: int = 6  # 3BB, 5BB, 6BB, 9BB, 12BB
    
    def get_cell_position(self, x: float, y: float, img_width: int, img_height: int) -> str:
        """Convert pixel coordinates to cell position (A1-Z26 format)"""
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
    isc_current_a: float = 0.0  # Short circuit current
    voc_voltage_v: float = 0.0  # Open circuit voltage
    test_current_a: float = 0.0  # Applied test current (typically 10% Isc)
    ambient_temp_c: float = 25.0
    module_temp_c: float = 25.0
    power_supply_make: str = ""
    power_supply_model: str = ""
    cable_length_m: float = 0.0
    
    # Camera settings
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
            "Ambient Temp (°C)": self.ambient_temp_c,
            "Module Temp (°C)": self.module_temp_c,
            "Power Supply": f"{self.power_supply_make} {self.power_supply_model}",
            "Cable Length (m)": self.cable_length_m,
            "Camera": f"{self.camera_make} {self.camera_model}",
            "Exposure (ms)": self.exposure_time_ms,
            "ISO": self.iso,
            "Resolution": self.resolution
        }


@dataclass
class DefectImpactAnalysis:
    """Comprehensive defect impact per IEC & MBJ standards"""
    area_loss_percent: float
    power_loss_w: float
    power_loss_percent: float
    performance_impact: str
    safety_risk: str
    reliability_impact: str
    financial_impact_usd: float
    mbj_classification: DefectClassification
    defect_grade: str  # A/B/C/D per MBJ


# ═══════════════════════════════════════════════════════════════════
# SESSION STATE
# ═══════════════════════════════════════════════════════════════════

def initialize_session_state():
    """Initialize all session state variables"""
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
# DEFECT ANALYSIS ENGINE WITH MBJ CLASSIFICATION
# ═══════════════════════════════════════════════════════════════════

class EnhancedDefectAnalyzer:
    """MBJ-compliant defect analyzer with standard/non-standard classification"""
    
    # MBJ Standard defects (expected in manufacturing)
    STANDARD_DEFECTS = [
        'crystal_dislocations', 'edge_wafer', 'striation_rings',
        'belt_marks', 'reduced_lifetime_cast_silicon'
    ]
    
    # MBJ Non-standard defects (quality issues)
    NON_STANDARD_DEFECTS = [
        'micro_crack', 'crack', 'branch_crack', 'dendritic_crack',
        'hotspot', 'isolated_area', 'dead_cell', 'finger_interruption',
        'shunted_cell', 'pid_effect', 'busbar_crack'
    ]
    
    def __init__(self, module_config: ModuleConfig):
        self.module_config = module_config
        
        # IEC TS 60904-13 severity thresholds
        self.severity_thresholds = {
            'micro_crack': {'low': 0.6, 'medium': 0.75, 'high': 0.85},
            'crack': {'low': 0.7, 'medium': 0.8, 'high': 0.9},
            'branch_crack': {'low': 0.7, 'medium': 0.82, 'high': 0.92},
            'hotspot': {'low': 0.65, 'medium': 0.78, 'high': 0.88},
            'isolated_area': {'low': 0.6, 'medium': 0.75, 'high': 0.85},
        }
    
    def analyze_defects(self, predictions: List[Dict], img_width: int, img_height: int) -> pd.DataFrame:
        """Comprehensive defect analysis with MBJ classification"""
        if not predictions:
            return pd.DataFrame()
        
        analysis_data = []
        
        for idx, pred in enumerate(predictions, 1):
            defect_type = pred.get('class', 'Unknown').lower().replace(' ', '_')
            confidence = pred.get('confidence', 0)
            x, y = pred.get('x', 0), pred.get('y', 0)
            width, height = pred.get('width', 0), pred.get('height', 0)
            
            # Cell location
            cell_position = self.module_config.get_cell_position(x, y, img_width, img_height)
            
            # MBJ Classification
            mbj_class = (DefectClassification.STANDARD if defect_type in self.STANDARD_DEFECTS 
                        else DefectClassification.NON_STANDARD)
            
            # Severity
            severity = self._calculate_severity(defect_type, confidence)
            
            # Impact analysis
            impact = self._calculate_impact(defect_type, width, height, img_width, img_height, confidence)
            
            # MBJ Grading
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
        """IEC TS 60904-13 severity calculation"""
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
        """Calculate impact using IEA PVPS models"""
        defect_area = width * height
        total_area = img_width * img_height
        area_loss_percent = (defect_area / total_area) * 100
        
        # IEA PVPS multipliers
        severity_multipliers = {
            'micro_crack': 0.5, 'crack': 1.2, 'branch_crack': 2.0,
            'hotspot': 2.5, 'isolated_area': 1.8, 'dead_cell': 5.0
        }
        
        multiplier = severity_multipliers.get(defect_type, 1.0)
        
        # Busbar redundancy factor (>6BB reduces impact)
        bb_factor = 1.0 if self.module_config.busbar_count <= 6 else 0.6
        
        base_power_loss = area_loss_percent * multiplier * confidence * bb_factor
        power_loss_w = (base_power_loss / 100) * self.module_config.rated_power_w
        power_loss_percent = (power_loss_w / self.module_config.rated_power_w) * 100
        
        # Performance classification
        if power_loss_percent >= 15:
            performance = 'Critical'
        elif power_loss_percent >= 8:
            performance = 'High'
        elif power_loss_percent >= 3:
            performance = 'Medium'
        else:
            performance = 'Low'
        
        # Safety assessment
        if defect_type in ['hotspot'] and confidence > 0.8:
            safety = 'High - Fire Risk'
        elif defect_type in ['crack', 'branch_crack'] and confidence > 0.85:
            safety = 'Medium - Structural'
        else:
            safety = 'Low - Monitor'
        
        # Reliability
        if power_loss_percent >= 10:
            reliability = 'High - Replace'
        elif power_loss_percent >= 5:
            reliability = 'Medium - Repair'
        else:
            reliability = 'Low - Monitor'
        
        # Financial (25-year, $0.10/kWh)
        annual_kwh_loss = (power_loss_w * 4 * 365) / 1000
        financial = annual_kwh_loss * 25 * 0.10
        
        # MBJ Classification
        mbj_class = (DefectClassification.STANDARD if defect_type in self.STANDARD_DEFECTS 
                    else DefectClassification.NON_STANDARD)
        
        # MBJ Grade
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
        """MBJ Rev 5.0 grading system"""
        if classification == DefectClassification.STANDARD:
            return 'A'  # Standard defects don't affect grade
        
        # Non-standard defects
        if power_loss_percent >= 20:
            return 'D'  # Fail
        elif power_loss_percent >= 10:
            return 'C'  # Fair (80-90% retention)
        elif power_loss_percent >= 5:
            return 'B'  # Good (90-95% retention)
        return 'A'  # Excellent (>95% retention)
    
    def create_cell_heatmap(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cell-level defect heatmap"""
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
# PLOTLY VISUALIZATION ENGINE
# ═══════════════════════════════════════════════════════════════════

class PlotlyVisualizationEngine:
    """Advanced Plotly-based visualizations"""
    
    @staticmethod
    def create_comprehensive_dashboard(df: pd.DataFrame, module_config: ModuleConfig):
        """Create interactive Plotly dashboard"""
        if df.empty or not PLOTLY_AVAILABLE:
            return None
        
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Defect Type Distribution',
                'Severity Analysis',
                'MBJ Classification',
                'Power Loss vs Confidence',
                'Cell Location Heatmap',
                'Financial Impact'
            ),
            specs=[
                [{'type': 'bar'}, {'type': 'pie'}],
                [{'type': 'pie'}, {'type': 'scatter'}],
                [{'type': 'heatmap'}, {'type': 'bar'}]
            ],
            vertical_spacing=0.12,
            horizontal_spacing=0.15
        )
        
        # 1. Defect Type Distribution
        defect_counts = df['Defect_Type'].value_counts()
        fig.add_trace(
            go.Bar(x=defect_counts.index, y=defect_counts.values, 
                  marker_color='crimson', name='Defects'),
            row=1, col=1
        )
        
        # 2. Severity Distribution
        severity_counts = df['Severity'].value_counts()
        severity_colors = {'Critical': '#FF4444', 'High': '#FFA500', 
                          'Medium': '#FFD700', 'Low': '#4CAF50'}
        fig.add_trace(
            go.Pie(labels=severity_counts.index, values=severity_counts.values,
                  marker=dict(colors=[severity_colors.get(s, 'gray') for s in severity_counts.index]),
                  name='Severity'),
            row=1, col=2
        )
        
        # 3. MBJ Classification
        mbj_counts = df['MBJ_Classification'].value_counts()
        fig.add_trace(
            go.Pie(labels=mbj_counts.index, values=mbj_counts.values,
                  marker=dict(colors=['#4CAF50', '#FF4444']),
                  name='MBJ'),
            row=2, col=1
        )
        
        # 4. Power Loss Scatter
        fig.add_trace(
            go.Scatter(
                x=df['Confidence_%'], y=df['Power_Loss_W'],
                mode='markers',
                marker=dict(
                    size=10,
                    color=df['Power_Loss_W'],
                    colorscale='Reds',
                    showscale=True,
                    colorbar=dict(x=0.47, len=0.3, y=0.35)
                ),
                text=df['Defect_Type'],
                hovertemplate='<b>%{text}</b><br>Confidence: %{x:.1f}%<br>Power Loss: %{y:.2f}W',
                name='Power Loss'
            ),
            row=2, col=2
        )
        
        # 5. Cell Heatmap
        analyzer = EnhancedDefectAnalyzer(module_config)
        cell_matrix = analyzer.create_cell_heatmap(df)
        fig.add_trace(
            go.Heatmap(
                z=cell_matrix.values,
                x=cell_matrix.columns.tolist(),
                y=cell_matrix.index.tolist(),
                colorscale='Reds',
                showscale=True,
                colorbar=dict(x=0.47, len=0.3, y=0.03),
                text=cell_matrix.values,
                texttemplate='%{text}',
                textfont={"size": 8}
            ),
            row=3, col=1
        )
        
        # 6. Financial Impact
        financial_by_type = df.groupby('Defect_Type')['Financial_Impact_USD'].sum().sort_values(ascending=False)
        fig.add_trace(
            go.Bar(
                x=financial_by_type.index,
                y=financial_by_type.values,
                marker_color='darkred',
                text=[f'${v:,.0f}' for v in financial_by_type.values],
                textposition='outside',
                name='Financial Impact'
            ),
            row=3, col=2
        )
        
        # Layout updates
        fig.update_xaxes(title_text="Defect Type", row=1, col=1)
        fig.update_yaxes(title_text="Count", row=1, col=1)
        fig.update_xaxes(title_text="Confidence (%)", row=2, col=2)
        fig.update_yaxes(title_text="Power Loss (W)", row=2, col=2)
        fig.update_xaxes(title_text="Column", row=3, col=1)
        fig.update_yaxes(title_text="Row", row=3, col=1)
        fig.update_xaxes(title_text="Defect Type", row=3, col=2)
        fig.update_yaxes(title_text="25Y Loss (USD)", row=3, col=2)
        
        fig.update_layout(
            height=1200,
            showlegend=False,
            title_text="<b>Comprehensive EL Defect Analysis Dashboard</b>",
            title_x=0.5,
            title_font_size=20
        )
        
        return fig
    
    @staticmethod
    def create_3d_defect_map(df: pd.DataFrame):
        """3D visualization of defects on module surface"""
        if df.empty or not PLOTLY_AVAILABLE:
            return None
        
        fig = go.Figure(data=[go.Scatter3d(
            x=df['X'],
            y=df['Y'],
            z=df['Power_Loss_W'],
            mode='markers',
            marker=dict(
                size=df['Width'] / 10,
                color=df['Power_Loss_W'],
                colorscale='Reds',
                showscale=True,
                colorbar=dict(title="Power Loss (W)")
            ),
            text=df['Defect_Type'],
            hovertemplate='<b>%{text}</b><br>Location: (%{x}, %{y})<br>Power Loss: %{z:.2f}W'
        )])
        
        fig.update_layout(
            title="3D Defect Distribution Map",
            scene=dict(
                xaxis_title="X Position (px)",
                yaxis_title="Y Position (px)",
                zaxis_title="Power Loss (W)"
            ),
            height=600
        )
        
        return fig


# ═══════════════════════════════════════════════════════════════════
# ROBOFLOW API WITH ENHANCED ERROR HANDLING
# ═══════════════════════════════════════════════════════════════════

class EnhancedRoboflowAPI:
    """Roboflow API with 403 error diagnostics"""
    
    @staticmethod
    def analyze_image(image: Image.Image, api_key: str, workspace: str,
                     project: str, version: int, confidence: float = 0.5) -> Optional[Dict]:
        """Call Roboflow with detailed error handling"""
        
        diagnostics = {
            'api_key_valid': bool(api_key and len(api_key) > 10),
            'workspace': workspace,
            'project': project,
            'version': version,
            'image_size': image.size,
            'request_url': f"https://detect.roboflow.com/{project}/{version}",
            'error': None,
            'status_code': None
        }
        
        try:
            # Convert image
            buffered = io.BytesIO()
            image.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            
            # CRITICAL: Correct API URL format
            # Format should be: https://detect.roboflow.com/PROJECT_ID/VERSION
            api_url = f"https://detect.roboflow.com/{project}/{version}"
            
            params = {
                "api_key": api_key,
                "confidence": int(confidence * 100),  # Convert to integer percentage
                "overlap": 30
            }
            
            # JSON request
            response = requests.post(
                api_url,
                params=params,
                data=img_str,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                timeout=30
            )
            
            diagnostics['status_code'] = response.status_code
            
            if response.status_code == 403:
                diagnostics['error'] = "403 Forbidden - Check API key permissions"
                diagnostics['response_text'] = response.text
                st.session_state.api_diagnostics = diagnostics
                
                st.error("🚫 **Roboflow API Error 403: Forbidden**")
                with st.expander("🔍 **Diagnostics & Solutions**"):
                    st.markdown("""
                    **Possible Causes:**
                    1. ❌ **Invalid API Key** - Key may be incorrect or expired
                    2. ❌ **Wrong Workspace/Project** - Project ID doesn't match your account
                    3. ❌ **API Access Disabled** - Check Roboflow dashboard permissions
                    4. ❌ **Rate Limit Exceeded** - Too many requests
                    
                    **How to Fix:**
