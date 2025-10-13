# ☀️ SolarVisionAI - Solar Panel Defect Detection System

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

AI-powered solar panel defect detection system using Roboflow API and Streamlit. Detect, classify, and report solar panel defects with automated workflows and elegant, user-friendly interface.

## 🎯 Features

### Core Functionality
- **🤖 AI-Powered Detection**: Integration with Roboflow API for state-of-the-art defect detection
- **📸 Image Upload**: Support for JPG, JPEG, and PNG formats
- **🔍 Quality Pre-check**: Automated image quality validation before analysis
- **📊 Real-time Analysis**: Instant defect detection with confidence scoring
- **🎨 Visual Results**: Annotated images with defect classifications

### Classification & Reporting
- **🚦 Severity Levels**: High, Medium, and Low severity classification
- **📈 Detailed Metrics**: Total defects, severity breakdown, and confidence scores
- **📋 Defect Legend**: Clear visual guide to defect classifications
- **📑 Report Export**: Export to PDF, Excel, or both formats
- **⏰ Timestamp Tracking**: Automated timestamp for each analysis

### Configuration & Customization
- **🏢 Company Branding**: Custom logo and company name
- **🔧 Configurable API**: Easy Roboflow workspace and model configuration
- **🎚️ Adjustable Threshold**: Dynamic confidence threshold adjustment
- **📝 Metadata Options**: Optional metadata inclusion in reports
- **👨‍💼 Admin Panel**: Advanced settings and controls for administrators

### Automated Workflows
- **✅ Image Quality Validation**: Pre-analysis quality checks
- **🔄 Result Parsing**: Automatic defect data extraction and formatting
- **📤 Export Automation**: One-click report generation
- **💾 Session Management**: State preservation across interactions

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- Roboflow account with trained solar defect detection model
- Roboflow API key

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/ganeshgowri-ASA/SolarVisionAI.git
   cd SolarVisionAI
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   streamlit run streamlit_app.py
   ```

4. **Access the application**
   - Open your browser and navigate to `http://localhost:8501`
   - The app will automatically open in your default browser

## 🔧 Configuration

### Roboflow API Setup

1. **Get API Key**
   - Sign up at [Roboflow](https://roboflow.com/)
   - Navigate to your workspace settings
   - Copy your API key

2. **Configure in App**
   - Open the sidebar in the Streamlit app
   - Enter your API key in the "API Configuration" section
   - Set your workspace ID, project ID, and model version

### Model Configuration

```python
# Example API configuration
workspace_id = "your-workspace-id"
project_id = "your-project-id"
model_version = 1
confidence_threshold = 0.5
```

### Company Branding

- Upload your company logo through the sidebar
- Set your company name in the configuration section
- Branding will appear in the app header and reports

## 📊 Usage Guide

### Step 1: Upload Image
1. Click on the file upload area
2. Select a solar panel image (JPG, JPEG, or PNG)
3. Wait for the image quality pre-check

### Step 2: Configure Settings (Optional)
- Adjust confidence threshold in the sidebar
- Configure report format preferences
- Enable/disable metadata inclusion

### Step 3: Analyze
1. Click the "🔍 Analyze Defects" button
2. Wait for AI analysis to complete
3. View detection results and metrics

### Step 4: Export Report
1. Review detected defects
2. Choose export format (PDF, Excel, or Both)
3. Click download button to save report

## 🏗️ Architecture

```
SolarVisionAI/
│
├── streamlit_app.py        # Main application file
├── requirements.txt        # Python dependencies
├── README.md              # Documentation (this file)
├── .gitignore             # Git ignore rules
│
└── [Future Extensions]
    ├── config/            # Configuration files
    ├── models/            # Model management
    ├── utils/             # Utility functions
    └── reports/           # Report templates
```

## 📦 Dependencies

- **streamlit** (>=1.28.0): Web application framework
- **Pillow** (>=10.0.0): Image processing
- **requests** (>=2.31.0): HTTP requests for API calls
- **pandas** (>=2.0.0): Data manipulation and analysis
- **openpyxl** (>=3.1.0): Excel file generation
- **reportlab** (>=4.0.0): PDF report generation
- **numpy** (>=1.24.0): Numerical computing support

## 🎨 UI Components

### Main Interface
- **Header**: Company branding and title
- **Upload Section**: Drag-and-drop image upload
- **Analysis Section**: Detection results and metrics
- **Export Section**: Report download options

### Sidebar
- **Company Branding**: Logo and name configuration
- **API Configuration**: Roboflow settings
- **Report Settings**: Export preferences
- **Admin Panel**: Advanced controls

## 🔍 Defect Classification

### Severity Levels

| Level | Confidence | Color | Action |
|-------|-----------|-------|--------|
| 🔴 High | >80% | Red | Immediate attention required |
| 🟡 Medium | 50-80% | Orange | Schedule maintenance |
| 🟢 Low | <50% | Green | Monitor regularly |

## 📈 Report Formats

### Excel Report
- Detailed defect data in tabular format
- Includes confidence scores, locations, and classifications
- Optional metadata and timestamps
- Easy to analyze and share

### PDF Report
- Professional formatted document
- Annotated images with defect markers
- Summary statistics and recommendations
- Company branding and timestamp

## 🔐 Security & Privacy

- **API Keys**: Stored securely in session state (not persisted)
- **Image Data**: Processed in memory, not stored on server
- **User Data**: No personal information collected
- **HTTPS**: Recommended for production deployment

## 🚀 Deployment

### Streamlit Community Cloud

1. **Fork this repository**
2. **Connect to Streamlit Cloud**
   - Visit [share.streamlit.io](https://share.streamlit.io/)
   - Click "New app"
   - Select your forked repository

3. **Configure Secrets** (Optional)
   ```toml
   # .streamlit/secrets.toml
   ROBOFLOW_API_KEY = "your-api-key"
   ```

4. **Deploy**
   - Click "Deploy"
   - Your app will be live in minutes!

### Docker Deployment

```dockerfile
# Coming soon - Docker configuration
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "streamlit_app.py"]
```

## 🛠️ Future Enhancements

### Planned Features
- [ ] **Batch Processing**: Analyze multiple images simultaneously
- [ ] **Historical Dashboard**: Track defects over time
- [ ] **AI Model Training**: Train custom models directly in the app
- [ ] **Mobile Support**: Responsive design for mobile devices
- [ ] **Database Integration**: Store analysis history
- [ ] **Email Notifications**: Automated alerts for critical defects
- [ ] **Multi-language Support**: Internationalization
- [ ] **Advanced Analytics**: Trends, patterns, and predictions
- [ ] **API Endpoint**: RESTful API for integration
- [ ] **User Authentication**: Multi-user support with roles

### Extension Points
- Custom defect types and classifications
- Integration with other AI platforms
- Advanced report customization
- Real-time monitoring capabilities
- IoT sensor data integration

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📧 Contact & Support

- **Email**: support@solarvisionai.com
- **Issues**: [GitHub Issues](https://github.com/ganeshgowri-ASA/SolarVisionAI/issues)
- **Documentation**: [Wiki](https://github.com/ganeshgowri-ASA/SolarVisionAI/wiki)

## 🙏 Acknowledgments

- **Roboflow**: For providing the excellent computer vision API
- **Streamlit**: For the amazing web app framework
- **Community**: For feedback and contributions

## 🌟 Star History

If you find this project useful, please consider giving it a star! ⭐

---

**Built with ❤️ by SolarVisionAI Team**

*Empowering renewable energy through AI-powered inspection*
