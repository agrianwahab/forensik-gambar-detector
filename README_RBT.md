# Risk-Based Testing (RBT) Framework
## Sistem Forensik Image Analysis

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Testing](https://img.shields.io/badge/testing-RBT-orange.svg)](docs/)

> **Framework komprehensif untuk implementasi Risk-Based Testing pada sistem forensik image analysis yang kompleks.**

## 📋 Daftar Isi

- [Overview](#overview)
- [Fitur Utama](#fitur-utama)
- [Instalasi](#instalasi)
- [Quick Start](#quick-start)
- [Konfigurasi](#konfigurasi)
- [Penggunaan](#penggunaan)
- [Dokumentasi](#dokumentasi)
- [Contoh Implementasi](#contoh-implementasi)
- [Monitoring dan Alerting](#monitoring-dan-alerting)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

Risk-Based Testing (RBT) Framework adalah solusi komprehensif untuk mengimplementasikan metodologi pengujian berbasis risiko pada sistem forensik image analysis. Framework ini dirancang khusus untuk menangani kompleksitas sistem forensik yang memerlukan akurasi tinggi dan reliabilitas dalam deteksi manipulasi gambar.

### Mengapa RBT untuk Sistem Forensik?

- **Kritikalitas Hasil**: False positive/negative dapat berdampak serius pada investigasi forensik
- **Kompleksitas Algoritma**: Multiple detection algorithms yang harus bekerja secara konsisten
- **Variasi Input**: Berbagai format, resolusi, dan jenis manipulasi gambar
- **Performance Requirements**: Analisis harus cepat namun akurat
- **Resource Optimization**: Fokus testing pada area berisiko tinggi

## 🚨 Catatan Penting - Unified RBT v2.0

> **🎆 MAJOR UPDATE**: Versi 2.0 menggabungkan `run_rbt.py` dan `run_enhanced_rbt.py` menjadi satu script unified yang lebih powerful!

### What's New in v2.0:
- ✅ **Unified Interface**: Satu script untuk semua mode pengujian
- ✅ **Enhanced Integration**: Mode `enhanced` dengan integrasi sistem forensik nyata
- ✅ **Component Testing**: Mode `test` untuk debugging individual components
- ✅ **Improved Error Handling**: Better error messages dan fallback mechanisms
- ✅ **Backward Compatibility**: Semua command lama tetap berfungsi
- ✅ **Verbose Logging**: Support untuk verbose output dengan `--verbose`

### Migration Required:
- ❌ **Deprecated**: `run_enhanced_rbt.py` (functionality moved to `run_rbt.py --mode enhanced`)
- ✅ **Recommended**: Gunakan `run_rbt.py` untuk semua pengujian

---

## ✨ Fitur Utama

### 🔍 Risk Analysis Engine
- **Automated Risk Identification**: Identifikasi otomatis risiko berdasarkan kompleksitas kode
- **Risk Scoring Matrix**: Perhitungan skor risiko dengan formula Probability × Impact
- **Component Risk Assessment**: Analisis risiko per komponen sistem
- **Historical Risk Tracking**: Pelacakan tren risiko dari waktu ke waktu

### 🧪 Test Case Generation
- **Risk-Driven Test Cases**: Generasi test case berdasarkan tingkat risiko
- **Forensic-Specific Scenarios**: Skenario pengujian khusus untuk forensik
- **Automated Test Data Management**: Manajemen dataset test otomatis
- **Cross-Algorithm Validation**: Validasi konsistensi antar algoritma

### 📊 Comprehensive Reporting
- **Executive Dashboard**: Dashboard eksekutif dengan metrik utama
- **Technical Reports**: Laporan teknis detail dengan analisis mendalam
- **Risk Coverage Matrix**: Matriks coverage risiko dan mitigasi
- **Forensic Validation Reports**: Laporan validasi khusus forensik

### 🔄 Continuous Monitoring
- **Real-time Risk Assessment**: Penilaian risiko real-time
- **Automated Alerting**: Sistem alert otomatis untuk critical failures
- **Performance Monitoring**: Monitoring performa sistem berkelanjutan
- **Trend Analysis**: Analisis tren untuk continuous improvement

### 🎛️ Advanced Configuration
- **Flexible Configuration**: Konfigurasi fleksibel via YAML
- **Environment-Specific Settings**: Pengaturan khusus per environment
- **Quality Gates**: Quality gates yang dapat dikonfigurasi
- **Integration Ready**: Siap integrasi dengan CI/CD pipeline

## 🚀 Instalasi dan Penggunaan

### Kompatibilitas Pengujian (Unified RBT)
Script `run_rbt.py` sekarang menggabungkan fitur dari pengujian RBT standar dan Enhanced RBT menjadi satu.

### Mode Pengujian:
- **Quick**: Penilaian risiko cepat untuk overview system
- **Full**: Siklus RBT lengkap dengan reporting standard
- **Enhanced**: Mode RBT dengan integrasi sistem forensik nyata
- **Critical**: Pengujian kritis saja untuk fast validation
- **Continuous**: Pemantauan berkelanjutan untuk production
- **Test**: Pengujian komponen individual untuk debugging

### Penjelasan Detail Mode:

#### Quick Mode
- **Tujuan**: Mendapatkan overview cepat status risiko sistem
- **Waktu**: < 5 menit
- **Output**: Risk count, critical issues, basic metrics
- **Cocok untuk**: Daily check, CI/CD pipeline gate

#### Full Mode
- **Tujuan**: Analisis komprehensif dengan reporting lengkap
- **Waktu**: 15-30 menit
- **Output**: Complete reports (JSON, HTML, CSV), detailed metrics
- **Cocok untuk**: Weekly assessment, release validation

#### Enhanced Mode
- **Tujuan**: Integrasi penuh dengan sistem forensik nyata
- **Waktu**: 30-60 menit
- **Output**: Real forensic validation, accuracy metrics
- **Cocok untuk**: Production validation, forensic accuracy testing

#### Critical Mode
- **Tujuan**: Fokus pada pengujian kritis yang harus lulus
- **Waktu**: 5-10 menit
- **Output**: Critical test results, pass/fail status
- **Cocok untuk**: Pre-deployment checks, emergency validation

#### Test Mode
- **Tujuan**: Debugging dan validasi komponen individual
- **Waktu**: 2-5 menit
- **Output**: Component status, integration health
- **Cocok untuk**: Development, troubleshooting

#### Continuous Mode
- **Tujuan**: Monitoring berkelanjutan untuk production
- **Waktu**: Always running
- **Output**: Scheduled reports, automated alerts
- **Cocok untuk**: Production monitoring, SLA compliance

Contoh Penggunaan:
```bash
python run_rbt.py --mode quick                # Penilaian risiko cepat
python run_rbt.py --mode full                 # Siklus RBT lengkap
python run_rbt.py --mode enhanced             # Enhanced RBT dengan integrasi nyata
python run_rbt.py --mode critical             # Pengujian kritis saja
python run_rbt.py --mode continuous           # Pemantauan berkelanjutan
python run_rbt.py --config custom_config.yaml # Menggunakan konfigurasi kustom
```

### Prerequisites

- Python 3.8 atau lebih tinggi
- pip package manager
- Git (untuk cloning repository)

### Step 1: Clone Repository

```bash
git clone https://github.com/your-org/forensic-rbt-framework.git
cd forensic-rbt-framework
```

### Step 2: Setup Virtual Environment

```bash
# Buat virtual environment
python -m venv rbt_env

# Aktivasi virtual environment
# Windows
rbt_env\Scripts\activate

# Linux/Mac
source rbt_env/bin/activate
```

### Step 3: Install Dependencies

```bash
# Install core dependencies
pip install -r requirements-rbt.txt

# Install development dependencies (optional)
pip install -r requirements-dev.txt
```

### Step 4: Verify Installation

```bash
# Test installation
python -c "from risk_based_testing_framework import ForensicRBTOrchestrator; print('✅ Installation successful')"

# Run quick test
python run_rbt.py --mode quick
```

## ⚡ Quick Start

### 1. Basic Risk Assessment

```bash
# Quick risk assessment
python run_rbt.py --mode quick
```

### 2. Complete RBT Cycle

```bash
# Full RBT cycle dengan reporting
python run_rbt.py --mode full
```

### 3. Enhanced RBT with Real Integration

```bash
# Enhanced RBT dengan integrasi sistem forensik nyata
python run_rbt.py --mode enhanced
```

### 4. Critical Tests Only

```bash
# Jalankan hanya critical tests
python run_rbt.py --mode critical
```

### 5. Component Testing

```bash
# Test komponen individual
python run_rbt.py --mode test
```

### 6. Continuous Monitoring

```bash
# Start continuous monitoring
python run_rbt.py --mode continuous
```

### 7. Custom Configuration

```bash
# Gunakan konfigurasi custom
python run_rbt.py --mode full --config my_config.yaml
```

### 8. Verbose Output

```bash
# Enable verbose logging
python run_rbt.py --mode full --verbose
```

## ⚙️ Konfigurasi

### File Konfigurasi Utama: `rbt_config.yaml`

```yaml
# Risk Assessment Configuration
risk_assessment:
  thresholds:
    critical: 0.15    # Risk score >= 0.15
    high: 0.10        # Risk score >= 0.10
    medium: 0.05      # Risk score >= 0.05
    low: 0.00         # Risk score < 0.05

# Test Execution Configuration
test_execution:
  timeouts:
    critical_tests: 300  # seconds
    high_tests: 180
    medium_tests: 120
    low_tests: 60

# Forensic Accuracy Configuration
forensic_accuracy:
  thresholds:
    overall_accuracy: 0.80   # 80% minimum
    precision: 0.85          # 85% minimum
    recall: 0.75             # 75% minimum
    f1_score: 0.80           # 80% minimum
```

### Environment-Specific Configuration

```yaml
# Development Environment
environments:
  development:
    risk_assessment:
      thresholds:
        critical: 0.20      # More lenient in dev
    
  # Production Environment
  production:
    forensic_accuracy:
      thresholds:
        overall_accuracy: 0.90  # Higher standards
```

## 📖 Penggunaan

### Programmatic Usage

```python
from risk_based_testing_framework import ForensicRBTOrchestrator

# Initialize orchestrator
orchestrator = ForensicRBTOrchestrator()

# Quick risk assessment
quick_result = orchestrator.run_quick_risk_assessment()
print(f"Critical risks: {quick_result['critical_risks']}")

# Complete RBT cycle
full_result = orchestrator.run_complete_rbt_cycle()
print(f"Overall status: {full_result['overall_status']}")
print(f"Reliability score: {full_result['forensic_reliability_score']:.1%}")
```

### Unified RBT Runner Usage

```python
# Using unified RBT runner
from run_rbt import RBTRunner

# Initialize runner
runner = RBTRunner(config_path='my_config.yaml', verbose=True)

# Run enhanced RBT
result = runner.run_enhanced_rbt()
print(f"Status: {result['status']}")

# Run component tests
component_result = runner.run_component_test()
print(f"Components tested: {component_result['components_tested']}")
```

### Custom Test Case Development

```python
from risk_based_testing_framework import TestCase, RiskLevel, RiskCategory

# Create custom test case
custom_test = TestCase(
    tc_id="RBT_CUSTOM_001",
    title="Custom Forensic Test",
    description="Test specific forensic scenario",
    risk_items=["R001"],
    risk_level=RiskLevel.HIGH,
    category=RiskCategory.ACCURACY,
    preconditions=["System initialized"],
    test_steps=["Load test image", "Run analysis", "Validate results"],
    expected_result="Accurate detection with >85% confidence",
    test_data={"test_images": ["test1.jpg", "test2.jpg"]},
    execution_time_limit=120,
    forensic_impact="High impact on investigation accuracy",
    validation_method="Ground truth comparison"
)
```

### Integration dengan CI/CD

#### GitHub Actions Example

```yaml
# .github/workflows/rbt.yml
name: Risk-Based Testing

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]
  schedule:
    - cron: '0 2 * * *'  # Daily at 2 AM

jobs:
  rbt-testing:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'
    
    - name: Install dependencies
      run: |
        pip install -r requirements-rbt.txt
    
    - name: Run RBT Framework
      run: |
        python run_rbt.py --mode full
    
    - name: Upload RBT Reports
      uses: actions/upload-artifact@v3
      with:
        name: rbt-reports
        path: rbt_reports/
    
    - name: Check Quality Gates
      run: |
        python -c "import json; result=json.load(open('rbt_reports/latest_result.json')); exit(0 if result['overall_status']=='PASS' else 1)"
```

## 📊 Monitoring dan Alerting

### Real-time Dashboard

```python
# Start monitoring dashboard
from risk_based_testing_framework import RBTDashboard

dashboard = RBTDashboard()
dashboard.start_server(port=8080)
# Access dashboard at http://localhost:8080
```

### Alert Configuration

```yaml
# Alert settings in rbt_config.yaml
alerting:
  channels:
    email:
      enabled: true
      recipients: ["forensic-team@company.com"]
    
    slack:
      enabled: true
      webhook_url: "https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK"
      channel: "#forensic-alerts"
  
  thresholds:
    critical_test_failure: 0      # Zero tolerance
    accuracy_degradation: 0.05    # 5% degradation
    performance_degradation: 0.20 # 20% degradation
```

### Continuous Monitoring Setup

```bash
# Setup as system service (Linux)
sudo cp scripts/rbt-monitor.service /etc/systemd/system/
sudo systemctl enable rbt-monitor
sudo systemctl start rbt-monitor

# Check status
sudo systemctl status rbt-monitor
```

## 📈 Metrik dan Formula

### Risk Assessment Metrics

```python
# Risk Score Calculation
Risk_Score = Probability × Impact

# Risk Coverage Percentage
Risk_Coverage = (Covered_Risks / Total_Risks) × 100

# Risk Mitigation Effectiveness
Mitigation_Effectiveness = (Successfully_Mitigated_Risks / Tested_Risks) × 100

# Residual Risk Score
Residual_Risk = Original_Risk_Score × (1 - Mitigation_Effectiveness)
```

### Forensic Accuracy Metrics

```python
# Overall Accuracy
Accuracy = (TP + TN) / (TP + TN + FP + FN)

# Precision (Positive Predictive Value)
Precision = TP / (TP + FP)

# Recall (Sensitivity)
Recall = TP / (TP + FN)

# F1-Score
F1_Score = 2 × (Precision × Recall) / (Precision + Recall)

# Specificity (True Negative Rate)
Specificity = TN / (TN + FP)
```

### Overall RBT Success Score

```python
Overall_RBT_Success = (
    0.30 × Risk_Coverage +
    0.40 × Mitigation_Effectiveness +
    0.25 × Forensic_Accuracy +
    0.05 × Test_Efficiency
)
```

## 🔧 Troubleshooting

### Common Issues

#### 1. Import Error: Module Not Found

```bash
# Problem: risk_based_testing_framework module not found
# Solution: Ensure you're in the correct directory and virtual environment is activated

cd /path/to/forensic-rbt-framework
source rbt_env/bin/activate  # Linux/Mac
# or
rbt_env\Scripts\activate     # Windows
```

#### 2. Enhanced RBT Mode Not Available

```bash
# Problem: Enhanced RBT modules not available
# Solution: Install enhanced RBT integration modules

pip install forensic-rbt-integration
# or if modules are in same directory
python -c "import forensic_rbt_integration; print('Enhanced RBT available')"
```

#### 3. Component Test Failures

```bash
# Problem: Component tests failing
# Solution: Check component availability and configuration

python run_rbt.py --mode test --verbose
# Check logs for specific component failures
```

#### 2. Configuration File Not Found

```bash
# Problem: rbt_config.yaml not found
# Solution: Create default configuration or specify path

python run_rbt.py --config /path/to/your/config.yaml
```

#### 3. Memory Issues with Large Images

```yaml
# Solution: Adjust memory limits in configuration
performance:
  limits:
    max_memory_gb: 16.0  # Increase memory limit
  
  # Enable image resizing for large files
  image_processing:
    max_resolution: "4K"  # Resize larger images
    enable_progressive_loading: true
```

#### 4. Test Timeout Issues

```yaml
# Solution: Increase timeout values
test_execution:
  timeouts:
    critical_tests: 600  # Increase from 300 to 600 seconds
    high_tests: 360      # Increase accordingly
```

### Debug Mode

```bash
# Enable verbose logging
python run_rbt.py --mode full --verbose

# Check logs
tail -f rbt_logs/rbt_run_*.log
```

### Performance Optimization

```python
# Enable parallel execution
test_execution:
  parallel:
    enabled: true
    max_workers: 8  # Adjust based on CPU cores
    
# Use memory profiling
from memory_profiler import profile

@profile
def run_rbt_with_profiling():
    orchestrator = ForensicRBTOrchestrator()
    return orchestrator.run_complete_rbt_cycle()
```

## 📚 Dokumentasi Lengkap

### File Dokumentasi

- [`RBT_Implementation_Guide.md`](RBT_Implementation_Guide.md) - Panduan implementasi lengkap
- [`rbt_config.yaml`](rbt_config.yaml) - File konfigurasi dengan dokumentasi
- [`risk_based_testing_framework.py`](risk_based_testing_framework.py) - Framework utama
- [`run_rbt.py`](run_rbt.py) - Script runner dengan contoh penggunaan

### API Documentation

```python
# Generate API documentation
python -m pydoc risk_based_testing_framework

# Or use Sphinx for comprehensive docs
sphinx-build -b html docs/ docs/_build/
```

## 🧪 Contoh Implementasi

### Skenario 1: Investigasi Dokumen Palsu

```python
# Test case untuk deteksi copy-move pada dokumen
test_case = TestCase(
    tc_id="RBT_FORENSIC_001",
    title="Document Forgery Detection",
    description="Detect copy-move manipulation in legal documents",
    risk_items=["R001", "R002"],
    risk_level=RiskLevel.CRITICAL,
    category=RiskCategory.ACCURACY,
    test_data={
        "original_documents": ["contract_original.pdf"],
        "forged_documents": ["contract_with_copied_signature.pdf"],
        "expected_detection": "copy_move_signature_area"
    },
    validation_method="Ground truth comparison with expert annotation"
)
```

### Skenario 2: Analisis Foto Bukti Kejahatan

```python
# Test case untuk verifikasi keaslian foto bukti
test_case = TestCase(
    tc_id="RBT_FORENSIC_002",
    title="Crime Scene Photo Verification",
    description="Verify authenticity of crime scene photographs",
    risk_items=["R003", "R004"],
    risk_level=RiskLevel.CRITICAL,
    category=RiskCategory.RELIABILITY,
    test_data={
        "crime_scene_photos": ["scene_001.jpg", "scene_002.jpg"],
        "metadata_requirements": ["timestamp", "gps", "camera_model"],
        "chain_of_custody": "required"
    },
    forensic_impact="Critical for legal proceedings"
)
```

## 🔄 Migrasi ke Unified RBT

### Dari run_rbt.py dan run_enhanced_rbt.py ke run_rbt.py (Unified)

Sebelumnya, pengujian RBT dijalankan dengan dua script terpisah:
- `run_rbt.py` - untuk pengujian RBT standard
- `run_enhanced_rbt.py` - untuk pengujian RBT dengan integrasi nyata

Sekarang semua fitur telah digabungkan ke dalam satu script `run_rbt.py`.

### Mapping Command Lama ke Baru:

```bash
# LAMA - RBT Standard
python run_rbt.py --mode quick
python run_rbt.py --mode full
python run_rbt.py --mode critical

# BARU - Tetap sama
python run_rbt.py --mode quick
python run_rbt.py --mode full
python run_rbt.py --mode critical

# LAMA - Enhanced RBT
python run_enhanced_rbt.py --mode quick
python run_enhanced_rbt.py --mode full
python run_enhanced_rbt.py --mode test

# BARU - Unified
python run_rbt.py --mode enhanced --config rbt_config.yaml
python run_rbt.py --mode enhanced --verbose
python run_rbt.py --mode test
```

### Keuntungan Unified Version:

1. **Single Point of Entry**: Satu script untuk semua mode pengujian
2. **Automatic Fallback**: Jika enhanced modules tidak tersedia, otomatis fallback ke standard
3. **Consistent Interface**: Interface yang konsisten untuk semua mode
4. **Better Error Handling**: Error handling yang lebih baik dan informatif
5. **Unified Configuration**: Satu file konfigurasi untuk semua mode

### Backward Compatibility:

- Semua command `run_rbt.py` lama tetap berfungsi
- Konfigurasi existing tetap kompatibel
- Output format tetap konsisten
- API programmatic tetap sama

## 🔄 Continuous Improvement

### Automated Learning

```python
# Framework untuk continuous improvement
class RBTContinuousImprovement:
    def analyze_test_effectiveness(self, test_results, defect_data):
        """Analyze effectiveness of current tests"""
        # Implementation untuk analisis efektivitas
        pass
    
    def recommend_test_improvements(self, effectiveness_metrics):
        """Recommend improvements based on analysis"""
        # Implementation untuk rekomendasi improvement
        pass
```

### Feedback Loop

```yaml
# Configuration untuk feedback loop
continuous_improvement:
  feedback_collection:
    enabled: true
    sources: ["test_results", "defect_reports", "user_feedback"]
  
  learning_algorithms:
    risk_prediction: true
    test_optimization: true
    threshold_tuning: true
```

## 🤝 Contributing

### Development Setup

```bash
# Clone repository
git clone https://github.com/your-org/forensic-rbt-framework.git
cd forensic-rbt-framework

# Setup development environment
python -m venv dev_env
source dev_env/bin/activate
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install
```

### Code Style

```bash
# Format code
black .
flake8 .
mypy .

# Run tests
pytest tests/ -v --cov=risk_based_testing_framework
```

### Contribution Guidelines

1. **Fork** repository
2. **Create** feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** changes (`git commit -m 'Add amazing feature'`)
4. **Push** to branch (`git push origin feature/amazing-feature`)
5. **Open** Pull Request

### Issue Reporting

Gunakan [GitHub Issues](https://github.com/your-org/forensic-rbt-framework/issues) untuk:
- Bug reports
- Feature requests
- Documentation improvements
- Performance issues

## 📄 License

Project ini dilisensikan di bawah MIT License - lihat file [LICENSE](LICENSE) untuk detail.

## 🙏 Acknowledgments

- Tim Forensic Testing untuk requirements dan feedback
- Community open source untuk tools dan libraries
- Research community untuk metodologi RBT

## 📞 Support

- **Documentation**: [Wiki](https://github.com/your-org/forensic-rbt-framework/wiki)
- **Issues**: [GitHub Issues](https://github.com/your-org/forensic-rbt-framework/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-org/forensic-rbt-framework/discussions)
- **Email**: forensic-rbt-support@company.com

---

**Made with ❤️ for the Forensic Testing Community**

> "Quality is not an act, it is a habit." - Aristotle

---

### Quick Links

- [🚀 Quick Start](#quick-start)
- [📜 Full Documentation](RBT_Implementation_Guide.md)
- [⚙️ Configuration Guide](rbt_config.yaml)
- [🧪 Example Tests](examples/)
- [📊 Dashboard Demo](http://demo.rbt-framework.com)
- [🔧 Troubleshooting](#troubleshooting)
- [🔄 Migration Guide](#migrasi-ke-unified-rbt)
- [🔍 Mode Comparison](#penjelasan-detail-mode)
- [🚀 Unified RBT Usage](#instalasi-dan-penggunaan)
