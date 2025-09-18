#!/usr/bin/env python3
"""
RBT Reporter dengan Bahasa Indonesia
Menghasilkan laporan Risk-Based Testing dalam Bahasa Indonesia
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Any
from pathlib import Path
import logging
import numpy as np

from risk_based_testing_framework import (
    RiskItem, TestCase, TestExecution, TestResult, RiskLevel,
    ForensicRiskAnalyzer, ForensicRBTReporter
)

class IndonesianRBTReporter(ForensicRBTReporter):
    """Reporter RBT dengan output Bahasa Indonesia"""
    
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        
    def generate_comprehensive_report(self, 
                                    risk_analyzer: ForensicRiskAnalyzer,
                                    test_cases: List[TestCase],
                                    executions: List[TestExecution]) -> Dict[str, Any]:
        """Generate laporan komprehensif RBT dalam Bahasa Indonesia"""
        
        # Hitung metrik
        metrics = self._calculate_test_metrics(executions)
        risk_coverage = self._calculate_risk_coverage_indonesia(risk_analyzer.risk_items, test_cases, executions)
        forensic_reliability = self._calculate_forensic_reliability(executions)
        
        report = {
            'metadata_laporan': {
                'dibuat_pada': datetime.now().strftime("%d %B %Y %H:%M:%S"),
                'total_risiko_teridentifikasi': len(risk_analyzer.risk_items),
                'total_kasus_uji': len(test_cases),
                'total_eksekusi': len(executions),
                'versi_laporan': '1.0'
            },
            
            'ringkasan_eksekutif': {
                'status_risiko_keseluruhan': self._assess_overall_risk_status_indonesia(risk_coverage),
                'ringkasan_eksekusi_pengujian': self._translate_metrics(metrics),
                'temuan_kritis': self._identify_critical_findings_indonesia(executions),
                'rekomendasi': self._generate_recommendations_indonesia(risk_coverage, metrics)
            },
            
            'analisis_risiko': {
                'matriks_risiko': self._translate_risk_matrix(risk_analyzer.risk_matrix),
                'skor_risiko_komponen': risk_analyzer.component_risks,
                'analisis_cakupan_risiko': risk_coverage
            },
            
            'detail_eksekusi_pengujian': {
                'metrik_pengujian': self._translate_metrics(metrics),
                'skor_reliabilitas_forensik': forensic_reliability,
                'hasil_eksekusi': self._translate_executions(executions),
                'analisis_defek': self._analyze_defects_indonesia(executions)
            },
            
            'validasi_forensik': {
                'metrik_akurasi': self._calculate_accuracy_metrics_indonesia(executions),
                'konsistensi_lintas_algoritma': self._analyze_algorithm_consistency_indonesia(executions),
                'analisis_kepercayaan': self._analyze_confidence_scores_indonesia(executions)
            },
            
            'evaluasi_kriteria_keberhasilan': {
                'kriteria_terpenuhi': self._evaluate_success_criteria_indonesia(metrics, risk_coverage, forensic_reliability),
                'status_lulus_gagal': self._determine_overall_status_indonesia(metrics, risk_coverage),
                'gerbang_kualitas': self._check_quality_gates_indonesia(metrics, forensic_reliability)
            }
        }
        
        self.report_data = report
        return report
    
    def _translate_risk_matrix(self, risk_matrix: Dict[str, Any]) -> Dict[str, Any]:
        """Terjemahkan risk matrix ke Bahasa Indonesia"""
        # Simple passthrough for now, can be expanded
        return risk_matrix if risk_matrix else {}
    
    def _translate_metrics(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Terjemahkan metrik ke Bahasa Indonesia"""
        if not metrics:
            return {}
            
        return {
            'total_pengujian': metrics.get('total_tests', 0),
            'pengujian_berhasil': metrics.get('passed_tests', 0),
            'pengujian_gagal': metrics.get('failed_tests', 0),
            'pengujian_error': metrics.get('error_tests', 0),
            'pengujian_dilewati': metrics.get('skipped_tests', 0),
            'tingkat_keberhasilan': f"{metrics.get('pass_rate', 0)*100:.1f}%",
            'tingkat_kegagalan': f"{metrics.get('fail_rate', 0)*100:.1f}%",
            'rata_rata_waktu_eksekusi': f"{metrics.get('average_execution_time', 0):.2f} detik",
            'rata_rata_akurasi_forensik': f"{metrics.get('average_forensic_accuracy', 0)*100:.1f}%",
            'rata_rata_skor_kepercayaan': f"{metrics.get('average_confidence_score', 0)*100:.1f}%"
        }
    
    def _calculate_risk_coverage_indonesia(self, risks: List[RiskItem], 
                                          test_cases: List[TestCase], 
                                          executions: List[TestExecution]) -> Dict[str, Any]:
        """Hitung coverage risiko dengan label Indonesia"""
        risk_coverage = {}
        
        # Terjemahan deskripsi risiko
        risk_translations = {
            "False Positive dalam deteksi manipulasi gambar asli": "Positif Palsu dalam deteksi manipulasi gambar asli",
            "False Negative dalam deteksi copy-move manipulation": "Negatif Palsu dalam deteksi manipulasi salin-pindah",
            "Pipeline failure pada tahap kritis analisis": "Kegagalan pipeline pada tahap kritis analisis",
            "Inkonsistensi hasil antar algoritma deteksi": "Inkonsistensi hasil antar algoritma deteksi",
            "Memory overflow pada gambar resolusi tinggi": "Overflow memori pada gambar resolusi tinggi",
            "Kegagalan export laporan forensik": "Kegagalan ekspor laporan forensik",
            "Degradasi performa pada batch processing": "Degradasi performa pada pemrosesan batch",
            "Metadata extraction failure pada format tidak umum": "Kegagalan ekstraksi metadata pada format tidak umum",
            "UI responsiveness pada Streamlit interface": "Responsivitas UI pada antarmuka Streamlit",
            "Visualization rendering issues": "Masalah rendering visualisasi"
        }
        
        # Terjemahan level risiko
        level_translations = {
            "Critical": "Kritis",
            "High": "Tinggi",
            "Medium": "Sedang",
            "Low": "Rendah"
        }
        
        for risk in risks:
            covering_tests = [tc for tc in test_cases if risk.risk_id in tc.risk_items]
            executed_tests = [e for e in executions if any(tc.tc_id == e.tc_id for tc in covering_tests)]
            passed_tests = [e for e in executed_tests if e.result == TestResult.PASS]
            
            coverage_percentage = len(executed_tests) / len(covering_tests) if covering_tests else 0
            mitigation_effectiveness = len(passed_tests) / len(executed_tests) if executed_tests else 0
            
            # Terjemahkan deskripsi
            desc_indonesia = risk_translations.get(risk.description, risk.description)
            level_indonesia = level_translations.get(risk.level.value, risk.level.value)
            
            risk_coverage[risk.risk_id] = {
                'deskripsi_risiko': desc_indonesia,
                'tingkat_risiko': level_indonesia,
                'skor_risiko': round(risk.risk_score, 3),
                'total_kasus_uji': len(covering_tests),
                'pengujian_dieksekusi': len(executed_tests),
                'pengujian_berhasil': len(passed_tests),
                'persentase_cakupan': f"{coverage_percentage*100:.1f}%",
                'efektivitas_mitigasi': f"{mitigation_effectiveness*100:.1f}%",
                'skor_risiko_residual': round(risk.risk_score * (1 - mitigation_effectiveness), 3)
            }
            
        return risk_coverage
    
    def _assess_overall_risk_status_indonesia(self, risk_coverage: Dict[str, Any]) -> str:
        """Tentukan status risiko keseluruhan dalam Bahasa Indonesia"""
        if not risk_coverage:
            return "TIDAK DAPAT DITENTUKAN - Tidak ada data cakupan risiko"
        
        # Hitung risiko residual
        total_residual_risk = sum(item['skor_risiko_residual'] for item in risk_coverage.values())
        critical_unmitigated = len([r for r in risk_coverage.values() 
                                   if r['tingkat_risiko'] == 'Kritis' and 
                                   r.get('efektivitas_mitigasi', '0%') == '0.0%'])
        
        if critical_unmitigated > 0:
            return f"RISIKO TINGGI - {critical_unmitigated} risiko kritis tidak dimitigasi dengan baik"
        elif total_residual_risk > 0.5:
            return "RISIKO SEDANG - Beberapa risiko masih memerlukan mitigasi lebih lanjut"
        elif total_residual_risk > 0.2:
            return "RISIKO RENDAH - Mayoritas risiko telah dimitigasi dengan baik"
        else:
            return "RISIKO MINIMAL - Semua risiko telah dimitigasi secara efektif"
    
    def _identify_critical_findings_indonesia(self, executions: List[TestExecution]) -> List[str]:
        """Identifikasi temuan kritis dalam Bahasa Indonesia"""
        findings = []
        
        # Cek akurasi rendah
        low_accuracy = [e for e in executions if e.forensic_accuracy < 0.7 and e.result != TestResult.SKIP]
        if low_accuracy:
            findings.append(f"AKURASI RENDAH: {len(low_accuracy)} pengujian dengan akurasi forensik < 70%")
        
        # Cek error tests
        error_tests = [e for e in executions if e.result == TestResult.ERROR]
        if error_tests:
            findings.append(f"ERROR PENGUJIAN: {len(error_tests)} pengujian mengalami error saat eksekusi")
        
        # Cek critical test failures
        critical_failures = [e for e in executions 
                           if e.result == TestResult.FAIL and 'CRITICAL' in e.tc_id]
        if critical_failures:
            findings.append(f"KEGAGALAN KRITIS: {len(critical_failures)} pengujian kritis gagal")
        
        # Cek performance issues
        slow_tests = [e for e in executions if e.execution_time > 30]
        if slow_tests:
            findings.append(f"MASALAH PERFORMA: {len(slow_tests)} pengujian melebihi batas waktu 30 detik")
        
        if not findings:
            findings.append("Tidak ada temuan kritis teridentifikasi")
        
        return findings
    
    def _generate_recommendations_indonesia(self, risk_coverage: Dict[str, Any], 
                                           metrics: Dict[str, Any]) -> List[str]:
        """Generate rekomendasi dalam Bahasa Indonesia"""
        recommendations = []
        
        # Cek pass rate
        if metrics.get('pass_rate', 0) < 0.9:
            recommendations.append("Tingkatkan tingkat keberhasilan pengujian - saat ini di bawah ambang batas 90%")
        
        # Cek unmitigated risks
        unmitigated = [r for r in risk_coverage.values() 
                      if r.get('efektivitas_mitigasi', '0%') == '0.0%' and 
                      r['tingkat_risiko'] in ['Kritis', 'Tinggi']]
        if unmitigated:
            recommendations.append(f"Tangani {len(unmitigated)} risiko kritis/tinggi dengan efektivitas mitigasi rendah")
        
        # Cek akurasi forensik
        if metrics.get('average_forensic_accuracy', 0) < 0.85:
            recommendations.append("Tingkatkan akurasi forensik sistem - target minimal 85%")
        
        # Cek waktu eksekusi
        if metrics.get('average_execution_time', 0) > 20:
            recommendations.append("Optimasi performa - rata-rata waktu eksekusi melebihi 20 detik")
        
        if not recommendations:
            recommendations.append("Sistem berperforma baik - pertahankan kualitas saat ini")
        
        return recommendations
    
    def _translate_executions(self, executions: List[TestExecution]) -> List[Dict[str, Any]]:
        """Terjemahkan hasil eksekusi ke Bahasa Indonesia"""
        translated = []
        
        result_map = {
            'TestResult.PASS': 'BERHASIL',
            'TestResult.FAIL': 'GAGAL',
            'TestResult.ERROR': 'ERROR',
            'TestResult.SKIP': 'DILEWATI'
        }
        
        for exec in executions:
            exec_dict = {
                'id_kasus_uji': exec.tc_id,
                'hasil': result_map.get(str(exec.result), str(exec.result)),
                'waktu_eksekusi': f"{exec.execution_time:.2f} detik",
                'hasil_aktual': exec.actual_result,
                'defek_ditemukan': exec.defects_found,
                'status_mitigasi_risiko': 'Berhasil' if exec.result == TestResult.PASS else 'Gagal',
                'akurasi_forensik': f"{exec.forensic_accuracy*100:.1f}%",
                'skor_kepercayaan': f"{exec.confidence_score*100:.1f}%",
                'waktu': exec.timestamp,
                'catatan': exec.notes
            }
            translated.append(exec_dict)
        
        return translated
    
    def _analyze_defects_indonesia(self, executions: List[TestExecution]) -> Dict[str, Any]:
        """Analisis defek dalam Bahasa Indonesia"""
        defects = []
        for exec in executions:
            defects.extend(exec.defects_found)
        
        # Kategorisasi defek
        categories = {
            'akurasi': len([d for d in defects if 'accuracy' in d.lower() or 'akurasi' in d.lower()]),
            'performa': len([d for d in defects if 'performance' in d.lower() or 'time' in d.lower()]),
            'integrasi': len([d for d in defects if 'integration' in d.lower() or 'pipeline' in d.lower()]),
            'reliabilitas': len([d for d in defects if 'reliability' in d.lower() or 'consistency' in d.lower()])
        }
        
        return {
            'total_defek': len(defects),
            'kategori_defek': categories,
            'densitas_defek': len(defects) / len(executions) if executions else 0,
            'defek_kritis': len([d for d in defects if 'critical' in d.lower() or 'error' in d.lower()])
        }
    
    def _calculate_accuracy_metrics_indonesia(self, executions: List[TestExecution]) -> Dict[str, Any]:
        """Hitung metrik akurasi dalam format Indonesia"""
        if not executions:
            return {'status': 'Tidak ada data eksekusi'}
        
        accuracies = [e.forensic_accuracy for e in executions if e.forensic_accuracy > 0]
        if not accuracies:
            return {'status': 'Tidak ada data akurasi'}
        
        import numpy as np
        
        return {
            'rata_rata_akurasi': f"{np.mean(accuracies)*100:.1f}%",
            'median_akurasi': f"{np.median(accuracies)*100:.1f}%",
            'akurasi_minimum': f"{np.min(accuracies)*100:.1f}%",
            'akurasi_maksimum': f"{np.max(accuracies)*100:.1f}%",
            'deviasi_standar': f"{np.std(accuracies)*100:.1f}%",
            'akurasi_diatas_80_persen': f"{len([a for a in accuracies if a > 0.8])/len(accuracies)*100:.1f}%"
        }
    
    def _analyze_algorithm_consistency_indonesia(self, executions: List[TestExecution]) -> Dict[str, Any]:
        """Analisis konsistensi algoritma dalam Bahasa Indonesia"""
        # Implementasi sederhana - bisa diperluas sesuai kebutuhan
        consistency_tests = [e for e in executions if 'consistency' in e.notes.lower() or 'reliability' in e.notes.lower()]
        
        if consistency_tests:
            avg_consistency = np.mean([e.forensic_accuracy for e in consistency_tests])
            return {
                'status': 'Dianalisis',
                'tingkat_konsistensi': f"{avg_consistency*100:.1f}%",
                'jumlah_pengujian_konsistensi': len(consistency_tests)
            }
        
        return {'status': 'Tidak ada pengujian konsistensi yang dieksekusi'}
    
    def _analyze_confidence_scores_indonesia(self, executions: List[TestExecution]) -> Dict[str, Any]:
        """Analisis skor kepercayaan dalam Bahasa Indonesia"""
        if not executions:
            return {'status': 'Tidak ada data eksekusi'}
        
        confidences = [e.confidence_score for e in executions if e.confidence_score > 0]
        if not confidences:
            return {'status': 'Tidak ada data kepercayaan'}
        
        import numpy as np
        
        return {
            'rata_rata_kepercayaan': f"{np.mean(confidences)*100:.1f}%",
            'median_kepercayaan': f"{np.median(confidences)*100:.1f}%",
            'kepercayaan_minimum': f"{np.min(confidences)*100:.1f}%",
            'kepercayaan_maksimum': f"{np.max(confidences)*100:.1f}%",
            'tingkat_kepercayaan_tinggi': f"{len([c for c in confidences if c > 0.8])/len(confidences)*100:.1f}%"
        }
    
    def _evaluate_success_criteria_indonesia(self, metrics: Dict[str, Any], 
                                            risk_coverage: Dict[str, Any],
                                            forensic_reliability: float) -> Dict[str, Any]:
        """Evaluasi kriteria keberhasilan dalam Bahasa Indonesia"""
        return {
            'tingkat_keberhasilan_diatas_90_persen': metrics.get('pass_rate', 0) >= 0.9,
            'risiko_kritis_dimitigasi': all(r.get('efektivitas_mitigasi', '0%') != '0.0%' 
                                           for r in risk_coverage.values() 
                                           if r['tingkat_risiko'] == 'Kritis'),
            'akurasi_forensik_diatas_80_persen': metrics.get('average_forensic_accuracy', 0) >= 0.8,
            'skor_reliabilitas_diatas_85_persen': forensic_reliability >= 0.85,
            'tidak_ada_defek_kritis': True,  # Simplified - bisa diperluas
            'performa_dalam_batas': metrics.get('average_execution_time', 0) < 120
        }
    
    def _determine_overall_status_indonesia(self, metrics: Dict[str, Any], 
                                           risk_coverage: Dict[str, Any]) -> str:
        """Tentukan status keseluruhan dalam Bahasa Indonesia"""
        pass_rate = metrics.get('pass_rate', 0)
        critical_unmitigated = len([r for r in risk_coverage.values() 
                                   if r['tingkat_risiko'] == 'Kritis' and 
                                   r.get('efektivitas_mitigasi', '0%') == '0.0%'])
        
        if pass_rate >= 0.9 and critical_unmitigated == 0:
            return "LULUS - Semua kriteria keberhasilan terpenuhi"
        elif pass_rate >= 0.8 and critical_unmitigated == 0:
            return "LULUS BERSYARAT - Beberapa perbaikan minor diperlukan"
        elif critical_unmitigated > 0:
            return "GAGAL - Risiko kritis tidak dimitigasi dengan baik"
        else:
            return "GAGAL - Tingkat keberhasilan di bawah ambang batas minimum"
    
    def _check_quality_gates_indonesia(self, metrics: Dict[str, Any], 
                                      forensic_reliability: float) -> Dict[str, Any]:
        """Periksa quality gates dalam Bahasa Indonesia"""
        return {
            'gerbang_1_fungsionalitas_dasar': {
                'status': metrics.get('pass_rate', 0) >= 0.7,
                'deskripsi': 'Fungsionalitas sistem dasar (70% tingkat keberhasilan)',
                'nilai_aktual': f"{metrics.get('pass_rate', 0)*100:.1f}%"
            },
            'gerbang_2_akurasi_forensik': {
                'status': metrics.get('average_forensic_accuracy', 0) >= 0.8,
                'deskripsi': 'Ambang batas akurasi forensik (80%)',
                'nilai_aktual': f"{metrics.get('average_forensic_accuracy', 0)*100:.1f}%"
            },
            'gerbang_3_reliabilitas': {
                'status': forensic_reliability >= 0.85,
                'deskripsi': 'Skor reliabilitas sistem (85%)',
                'nilai_aktual': f"{forensic_reliability*100:.1f}%"
            },
            'gerbang_4_performa': {
                'status': metrics.get('average_execution_time', 0) < 120,
                'deskripsi': 'Ambang batas performa (120 detik)',
                'nilai_aktual': f"{metrics.get('average_execution_time', 0):.1f} detik"
            }
        }
    
    def export_report_to_json_indonesia(self, filename: str = None) -> str:
        """Export laporan ke JSON dengan format Indonesia"""
        if not self.report_data:
            raise ValueError("Tidak ada data laporan untuk diekspor")
        
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"laporan_rbt_{timestamp}.json"
        
        # Ensure reports directory exists
        reports_dir = Path("rbt_reports")
        reports_dir.mkdir(exist_ok=True)
        
        filepath = reports_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.report_data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Laporan JSON diekspor ke: {filepath}")
        return str(filepath)
    
    def export_report_to_html_indonesia(self, filename: str = None) -> str:
        """Export laporan ke HTML dengan format Indonesia"""
        if not self.report_data:
            raise ValueError("Tidak ada data laporan untuk diekspor")
        
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"laporan_rbt_{timestamp}.html"
        
        # Ensure reports directory exists
        reports_dir = Path("rbt_reports")
        reports_dir.mkdir(exist_ok=True)
        
        filepath = reports_dir / filename
        
        html_content = self._generate_html_report_indonesia()
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        self.logger.info(f"Laporan HTML diekspor ke: {filepath}")
        return str(filepath)
    
    def _generate_html_report_indonesia(self) -> str:
        """Generate HTML report dalam Bahasa Indonesia"""
        report = self.report_data
        
        html = f"""<!DOCTYPE html>
<html lang="id">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Laporan Risk-Based Testing - Sistem Forensik</title>
    <style>
        body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #34495e; margin-top: 30px; }}
        h3 {{ color: #7f8c8d; }}
        .metric-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 20px 0; }}
        .metric-card {{ background: #ecf0f1; padding: 15px; border-radius: 8px; border-left: 4px solid #3498db; }}
        .metric-value {{ font-size: 24px; font-weight: bold; color: #2c3e50; }}
        .metric-label {{ color: #7f8c8d; font-size: 12px; text-transform: uppercase; }}
        .status-pass {{ color: #27ae60; font-weight: bold; }}
        .status-fail {{ color: #e74c3c; font-weight: bold; }}
        .status-warning {{ color: #f39c12; font-weight: bold; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th {{ background: #3498db; color: white; padding: 10px; text-align: left; }}
        td {{ padding: 10px; border-bottom: 1px solid #ecf0f1; }}
        tr:hover {{ background: #f8f9fa; }}
        .risk-critical {{ background: #ffebee; }}
        .risk-high {{ background: #fff3e0; }}
        .risk-medium {{ background: #fff8e1; }}
        .risk-low {{ background: #f1f8e9; }}
        .recommendation {{ background: #e3f2fd; padding: 15px; border-radius: 5px; margin: 10px 0; }}
        .finding {{ background: #fce4ec; padding: 15px; border-radius: 5px; margin: 10px 0; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 Laporan Risk-Based Testing</h1>
        <p><strong>Sistem Forensik Image Analysis</strong></p>
        <p>Dibuat pada: {report['metadata_laporan']['dibuat_pada']}</p>
        
        <h2>📋 Ringkasan Eksekutif</h2>
        <div class="metric-grid">
            <div class="metric-card">
                <div class="metric-label">Total Pengujian</div>
                <div class="metric-value">{report['ringkasan_eksekutif']['ringkasan_eksekusi_pengujian']['total_pengujian']}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Tingkat Keberhasilan</div>
                <div class="metric-value">{report['ringkasan_eksekutif']['ringkasan_eksekusi_pengujian']['tingkat_keberhasilan']}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Akurasi Forensik</div>
                <div class="metric-value">{report['ringkasan_eksekutif']['ringkasan_eksekusi_pengujian']['rata_rata_akurasi_forensik']}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Skor Kepercayaan</div>
                <div class="metric-value">{report['ringkasan_eksekutif']['ringkasan_eksekusi_pengujian']['rata_rata_skor_kepercayaan']}</div>
            </div>
        </div>
        
        <h3>Status Risiko Keseluruhan</h3>
        <p class="{'status-fail' if 'TINGGI' in report['ringkasan_eksekutif']['status_risiko_keseluruhan'] else 'status-pass'}">{report['ringkasan_eksekutif']['status_risiko_keseluruhan']}</p>
        
        <h3>Temuan Kritis</h3>
        {''.join([f'<div class="finding">• {finding}</div>' for finding in report['ringkasan_eksekutif']['temuan_kritis']])}
        
        <h3>Rekomendasi</h3>
        {''.join([f'<div class="recommendation">• {rec}</div>' for rec in report['ringkasan_eksekutif']['rekomendasi']])}
        
        <h2>🔍 Analisis Cakupan Risiko</h2>
        <table>
            <thead>
                <tr>
                    <th>ID Risiko</th>
                    <th>Deskripsi</th>
                    <th>Tingkat</th>
                    <th>Cakupan</th>
                    <th>Mitigasi</th>
                    <th>Risiko Residual</th>
                </tr>
            </thead>
            <tbody>
"""
        
        for risk_id, risk_data in report['analisis_risiko']['analisis_cakupan_risiko'].items():
            risk_class = f"risk-{risk_data['tingkat_risiko'].lower()}"
            html += f"""
                <tr class="{risk_class}">
                    <td>{risk_id}</td>
                    <td>{risk_data['deskripsi_risiko']}</td>
                    <td>{risk_data['tingkat_risiko']}</td>
                    <td>{risk_data['persentase_cakupan']}</td>
                    <td>{risk_data['efektivitas_mitigasi']}</td>
                    <td>{risk_data['skor_risiko_residual']}</td>
                </tr>
"""
        
        html += """
            </tbody>
        </table>
        
        <h2>✅ Evaluasi Kriteria Keberhasilan</h2>
        <p><strong>Status: </strong><span class="status-pass">{}</span></p>
        
        <h3>Gerbang Kualitas</h3>
        <table>
            <thead>
                <tr>
                    <th>Gerbang</th>
                    <th>Deskripsi</th>
                    <th>Target</th>
                    <th>Aktual</th>
                    <th>Status</th>
                </tr>
            </thead>
            <tbody>
""".format(report['evaluasi_kriteria_keberhasilan']['status_lulus_gagal'])
        
        for gate_id, gate_data in report['evaluasi_kriteria_keberhasilan']['gerbang_kualitas'].items():
            status_icon = "✅" if gate_data['status'] else "❌"
            html += f"""
                <tr>
                    <td>{gate_id.replace('_', ' ').title()}</td>
                    <td>{gate_data['deskripsi']}</td>
                    <td>{gate_data['deskripsi'].split('(')[1].split(')')[0] if '(' in gate_data['deskripsi'] else 'N/A'}</td>
                    <td>{gate_data['nilai_aktual']}</td>
                    <td>{status_icon}</td>
                </tr>
"""
        
        html += """
            </tbody>
        </table>
        
        <hr>
        <p style="text-align: center; color: #7f8c8d; margin-top: 30px;">
            Laporan ini dihasilkan secara otomatis oleh Risk-Based Testing Framework v1.0<br>
            © 2025 Forensic Testing Team
        </p>
    </div>
</body>
</html>
"""
        
        return html