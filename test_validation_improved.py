#!/usr/bin/env python3
"""
Test script untuk memverifikasi peningkatan skor validasi setelah perbaikan
"""

from validator import ForensicValidator
from main import analyze_image_comprehensive_advanced

def test_validation_improvements():
    """Test peningkatan skor validasi"""
    print("=" * 60)
    print("TEST VALIDASI SETELAH PERBAIKAN")
    print("=" * 60)
    
    # Analisis gambar
    print("Menganalisis gambar splicing.jpg...")
    results = analyze_image_comprehensive_advanced('splicing.jpg', test_mode=True)
    
    # Inisialisasi validator
    validator = ForensicValidator()
    
    # Validasi setiap komponen
    cluster_conf, cluster_det = validator.validate_clustering(results)
    loc_conf, loc_det = validator.validate_localization(results)
    ela_conf, ela_det = validator.validate_ela(results)
    feat_conf, feat_det = validator.validate_feature_matching(results)
    metadata_conf, metadata_det = validator.validate_metadata(results)
    
    print("\n=== HASIL VALIDASI SETELAH PERBAIKAN ===")
    print(f"K-Means Clustering: {cluster_conf*100:.1f}%")
    print(f"Lokalisasi: {loc_conf*100:.1f}%")
    print(f"ELA: {ela_conf*100:.1f}%")
    print(f"SIFT Feature Matching: {feat_conf*100:.1f}%")
    print(f"Metadata: {metadata_conf*100:.1f}%")
    
    print("\n=== STATUS VALIDASI ===")
    print(f"Clustering: {'✅ LULUS' if cluster_conf >= validator.thresholds['clustering'] else '❌ GAGAL'} - Threshold: {validator.thresholds['clustering']*100:.1f}%")
    print(f"Lokalisasi: {'✅ LULUS' if loc_conf >= validator.thresholds['localization'] else '❌ GAGAL'} - Threshold: {validator.thresholds['localization']*100:.1f}%")
    print(f"ELA: {'✅ LULUS' if ela_conf >= validator.thresholds['ela'] else '❌ GAGAL'} - Threshold: {validator.thresholds['ela']*100:.1f}%")
    print(f"Feature Matching: {'✅ LULUS' if feat_conf >= validator.thresholds['feature_matching'] else '❌ GAGAL'} - Threshold: {validator.thresholds['feature_matching']*100:.1f}%")
    print(f"Metadata: {'✅ LULUS' if metadata_conf >= validator.thresholds['metadata'] else '❌ GAGAL'} - Threshold: {validator.thresholds['metadata']*100:.1f}%")
    
    # Hitung jumlah yang lulus
    passed_tests = 0
    total_tests = 5
    
    if cluster_conf >= validator.thresholds['clustering']:
        passed_tests += 1
    if loc_conf >= validator.thresholds['localization']:
        passed_tests += 1
    if ela_conf >= validator.thresholds['ela']:
        passed_tests += 1
    if feat_conf >= validator.thresholds['feature_matching']:
        passed_tests += 1
    if metadata_conf >= validator.thresholds['metadata']:
        passed_tests += 1
    
    print("\n=== RINGKASAN ===")
    print(f"Tests Passed: {passed_tests}/{total_tests}")
    print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
    
    if passed_tests >= 4:
        print("🎉 EXCELLENT! Perbaikan berhasil meningkatkan performa sistem!")
    elif passed_tests >= 3:
        print("👍 GOOD! Perbaikan menunjukkan peningkatan yang signifikan!")
    elif passed_tests >= 2:
        print("👌 FAIR! Ada peningkatan, tapi masih perlu perbaikan lebih lanjut.")
    else:
        print("⚠️ NEEDS IMPROVEMENT! Perbaikan belum optimal.")
    
    return passed_tests, total_tests

if __name__ == "__main__":
    test_validation_improvements()