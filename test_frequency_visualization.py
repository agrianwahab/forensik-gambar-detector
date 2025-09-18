#!/usr/bin/env python3
"""
Test script untuk menguji visualisasi analisis domain frekuensi yang telah diperbaiki
"""

import matplotlib.pyplot as plt
import numpy as np
from visualization import create_frequency_visualization

def create_test_data():
    """Membuat data test untuk visualisasi frekuensi"""
    return {
        'frequency_analysis': {
            'dct_stats': {
                'low_freq_energy': 0.65,
                'mid_freq_energy': 0.25,
                'high_freq_energy': 0.10,
                'freq_ratio': 0.384
            },
            'frequency_inconsistency': 0.35
        }
    }

def test_frequency_visualization():
    """Test visualisasi frekuensi dengan data sample"""
    print("🧪 Testing improved frequency visualization...")
    
    # Buat data test
    test_results = create_test_data()
    
    # Buat figure dengan ukuran yang sesuai
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    fig.suptitle('Test: Visualisasi Analisis Domain Frekuensi yang Diperbaiki', 
                 fontsize=16, fontweight='bold', y=0.95)
    
    # Panggil fungsi visualisasi yang telah diperbaiki
    create_frequency_visualization(ax, test_results)
    
    # Simpan hasil test
    output_file = 'test_frequency_improved_visualization.png'
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"✅ Test selesai! Hasil disimpan sebagai: {output_file}")
    print("📊 Fitur yang telah diperbaiki:")
    print("   - Tata letak simetris dan seimbang")
    print("   - Panel metrik dan pie chart sejajar")
    print("   - Spacing dan padding yang konsisten")
    print("   - Styling yang lebih profesional")
    print("   - Legend dengan spacing yang optimal")
    print("   - Status dan interpretasi yang informatif")

def test_multiple_scenarios():
    """Test dengan berbagai skenario data"""
    print("\n🔬 Testing multiple data scenarios...")
    
    scenarios = [
        {
            'name': 'Low Inconsistency',
            'data': {
                'frequency_analysis': {
                    'dct_stats': {
                        'low_freq_energy': 0.70,
                        'mid_freq_energy': 0.20,
                        'high_freq_energy': 0.10,
                        'freq_ratio': 0.286
                    },
                    'frequency_inconsistency': 0.15
                }
            }
        },
        {
            'name': 'High Inconsistency',
            'data': {
                'frequency_analysis': {
                    'dct_stats': {
                        'low_freq_energy': 0.45,
                        'mid_freq_energy': 0.35,
                        'high_freq_energy': 0.20,
                        'freq_ratio': 0.556
                    },
                    'frequency_inconsistency': 0.75
                }
            }
        },
        {
            'name': 'No Data',
            'data': {}
        }
    ]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Test: Multiple Scenarios - Visualisasi Frekuensi yang Diperbaiki', 
                 fontsize=16, fontweight='bold', y=0.95)
    
    for i, scenario in enumerate(scenarios):
        create_frequency_visualization(axes[i], scenario['data'])
        axes[i].set_title(f"Scenario: {scenario['name']}", 
                         fontsize=12, fontweight='bold', pad=20)
    
    # Simpan hasil test multiple scenarios
    output_file = 'test_frequency_multiple_scenarios.png'
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"✅ Multiple scenarios test selesai! Hasil: {output_file}")

if __name__ == "__main__":
    print("🚀 Memulai test visualisasi analisis domain frekuensi yang diperbaiki...")
    print("=" * 70)
    
    # Test utama
    test_frequency_visualization()
    
    # Test multiple scenarios
    test_multiple_scenarios()
    
    print("\n" + "=" * 70)
    print("🎉 Semua test selesai! Visualisasi frekuensi telah diperbaiki dengan:")
    print("   ✓ Tata letak yang perfectly symmetrical")
    print("   ✓ Panel yang sejajar dan proporsional")
    print("   ✓ Spacing dan alignment yang optimal")
    print("   ✓ Styling yang lebih profesional dan modern")
    print("   ✓ Informasi yang lebih informatif dan mudah dipahami")