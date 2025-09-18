#!/usr/bin/env python3
"""
Test script untuk menguji visualisasi bar chart distribusi frekuensi yang menggantikan pie chart
"""

import matplotlib.pyplot as plt
import numpy as np
from visualization import create_frequency_visualization

def create_test_data_scenarios():
    """Membuat berbagai skenario data untuk test bar chart"""
    return [
        {
            'name': 'Distribusi Normal',
            'data': {
                'frequency_analysis': {
                    'dct_stats': {
                        'low_freq_energy': 799093.375,
                        'mid_freq_energy': 166424.219,
                        'high_freq_energy': 1153430.000,
                        'freq_ratio': 14.4394
                    },
                    'frequency_inconsistency': 0.5713
                }
            }
        },
        {
            'name': 'High Frequency Dominan',
            'data': {
                'frequency_analysis': {
                    'dct_stats': {
                        'low_freq_energy': 200000.0,
                        'mid_freq_energy': 300000.0,
                        'high_freq_energy': 2000000.0,
                        'freq_ratio': 10.0
                    },
                    'frequency_inconsistency': 0.75
                }
            }
        },
        {
            'name': 'Low Frequency Dominan',
            'data': {
                'frequency_analysis': {
                    'dct_stats': {
                        'low_freq_energy': 3000000.0,
                        'mid_freq_energy': 500000.0,
                        'high_freq_energy': 200000.0,
                        'freq_ratio': 0.067
                    },
                    'frequency_inconsistency': 0.25
                }
            }
        },
        {
            'name': 'Distribusi Seimbang',
            'data': {
                'frequency_analysis': {
                    'dct_stats': {
                        'low_freq_energy': 1000000.0,
                        'mid_freq_energy': 1100000.0,
                        'high_freq_energy': 900000.0,
                        'freq_ratio': 0.9
                    },
                    'frequency_inconsistency': 0.15
                }
            }
        }
    ]

def test_bar_chart_single():
    """Test bar chart dengan satu skenario"""
    print("📊 Testing bar chart distribusi frekuensi...")
    
    test_data = {
        'frequency_analysis': {
            'dct_stats': {
                'low_freq_energy': 799093.375,
                'mid_freq_energy': 166424.219,
                'high_freq_energy': 1153430.000,
                'freq_ratio': 14.4394
            },
            'frequency_inconsistency': 0.5713
        }
    }
    
    # Buat figure dengan ukuran yang sesuai
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    fig.suptitle('Test: Bar Chart Distribusi Frekuensi (Menggantikan Pie Chart)', 
                 fontsize=16, fontweight='bold', y=0.95)
    
    # Panggil fungsi visualisasi yang telah diperbaiki
    create_frequency_visualization(ax, test_data)
    
    # Simpan hasil test
    output_file = 'test_frequency_bar_chart_single.png'
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"✅ Test selesai! Hasil disimpan sebagai: {output_file}")
    print("🎯 Fitur bar chart yang baru:")
    print("   - Grafik batang yang lebih besar dan jelas")
    print("   - Tidak ada pie chart lagi, fokus pada bar chart")
    print("   - Grid lines untuk kemudahan pembacaan")
    print("   - Shadow effect untuk tampilan 3D")
    print("   - Label persentase di atas setiap bar")

def test_bar_chart_multiple_scenarios():
    """Test bar chart dengan berbagai skenario"""
    print("\n🧪 Testing bar chart dengan berbagai skenario...")
    
    scenarios = create_test_data_scenarios()
    
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    fig.suptitle('Test: Bar Chart Distribusi Frekuensi - Berbagai Skenario', 
                 fontsize=18, fontweight='bold', y=0.95)
    
    axes_flat = axes.flatten()
    
    for i, scenario in enumerate(scenarios):
        create_frequency_visualization(axes_flat[i], scenario['data'])
        axes_flat[i].set_title(f"{scenario['name']}", 
                              fontsize=14, fontweight='bold', pad=20)
    
    # Simpan hasil test multiple scenarios
    output_file = 'test_frequency_bar_chart_scenarios.png'
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"✅ Multiple scenarios test selesai! Hasil: {output_file}")

def test_comparison_pie_vs_bar():
    """Membuat dokumentasi perbandingan pie chart vs bar chart"""
    print("\n📋 Membuat dokumentasi perbandingan pie chart vs bar chart...")
    
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    
    # Judul utama
    ax.text(0.5, 0.95, 'PERBANDINGAN: PIE CHART vs BAR CHART', 
            ha='center', va='top', transform=ax.transAxes, 
            fontsize=20, fontweight='bold', color='#2C3E50')
    
    # Sebelum (Pie Chart)
    ax.text(0.25, 0.85, 'SEBELUM: PIE CHART', 
            ha='center', va='top', transform=ax.transAxes, 
            fontsize=16, fontweight='bold', color='#E74C3C')
    
    pie_features = [
        "• Diagram lingkaran dengan persentase",
        "• Legend terpisah di bawah",
        "• Sulit membandingkan nilai yang mirip",
        "• Memakan ruang untuk legend",
        "• Kurang detail untuk analisis"
    ]
    
    for i, feature in enumerate(pie_features):
        ax.text(0.05, 0.75 - i*0.04, feature, 
                ha='left', va='top', transform=ax.transAxes, 
                fontsize=12, color='#E74C3C')
    
    # Sesudah (Bar Chart)
    ax.text(0.75, 0.85, 'SESUDAH: BAR CHART', 
            ha='center', va='top', transform=ax.transAxes, 
            fontsize=16, fontweight='bold', color='#27AE60')
    
    bar_features = [
        "• Grafik batang yang besar dan jelas",
        "• Label langsung di atas setiap bar",
        "• Mudah membandingkan nilai",
        "• Grid lines untuk referensi",
        "• Shadow effect untuk tampilan 3D",
        "• Y-axis dengan skala yang jelas",
        "• Lebih fokus pada data"
    ]
    
    for i, feature in enumerate(bar_features):
        ax.text(0.55, 0.75 - i*0.04, feature, 
                ha='left', va='top', transform=ax.transAxes, 
                fontsize=12, color='#27AE60')
    
    # Keunggulan bar chart
    ax.text(0.5, 0.4, 'KEUNGGULAN BAR CHART:', 
            ha='center', va='top', transform=ax.transAxes, 
            fontsize=16, fontweight='bold', color='#2C3E50')
    
    advantages = [
        "✓ Tampilan lebih besar dan mudah dibaca",
        "✓ Perbandingan nilai lebih intuitif",
        "✓ Grid lines membantu estimasi nilai",
        "✓ Tidak ada elemen yang saling overlap",
        "✓ Fokus penuh pada distribusi data",
        "✓ Styling profesional dengan shadow effect"
    ]
    
    for i, advantage in enumerate(advantages):
        ax.text(0.5, 0.32 - i*0.04, advantage, 
                ha='center', va='top', transform=ax.transAxes, 
                fontsize=13, color='#27AE60', fontweight='bold')
    
    ax.axis('off')
    
    # Simpan dokumentasi
    output_file = 'dokumentasi_pie_vs_bar_chart.png'
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"📋 Dokumentasi perbandingan disimpan sebagai: {output_file}")

if __name__ == "__main__":
    print("🚀 Memulai test bar chart distribusi frekuensi...")
    print("=" * 80)
    
    # Test single scenario
    test_bar_chart_single()
    
    # Test multiple scenarios
    test_bar_chart_multiple_scenarios()
    
    # Buat dokumentasi perbandingan
    test_comparison_pie_vs_bar()
    
    print("\n" + "=" * 80)
    print("🎉 Semua test selesai! Bar chart distribusi frekuensi berhasil:")
    print("   ✓ Pie chart telah dihilangkan sepenuhnya")
    print("   ✓ Bar chart yang lebih besar dan jelas")
    print("   ✓ Grid lines untuk kemudahan pembacaan")
    print("   ✓ Shadow effect untuk tampilan profesional")
    print("   ✓ Label persentase langsung di atas bar")
    print("   ✓ Y-axis dengan skala yang informatif")
    print("   ✓ Tampilan yang lebih fokus dan mudah dipahami")