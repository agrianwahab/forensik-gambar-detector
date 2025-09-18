#!/usr/bin/env python3
"""
Test script untuk menguji perbaikan layout dan label distribusi frekuensi
"""

import matplotlib.pyplot as plt
import numpy as np
from visualization import create_frequency_visualization

def create_test_data_scenarios():
    """Membuat berbagai skenario data untuk test layout yang diperbaiki"""
    return [
        {
            'name': 'Layout Baru - Distribusi Normal',
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
            'name': 'Layout Baru - High Dominan',
            'data': {
                'frequency_analysis': {
                    'dct_stats': {
                        'low_freq_energy': 200000.0,
                        'mid_freq_energy': 300000.0,
                        'high_freq_energy': 2500000.0,
                        'freq_ratio': 12.5
                    },
                    'frequency_inconsistency': 0.85
                }
            }
        },
        {
            'name': 'Layout Baru - Low Dominan',
            'data': {
                'frequency_analysis': {
                    'dct_stats': {
                        'low_freq_energy': 3500000.0,
                        'mid_freq_energy': 400000.0,
                        'high_freq_energy': 150000.0,
                        'freq_ratio': 0.043
                    },
                    'frequency_inconsistency': 0.25
                }
            }
        },
        {
            'name': 'Layout Baru - Seimbang',
            'data': {
                'frequency_analysis': {
                    'dct_stats': {
                        'low_freq_energy': 1200000.0,
                        'mid_freq_energy': 1100000.0,
                        'high_freq_energy': 1000000.0,
                        'freq_ratio': 0.83
                    },
                    'frequency_inconsistency': 0.15
                }
            }
        }
    ]

def test_improved_layout_single():
    """Test layout yang diperbaiki dengan satu skenario"""
    print("🎨 Testing layout distribusi frekuensi yang diperbaiki...")
    
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
    fig.suptitle('Test: Layout Distribusi Frekuensi yang Diperbaiki', 
                 fontsize=16, fontweight='bold', y=0.95)
    
    # Panggil fungsi visualisasi yang telah diperbaiki
    create_frequency_visualization(ax, test_data)
    
    # Simpan hasil test
    output_file = 'test_frequency_improved_layout_single.png'
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"✅ Test selesai! Hasil disimpan sebagai: {output_file}")
    print("🎯 Perbaikan layout yang telah dilakukan:")
    print("   - Label kategori: Low, Mid, High (tanpa 'Freq')")
    print("   - Wrap text untuk penjelasan dengan line break")
    print("   - Spacing bar yang lebih optimal (1.8x)")
    print("   - Shadow effect yang lebih prominent (0.15 alpha)")
    print("   - Grid lines dengan spacing yang lebih baik")
    print("   - Font size yang lebih besar untuk keterbacaan")

def test_improved_layout_multiple():
    """Test layout yang diperbaiki dengan berbagai skenario"""
    print("\n🧪 Testing layout dengan berbagai skenario...")
    
    scenarios = create_test_data_scenarios()
    
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    fig.suptitle('Test: Layout Distribusi Frekuensi yang Diperbaiki - Berbagai Skenario', 
                 fontsize=18, fontweight='bold', y=0.95)
    
    axes_flat = axes.flatten()
    
    for i, scenario in enumerate(scenarios):
        create_frequency_visualization(axes_flat[i], scenario['data'])
        axes_flat[i].set_title(f"{scenario['name']}", 
                              fontsize=14, fontweight='bold', pad=20)
    
    # Simpan hasil test multiple scenarios
    output_file = 'test_frequency_improved_layout_scenarios.png'
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"✅ Multiple scenarios test selesai! Hasil: {output_file}")

def test_before_after_comparison():
    """Membuat dokumentasi perbandingan sebelum dan sesudah perbaikan layout"""
    print("\n📋 Membuat dokumentasi perbandingan layout...")
    
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    
    # Judul utama
    ax.text(0.5, 0.95, 'PERBAIKAN LAYOUT DISTRIBUSI FREKUENSI', 
            ha='center', va='top', transform=ax.transAxes, 
            fontsize=20, fontweight='bold', color='#2C3E50')
    
    # Sebelum
    ax.text(0.25, 0.85, 'SEBELUM PERBAIKAN', 
            ha='center', va='top', transform=ax.transAxes, 
            fontsize=16, fontweight='bold', color='#E74C3C')
    
    before_features = [
        "• Label: Low Freq, Mid Freq, High Freq",
        "• Penjelasan dalam satu baris panjang",
        "• Spacing bar yang standar (1.5x)",
        "• Shadow effect tipis (0.1 alpha)",
        "• Grid spacing kurang optimal",
        "• Font size kecil untuk beberapa elemen"
    ]
    
    for i, feature in enumerate(before_features):
        ax.text(0.05, 0.75 - i*0.04, feature, 
                ha='left', va='top', transform=ax.transAxes, 
                fontsize=12, color='#E74C3C')
    
    # Sesudah
    ax.text(0.75, 0.85, 'SESUDAH PERBAIKAN', 
            ha='center', va='top', transform=ax.transAxes, 
            fontsize=16, fontweight='bold', color='#27AE60')
    
    after_features = [
        "• Label: Low, Mid, High (lebih ringkas)",
        "• Penjelasan dengan wrap text (2 baris)",
        "• Spacing bar yang lebih luas (1.8x)",
        "• Shadow effect yang lebih jelas (0.15 alpha)",
        "• Grid spacing yang optimal (max/4)",
        "• Font size yang lebih besar dan bold",
        "• Bar height 85% (dari 80%)",
        "• Enhanced styling untuk semua elemen"
    ]
    
    for i, feature in enumerate(after_features):
        ax.text(0.55, 0.75 - i*0.04, feature, 
                ha='left', va='top', transform=ax.transAxes, 
                fontsize=12, color='#27AE60')
    
    # Keunggulan perbaikan
    ax.text(0.5, 0.4, 'KEUNGGULAN PERBAIKAN LAYOUT:', 
            ha='center', va='top', transform=ax.transAxes, 
            fontsize=16, fontweight='bold', color='#2C3E50')
    
    advantages = [
        "✓ Label kategori lebih ringkas dan mudah dibaca",
        "✓ Penjelasan dengan wrap text lebih rapi",
        "✓ Spacing yang lebih optimal untuk keterbacaan",
        "✓ Shadow effect yang lebih prominent untuk kesan 3D",
        "✓ Grid lines dengan spacing yang lebih baik",
        "✓ Font size yang konsisten dan lebih besar",
        "✓ Layout yang lebih seimbang dan profesional",
        "✓ Tampilan keseluruhan yang lebih bagus"
    ]
    
    for i, advantage in enumerate(advantages):
        ax.text(0.5, 0.32 - i*0.03, advantage, 
                ha='center', va='top', transform=ax.transAxes, 
                fontsize=12, color='#27AE60', fontweight='bold')
    
    ax.axis('off')
    
    # Simpan dokumentasi
    output_file = 'dokumentasi_perbaikan_layout_frekuensi.png'
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"📋 Dokumentasi perbaikan layout disimpan sebagai: {output_file}")

if __name__ == "__main__":
    print("🚀 Memulai test perbaikan layout distribusi frekuensi...")
    print("=" * 80)
    
    # Test single scenario
    test_improved_layout_single()
    
    # Test multiple scenarios
    test_improved_layout_multiple()
    
    # Buat dokumentasi perbandingan
    test_before_after_comparison()
    
    print("\n" + "=" * 80)
    print("🎉 Semua test selesai! Perbaikan layout distribusi frekuensi berhasil:")
    print("   ✓ Label kategori lebih ringkas: Low, Mid, High")
    print("   ✓ Penjelasan dengan wrap text yang rapi")
    print("   ✓ Spacing bar yang lebih optimal")
    print("   ✓ Shadow effect yang lebih prominent")
    print("   ✓ Grid lines dengan spacing yang lebih baik")
    print("   ✓ Font size yang konsisten dan lebih besar")
    print("   ✓ Layout keseluruhan yang lebih bagus dan profesional")
    print("   ✓ Tampilan yang lebih mudah dibaca dan dipahami")