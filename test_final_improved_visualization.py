#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test script for final improved probability visualization with modern aesthetics
"""

import sys
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from visualization import create_probability_bars, create_uncertainty_visualization

# Create test analysis results with complete data
analysis_results = {
    'classification': {
        'type': 'Splicing Detected',
        'confidence': 0.78,
        'uncertainty_analysis': {
            'probabilities': {
                'authentic_probability': 0.22,
                'copy_move_probability': 0.36,
                'splicing_probability': 0.42,
                'uncertainty_level': 0.157,
                'confidence_intervals': {
                    'authentic': {'lower': 0.17, 'upper': 0.27},
                    'copy_move': {'lower': 0.31, 'upper': 0.41},
                    'splicing': {'lower': 0.37, 'upper': 0.47}
                }
            },
            'report': {
                'primary_assessment': 'Indikasi: Manipulasi Splicing Terdeteksi',
                'assessment_reliability': 'Sedang',
                'indicator_coherence': 'Sedang: Sebagian besar indikator menunjukkan koherensi yang cukup baik.',
                'recommendation': 'Manual review recommended'
            }
        }
    }
}

print("Testing final improved visualizations...")

# Test 1: Modern probability visualization
print("1. Testing modern probability visualization...")
fig1, ax1 = plt.subplots(figsize=(10, 8))
create_probability_bars(ax1, analysis_results)
output_file1 = "test_probability_modern.png"
plt.savefig(output_file1, dpi=150, bbox_inches='tight')
plt.close()
print(f"   Saved to: {output_file1}")

# Test 2: Uncertainty visualization (unchanged, already good)
print("2. Testing uncertainty visualization...")
fig2, ax2 = plt.subplots(figsize=(8, 6))
create_uncertainty_visualization(ax2, analysis_results)
output_file2 = "test_uncertainty_final.png"
plt.savefig(output_file2, dpi=150, bbox_inches='tight')
plt.close()
print(f"   Saved to: {output_file2}")

# Test 3: Combined visualization with perfect spacing
print("3. Testing combined final visualization...")
fig3, (ax3, ax4) = plt.subplots(1, 2, figsize=(20, 8))

# Perfect spacing adjustments
plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, wspace=0.4)

create_uncertainty_visualization(ax3, analysis_results)
create_probability_bars(ax4, analysis_results)

# Add overall title
fig3.suptitle('Tahap 4: Laporan Akhir dan Interpretasi Forensik', fontsize=16, fontweight='bold', y=0.98)

output_file3 = "test_final_combined.png"
plt.savefig(output_file3, dpi=150, bbox_inches='tight')
plt.close()
print(f"   Saved to: {output_file3}")

print("\n" + "="*60)
print("FINAL VISUALIZATION IMPROVEMENTS COMPLETED!")
print("="*60)
print("Output files:")
print(f"   - Modern Probability: {output_file1}")
print(f"   - Uncertainty: {output_file2}")
print(f"   - Combined Final: {output_file3}")
print("\nKey Improvements:")
print("   * Probability analysis now more symmetrical and modern")
print("   * Wrap text for Evidence Copy-Move and Splicing labels")
print("   * Modern color scheme and styling")
print("   * Better spacing and no text overlap")
print("   * Professional aesthetics with subtle shadows")
print("   * Perfect balance between both analysis panels")