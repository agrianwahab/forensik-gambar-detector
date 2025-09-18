#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test script for fixed visualizations - both uncertainty and probability analysis
"""

import sys
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from visualization import create_uncertainty_visualization, create_probability_bars

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
                'uncertainty_level': 0.157
            },
            'report': {
                'primary_assessment': 'Indikasi: Manipulasi Splicing Terdeteksi',
                'assessment_reliability': 'Sedang',
                'indicator_coherence': 'Sedang: Sebagian besar indikator menunjukkan koherensi yang cukup baik namun terdapat beberapa anomali.',
                'recommendation': 'Manual review recommended'
            }
        }
    }
}

print("Testing fixed visualizations...")

# Test 1: Uncertainty visualization (simplified)
print("1. Testing simplified uncertainty visualization...")
fig1, ax1 = plt.subplots(figsize=(6, 4))
create_uncertainty_visualization(ax1, analysis_results)
output_file1 = "test_uncertainty_fixed.png"
plt.savefig(output_file1, dpi=150, bbox_inches='tight')
plt.close()
print(f"   Saved to: {output_file1}")

# Test 2: Probability visualization (with Keandalan Asesmen)
print("2. Testing probability visualization with Keandalan Asesmen...")
fig2, ax2 = plt.subplots(figsize=(8, 6))
create_probability_bars(ax2, analysis_results)
output_file2 = "test_probability_fixed.png"
plt.savefig(output_file2, dpi=150, bbox_inches='tight')
plt.close()
print(f"   Saved to: {output_file2}")

# Test 3: Combined visualization with proper spacing
print("3. Testing combined visualization with proper spacing...")
fig3, (ax3, ax4) = plt.subplots(1, 2, figsize=(16, 6))

# Adjust subplot parameters for better spacing
plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, wspace=0.3)

create_uncertainty_visualization(ax3, analysis_results)
create_probability_bars(ax4, analysis_results)

output_file3 = "test_combined_spacing.png"
plt.savefig(output_file3, dpi=150, bbox_inches='tight')
plt.close()
print(f"   Saved to: {output_file3}")

print("All visualization tests completed successfully!")
print("Check the output files to verify the spacing and layout improvements.")