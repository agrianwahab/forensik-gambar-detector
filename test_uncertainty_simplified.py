#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test script for simplified uncertainty visualization
"""

import sys
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from visualization import create_uncertainty_visualization

# Create test analysis results with uncertainty data
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
                'indicator_coherence': 'Sedang: Sebagian besar indikator menunjukkan koherensi yang cukup baik namun terdapat beberapa anomali pada analisis frekuensi dan tekstur yang perlu diteliti lebih lanjut.',
                'recommendation': 'Manual review recommended'
            }
        }
    }
}

print("Testing simplified uncertainty visualization...")

# Create figure and test the visualization
fig, ax = plt.subplots(figsize=(8, 6))
create_uncertainty_visualization(ax, analysis_results)

# Save the result
output_file = "test_uncertainty_simplified.png"
plt.savefig(output_file, dpi=150, bbox_inches='tight')
plt.close()

print(f"Test visualization saved to: {output_file}")
print("Simplified uncertainty visualization test completed successfully!")