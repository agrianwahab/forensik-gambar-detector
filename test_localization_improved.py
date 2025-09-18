#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test script for improved K-Means localization visualization
"""

import sys
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from visualization import create_localization_visualization

# Create test image
test_image = Image.open("splicing.jpg") if os.path.exists("splicing.jpg") else Image.new('RGB', (400, 300), color='blue')

# Create test analysis results with localization data
h, w = test_image.height, test_image.width

# Create synthetic localization data
combined_mask = np.zeros((h, w), dtype=bool)
# Add some suspicious regions
combined_mask[50:150, 100:200] = True  # Top-left region
combined_mask[200:280, 250:350] = True  # Bottom-right region

# Create K-means cluster data
cluster_means = [
    (0, 15.2, 3.1),   # Normal cluster
    (1, 85.7, 12.3),  # Suspicious cluster
    (2, 78.4, 8.7),   # Suspicious cluster
    (3, 25.6, 4.2),   # Normal cluster
    (4, 45.3, 6.8),   # Medium cluster
]

analysis_results = {
    'metadata': {
        'Filename': 'test_image.jpg',
        'FileSize': '123456 bytes',
        'Dimensions': f'{w}x{h}',
    },
    'ela_mean': 32.5,
    'ela_std': 15.2,
    'localization_analysis': {
        'kmeans_localization': {
            'cluster_ela_means': cluster_means,
            'suspicious_clusters': [1, 2],
            'tampering_percentage': 12.5
        },
        'combined_tampering_mask': combined_mask,
        'tampering_percentage': 12.5
    }
}

print("Testing improved K-Means localization visualization...")

# Create figure and test the visualization
fig, ax = plt.subplots(figsize=(10, 8))
create_localization_visualization(ax, test_image, analysis_results)

# Save the result
output_file = "test_localization_improved.png"
plt.savefig(output_file, dpi=150, bbox_inches='tight')
plt.close()

print(f"Test visualization saved to: {output_file}")
print("Improved K-Means localization visualization test completed successfully!")