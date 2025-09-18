#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test comprehensive PNG visualization with all 22 panels including MM Fusion and TruFor
"""

import sys
import os
import numpy as np
from PIL import Image

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from visualization import visualize_results_advanced

# Create test image
test_image = Image.open("splicing.jpg") if os.path.exists("splicing.jpg") else Image.new('RGB', (400, 300), color='blue')

# Create comprehensive test results with MM Fusion and TruFor
analysis_results = {
    'metadata': {
        'Filename': 'test_image.jpg',
        'FileSize (bytes)': 123456,
        'Metadata_Authenticity_Score': 85,
        'Metadata_Inconsistency': ['EXIF timestamp mismatch']
    },
    'classification': {
        'type': 'Copy-Move Forgery Detected',
        'confidence': 'HIGH',
        'details': [
            'RANSAC inliers detected: 15 matches',
            'Block matching found 25 similar blocks',
            'ELA shows compression inconsistencies'
        ],
        'uncertainty_analysis': {
            'probabilities': {
                'authentic_probability': 0.1,
                'copy_move_probability': 0.85,
                'splicing_probability': 0.05,
                'uncertainty_level': 0.15
            },
            'report': {
                'primary_assessment': 'Copy-Move Forgery Detected',
                'assessment_reliability': 'HIGH',
                'indicator_coherence': 'HIGH',
                'recommendation': 'Manual review recommended'
            }
        }
    },
    'ela_image': Image.fromarray((np.random.rand(300, 400, 3) * 255).astype(np.uint8)).convert('RGB'),
    'ela_mean': 25.5,
    'ela_std': 12.3,
    # Generate non-zero JPEG ghost data
    'jpeg_ghost': np.random.rand(300, 400) * 0.5 + 0.2,  # Values between 0.2 and 0.7
    'noise_map': np.random.rand(300, 400),
    'mm_fusion_analysis': {
        'forgery_detected': True,
        'forgery_confidence_score': 87.5,
        'confidence_level': 'HIGH',
        'confidence_factors': [
            'High ELA anomaly detected in regions',
            'Copy-move patterns confirmed by SIFT',
            'JPEG ghost analysis shows double compression',
            'Noise inconsistency detected',
            'Texture patterns show manipulation'
        ],
        'heatmap_data': {
            'fusion_map': np.random.rand(300, 400) * 0.8 + 0.2
        }
    },
    'trufor_analysis': {
        'forensic_confidence': 92.3,
        'is_authentic': False,
        'risk_level': 'HIGH',
        'forgery_localization_map': np.random.rand(300, 400) * 0.6,
        'multi_scale_analysis': [
            {'scale': 1.0, 'confidence': 0.89},
            {'scale': 0.5, 'confidence': 0.91},
            {'scale': 2.0, 'confidence': 0.85}
        ],
        'forensic_report': {
            'summary': 'High confidence forgery detected with multiple indicators'
        }
    },
    'error_level_analysis': {
        'ela_score': 0.75,
        'ela_mean': 25.5,
        'ela_std': 12.3
    },
    'copy_move_detection': {
        'detected': True,
        'num_matches': 25,
        'keypoints': [],
        'matches': [],
        'ransac_inliers': 15
    },
    'block_matches': [
        {'block1_pos': (i*10, i*10), 'block2_pos': (i*10+50, i*10+50), 'similarity': 0.9+i*0.01}
        for i in range(5)
    ],
    'localization_analysis': {
        'combined_tampering_mask': np.random.rand(300, 400) > 0.7,
        'tampering_percentage': 15.5
    },
    'edge_analysis': {
        'edge_inconsistency': 0.35
    },
    'illumination_analysis': {
        'overall_illumination_inconsistency': 0.25
    },
    'frequency_analysis': {
        'frequency_inconsistency': 1.2,
        'dct_stats': {
            'low_freq_energy': 0.7,
            'mid_freq_energy': 0.2,
            'high_freq_energy': 0.1
        }
    },
    'texture_analysis': {
        'texture_inconsistency': 0.45,
        'texture_consistency': {
            'contrast_consistency': 0.3,
            'homogeneity_consistency': 0.4,
            'energy_consistency': 0.35
        }
    },
    'statistical_analysis': {
        'R_entropy': 7.2,
        'G_entropy': 7.5,
        'B_entropy': 7.1,
        'overall_entropy': 7.3,
        'rg_correlation': 0.85,
        'rb_correlation': 0.82,
        'gb_correlation': 0.88
    },
    'jpeg_analysis': {
        'basic_analysis': {
            'quality_responses': [
                {'quality': q, 'response_mean': np.random.rand() * 10}
                for q in range(50, 100, 5)
            ],
            'estimated_original_quality': 85
        }
    },
    'noise_analysis': {
        'overall_inconsistency': 0.32
    },
    'pipeline_status': {
        'total_stages': 19,
        'completed_stages': 19,
        'failed_stages': [],
        'stage_details': {}
    }
}

# Test visualization
print("Testing comprehensive visualization with 22 panels...")
output_file = "test_full_22_visualization.png"

try:
    result = visualize_results_advanced(test_image, analysis_results, output_file)
    
    if result and os.path.exists(result):
        size = os.path.getsize(result)
        print(f"\n✅ SUCCESS!")
        print(f"Visualization created: {result}")
        print(f"File size: {size:,} bytes")
        print(f"\nThe visualization contains all 22 panels:")
        print("  Row 1: Original, ELA, Copy-Move, Block Matching")
        print("  Row 2: K-Means, Edge, Illumination, JPEG Ghost (Enhanced)")
        print("  Row 3: Combined Heatmap, Frequency, Texture, Statistics")
        print("  Row 4: Quality Response, Noise Map, Metadata, DCT")
        print("  Row 5: MM Fusion, TruFor, Summary, Pipeline Status")
        print("  Row 6: Probability Analysis, Uncertainty Analysis")
    else:
        print("\n❌ FAILED: Visualization not created")
        
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()