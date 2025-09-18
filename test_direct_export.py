#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Direct test for ZIP export functionality
"""

import sys
import os
import numpy as np
from PIL import Image

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from export_utils import export_comprehensive_package

# Load real image
image_path = "splicing.jpg"
if os.path.exists(image_path):
    original_image = Image.open(image_path)
else:
    # Create test image if real image not found
    original_image = Image.new('RGB', (400, 300), color='blue')
    print("Using test image (real image not found)")

# Create comprehensive analysis results 
analysis_results = {
    'classification': {
        'type': 'Copy-Move Forgery',
        'confidence': 'HIGH',
        'copy_move_score': 85,
        'splicing_score': 45,
        'uncertainty_analysis': {
            'uncertainty_level': 0.15,
            'indicator_coherence': 'HIGH',
            'probabilities': {
                'authentic_probability': 0.1,
                'copy_move_probability': 0.85,
                'splicing_probability': 0.05,
                'uncertainty_level': 0.15
            },
            'report': {
                'primary_assessment': 'Copy-Move Forgery Detected',
                'assessment_reliability': 'HIGH',
                'recommendation': 'Manual review recommended'
            }
        }
    },
    'error_level_analysis': {
        'ela_score': 0.75,
        'ela_mean': 25.5,
        'ela_std': 12.3,
        'ela_data': np.random.rand(300, 400) * 255
    },
    'metadata': {
        'width': 400,
        'height': 300,
        'format': 'JPEG',
        'Metadata_Authenticity_Score': 85,
        'Metadata_Inconsistency': ['EXIF timestamp mismatch']
    },
    'copy_move_detection': {
        'detected': True,
        'num_matches': 25,
        'keypoints': [],
        'matches': []
    },
    'jpeg_ghost_analysis': {
        'ghost_detected': True,
        'quality_levels': [70, 85, 90],
        'ghost_map': np.random.rand(300, 400)
    },
    'noise_analysis': {
        'noise_score': 0.6,
        'noise_map': np.random.rand(300, 400)
    },
    'pipeline_status': {
        'total_stages': 19,
        'completed_stages': 17,
        'failed_stages': ['trufor_analysis'],
        'stage_details': {}
    }
}

# Test export
print("Testing comprehensive ZIP export...")
try:
    result = export_comprehensive_package(
        original_image, 
        analysis_results, 
        base_filename="forensic_test"
    )
    
    if result and 'complete_zip' in result:
        zip_path = result['complete_zip']
        if os.path.exists(zip_path):
            size = os.path.getsize(zip_path)
            print(f"\n=== SUCCESS ===")
            print(f"ZIP file created: {zip_path}")
            print(f"File size: {size:,} bytes")
            
            # List contents
            import zipfile
            with zipfile.ZipFile(zip_path, 'r') as zf:
                print(f"\nZIP contents ({len(zf.namelist())} files):")
                for name in zf.namelist()[:10]:  # Show first 10 files
                    info = zf.getinfo(name)
                    print(f"  - {name} ({info.file_size:,} bytes)")
                if len(zf.namelist()) > 10:
                    print(f"  ... and {len(zf.namelist()) - 10} more files")
        else:
            print("\n=== FAILED ===")
            print("ZIP file path returned but file doesn't exist")
    else:
        print("\n=== FAILED ===")
        print("ZIP file not created")
        print(f"Result: {result}")
except Exception as e:
    print(f"\n=== ERROR ===")
    print(f"Exception: {e}")
    import traceback
    traceback.print_exc()