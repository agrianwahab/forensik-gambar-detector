#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test script for ZIP export functionality
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from PIL import Image
from export_utils import export_comprehensive_package
import numpy as np

# Create a simple test image
test_image = Image.new('RGB', (100, 100), color='red')

# Create minimal analysis results
analysis_results = {
    'classification': {
        'type': 'TEST',
        'confidence': 'HIGH',
        'copy_move_score': 50,
        'splicing_score': 30
    },
    'error_level_analysis': {
        'ela_score': 0.5,
        'ela_mean': 10,
        'ela_std': 5,
        'ela_data': np.zeros((100, 100))
    },
    'metadata': {
        'width': 100,
        'height': 100,
        'format': 'JPEG'
    }
}

# Test export
print("Testing ZIP export...")
try:
    result = export_comprehensive_package(
        test_image, 
        analysis_results, 
        base_filename="test_export"
    )
    
    if result and 'complete_zip' in result:
        print(f"SUCCESS: ZIP file created at {result['complete_zip']}")
    else:
        print("FAILED: ZIP file not created")
        print(f"Result: {result}")
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()