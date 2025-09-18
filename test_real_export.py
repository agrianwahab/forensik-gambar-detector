#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test export with real image
"""

import os
import sys

# Set environment variable to handle unicode output
os.environ['PYTHONIOENCODING'] = 'utf-8'

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main import analyze_image_comprehensive_advanced
from export_utils import export_comprehensive_package
from PIL import Image

# Analyze real image
image_path = "splicing.jpg"
if not os.path.exists(image_path):
    print(f"Error: {image_path} not found")
    sys.exit(1)

print(f"Analyzing {image_path}...")
analysis_results = analyze_image_comprehensive_advanced(image_path, "./results", test_mode=False)

if analysis_results:
    print("\nAnalysis complete. Creating export package...")
    
    # Open the image for export
    original_image = Image.open(image_path)
    
    # Export comprehensive package
    result = export_comprehensive_package(
        original_image, 
        analysis_results, 
        base_filename="splicing_forensic"
    )
    
    if result and 'complete_zip' in result:
        print(f"\n*** SUCCESS: ZIP file created at {result['complete_zip']} ***")
    else:
        print("\n*** FAILED: ZIP file not created ***")
else:
    print("Analysis failed")