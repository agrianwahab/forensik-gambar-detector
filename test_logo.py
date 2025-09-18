#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test script to verify logo path
"""

import os
from PIL import Image

# Check if logo exists
logo_path = "asset/Logo-UHO-Normal.png"

print("=" * 50)
print("LOGO UHO VERIFICATION")
print("=" * 50)

# Check existence
if os.path.exists(logo_path):
    print(f"✅ Logo found at: {logo_path}")
    
    # Check if it's readable
    try:
        img = Image.open(logo_path)
        width, height = img.size
        print(f"✅ Logo dimensions: {width}x{height} pixels")
        print(f"✅ Logo format: {img.format}")
        print(f"✅ Logo mode: {img.mode}")
        
        # Get file size
        file_size = os.path.getsize(logo_path)
        print(f"✅ Logo file size: {file_size:,} bytes")
        
        print("\n✅ LOGO READY TO USE IN STREAMLIT!")
        print(f"   Use this path in app.py: '{logo_path}'")
        
    except Exception as e:
        print(f"❌ Error reading logo: {e}")
else:
    print(f"❌ Logo NOT found at: {logo_path}")
    print("\nChecking alternative locations...")
    
    # Check other possible locations
    possible_paths = [
        "Logo-UHO-Normal.png",
        "assets/Logo-UHO-Normal.png",
        "../asset/Logo-UHO-Normal.png",
        "Logo-UHO-Normal-1-294x300[1].png"
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            print(f"  Found at: {path}")
            
print("\n" + "=" * 50)