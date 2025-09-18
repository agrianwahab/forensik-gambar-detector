#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test if app runs correctly with logo
"""

import os
import sys

print("=" * 60)
print("TESTING STREAMLIT APP WITH LOGO UHO")
print("=" * 60)

# Check logo
logo_path = "asset/Logo-UHO-Normal.png"
if os.path.exists(logo_path):
    print(f"✅ Logo found: {logo_path}")
    file_size = os.path.getsize(logo_path)
    print(f"   Size: {file_size:,} bytes")
else:
    print(f"❌ Logo NOT found: {logo_path}")

# Check app.py
if os.path.exists("app.py"):
    print("✅ app.py found")
    
    # Check if logo path is correctly referenced in app.py
    with open("app.py", "r", encoding="utf-8") as f:
        content = f.read()
        
    correct_path_count = content.count('asset/Logo-UHO-Normal.png')
    if correct_path_count > 0:
        print(f"✅ Logo path correctly referenced {correct_path_count} times in app.py")
    else:
        print("❌ Logo path not found in app.py")
        
    # Check for old incorrect paths
    old_paths = [
        'Logo-UHO-Normal-1-294x300[1].png',
        'Logo-UHO-Normal.png'  # without asset/ folder
    ]
    
    for old_path in old_paths:
        if old_path in content:
            print(f"⚠️  Old/incorrect path still found: {old_path}")
else:
    print("❌ app.py NOT found")

print("\n" + "=" * 60)
print("INSTRUCTIONS TO RUN THE APP:")
print("=" * 60)
print("Run this command to start the Streamlit app:")
print("\n  streamlit run app.py")
print("\nThe app should now display:")
print("  1. Logo UHO in the sidebar (100px width)")
print("  2. Logo UHO in the main header (80px width)") 
print("  3. Professional header with university name")
print("  4. Footer with copyright information")
print("\n" + "=" * 60)