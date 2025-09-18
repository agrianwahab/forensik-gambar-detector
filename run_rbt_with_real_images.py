#!/usr/bin/env python3
"""
Risk-Based Testing Runner with Real Images
Runs RBT tests using actual available images in the system
"""

import os
import sys
import shutil
from pathlib import Path

def setup_test_environment():
    """Setup test environment with actual images"""
    
    # Create temp_uploads directory if it doesn't exist
    temp_dir = Path("temp_uploads")
    temp_dir.mkdir(exist_ok=True)
    
    # Copy available images to temp_uploads for testing
    test_images = [
        "splicing.jpg",
        "splicing_image.jpg",
        "asset/Logo-UHO-Normal.png"
    ]
    
    copied_images = []
    for img in test_images:
        src = Path(img)
        if src.exists():
            # Create different test copies for different test scenarios
            if img == "splicing.jpg":
                # Copy for pipeline test
                dst1 = temp_dir / "dummy_pipeline_test.jpg"
                shutil.copy2(src, dst1)
                copied_images.append(str(dst1))
                
                # Copy for vector test
                dst2 = temp_dir / "dummy_vector_test.jpg"
                shutil.copy2(src, dst2)
                copied_images.append(str(dst2))
                
            elif img == "splicing_image.jpg":
                # Copy for manipulation test
                dst = temp_dir / "test_manipulated.jpg"
                shutil.copy2(src, dst)
                copied_images.append(str(dst))
                
            elif "Logo-UHO-Normal.png" in img:
                # Copy for authentic image test
                dst = temp_dir / "test_authentic.png"
                shutil.copy2(src, dst)
                copied_images.append(str(dst))
    
    print(f"✅ Test environment setup complete")
    print(f"   Created {len(copied_images)} test images in temp_uploads/")
    for img in copied_images:
        print(f"   - {img}")
    
    return copied_images

def run_enhanced_rbt():
    """Run the enhanced RBT test suite"""
    
    print("\n" + "="*60)
    print("RISK-BASED TESTING WITH REAL IMAGES")
    print("="*60 + "\n")
    
    # Setup test environment
    print("📁 Setting up test environment...")
    test_images = setup_test_environment()
    
    if not test_images:
        print("❌ No test images available. Cannot proceed with testing.")
        return False
    
    print("\n🔍 Starting Risk-Based Testing...")
    print("-" * 60)
    
    # Import and run the RBT framework
    try:
        from forensic_rbt_integration import EnhancedForensicRBTOrchestrator
        
        # Use the enhanced orchestrator with integrated test executor
        orchestrator = EnhancedForensicRBTOrchestrator(config_path="rbt_config.yaml")
        
        # Run enhanced RBT cycle
        print("\n🚀 Running Enhanced RBT Cycle...")
        results = orchestrator.run_enhanced_rbt_cycle()
        
        # Display results
        print("\n" + "="*60)
        print("📊 TEST RESULTS SUMMARY")
        print("="*60)
        
        if results:
            print(f"\n✅ Overall Status: {results.get('overall_status', 'Unknown')}")
            print(f"📈 Forensic Reliability: {results.get('forensic_reliability', 0):.1%}")
            print(f"🎯 Risk Coverage: {results.get('risk_coverage', 0):.1%}")
            print(f"✔️  Test Pass Rate: {results.get('test_pass_rate', 0):.1%}")
            
            # Quality gates
            gates = results.get('quality_gates', {})
            if gates:
                print(f"\n🚪 Quality Gates:")
                for gate, passed in gates.items():
                    status = "✅" if passed else "❌"
                    print(f"   {status} {gate}")
            
            # Risk level
            risk_level = results.get('risk_level', 'Unknown')
            if risk_level == 'LOW':
                print(f"\n🟢 Risk Level: {risk_level}")
            elif risk_level == 'MEDIUM':
                print(f"\n🟡 Risk Level: {risk_level}")
            elif risk_level == 'HIGH':
                print(f"\n🟠 Risk Level: {risk_level}")
            else:
                print(f"\n🔴 Risk Level: {risk_level}")
            
            # Test breakdown
            test_results = results.get('test_results_by_level', {})
            if test_results:
                print(f"\n📋 Test Results by Risk Level:")
                for level, data in test_results.items():
                    if isinstance(data, dict):
                        passed = data.get('passed', 0)
                        total = data.get('total', 0)
                        rate = data.get('pass_rate', 0)
                        print(f"   {level}: {passed}/{total} passed ({rate:.1%})")
            
            # Recommendations
            recommendations = results.get('recommendations', [])
            if recommendations:
                print(f"\n💡 Recommendations:")
                for i, rec in enumerate(recommendations, 1):
                    print(f"   {i}. {rec}")
            
            # Report paths
            print(f"\n📄 Reports Generated:")
            if 'report_paths' in results:
                for report_type, path in results['report_paths'].items():
                    if path and os.path.exists(path):
                        print(f"   ✅ {report_type}: {path}")
            
            return True
        else:
            print("❌ No results returned from RBT cycle")
            return False
            
    except Exception as e:
        print(f"\n❌ Error running RBT tests: {e}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return False
    
    finally:
        # Cleanup is optional - keep test images for inspection
        print("\n📁 Test images retained in temp_uploads/ for inspection")

def main():
    """Main entry point"""
    success = run_enhanced_rbt()
    
    if success:
        print("\n✅ Risk-Based Testing completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Risk-Based Testing encountered issues")
        sys.exit(1)

if __name__ == "__main__":
    main()