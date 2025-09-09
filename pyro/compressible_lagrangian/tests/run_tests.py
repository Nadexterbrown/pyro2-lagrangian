#!/usr/bin/env python3
"""
Test runner for compressible Lagrangian solver tests.

This script runs all validation tests for the Lagrangian solver
and provides a summary of results.
"""

import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing

# Add pyro path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../'))

from test_sod import run_sod_test
from test_constant_velocity_piston import run_constant_velocity_piston_test


def run_all_tests():
    """Run complete test suite."""
    print("="*70)
    print("COMPRESSIBLE LAGRANGIAN SOLVER - VALIDATION TEST SUITE")
    print("="*70)
    
    test_results = {}
    
    # Test 1: Sod shock tube
    print("\n" + "="*50)
    print("TEST 1: Sod Shock Tube")
    print("="*50)
    
    try:
        sod_passed = run_sod_test(nx=100, riemann_solver_type='hllc', plot=False)
        test_results['sod_shock_tube'] = sod_passed
        print(f"Sod test: {'PASSED' if sod_passed else 'FAILED'}")
    except Exception as e:
        print(f"Sod test: FAILED (Exception: {str(e)})")
        test_results['sod_shock_tube'] = False
    
    # Test 2: Constant velocity piston
    print("\n" + "="*50)
    print("TEST 2: Constant Velocity Piston")
    print("="*50)
    
    try:
        piston_passed = run_constant_velocity_piston_test(nx=50, plot=False)
        test_results['constant_velocity_piston'] = piston_passed
        print(f"Piston test: {'PASSED' if piston_passed else 'FAILED'}")
    except Exception as e:
        print(f"Piston test: FAILED (Exception: {str(e)})")
        test_results['constant_velocity_piston'] = False
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUITE SUMMARY")
    print("="*70)
    
    total_tests = len(test_results)
    passed_tests = sum(test_results.values())
    
    for test_name, result in test_results.items():
        status = "PASSED" if result else "FAILED"
        print(f"{test_name:.<40} {status}")
    
    print("-" * 50)
    print(f"Total tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {total_tests - passed_tests}")
    print(f"Success rate: {100.0 * passed_tests / total_tests:.1f}%")
    
    overall_success = passed_tests == total_tests
    print(f"\nOVERALL RESULT: {'PASSED' if overall_success else 'FAILED'}")
    print("="*70)
    
    return overall_success, test_results


if __name__ == "__main__":
    success, results = run_all_tests()
    sys.exit(0 if success else 1)