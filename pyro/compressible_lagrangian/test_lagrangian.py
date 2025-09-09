#!/usr/bin/env python3
"""
Test script for Lagrangian compressible solver validation.

This script performs basic validation tests for the Lagrangian solver
components including grid motion, Riemann solvers, and piston dynamics.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from .grid import LagrangianGrid1d
from .gas_state import GasState
from .riemann import LagrangianRiemannSolver, HLLCRiemannSolver
from .piston import Piston


def test_grid_motion():
    """Test basic grid motion and geometric conservation."""
    print("Testing grid motion...")
    
    # Create uniform grid
    grid = LagrangianGrid1d(n_cells=10, x_left=0.0, x_right=1.0)
    
    # Test uniform expansion
    u_face = np.full(grid.n_faces, 1.0)  # Uniform velocity
    dt = 0.01
    
    initial_volume = np.sum(grid.volume)
    grid.update_positions(u_face, dt)
    final_volume = np.sum(grid.volume)
    
    print(f"Initial total volume: {initial_volume:.6f}")
    print(f"Final total volume: {final_volume:.6f}")
    print(f"Volume change rate: {(final_volume - initial_volume) / dt:.6f}")
    print(f"Expected volume change rate: {grid.area * (u_face[-1] - u_face[0]):.6f}")
    
    # Check geometric conservation
    volume_rate = grid.compute_volume_change_rate()
    expected_rate = grid.area * (u_face[1:] - u_face[:-1])
    
    print("Grid motion test: PASSED\n" if np.allclose(volume_rate, expected_rate) else "Grid motion test: FAILED\n")


def test_riemann_solver():
    """Test Riemann solver against known solutions."""
    print("Testing Riemann solvers...")
    
    # Sod problem states
    rho_L, u_L, p_L = 1.0, 0.0, 1.0
    rho_R, u_R, p_R = 0.125, 0.0, 0.1
    
    # Test exact solver
    exact_solver = LagrangianRiemannSolver(gamma=1.4)
    u_star_exact, p_star_exact, rho_star_exact = exact_solver.solve_interface(
        rho_L, u_L, p_L, rho_R, u_R, p_R)
    
    # Test HLLC solver
    hllc_solver = HLLCRiemannSolver(gamma=1.4)
    u_star_hllc, p_star_hllc, rho_star_hllc = hllc_solver.solve_interface(
        rho_L, u_L, p_L, rho_R, u_R, p_R)
    
    print(f"Exact solver - u*: {u_star_exact:.6f}, p*: {p_star_exact:.6f}")
    print(f"HLLC solver  - u*: {u_star_hllc:.6f}, p*: {p_star_hllc:.6f}")
    
    # Expected values (approximate)
    u_expected = 0.9274  # From analytical solution
    p_expected = 0.3031
    
    exact_error = abs(u_star_exact - u_expected) + abs(p_star_exact - p_expected)
    hllc_error = abs(u_star_hllc - u_expected) + abs(p_star_hllc - p_expected)
    
    print("Riemann solver test: PASSED\n" if exact_error < 0.1 and hllc_error < 0.1 else "Riemann solver test: FAILED\n")


def test_gas_state():
    """Test gas state management and EOS calculations."""
    print("Testing gas state management...")
    
    # Create gas state
    gas = GasState(n_cells=5, gamma=1.4, gas_constant=287.0)
    
    # Set uniform conditions
    density = np.full(5, 1.225)
    velocity = np.full(5, 0.0)
    pressure = np.full(5, 101325.0)
    
    gas.set_initial_conditions(density, velocity, pressure)
    
    # Test conversions
    original_energy = gas.energy.copy()
    gas.cons_to_prim()
    gas.prim_to_cons()
    
    energy_error = np.max(np.abs(gas.energy - original_energy))
    
    # Test thermodynamic consistency
    expected_temp = pressure[0] / (density[0] * gas.gas_constant)
    temp_error = abs(gas.temperature[0] - expected_temp)
    
    print(f"Energy conservation error: {energy_error:.2e}")
    print(f"Temperature calculation error: {temp_error:.2e}")
    print("Gas state test: PASSED\n" if energy_error < 1e-10 and temp_error < 1e-6 else "Gas state test: FAILED\n")


def test_piston_dynamics():
    """Test piston dynamics and force calculations."""
    print("Testing piston dynamics...")
    
    # Create piston
    piston = Piston(mass=1.0, area=0.01, x0=0.0, v0=0.0)
    
    # Set pressures
    piston.pressure_front = 101325.0  # Atmospheric
    piston.pressure_back = 202650.0   # Higher pressure
    
    # Calculate force
    expected_force = (piston.pressure_front - piston.pressure_back) * piston.area
    actual_force = piston.compute_pressure_force()
    
    # Test acceleration
    expected_acceleration = expected_force / piston.mass
    actual_acceleration = piston.compute_acceleration(0.0)
    
    print(f"Expected force: {expected_force:.3f} N")
    print(f"Actual force: {actual_force:.3f} N")
    print(f"Expected acceleration: {expected_acceleration:.3f} m/s²")
    print(f"Actual acceleration: {actual_acceleration:.3f} m/s²")
    
    force_error = abs(actual_force - expected_force)
    accel_error = abs(actual_acceleration - expected_acceleration)
    
    print("Piston dynamics test: PASSED\n" if force_error < 1e-10 and accel_error < 1e-10 else "Piston dynamics test: FAILED\n")


def test_conservation():
    """Test conservation properties in simple scenarios."""
    print("Testing conservation properties...")
    
    # Create simple system
    grid = LagrangianGrid1d(n_cells=20, x_left=0.0, x_right=1.0)
    gas = GasState(n_cells=20, gamma=1.4)
    
    # Set uniform conditions
    density = np.full(20, 1.0)
    velocity = np.full(20, 0.1)  # Small uniform velocity
    pressure = np.full(20, 1.0)
    
    gas.set_initial_conditions(density, velocity, pressure)
    
    # Store initial conserved quantities
    initial = gas.compute_conserved_totals(grid)
    
    # Simple uniform translation
    dt = 0.001
    u_face = np.full(grid.n_faces, 0.1)
    
    for step in range(10):
        grid.update_positions(u_face, dt)
        # In uniform translation, gas state shouldn't change
    
    final = gas.compute_conserved_totals(grid)
    
    mass_error = abs(final['mass'] - initial['mass']) / initial['mass']
    momentum_error = abs(final['momentum'] - initial['momentum']) / max(abs(initial['momentum']), 1e-10)
    energy_error = abs(final['energy'] - initial['energy']) / initial['energy']
    
    print(f"Mass conservation error: {mass_error:.2e}")
    print(f"Momentum conservation error: {momentum_error:.2e}")
    print(f"Energy conservation error: {energy_error:.2e}")
    
    print("Conservation test: PASSED\n" if mass_error < 1e-14 and energy_error < 1e-14 else "Conservation test: FAILED\n")


def run_all_tests():
    """Run all validation tests."""
    print("=" * 50)
    print("LAGRANGIAN SOLVER VALIDATION TESTS")
    print("=" * 50)
    
    test_grid_motion()
    test_riemann_solver()
    test_gas_state()
    test_piston_dynamics()
    test_conservation()
    
    print("=" * 50)
    print("VALIDATION TESTS COMPLETE")
    print("=" * 50)


if __name__ == "__main__":
    run_all_tests()