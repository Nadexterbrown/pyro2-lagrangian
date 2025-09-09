"""
Test suite for compressible Lagrangian solver.

This module contains validation tests for the Lagrangian compressible
flow solver including:
- Sod shock tube with stationary boundaries
- Piston-driven compression tests
- Riemann solver validation
- Conservation property tests
- Grid motion verification
"""

__all__ = ["test_sod", "test_constant_velocity_piston", "test_sod_stationary"]