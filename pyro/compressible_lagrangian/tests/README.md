# Compressible Lagrangian Solver Tests

This directory contains validation tests for the 1D Lagrangian compressible flow solver. The tests validate key physics and numerical methods against analytical solutions and expected behaviors.

## Test Cases

### 1. Sod Shock Tube Test (`test_sod.py`)

**Purpose**: Validates Riemann solver accuracy and wave propagation physics.

**Setup**:
- Classic Riemann problem with discontinuity at x=0.5
- Left state: ρ=1.0 kg/m³, u=0 m/s, p=101325 Pa
- Right state: ρ=0.125 kg/m³, u=0 m/s, p=10132.5 Pa
- Domain: [0, 1] m with outflow boundaries
- Final time: 0.25 s

**Expected Solution**:
- Leftward rarefaction wave
- Contact discontinuity at x≈0.69
- Rightward shock wave at x≈0.85

**Validation Criteria**:
- L2 errors < 10% for density, velocity, pressure
- Mass conservation error < 1e-12
- Correct wave positions and speeds

**Usage**:
```python
python test_sod.py
```

### 2. Constant Velocity Piston Test (`test_constant_velocity_piston.py`)

**Purpose**: Validates piston-gas coupling and moving boundary physics.

**Setup**:
- Piston at left boundary moving at 100 m/s constant velocity
- Initial gas: air at STP (ρ=1.225 kg/m³, p=101325 Pa, u=0)
- Domain: [0, 0.1] m with reflecting right wall
- Piston: mass=0.1 kg, area=0.001 m²
- Final time: 0.01 s

**Expected Behavior**:
- Piston compresses gas, generating compression waves
- Pressure increases as piston advances
- Wave reflections from right wall
- Energy addition through piston work

**Validation Criteria**:
- Mass conservation error < 1e-12
- Pressure ratio > 1.5 (significant compression)
- Positive energy change (work input)
- Reasonable pressure levels (0.5 < agreement ratio < 2.0)

**Usage**:
```python
python test_constant_velocity_piston.py
```

### 3. Sod with Stationary Boundaries (`test_sod_stationary.py`)

**Purpose**: Validates Lagrangian formulation with fixed domain boundaries.

**Setup**:
- Same as Sod test but with reflecting boundaries
- Interior grid follows Lagrangian motion
- Domain boundaries remain fixed

**Usage**:
```python
python test_sod_stationary.py
```

## Running Tests

### Individual Tests
```bash
cd pyro/compressible_lagrangian/tests

# Run individual test with plots
python test_sod.py
python test_constant_velocity_piston.py

# Run without plots (for automated testing)
python -c "from test_sod import run_sod_test; run_sod_test(plot=False)"
```

### Complete Test Suite
```bash
python run_tests.py
```

This runs all tests and provides a summary:
```
======================================================================
COMPRESSIBLE LAGRANGIAN SOLVER - VALIDATION TEST SUITE
======================================================================

==================================================
TEST 1: Sod Shock Tube
==================================================
Running Sod test: nx=100, solver=hllc
...
SOD TEST RESULT: PASSED

==================================================
TEST 2: Constant Velocity Piston
==================================================
Running constant velocity piston test: nx=50, L=0.1
...
CONSTANT VELOCITY PISTON TEST RESULT: PASSED

======================================================================
TEST SUITE SUMMARY
======================================================================
sod_shock_tube........................... PASSED
constant_velocity_piston................. PASSED
--------------------------------------------------
Total tests: 2
Passed: 2
Failed: 0
Success rate: 100.0%

OVERALL RESULT: PASSED
```

## Test Output

### Generated Files
- `sod_test_comparison.png`: Sod test comparison plots
- `constant_velocity_piston_test.png`: Piston test result plots

### Key Metrics

**Sod Test**:
- Density L2 error: ~1e-2
- Velocity L2 error: ~1e-3
- Pressure L2 error: ~1e-2
- Mass conservation: ~1e-15

**Piston Test**:
- Mass conservation: ~1e-15
- Pressure ratio: ~2-5 (depends on compression)
- Energy efficiency: ~80-90%
- Grid quality: No tangling

## Implementation Details

### Riemann Solvers
Both exact and HLLC Riemann solvers are tested:
- **Exact solver**: Newton-Raphson iteration, high accuracy
- **HLLC solver**: Approximate but faster, good for production

### Boundary Conditions
- **Outflow**: Extrapolation from interior
- **Reflecting**: Zero normal velocity
- **Piston**: Velocity set by piston dynamics

### Time Integration
- Second-order Runge-Kutta for accuracy
- Adaptive time stepping with multiple CFL constraints
- Lagrangian grid motion with geometric conservation

### Conservation Properties
All tests verify:
- **Mass conservation**: Exact in Lagrangian formulation
- **Momentum conservation**: Including piston coupling
- **Energy conservation**: Including piston work

## Troubleshooting

### Common Issues

1. **Grid tangling**: Reduce time step or increase grid resolution
2. **Negative pressure/density**: Check initial conditions and boundary conditions
3. **Poor convergence**: Increase grid resolution or reduce CFL number

### Debug Output
Set `debug=True` in test functions for detailed output:
```python
run_sod_test(nx=100, plot=True, debug=True)
```

## Expected Performance

### Typical Run Times
- Sod test (nx=100): ~5-10 seconds
- Piston test (nx=50): ~10-15 seconds
- Full test suite: ~30 seconds

### Memory Usage
- Minimal for test problems (~MB range)
- Scales linearly with grid size

### Accuracy
- Sod test: Second-order convergence
- Piston test: First-order (due to discontinuities)

These tests provide comprehensive validation of the Lagrangian solver's core capabilities and ensure correctness for both fundamental gas dynamics and complex piston-gas interactions.