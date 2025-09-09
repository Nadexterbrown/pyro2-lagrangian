"""
Sod shock tube test with stationary boundary conditions.

This test validates the Lagrangian solver against the analytical Sod solution
with fixed boundaries. The domain boundaries do not move, while interior
cell boundaries follow the Lagrangian motion.

Test setup:
- Domain: [0, 1] with fixed boundaries
- Initial discontinuity at x = 0.5
- Left state: ρ=1.0, u=0.0, p=1.0  
- Right state: ρ=0.125, u=0.0, p=0.1
- Boundary conditions: stationary (u_boundary = 0)
- Final time: t = 0.25

Expected solution:
- Leftward rarefaction: head at x ≈ 0.26, tail at x = 0.5
- Contact discontinuity at x ≈ 0.69
- Rightward shock at x ≈ 0.85
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any
import os
import sys

# Add pyro path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../'))

from pyro.compressible_lagrangian.grid import LagrangianGrid1d
from pyro.compressible_lagrangian.gas_state import GasState
from pyro.compressible_lagrangian.riemann import LagrangianRiemannSolver, HLLCRiemannSolver
from pyro.util import runparams


def sod_exact_solution(x: np.ndarray, t: float, gamma: float = 1.4) -> Dict[str, np.ndarray]:
    """
    Compute exact Sod shock tube solution.
    
    Args:
        x: Position array
        t: Time
        gamma: Ratio of specific heats
        
    Returns:
        Dictionary with exact solution arrays
    """
    # Initial conditions
    rho_L, u_L, p_L = 1.0, 0.0, 1.0
    rho_R, u_R, p_R = 0.125, 0.0, 0.1
    
    # Sound speeds
    c_L = np.sqrt(gamma * p_L / rho_L)
    c_R = np.sqrt(gamma * p_R / rho_R)
    
    # Star region properties (from exact Riemann solver)
    p_star = 0.30313  # Exact value
    u_star = 0.92745  # Exact value
    
    # Wave speeds
    # Left rarefaction
    rho_star_L = rho_L * (p_star / p_L)**(1.0 / gamma)
    c_star_L = np.sqrt(gamma * p_star / rho_star_L)
    
    # Right shock
    p_ratio = p_star / p_R
    rho_star_R = rho_R * ((gamma + 1) * p_ratio + (gamma - 1)) / ((gamma - 1) * p_ratio + (gamma + 1))
    shock_speed = u_R + c_R * np.sqrt((gamma + 1) * p_ratio / (2 * gamma) + (gamma - 1) / (2 * gamma))
    
    # Critical positions
    x_interface = 0.5  # Initial interface position
    
    # Left rarefaction: head and tail
    x_rare_head = x_interface + (u_L - c_L) * t
    x_rare_tail = x_interface + (u_star - c_star_L) * t
    
    # Contact discontinuity
    x_contact = x_interface + u_star * t
    
    # Right shock
    x_shock = x_interface + shock_speed * t
    
    # Initialize solution arrays
    n = len(x)
    density = np.zeros(n)
    velocity = np.zeros(n)
    pressure = np.zeros(n)
    
    for i in range(n):
        xi = x[i]
        
        if xi < x_rare_head:
            # Left state (undisturbed)
            density[i] = rho_L
            velocity[i] = u_L
            pressure[i] = p_L
            
        elif xi < x_rare_tail:
            # Inside rarefaction fan
            # Isentropic relations
            c_local = c_L + (gamma - 1) * (xi - x_interface) / (2 * t)
            velocity[i] = (gamma - 1) * (xi - x_interface) / ((gamma + 1) * t)
            density[i] = rho_L * (c_local / c_L)**(2 / (gamma - 1))
            pressure[i] = p_L * (c_local / c_L)**(2 * gamma / (gamma - 1))
            
        elif xi < x_contact:
            # Left star region
            density[i] = rho_star_L
            velocity[i] = u_star
            pressure[i] = p_star
            
        elif xi < x_shock:
            # Right star region  
            density[i] = rho_star_R
            velocity[i] = u_star
            pressure[i] = p_star
            
        else:
            # Right state (undisturbed)
            density[i] = rho_R
            velocity[i] = u_R
            pressure[i] = p_R
    
    return {
        'density': density,
        'velocity': velocity,
        'pressure': pressure,
        'positions': {
            'rare_head': x_rare_head,
            'rare_tail': x_rare_tail, 
            'contact': x_contact,
            'shock': x_shock
        }
    }


def run_sod_test(nx: int = 100, final_time: float = 0.25, 
                riemann_solver_type: str = 'hllc') -> Dict[str, Any]:
    """
    Run Sod shock tube test with stationary boundaries.
    
    Args:
        nx: Number of grid cells
        final_time: Final simulation time
        riemann_solver_type: 'exact' or 'hllc'
        
    Returns:
        Dictionary with test results
    """
    # Initialize grid with stationary boundaries
    grid = LagrangianGrid1d(n_cells=nx, x_left=0.0, x_right=1.0)
    
    # Initialize gas state
    gamma = 1.4
    gas_constant = 1.0  # Normalized units
    gas_state = GasState(n_cells=nx, gamma=gamma, gas_constant=gas_constant)
    
    # Set initial Sod conditions
    density = np.zeros(nx)
    velocity = np.zeros(nx)
    pressure = np.zeros(nx)
    
    interface_pos = 0.5
    for i in range(nx):
        if grid.x_center[i] < interface_pos:
            # Left state
            density[i] = 1.0
            velocity[i] = 0.0
            pressure[i] = 1.0
        else:
            # Right state  
            density[i] = 0.125
            velocity[i] = 0.0
            pressure[i] = 0.1
    
    gas_state.set_initial_conditions(density, velocity, pressure)
    
    # Initialize Riemann solver
    if riemann_solver_type.lower() == 'exact':
        riemann_solver = LagrangianRiemannSolver(gamma=gamma)
    else:
        riemann_solver = HLLCRiemannSolver(gamma=gamma)
    
    # Time integration parameters
    cfl = 0.5
    t = 0.0
    dt_history = []
    
    print(f"Running Sod test: nx={nx}, solver={riemann_solver_type}")
    print(f"Initial mass: {np.sum(gas_state.density * grid.volume):.6f}")
    
    # Time integration loop
    step = 0
    while t < final_time:
        # Compute time step
        dt = grid.get_cfl_timestep(gas_state.sound_speed, cfl)
        dt = min(dt, final_time - t)
        dt_history.append(dt)
        
        # Store old state
        density_old = gas_state.density.copy()
        momentum_old = gas_state.momentum.copy()
        energy_old = gas_state.energy.copy()
        
        # Compute face velocities with stationary boundaries
        u_face = grid.compute_face_velocities(gas_state.velocity)
        
        # Apply stationary boundary conditions
        u_face[0] = 0.0   # Left boundary fixed
        u_face[-1] = 0.0  # Right boundary fixed
        
        # Compute Riemann solutions at interior faces
        rho_L = gas_state.density[:-1]
        rho_R = gas_state.density[1:]
        u_L = gas_state.velocity[:-1] 
        u_R = gas_state.velocity[1:]
        p_L = gas_state.pressure[:-1]
        p_R = gas_state.pressure[1:]
        
        # Solve Riemann problems
        if hasattr(riemann_solver, 'solve_array'):
            u_star, p_star, _ = riemann_solver.solve_array(rho_L, u_L, p_L, rho_R, u_R, p_R)
            # Update interior face velocities
            u_face[1:-1] = u_star
        
        # Compute pressure forces
        p_face = np.zeros(grid.n_faces)
        p_face[0] = gas_state.pressure[0]
        p_face[-1] = gas_state.pressure[-1]
        if len(p_star) == len(p_face) - 2:
            p_face[1:-1] = p_star
        
        # Update momentum (Lagrangian momentum equation)
        for i in range(nx):
            dp_dx = (p_face[i+1] - p_face[i]) / grid.dx[i]
            gas_state.momentum[i] -= dt * dp_dx * grid.volume[i]
        
        # Update energy (work done by pressure)
        for i in range(nx):
            work_left = p_face[i] * u_face[i] * grid.area
            work_right = p_face[i+1] * u_face[i+1] * grid.area
            work_net = work_right - work_left
            gas_state.energy[i] -= dt * work_net / grid.dx[i]
        
        # Update grid positions (interior only, boundaries fixed)
        x_face_new = grid.x_face.copy()
        x_face_new[1:-1] = grid.x_face[1:-1] + dt * u_face[1:-1]
        grid.x_face = x_face_new
        grid._compute_geometry()
        
        # Update primitive variables
        gas_state.cons_to_prim()
        gas_state.update_thermodynamics()
        
        # Check for negative pressure/density
        if np.any(gas_state.pressure <= 0) or np.any(gas_state.density <= 0):
            print(f"WARNING: Negative values at step {step}, t={t:.4f}")
            break
        
        t += dt
        step += 1
        
        if step % 50 == 0:
            print(f"Step {step}: t={t:.4f}, dt={dt:.2e}")
    
    print(f"Final time: {t:.6f}, steps: {step}")
    print(f"Final mass: {np.sum(gas_state.density * grid.volume):.6f}")
    
    return {
        'grid': grid,
        'gas_state': gas_state,
        'final_time': t,
        'steps': step,
        'dt_history': dt_history,
        'solver_type': riemann_solver_type
    }


def compare_with_exact(results: Dict[str, Any], plot: bool = True) -> Dict[str, float]:
    """
    Compare numerical solution with exact Sod solution.
    
    Args:
        results: Results from run_sod_test
        plot: Whether to create comparison plots
        
    Returns:
        Dictionary with error metrics
    """
    grid = results['grid']
    gas_state = results['gas_state']
    t_final = results['final_time']
    
    # Get exact solution
    exact = sod_exact_solution(grid.x_center, t_final)
    
    # Compute L1 and L2 errors
    rho_error_l1 = np.mean(np.abs(gas_state.density - exact['density']))
    rho_error_l2 = np.sqrt(np.mean((gas_state.density - exact['density'])**2))
    
    u_error_l1 = np.mean(np.abs(gas_state.velocity - exact['velocity']))
    u_error_l2 = np.sqrt(np.mean((gas_state.velocity - exact['velocity'])**2))
    
    p_error_l1 = np.mean(np.abs(gas_state.pressure - exact['pressure']))
    p_error_l2 = np.sqrt(np.mean((gas_state.pressure - exact['pressure'])**2))
    
    errors = {
        'density_l1': rho_error_l1,
        'density_l2': rho_error_l2,
        'velocity_l1': u_error_l1,
        'velocity_l2': u_error_l2,
        'pressure_l1': p_error_l1,
        'pressure_l2': p_error_l2
    }
    
    print(f"\nError Analysis (t={t_final:.3f}):")
    print(f"Density  L1: {rho_error_l1:.4e}, L2: {rho_error_l2:.4e}")
    print(f"Velocity L1: {u_error_l1:.4e}, L2: {u_error_l2:.4e}")
    print(f"Pressure L1: {p_error_l1:.4e}, L2: {p_error_l2:.4e}")
    
    if plot:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle(f'Sod Test Comparison (t={t_final:.3f}, solver={results["solver_type"]})')
        
        # Density
        axes[0,0].plot(grid.x_center, gas_state.density, 'b-', linewidth=2, label='Numerical')
        axes[0,0].plot(grid.x_center, exact['density'], 'r--', linewidth=1, label='Exact')
        axes[0,0].set_xlabel('Position')
        axes[0,0].set_ylabel('Density')
        axes[0,0].set_title('Density')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # Velocity
        axes[0,1].plot(grid.x_center, gas_state.velocity, 'b-', linewidth=2, label='Numerical')
        axes[0,1].plot(grid.x_center, exact['velocity'], 'r--', linewidth=1, label='Exact')
        axes[0,1].set_xlabel('Position')
        axes[0,1].set_ylabel('Velocity')
        axes[0,1].set_title('Velocity')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        
        # Pressure
        axes[1,0].plot(grid.x_center, gas_state.pressure, 'b-', linewidth=2, label='Numerical')
        axes[1,0].plot(grid.x_center, exact['pressure'], 'r--', linewidth=1, label='Exact')
        axes[1,0].set_xlabel('Position')
        axes[1,0].set_ylabel('Pressure') 
        axes[1,0].set_title('Pressure')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
        
        # Grid
        axes[1,1].plot(grid.x_face, np.zeros_like(grid.x_face), 'ko-', markersize=3)
        axes[1,1].axvline(exact['positions']['rare_head'], color='g', linestyle=':', label='Rare. head')
        axes[1,1].axvline(exact['positions']['rare_tail'], color='g', linestyle='--', label='Rare. tail')
        axes[1,1].axvline(exact['positions']['contact'], color='b', linestyle='--', label='Contact')
        axes[1,1].axvline(exact['positions']['shock'], color='r', linestyle='--', label='Shock')
        axes[1,1].set_xlabel('Position')
        axes[1,1].set_ylabel('Grid')
        axes[1,1].set_title('Grid and Wave Positions')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('sod_test_stationary_boundaries.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    return errors


def test_convergence(grid_sizes: list = [50, 100, 200], 
                    riemann_solver_type: str = 'hllc') -> Dict[str, Any]:
    """
    Test convergence of the solver with grid refinement.
    
    Args:
        grid_sizes: List of grid resolutions to test
        riemann_solver_type: Riemann solver type
        
    Returns:
        Convergence test results
    """
    print(f"\n{'='*50}")
    print(f"CONVERGENCE TEST ({riemann_solver_type.upper()} solver)")
    print(f"{'='*50}")
    
    convergence_data = {
        'nx': [],
        'dx': [],
        'density_l2': [],
        'velocity_l2': [],
        'pressure_l2': []
    }
    
    for nx in grid_sizes:
        print(f"\nTesting nx = {nx}")
        results = run_sod_test(nx=nx, riemann_solver_type=riemann_solver_type)
        errors = compare_with_exact(results, plot=False)
        
        convergence_data['nx'].append(nx)
        convergence_data['dx'].append(1.0 / nx)
        convergence_data['density_l2'].append(errors['density_l2'])
        convergence_data['velocity_l2'].append(errors['velocity_l2'])
        convergence_data['pressure_l2'].append(errors['pressure_l2'])
    
    # Compute convergence rates
    dx = np.array(convergence_data['dx'])
    rho_errors = np.array(convergence_data['density_l2'])
    u_errors = np.array(convergence_data['velocity_l2'])
    p_errors = np.array(convergence_data['pressure_l2'])
    
    # Linear fit in log space: log(error) = rate * log(dx) + const
    rho_rate = np.polyfit(np.log(dx), np.log(rho_errors), 1)[0]
    u_rate = np.polyfit(np.log(dx), np.log(u_errors), 1)[0]
    p_rate = np.polyfit(np.log(dx), np.log(p_errors), 1)[0]
    
    print(f"\nConvergence Rates:")
    print(f"Density:  {rho_rate:.2f}")
    print(f"Velocity: {u_rate:.2f}")
    print(f"Pressure: {p_rate:.2f}")
    
    return {
        'data': convergence_data,
        'rates': {
            'density': rho_rate,
            'velocity': u_rate,
            'pressure': p_rate
        }
    }


def main():
    """Main test function."""
    print("="*60)
    print("SOD SHOCK TUBE TEST - STATIONARY BOUNDARIES")
    print("="*60)
    
    # Single test with visualization
    print("\n1. Running single test with HLLC solver...")
    results = run_sod_test(nx=100, riemann_solver_type='hllc')
    errors = compare_with_exact(results, plot=True)
    
    # Test with exact solver
    print("\n2. Running test with exact solver...")
    results_exact = run_sod_test(nx=100, riemann_solver_type='exact')
    errors_exact = compare_with_exact(results_exact, plot=False)
    
    print(f"\nSolver Comparison (L2 errors):")
    print(f"HLLC  - Density: {errors['density_l2']:.4e}, Velocity: {errors['velocity_l2']:.4e}, Pressure: {errors['pressure_l2']:.4e}")
    print(f"Exact - Density: {errors_exact['density_l2']:.4e}, Velocity: {errors_exact['velocity_l2']:.4e}, Pressure: {errors_exact['pressure_l2']:.4e}")
    
    # Convergence test
    print("\n3. Running convergence test...")
    convergence = test_convergence([50, 100, 200], 'hllc')
    
    # Pass/fail criteria
    density_error = errors['density_l2']
    velocity_error = errors['velocity_l2'] 
    pressure_error = errors['pressure_l2']
    
    tolerance = 0.05  # 5% error tolerance
    
    passed = (density_error < tolerance and 
              velocity_error < tolerance and 
              pressure_error < tolerance)
    
    print(f"\n{'='*60}")
    print(f"TEST RESULT: {'PASSED' if passed else 'FAILED'}")
    print(f"Tolerance: {tolerance:.1%}")
    print(f"Max error: {max(density_error, velocity_error, pressure_error):.4e}")
    print(f"{'='*60}")
    
    return passed


if __name__ == "__main__":
    main()