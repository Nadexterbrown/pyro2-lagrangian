"""
Sod shock tube test for Lagrangian compressible solver.

This test validates the Lagrangian solver against the analytical Sod solution.
The classic Riemann problem with left/right states separated by a diaphragm
that breaks at t=0.

Test setup:
- Domain: [0, 1] 
- Initial discontinuity at x = 0.5
- Left state: ρ=1.0, u=0.0, p=1.0  
- Right state: ρ=0.125, u=0.0, p=0.1
- Boundary conditions: outflow
- Final time: t = 0.25

Expected solution:
- Leftward rarefaction wave
- Contact discontinuity  
- Rightward shock wave
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, Tuple
import os
import sys

# Add pyro path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../'))

from pyro.compressible_lagrangian.grid import LagrangianGrid1d
from pyro.compressible_lagrangian.gas_state import GasState
from pyro.compressible_lagrangian.riemann import LagrangianRiemannSolver, HLLCRiemannSolver


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
    p_star = 0.30313017805064697  # Exact value
    u_star = 0.9274526200489498   # Exact value
    
    # Wave speeds and densities
    # Left rarefaction
    rho_star_L = rho_L * (p_star / p_L)**(1.0 / gamma)
    c_star_L = np.sqrt(gamma * p_star / rho_star_L)
    
    # Right shock
    p_ratio = p_star / p_R
    rho_star_R = rho_R * ((gamma + 1) * p_ratio + (gamma - 1)) / ((gamma - 1) * p_ratio + (gamma + 1))
    shock_speed = u_R + c_R * np.sqrt((gamma + 1) * p_ratio / (2 * gamma) + (gamma - 1) / (2 * gamma))
    
    # Critical positions (relative to initial interface at x=0.5)
    x_interface = 0.5
    
    x_rare_head = x_interface + (u_L - c_L) * t  # Left rarefaction head
    x_rare_tail = x_interface + (u_star - c_star_L) * t  # Left rarefaction tail
    x_contact = x_interface + u_star * t  # Contact discontinuity
    x_shock = x_interface + shock_speed * t  # Right shock
    
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
            # Inside rarefaction fan - isentropic relations
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
        'temperature': pressure / (density * 287.0),  # Assuming R = 287 J/(kg·K)
        'positions': {
            'rare_head': x_rare_head,
            'rare_tail': x_rare_tail, 
            'contact': x_contact,
            'shock': x_shock
        }
    }


class SodTest:
    """Sod shock tube test class."""
    
    def __init__(self, nx: int = 100, gamma: float = 1.4, gas_constant: float = 287.0):
        """
        Initialize Sod test.
        
        Args:
            nx: Number of grid cells
            gamma: Ratio of specific heats
            gas_constant: Specific gas constant [J/(kg·K)]
        """
        self.nx = nx
        self.gamma = gamma
        self.gas_constant = gas_constant
        self.final_time = 0.25
        self.cfl = 0.5
        
        # Initialize components
        self.grid = LagrangianGrid1d(n_cells=nx, x_left=0.0, x_right=1.0)
        self.gas_state = GasState(n_cells=nx, gamma=gamma, gas_constant=gas_constant)
        
        # Test results
        self.results = {}
    
    def set_initial_conditions(self):
        """Set initial Sod conditions."""
        density = np.zeros(self.nx)
        velocity = np.zeros(self.nx)
        pressure = np.zeros(self.nx)
        
        interface_pos = 0.5
        for i in range(self.nx):
            if self.grid.x_center[i] < interface_pos:
                # Left state
                density[i] = 1.0
                velocity[i] = 0.0
                pressure[i] = 101325.0  # 1 atm in Pa
            else:
                # Right state  
                density[i] = 0.125
                velocity[i] = 0.0
                pressure[i] = 10132.5   # 0.1 atm in Pa
        
        self.gas_state.set_initial_conditions(density, velocity, pressure)
        
        # Store initial conserved quantities
        initial_conserved = self.gas_state.compute_conserved_totals(self.grid)
        self.initial_mass = initial_conserved['mass']
        self.initial_momentum = initial_conserved['momentum']
        self.initial_energy = initial_conserved['energy']
    
    def run_simulation(self, riemann_solver_type: str = 'hllc') -> Dict[str, Any]:
        """
        Run the Sod simulation.
        
        Args:
            riemann_solver_type: 'exact' or 'hllc'
            
        Returns:
            Dictionary with simulation results
        """
        print(f"Running Sod test: nx={self.nx}, solver={riemann_solver_type}")
        
        # Initialize Riemann solver
        if riemann_solver_type.lower() == 'exact':
            riemann_solver = LagrangianRiemannSolver(gamma=self.gamma)
        else:
            riemann_solver = HLLCRiemannSolver(gamma=self.gamma)
        
        # Set initial conditions
        self.set_initial_conditions()
        
        # Time integration
        t = 0.0
        step = 0
        dt_history = []
        time_history = [0.0]
        
        print(f"Initial mass: {self.initial_mass:.6f} kg")
        
        while t < self.final_time:
            # Compute time step
            dt = self.grid.get_cfl_timestep(self.gas_state.sound_speed, self.cfl)
            dt = min(dt, self.final_time - t)
            dt_history.append(dt)
            
            # Compute face velocities (outflow boundaries)
            u_face = self.grid.compute_face_velocities(self.gas_state.velocity)
            
            # Get interface states for Riemann problems
            rho_L, rho_R, u_L, u_R, p_L, p_R = self._get_interface_states()
            
            # Solve Riemann problems at interior faces
            if hasattr(riemann_solver, 'solve_array'):
                u_star, p_star, _ = riemann_solver.solve_array(rho_L, u_L, p_L, rho_R, u_R, p_R)
                u_face[1:-1] = u_star
            else:
                # Fallback for single interface solver
                for i in range(1, len(u_face) - 1):
                    u_s, p_s, _ = riemann_solver.solve_interface(
                        rho_L[i-1], u_L[i-1], p_L[i-1], 
                        rho_R[i-1], u_R[i-1], p_R[i-1])
                    u_face[i] = u_s
            
            # Update using Lagrangian equations
            self._update_lagrangian_step(dt, u_face, riemann_solver)
            
            # Update grid positions
            self.grid.update_positions(u_face, dt)
            
            # Check for problems
            if np.any(self.gas_state.pressure <= 0) or np.any(self.gas_state.density <= 0):
                print(f"WARNING: Non-physical values at step {step}, t={t:.4f}")
                break
            
            t += dt
            step += 1
            time_history.append(t)
            
            if step % 100 == 0:
                print(f"Step {step}: t={t:.4f}, dt={dt:.2e}")
        
        # Final diagnostics
        final_conserved = self.gas_state.compute_conserved_totals(self.grid)
        mass_error = abs(final_conserved['mass'] - self.initial_mass) / self.initial_mass
        
        print(f"Final time: {t:.6f}, steps: {step}")
        print(f"Final mass: {final_conserved['mass']:.6f} kg")
        print(f"Mass conservation error: {mass_error:.2e}")
        
        self.results = {
            'final_time': t,
            'steps': step,
            'dt_history': dt_history,
            'time_history': time_history,
            'solver_type': riemann_solver_type,
            'mass_error': mass_error,
            'initial_conserved': {
                'mass': self.initial_mass,
                'momentum': self.initial_momentum,
                'energy': self.initial_energy
            },
            'final_conserved': final_conserved
        }
        
        return self.results
    
    def _get_interface_states(self) -> Tuple[np.ndarray, ...]:
        """Get left and right states at interior interfaces."""
        n_interior = self.nx - 1
        
        rho_L = self.gas_state.density[:-1]
        rho_R = self.gas_state.density[1:]
        u_L = self.gas_state.velocity[:-1]
        u_R = self.gas_state.velocity[1:]
        p_L = self.gas_state.pressure[:-1]
        p_R = self.gas_state.pressure[1:]
        
        return rho_L, rho_R, u_L, u_R, p_L, p_R
    
    def _update_lagrangian_step(self, dt: float, u_face: np.ndarray, riemann_solver):
        """Update gas state using Lagrangian equations."""
        # Get interface states
        rho_L, rho_R, u_L, u_R, p_L, p_R = self._get_interface_states()
        
        # Compute interface pressures
        if hasattr(riemann_solver, 'solve_array'):
            _, p_star, _ = riemann_solver.solve_array(rho_L, u_L, p_L, rho_R, u_R, p_R)
        else:
            p_star = np.zeros(len(rho_L))
            for i in range(len(rho_L)):
                _, p_star[i], _ = riemann_solver.solve_interface(
                    rho_L[i], u_L[i], p_L[i], rho_R[i], u_R[i], p_R[i])
        
        # Construct face pressure array
        p_face = np.zeros(self.grid.n_faces)
        p_face[0] = self.gas_state.pressure[0]      # Left boundary
        p_face[-1] = self.gas_state.pressure[-1]   # Right boundary
        p_face[1:-1] = p_star                      # Interior faces
        
        # Update momentum: d(ρu)/dt = -∂p/∂x
        for i in range(self.nx):
            dp_dx = (p_face[i+1] - p_face[i]) / self.grid.dx[i]
            self.gas_state.momentum[i] -= dt * dp_dx * self.grid.volume[i]
        
        # Update energy: d(ρE)/dt = -∂(pu)/∂x
        for i in range(self.nx):
            pu_left = p_face[i] * u_face[i]
            pu_right = p_face[i+1] * u_face[i+1]
            div_pu = (pu_right - pu_left) / self.grid.dx[i] * self.grid.area
            self.gas_state.energy[i] -= dt * div_pu
        
        # Update primitive variables
        self.gas_state.cons_to_prim()
        self.gas_state.update_thermodynamics()
    
    def compare_with_exact(self, plot: bool = True) -> Dict[str, float]:
        """Compare numerical solution with exact solution."""
        # Get exact solution
        exact = sod_exact_solution(self.grid.x_center, self.results['final_time'], self.gamma)
        
        # Scale exact solution to match our pressure scaling
        pressure_scale = 101325.0  # Scale from normalized to Pa
        exact['pressure'] *= pressure_scale
        exact['density'] *= 1.0  # Density scaling
        
        # Compute errors
        rho_error_l1 = np.mean(np.abs(self.gas_state.density - exact['density']))
        rho_error_l2 = np.sqrt(np.mean((self.gas_state.density - exact['density'])**2))
        
        u_error_l1 = np.mean(np.abs(self.gas_state.velocity - exact['velocity']))
        u_error_l2 = np.sqrt(np.mean((self.gas_state.velocity - exact['velocity'])**2))
        
        p_error_l1 = np.mean(np.abs(self.gas_state.pressure - exact['pressure']))
        p_error_l2 = np.sqrt(np.mean((self.gas_state.pressure - exact['pressure'])**2))
        
        # Relative errors
        rho_rel_l2 = rho_error_l2 / np.mean(exact['density'])
        u_rel_l2 = u_error_l2 / (np.mean(np.abs(exact['velocity'])) + 1e-10)
        p_rel_l2 = p_error_l2 / np.mean(exact['pressure'])
        
        errors = {
            'density_l1': rho_error_l1,
            'density_l2': rho_error_l2,
            'density_rel_l2': rho_rel_l2,
            'velocity_l1': u_error_l1,
            'velocity_l2': u_error_l2,
            'velocity_rel_l2': u_rel_l2,
            'pressure_l1': p_error_l1,
            'pressure_l2': p_error_l2,
            'pressure_rel_l2': p_rel_l2
        }
        
        print(f"\nError Analysis (t={self.results['final_time']:.3f}):")
        print(f"Density  - L1: {rho_error_l1:.4e}, L2: {rho_error_l2:.4e}, Rel L2: {rho_rel_l2:.4e}")
        print(f"Velocity - L1: {u_error_l1:.4e}, L2: {u_error_l2:.4e}, Rel L2: {u_rel_l2:.4e}")
        print(f"Pressure - L1: {p_error_l1:.4e}, L2: {p_error_l2:.4e}, Rel L2: {p_rel_l2:.4e}")
        
        if plot:
            self._plot_comparison(exact, errors)
        
        return errors
    
    def _plot_comparison(self, exact: Dict[str, np.ndarray], errors: Dict[str, float]):
        """Create comparison plots."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Sod Test Comparison (t={self.results["final_time"]:.3f}, '
                    f'nx={self.nx}, solver={self.results["solver_type"]})', fontsize=14)
        
        # Density
        axes[0,0].plot(self.grid.x_center, self.gas_state.density, 'b-', linewidth=2, label='Numerical')
        axes[0,0].plot(self.grid.x_center, exact['density'], 'r--', linewidth=1.5, label='Exact')
        axes[0,0].set_xlabel('Position [m]')
        axes[0,0].set_ylabel('Density [kg/m³]')
        axes[0,0].set_title(f'Density (L2 error: {errors["density_rel_l2"]:.2e})')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # Velocity
        axes[0,1].plot(self.grid.x_center, self.gas_state.velocity, 'b-', linewidth=2, label='Numerical')
        axes[0,1].plot(self.grid.x_center, exact['velocity'], 'r--', linewidth=1.5, label='Exact')
        axes[0,1].set_xlabel('Position [m]')
        axes[0,1].set_ylabel('Velocity [m/s]')
        axes[0,1].set_title(f'Velocity (L2 error: {errors["velocity_rel_l2"]:.2e})')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        
        # Pressure
        axes[1,0].plot(self.grid.x_center, self.gas_state.pressure/1000, 'b-', linewidth=2, label='Numerical')
        axes[1,0].plot(self.grid.x_center, exact['pressure']/1000, 'r--', linewidth=1.5, label='Exact')
        axes[1,0].set_xlabel('Position [m]')
        axes[1,0].set_ylabel('Pressure [kPa]')
        axes[1,0].set_title(f'Pressure (L2 error: {errors["pressure_rel_l2"]:.2e})')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
        
        # Grid and wave positions
        axes[1,1].plot(self.grid.x_face, np.zeros_like(self.grid.x_face), 'ko-', markersize=3, alpha=0.7)
        axes[1,1].axvline(exact['positions']['rare_head'], color='g', linestyle=':', linewidth=2, label='Rare. head')
        axes[1,1].axvline(exact['positions']['rare_tail'], color='g', linestyle='--', linewidth=2, label='Rare. tail')
        axes[1,1].axvline(exact['positions']['contact'], color='b', linestyle='--', linewidth=2, label='Contact')
        axes[1,1].axvline(exact['positions']['shock'], color='r', linestyle='--', linewidth=2, label='Shock')
        axes[1,1].set_xlabel('Position [m]')
        axes[1,1].set_ylabel('Grid Points')
        axes[1,1].set_title('Grid and Wave Positions')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)
        axes[1,1].set_ylim(-0.1, 0.1)
        
        plt.tight_layout()
        plt.savefig('sod_test_comparison.png', dpi=150, bbox_inches='tight')
        plt.show()


def run_sod_test(nx: int = 100, riemann_solver_type: str = 'hllc', plot: bool = True) -> bool:
    """
    Run complete Sod test.
    
    Args:
        nx: Number of grid cells
        riemann_solver_type: 'exact' or 'hllc'
        plot: Whether to create plots
        
    Returns:
        True if test passed, False otherwise
    """
    # Create and run test
    test = SodTest(nx=nx)
    results = test.run_simulation(riemann_solver_type=riemann_solver_type)
    errors = test.compare_with_exact(plot=plot)
    
    # Pass/fail criteria
    tolerance = 0.1  # 10% relative error tolerance
    
    passed = (errors['density_rel_l2'] < tolerance and 
              errors['velocity_rel_l2'] < tolerance and 
              errors['pressure_rel_l2'] < tolerance and
              results['mass_error'] < 1e-12)
    
    print(f"\n{'='*60}")
    print(f"SOD TEST RESULT: {'PASSED' if passed else 'FAILED'}")
    print(f"Tolerance: {tolerance:.1%}")
    print(f"Max relative error: {max(errors['density_rel_l2'], errors['velocity_rel_l2'], errors['pressure_rel_l2']):.4e}")
    print(f"Mass conservation error: {results['mass_error']:.2e}")
    print(f"{'='*60}")
    
    return passed


if __name__ == "__main__":
    print("SOD SHOCK TUBE TEST")
    print("="*50)
    
    # Test with HLLC solver
    print("\n1. Testing with HLLC solver...")
    hllc_passed = run_sod_test(nx=100, riemann_solver_type='hllc', plot=True)
    
    # Test with exact solver
    print("\n2. Testing with exact solver...")
    exact_passed = run_sod_test(nx=100, riemann_solver_type='exact', plot=False)
    
    # Overall result
    overall_passed = hllc_passed and exact_passed
    print(f"\nOVERALL SOD TEST: {'PASSED' if overall_passed else 'FAILED'}")