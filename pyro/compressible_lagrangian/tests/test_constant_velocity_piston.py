"""
Constant velocity piston test for Lagrangian compressible solver.

This test validates the piston-gas coupling with a piston moving at constant
velocity at the left boundary. The piston compresses initially quiescent gas,
generating compression waves that propagate through the domain.

Test setup:
- Domain: [0, 0.1] m
- Left boundary: Piston with constant velocity
- Right boundary: Reflecting wall
- Initial gas: ρ=1.225 kg/m³, u=0 m/s, p=101325 Pa (air at STP)
- Piston: mass=0.1 kg, area=0.001 m², velocity=100 m/s (constant)

Expected behavior:
- Piston compresses gas, generating compression waves
- Waves reflect from right wall
- Pressure rises as piston advances
- Gas velocity increases near piston
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, List
import os
import sys

# Add pyro path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../'))

from pyro.compressible_lagrangian.grid import LagrangianGrid1d
from pyro.compressible_lagrangian.gas_state import GasState
from pyro.compressible_lagrangian.riemann import HLLCRiemannSolver
from pyro.compressible_lagrangian.piston import Piston


class ConstantVelocityPistonTest:
    """Constant velocity piston test class."""
    
    def __init__(self, nx: int = 50, domain_length: float = 0.1):
        """
        Initialize constant velocity piston test.
        
        Args:
            nx: Number of grid cells
            domain_length: Domain length [m]
        """
        self.nx = nx
        self.domain_length = domain_length
        self.gamma = 1.4
        self.gas_constant = 287.0  # Air
        
        # Simulation parameters
        self.final_time = 0.01  # 10 ms
        self.cfl = 0.3
        
        # Initialize components
        self.grid = LagrangianGrid1d(n_cells=nx, x_left=0.0, x_right=domain_length)
        self.gas_state = GasState(n_cells=nx, gamma=self.gamma, gas_constant=self.gas_constant)
        self.riemann_solver = HLLCRiemannSolver(gamma=self.gamma)
        
        # Initialize piston
        piston_mass = 0.1      # kg
        piston_area = 0.001    # m²
        piston_velocity = 100.0  # m/s
        self.piston = Piston(mass=piston_mass, area=piston_area, x0=0.0, v0=piston_velocity)
        
        # Make piston maintain constant velocity (no dynamics)
        self.piston.mass = 1e6  # Very large mass for constant velocity
        
        # Storage for history
        self.time_history = []
        self.piston_position_history = []
        self.piston_velocity_history = []
        self.pressure_history = []
        self.max_pressure_history = []
        
        # Test results
        self.results = {}
    
    def set_initial_conditions(self):
        """Set initial conditions for quiescent gas."""
        # Air at standard conditions
        rho0 = 1.225      # kg/m³
        u0 = 0.0          # m/s
        p0 = 101325.0     # Pa
        
        density = np.full(self.nx, rho0)
        velocity = np.full(self.nx, u0)
        pressure = np.full(self.nx, p0)
        
        self.gas_state.set_initial_conditions(density, velocity, pressure)
        
        # Store initial conserved quantities
        initial_conserved = self.gas_state.compute_conserved_totals(self.grid)
        self.initial_mass = initial_conserved['mass']
        self.initial_momentum = initial_conserved['momentum']
        self.initial_energy = initial_conserved['energy']
        
        # Set initial piston pressures
        self.piston.pressure_front = 101325.0  # External pressure
        self.piston.pressure_back = p0         # Initial gas pressure
    
    def run_simulation(self) -> Dict[str, Any]:
        """
        Run the constant velocity piston simulation.
        
        Returns:
            Dictionary with simulation results
        """
        print(f"Running constant velocity piston test: nx={self.nx}, L={self.domain_length}")
        print(f"Piston: mass={self.piston.mass:.3f} kg, area={self.piston.area:.6f} m²")
        print(f"Initial velocity: {self.piston.velocity:.1f} m/s")
        
        # Set initial conditions
        self.set_initial_conditions()
        
        # Time integration
        t = 0.0
        step = 0
        dt_history = []
        
        print(f"Initial mass: {self.initial_mass:.6e} kg")
        
        while t < self.final_time:
            # Compute time step considering both gas and piston
            dt_gas = self.grid.get_cfl_timestep(self.gas_state.sound_speed, self.cfl)
            dt_piston = self.piston.get_cfl_timestep(np.min(self.grid.dx), safety_factor=0.1)
            dt = min(dt_gas, dt_piston, self.final_time - t)
            dt_history.append(dt)
            
            # Update piston (maintain constant velocity)
            old_piston_pos = self.piston.position
            self.piston.velocity = 100.0  # Enforce constant velocity
            self.piston.position = old_piston_pos + self.piston.velocity * dt
            
            # Update piston pressure from gas
            self.piston.pressure_back = self.gas_state.pressure[0]
            
            # Compute face velocities
            u_face = self.grid.compute_face_velocities(self.gas_state.velocity)
            
            # Apply boundary conditions
            u_face[0] = self.piston.velocity      # Left boundary: piston velocity
            u_face[-1] = 0.0                     # Right boundary: reflecting wall
            
            # Update Lagrangian step
            self._update_lagrangian_step(dt, u_face)
            
            # Update grid positions with boundary constraints
            x_face_new = self.grid.x_face.copy()
            x_face_new[0] = self.piston.position  # Left boundary follows piston
            x_face_new[1:-1] = self.grid.x_face[1:-1] + dt * u_face[1:-1]  # Interior faces
            x_face_new[-1] = self.domain_length   # Right boundary fixed
            
            # Check for grid problems
            if np.any(np.diff(x_face_new) <= 0):
                print(f"WARNING: Grid tangling at step {step}, t={t:.4f}")
                break
            
            self.grid.x_face = x_face_new
            self.grid._compute_geometry()
            
            # Check for non-physical values
            if np.any(self.gas_state.pressure <= 0) or np.any(self.gas_state.density <= 0):
                print(f"WARNING: Non-physical values at step {step}, t={t:.4f}")
                break
            
            # Store history
            self.time_history.append(t)
            self.piston_position_history.append(self.piston.position)
            self.piston_velocity_history.append(self.piston.velocity)
            self.pressure_history.append(self.gas_state.pressure.copy())
            self.max_pressure_history.append(np.max(self.gas_state.pressure))
            
            t += dt
            step += 1
            
            if step % 100 == 0 or step < 10:
                max_p = np.max(self.gas_state.pressure)
                print(f"Step {step}: t={t:.4f}, dt={dt:.2e}, max_p={max_p:.0f} Pa, piston_x={self.piston.position:.6f} m")
        
        # Final diagnostics
        final_conserved = self.gas_state.compute_conserved_totals(self.grid)
        
        # Mass should be exactly conserved in Lagrangian
        mass_error = abs(final_conserved['mass'] - self.initial_mass) / self.initial_mass
        
        # Momentum increases due to piston work
        momentum_change = final_conserved['momentum'] - self.initial_momentum
        
        # Energy increases due to piston work
        energy_change = final_conserved['energy'] - self.initial_energy
        
        print(f"\nFinal time: {t:.6f} s, steps: {step}")
        print(f"Final piston position: {self.piston.position:.6f} m")
        print(f"Piston displacement: {self.piston.position:.6f} m")
        print(f"Final max pressure: {np.max(self.gas_state.pressure):.0f} Pa")
        print(f"Pressure ratio: {np.max(self.gas_state.pressure)/101325:.2f}")
        print(f"Mass conservation error: {mass_error:.2e}")
        print(f"Momentum change: {momentum_change:.4e} kg·m/s")
        print(f"Energy change: {energy_change:.4e} J")
        
        self.results = {
            'final_time': t,
            'steps': step,
            'dt_history': dt_history,
            'mass_error': mass_error,
            'momentum_change': momentum_change,
            'energy_change': energy_change,
            'final_max_pressure': np.max(self.gas_state.pressure),
            'pressure_ratio': np.max(self.gas_state.pressure) / 101325.0,
            'piston_displacement': self.piston.position,
            'initial_conserved': {
                'mass': self.initial_mass,
                'momentum': self.initial_momentum,
                'energy': self.initial_energy
            },
            'final_conserved': final_conserved
        }
        
        return self.results
    
    def _update_lagrangian_step(self, dt: float, u_face: np.ndarray):
        """Update gas state using Lagrangian equations with piston coupling."""
        # Get interface states (interior faces only)
        rho_L = self.gas_state.density[:-1]
        rho_R = self.gas_state.density[1:]
        u_L = self.gas_state.velocity[:-1]
        u_R = self.gas_state.velocity[1:]
        p_L = self.gas_state.pressure[:-1]
        p_R = self.gas_state.pressure[1:]
        
        # Solve Riemann problems at interior faces
        u_star, p_star, _ = self.riemann_solver.solve_array(rho_L, u_L, p_L, rho_R, u_R, p_R)
        
        # Construct face pressure array
        p_face = np.zeros(self.grid.n_faces)
        
        # Left boundary: piston pressure (external atmospheric)
        p_face[0] = self.piston.pressure_front
        
        # Interior faces: Riemann solution
        p_face[1:-1] = p_star
        
        # Right boundary: reflecting wall (use cell pressure)
        p_face[-1] = self.gas_state.pressure[-1]
        
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
    
    def analyze_results(self) -> Dict[str, Any]:
        """Analyze simulation results for expected behavior."""
        if not self.results:
            raise RuntimeError("No results to analyze. Run simulation first.")
        
        analysis = {}
        
        # Piston work calculation
        piston_displacement = self.results['piston_displacement']
        avg_pressure = np.mean([p[0] for p in self.pressure_history])  # Average pressure at piston face
        work_done = avg_pressure * self.piston.area * piston_displacement
        
        analysis['piston_work'] = work_done
        analysis['energy_efficiency'] = self.results['energy_change'] / work_done if work_done > 0 else 0
        
        # Wave propagation analysis
        # Estimate compression wave speed
        c0 = np.sqrt(self.gamma * 101325.0 / 1.225)  # Initial sound speed
        expected_wave_travel = c0 * self.results['final_time']
        
        analysis['initial_sound_speed'] = c0
        analysis['expected_wave_travel'] = expected_wave_travel
        analysis['domain_crossings'] = expected_wave_travel / self.domain_length
        
        # Pressure analysis
        max_pressure = self.results['final_max_pressure']
        initial_pressure = 101325.0
        
        # Theoretical compression ratio for constant velocity piston (approximate)
        # Using simple compression relationship
        volume_ratio = (self.domain_length - piston_displacement) / self.domain_length
        theoretical_pressure = initial_pressure / (volume_ratio**self.gamma)
        
        analysis['volume_ratio'] = volume_ratio
        analysis['theoretical_max_pressure'] = theoretical_pressure
        analysis['pressure_agreement'] = max_pressure / theoretical_pressure
        
        print(f"\nAnalysis Results:")
        print(f"Piston work done: {work_done:.2f} J")
        print(f"Energy efficiency: {analysis['energy_efficiency']:.1%}")
        print(f"Initial sound speed: {c0:.1f} m/s")
        print(f"Expected wave travel: {expected_wave_travel:.3f} m")
        print(f"Domain crossings: {analysis['domain_crossings']:.1f}")
        print(f"Volume compression ratio: {volume_ratio:.3f}")
        print(f"Theoretical max pressure: {theoretical_pressure/1000:.1f} kPa")
        print(f"Actual max pressure: {max_pressure/1000:.1f} kPa")
        print(f"Pressure agreement: {analysis['pressure_agreement']:.3f}")
        
        return analysis
    
    def plot_results(self):
        """Create comprehensive result plots."""
        if not self.results:
            raise RuntimeError("No results to plot. Run simulation first.")
        
        fig = plt.figure(figsize=(16, 12))
        
        # Create 6 subplots
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Final state profiles
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(self.grid.x_center, self.gas_state.density, 'b-', linewidth=2)
        ax1.set_xlabel('Position [m]')
        ax1.set_ylabel('Density [kg/m³]')
        ax1.set_title('Final Density Profile')
        ax1.grid(True, alpha=0.3)
        
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(self.grid.x_center, self.gas_state.velocity, 'r-', linewidth=2)
        ax2.set_xlabel('Position [m]')
        ax2.set_ylabel('Velocity [m/s]')
        ax2.set_title('Final Velocity Profile')
        ax2.grid(True, alpha=0.3)
        
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.plot(self.grid.x_center, self.gas_state.pressure/1000, 'g-', linewidth=2)
        ax3.set_xlabel('Position [m]')
        ax3.set_ylabel('Pressure [kPa]')
        ax3.set_title('Final Pressure Profile')
        ax3.grid(True, alpha=0.3)
        
        # 2. Time evolution
        time_array = np.array(self.time_history) * 1000  # Convert to ms
        
        ax4 = fig.add_subplot(gs[1, 0])
        ax4.plot(time_array, np.array(self.piston_position_history)*1000, 'b-', linewidth=2)
        ax4.set_xlabel('Time [ms]')
        ax4.set_ylabel('Piston Position [mm]')
        ax4.set_title('Piston Motion')
        ax4.grid(True, alpha=0.3)
        
        ax5 = fig.add_subplot(gs[1, 1])
        ax5.plot(time_array, self.piston_velocity_history, 'r-', linewidth=2)
        ax5.set_xlabel('Time [ms]')
        ax5.set_ylabel('Piston Velocity [m/s]')
        ax5.set_title('Piston Velocity')
        ax5.grid(True, alpha=0.3)
        
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.plot(time_array, np.array(self.max_pressure_history)/1000, 'g-', linewidth=2)
        ax6.set_xlabel('Time [ms]')
        ax6.set_ylabel('Max Pressure [kPa]')
        ax6.set_title('Maximum Pressure Evolution')
        ax6.grid(True, alpha=0.3)
        
        # 3. Grid evolution and pressure contour
        ax7 = fig.add_subplot(gs[2, :2])
        
        # Sample pressure history for contour plot
        n_time_samples = min(50, len(self.pressure_history))
        time_indices = np.linspace(0, len(self.pressure_history)-1, n_time_samples, dtype=int)
        
        pressure_matrix = np.array([self.pressure_history[i]/1000 for i in time_indices])
        time_samples = np.array([self.time_history[i]*1000 for i in time_indices])
        
        X, T = np.meshgrid(self.grid.x_center*1000, time_samples)  # Convert to mm and ms
        
        contour = ax7.contourf(X, T, pressure_matrix, levels=20, cmap='viridis')
        ax7.set_xlabel('Position [mm]')
        ax7.set_ylabel('Time [ms]')
        ax7.set_title('Pressure Evolution')
        cbar = plt.colorbar(contour, ax=ax7)
        cbar.set_label('Pressure [kPa]')
        
        # 4. Conservation check
        ax8 = fig.add_subplot(gs[2, 2])
        
        # Plot conservation quantities over time (if we stored them)
        ax8.text(0.1, 0.8, f"Mass error: {self.results['mass_error']:.2e}", transform=ax8.transAxes, fontsize=10)
        ax8.text(0.1, 0.7, f"Momentum Δ: {self.results['momentum_change']:.2e} kg·m/s", transform=ax8.transAxes, fontsize=10)
        ax8.text(0.1, 0.6, f"Energy Δ: {self.results['energy_change']:.2e} J", transform=ax8.transAxes, fontsize=10)
        ax8.text(0.1, 0.5, f"Pressure ratio: {self.results['pressure_ratio']:.2f}", transform=ax8.transAxes, fontsize=10)
        ax8.text(0.1, 0.4, f"Final time: {self.results['final_time']:.4f} s", transform=ax8.transAxes, fontsize=10)
        ax8.text(0.1, 0.3, f"Steps: {self.results['steps']}", transform=ax8.transAxes, fontsize=10)
        ax8.set_title('Simulation Summary')
        ax8.set_xticks([])
        ax8.set_yticks([])
        
        plt.suptitle(f'Constant Velocity Piston Test (v={self.piston.velocity:.0f} m/s, nx={self.nx})', fontsize=14)
        plt.savefig('constant_velocity_piston_test.png', dpi=150, bbox_inches='tight')
        plt.show()


def run_constant_velocity_piston_test(nx: int = 50, plot: bool = True) -> bool:
    """
    Run complete constant velocity piston test.
    
    Args:
        nx: Number of grid cells
        plot: Whether to create plots
        
    Returns:
        True if test passed, False otherwise
    """
    # Create and run test
    test = ConstantVelocityPistonTest(nx=nx)
    results = test.run_simulation()
    analysis = test.analyze_results()
    
    if plot:
        test.plot_results()
    
    # Pass/fail criteria
    criteria = {
        'mass_conservation': results['mass_error'] < 1e-12,
        'pressure_increase': results['pressure_ratio'] > 1.5,  # Expect significant compression
        'positive_work': results['energy_change'] > 0,         # Energy should increase
        'reasonable_pressure': analysis['pressure_agreement'] > 0.5 and analysis['pressure_agreement'] < 2.0
    }
    
    passed = all(criteria.values())
    
    print(f"\n{'='*70}")
    print(f"CONSTANT VELOCITY PISTON TEST RESULT: {'PASSED' if passed else 'FAILED'}")
    print(f"{'='*70}")
    
    for criterion, result in criteria.items():
        status = 'PASS' if result else 'FAIL'
        print(f"{criterion:.<45} {status}")
    
    print(f"{'='*70}")
    
    return passed


if __name__ == "__main__":
    print("CONSTANT VELOCITY PISTON TEST")
    print("="*70)
    
    # Run test
    passed = run_constant_velocity_piston_test(nx=50, plot=True)
    
    print(f"\nOVERALL TEST: {'PASSED' if passed else 'FAILED'}")