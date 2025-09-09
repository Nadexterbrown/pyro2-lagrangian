"""
Main Lagrangian compressible flow simulation class.

This module implements the complete 1D Lagrangian gas dynamics solver that
integrates the grid, gas state, Riemann solvers, and piston dynamics into
a unified simulation framework compatible with pyro2.

Based on the L1d4 solver from GDTk repository.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional, List
import warnings
import time

from pyro.simulation_null import NullSimulation, grid_setup, bc_setup
from pyro.util import msg
from .grid import LagrangianGrid1d
from .gas_state import GasState
from .riemann import LagrangianRiemannSolver, HLLCRiemannSolver
from .piston import Piston, PistonGroup


class Variables:
    """
    Container class for easy access to Lagrangian compressible variables.
    
    In Lagrangian formulation, we track:
    - Conservative variables: density, momentum, energy
    - Primitive variables: density, velocity, pressure  
    - Grid variables: positions, volumes
    """
    def __init__(self):
        # Conservative variables (for time integration)
        self.idens = 0  # Density
        self.imom = 1   # Momentum density
        self.iener = 2  # Total energy density
        
        # Primitive variables (for physics)
        self.irho = 0   # Density
        self.ivel = 1   # Velocity
        self.ipres = 2  # Pressure
        self.itemp = 3  # Temperature
        
        self.nvar = 3   # Number of conservative variables
        self.nprim = 4  # Number of primitive variables


class Simulation(NullSimulation):
    """
    Lagrangian compressible hydrodynamics simulation.
    
    This class orchestrates the complete Lagrangian simulation including:
    - Grid motion and geometric conservation
    - Gas dynamics with Riemann solvers
    - Piston dynamics and boundary coupling
    - Adaptive time stepping
    - Output and visualization
    
    Theory:
    The Lagrangian formulation solves the conservation laws in a moving
    coordinate system where cell boundaries follow fluid motion. This
    eliminates advection terms and ensures exact mass conservation.
    
    The system of equations becomes:
    dρ/dt = 0  (mass conservation - exact in Lagrangian)
    d(ρu)/dt = -∂p/∂x  (momentum conservation)
    d(ρE)/dt = -∂(pu)/∂x  (energy conservation)
    dx/dt = u  (grid motion)
    
    Attributes:
        grid (LagrangianGrid1d): Computational grid
        gas_state (GasState): Thermodynamic state
        riemann_solver: Riemann solver for interfaces
        pistons (list): List of piston objects
        piston_group (PistonGroup): Coupled piston system
        
        # Simulation parameters
        gamma (float): Ratio of specific heats
        cfl_factor (float): CFL safety factor
        riemann_solver_type (str): 'exact' or 'hllc'
        
        # Time stepping
        dt (float): Current time step
        dt_history (list): Time step history
        
        # Conservation tracking
        mass_initial (float): Initial total mass
        momentum_initial (float): Initial total momentum  
        energy_initial (float): Initial total energy
    """
    
    def __init__(self, solver_name: str):
        """
        Initialize Lagrangian simulation.
        
        Args:
            solver_name: Name of the solver (for pyro2 compatibility)
        """
        super().__init__(solver_name)
        
        # Core components (initialized in initialize())
        self.grid = None
        self.gas_state = None
        self.riemann_solver = None
        self.pistons = []
        self.piston_group = None
        
        # Physics parameters
        self.gamma = 1.4
        self.gas_constant = 287.0
        
        # Numerical parameters
        self.cfl_factor = 0.5
        self.riemann_solver_type = 'hllc'  # 'exact' or 'hllc'
        self.time_integration_order = 2  # 1 or 2
        
        # Time stepping
        self.dt = 0.0
        self.dt_history = []
        
        # Conservation tracking
        self.mass_initial = 0.0
        self.momentum_initial = 0.0
        self.energy_initial = 0.0
        
        # Output and diagnostics
        self.output_frequency = 10
        self.diagnostic_frequency = 1
        
    def initialize(self):
        """Initialize all simulation components."""
        
        # Get grid parameters from runtime parameters
        nx = self.rp.get_param("mesh.nx")
        xmin = self.rp.get_param("mesh.xmin") 
        xmax = self.rp.get_param("mesh.xmax")
        
        # Physics parameters
        self.gamma = self.rp.get_param("eos.gamma")
        self.gas_constant = self.rp.get_param("eos.gas_constant")
        
        # Numerical parameters
        self.cfl_factor = self.rp.get_param("driver.cfl")
        self.riemann_solver_type = self.rp.get_param("compressible_lagrangian.riemann_solver")
        
        # Initialize grid
        area = self.rp.get_param("compressible_lagrangian.area")
        self.grid = LagrangianGrid1d(nx, xmin, xmax, area)
        
        # Initialize gas state
        self.gas_state = GasState(nx, self.gamma, self.gas_constant)
        
        # Initialize Riemann solver
        if self.riemann_solver_type.lower() == 'exact':
            self.riemann_solver = LagrangianRiemannSolver(self.gamma)
        else:
            self.riemann_solver = HLLCRiemannSolver(self.gamma)
        
        # Initialize pistons if specified
        self._initialize_pistons()
        
        # Set initial conditions
        self._set_initial_conditions()
        
        # Store initial conserved quantities
        conserved = self.gas_state.compute_conserved_totals(self.grid)
        self.mass_initial = conserved['mass']
        self.momentum_initial = conserved['momentum']
        self.energy_initial = conserved['energy']
        
        msg.success("Lagrangian simulation initialized successfully")
    
    def _initialize_pistons(self):
        """Initialize piston systems from runtime parameters."""
        # Check if pistons are enabled
        if not self.rp.get_param("compressible_lagrangian.enable_pistons"):
            return
        
        # Left piston
        if self.rp.get_param("compressible_lagrangian.left_piston.enabled"):
            mass = self.rp.get_param("compressible_lagrangian.left_piston.mass")
            area = self.rp.get_param("compressible_lagrangian.left_piston.area")
            x0 = self.rp.get_param("compressible_lagrangian.left_piston.initial_position")
            v0 = self.rp.get_param("compressible_lagrangian.left_piston.initial_velocity")
            
            left_piston = Piston(mass, area, x0, v0)
            self.pistons.append(left_piston)
        
        # Right piston  
        if self.rp.get_param("compressible_lagrangian.right_piston.enabled"):
            mass = self.rp.get_param("compressible_lagrangian.right_piston.mass")
            area = self.rp.get_param("compressible_lagrangian.right_piston.area")
            x0 = self.rp.get_param("compressible_lagrangian.right_piston.initial_position") 
            v0 = self.rp.get_param("compressible_lagrangian.right_piston.initial_velocity")
            
            right_piston = Piston(mass, area, x0, v0)
            self.pistons.append(right_piston)
        
        # Create piston group if multiple pistons
        if len(self.pistons) > 1:
            self.piston_group = PistonGroup(self.pistons)
    
    def _set_initial_conditions(self):
        """Set initial conditions based on problem type."""
        problem_name = self.rp.get_param("compressible_lagrangian.problem")
        
        if problem_name == "sod":
            self._init_sod_problem()
        elif problem_name == "piston_compression":
            self._init_piston_compression()
        elif problem_name == "shock_tube":
            self._init_shock_tube()
        else:
            # Default uniform conditions
            rho0 = self.rp.get_param("compressible_lagrangian.rho0")
            u0 = self.rp.get_param("compressible_lagrangian.u0") 
            p0 = self.rp.get_param("compressible_lagrangian.p0")
            
            density = np.full(self.grid.n_cells, rho0)
            velocity = np.full(self.grid.n_cells, u0)
            pressure = np.full(self.grid.n_cells, p0)
            
            self.gas_state.set_initial_conditions(density, velocity, pressure)
    
    def _init_sod_problem(self):
        """Initialize Sod shock tube problem."""
        # Sod problem parameters
        rho_L = self.rp.get_param("sod.rho_L")
        u_L = self.rp.get_param("sod.u_L")
        p_L = self.rp.get_param("sod.p_L")
        
        rho_R = self.rp.get_param("sod.rho_R")
        u_R = self.rp.get_param("sod.u_R")
        p_R = self.rp.get_param("sod.p_R")
        
        interface_pos = self.rp.get_param("sod.interface_pos")
        
        # Set up initial conditions
        density = np.zeros(self.grid.n_cells)
        velocity = np.zeros(self.grid.n_cells)
        pressure = np.zeros(self.grid.n_cells)
        
        for i in range(self.grid.n_cells):
            if self.grid.x_center[i] < interface_pos:
                density[i] = rho_L
                velocity[i] = u_L
                pressure[i] = p_L
            else:
                density[i] = rho_R
                velocity[i] = u_R
                pressure[i] = p_R
        
        self.gas_state.set_initial_conditions(density, velocity, pressure)
    
    def _init_piston_compression(self):
        """Initialize piston-driven compression problem."""
        # Uniform initial gas state
        rho0 = self.rp.get_param("piston_compression.rho0")
        u0 = self.rp.get_param("piston_compression.u0")
        p0 = self.rp.get_param("piston_compression.p0")
        
        density = np.full(self.grid.n_cells, rho0)
        velocity = np.full(self.grid.n_cells, u0) 
        pressure = np.full(self.grid.n_cells, p0)
        
        self.gas_state.set_initial_conditions(density, velocity, pressure)
    
    def compute_timestep(self) -> float:
        """
        Compute adaptive timestep based on CFL conditions.
        
        Returns:
            Maximum stable timestep [s]
        """
        # Gas dynamics CFL condition
        dt_gas = self.grid.get_cfl_timestep(self.gas_state.sound_speed, self.cfl_factor)
        
        # Piston dynamics constraints
        dt_piston = np.inf
        if self.pistons:
            dx_min = np.min(self.grid.dx)
            for piston in self.pistons:
                dt_p = piston.get_cfl_timestep(dx_min)
                dt_piston = min(dt_piston, dt_p)
        
        # Take minimum
        dt = min(dt_gas, dt_piston)
        
        # Apply maximum timestep limit
        dt_max = self.rp.get_param("driver.dt_max")
        if dt_max > 0:
            dt = min(dt, dt_max)
        
        return dt
    
    def construct_rhs(self):
        """
        Construct right-hand side for time integration.
        
        In Lagrangian formulation:
        - Mass is exactly conserved (no RHS term)
        - Momentum RHS = -pressure gradient
        - Energy RHS = -divergence of pressure work
        """
        n_cells = self.grid.n_cells
        rhs = np.zeros((n_cells, 3))  # [density, momentum, energy]
        
        # Get interface states for Riemann problems
        rho_L, rho_R, u_L, u_R, p_L, p_R = self._compute_interface_states()
        
        # Solve Riemann problems at interfaces
        if hasattr(self.riemann_solver, 'solve_array'):
            u_face, p_face, rho_face = self.riemann_solver.solve_array(
                rho_L, u_L, p_L, rho_R, u_R, p_R)
        else:
            # Fallback to single interface solver
            u_face = np.zeros(n_cells + 1)
            p_face = np.zeros(n_cells + 1)
            for i in range(n_cells + 1):
                if i == 0 or i == n_cells:
                    # Boundary faces
                    u_face[i] = u_L[i] if i == 0 else u_R[i-1]
                    p_face[i] = p_L[i] if i == 0 else p_R[i-1]
                else:
                    # Interior faces
                    u_star, p_star, _ = self.riemann_solver.solve_interface(
                        rho_L[i], u_L[i], p_L[i], rho_R[i], u_R[i], p_R[i])
                    u_face[i] = u_star
                    p_face[i] = p_star
        
        # Update face velocities in grid
        self._apply_boundary_conditions(u_face)
        
        # Mass conservation: dρ/dt = 0 (exact in Lagrangian)
        rhs[:, 0] = 0.0
        
        # Momentum conservation: d(ρu)/dt = -∂p/∂x
        for i in range(n_cells):
            pressure_gradient = (p_face[i+1] - p_face[i]) / self.grid.dx[i]
            rhs[i, 1] = -pressure_gradient * self.grid.volume[i]
        
        # Energy conservation: d(ρE)/dt = -∂(pu)/∂x  
        for i in range(n_cells):
            work_flux_left = p_face[i] * u_face[i] * self.grid.area
            work_flux_right = p_face[i+1] * u_face[i+1] * self.grid.area
            work_divergence = (work_flux_right - work_flux_left) / self.grid.dx[i]
            rhs[i, 2] = -work_divergence
        
        return rhs
    
    def _compute_interface_states(self):
        """Compute left and right states at cell interfaces."""
        n_faces = self.grid.n_faces
        rho_L = np.zeros(n_faces)
        rho_R = np.zeros(n_faces)
        u_L = np.zeros(n_faces)
        u_R = np.zeros(n_faces) 
        p_L = np.zeros(n_faces)
        p_R = np.zeros(n_faces)
        
        # Interior faces
        for i in range(1, n_faces - 1):
            # Left state (from left cell)
            rho_L[i] = self.gas_state.density[i-1]
            u_L[i] = self.gas_state.velocity[i-1]
            p_L[i] = self.gas_state.pressure[i-1]
            
            # Right state (from right cell)
            rho_R[i] = self.gas_state.density[i]
            u_R[i] = self.gas_state.velocity[i]
            p_R[i] = self.gas_state.pressure[i]
        
        # Boundary faces - use cell values
        # Left boundary
        rho_L[0] = self.gas_state.density[0]
        rho_R[0] = self.gas_state.density[0]
        u_L[0] = self.gas_state.velocity[0]
        u_R[0] = self.gas_state.velocity[0]
        p_L[0] = self.gas_state.pressure[0]
        p_R[0] = self.gas_state.pressure[0]
        
        # Right boundary
        rho_L[-1] = self.gas_state.density[-1]
        rho_R[-1] = self.gas_state.density[-1]
        u_L[-1] = self.gas_state.velocity[-1]
        u_R[-1] = self.gas_state.velocity[-1]
        p_L[-1] = self.gas_state.pressure[-1]
        p_R[-1] = self.gas_state.pressure[-1]
        
        return rho_L, rho_R, u_L, u_R, p_L, p_R
    
    def _apply_boundary_conditions(self, u_face: np.ndarray):
        """Apply boundary conditions to face velocities."""
        # Left boundary
        left_bc = self.rp.get_param("mesh.xlboundary")
        if left_bc == "reflect":
            u_face[0] = 0.0
        elif left_bc == "piston" and self.pistons:
            u_face[0] = self.pistons[0].velocity
        
        # Right boundary
        right_bc = self.rp.get_param("mesh.xrboundary")
        if right_bc == "reflect":
            u_face[-1] = 0.0
        elif right_bc == "piston" and len(self.pistons) > 1:
            u_face[-1] = self.pistons[1].velocity
    
    def advance_timestep(self, dt: float):
        """
        Advance solution by one timestep.
        
        Args:
            dt: Time step size [s]
        """
        # Store current state for multi-stage methods
        density_old = self.gas_state.density.copy()
        momentum_old = self.gas_state.momentum.copy()
        energy_old = self.gas_state.energy.copy()
        
        if self.time_integration_order == 1:
            # Forward Euler
            rhs = self.construct_rhs()
            
            self.gas_state.momentum += dt * rhs[:, 1]
            self.gas_state.energy += dt * rhs[:, 2]
            
        else:
            # Second-order Runge-Kutta (RK2)
            # Stage 1
            rhs1 = self.construct_rhs()
            
            # Half step
            momentum_half = momentum_old + 0.5 * dt * rhs1[:, 1]
            energy_half = energy_old + 0.5 * dt * rhs1[:, 2]
            
            # Update primitive variables for stage 2
            self.gas_state.momentum[:] = momentum_half
            self.gas_state.energy[:] = energy_half
            self.gas_state.cons_to_prim()
            self.gas_state.update_thermodynamics()
            
            # Stage 2
            rhs2 = self.construct_rhs()
            
            # Full step with stage 2 RHS
            self.gas_state.momentum[:] = momentum_old + dt * rhs2[:, 1]
            self.gas_state.energy[:] = energy_old + dt * rhs2[:, 2]
        
        # Update primitive variables
        self.gas_state.cons_to_prim()
        self.gas_state.update_thermodynamics()
        
        # Update grid motion
        u_face = self.grid.compute_face_velocities(self.gas_state.velocity)
        self._apply_boundary_conditions(u_face)
        self.grid.update_positions(u_face, dt)
        
        # Update piston dynamics
        self._update_pistons(dt)
        
        # Update time
        self.cc_data.t += dt
        self.dt = dt
        self.dt_history.append(dt)
    
    def _update_pistons(self, dt: float):
        """Update piston positions and dynamics."""
        if not self.pistons:
            return
        
        # Update pressures on piston faces
        if len(self.pistons) >= 1:
            # Left piston
            self.pistons[0].pressure_front = 101325.0  # External pressure
            self.pistons[0].pressure_back = self.gas_state.pressure[0]
            
        if len(self.pistons) >= 2:
            # Right piston
            self.pistons[1].pressure_front = self.gas_state.pressure[-1]
            self.pistons[1].pressure_back = 101325.0  # External pressure
        
        # Update piston motion
        if self.piston_group:
            self.piston_group.update_coupled_motion(dt, self.cc_data.t)
        else:
            for piston in self.pistons:
                piston.update_state(dt, self.cc_data.t)
    
    def check_conservation(self) -> Dict[str, float]:
        """Check conservation of mass, momentum, and energy."""
        current = self.gas_state.compute_conserved_totals(self.grid)
        
        mass_error = abs(current['mass'] - self.mass_initial) / self.mass_initial
        momentum_error = abs(current['momentum'] - self.momentum_initial) / max(abs(self.momentum_initial), 1e-10)
        energy_error = abs(current['energy'] - self.energy_initial) / self.energy_initial
        
        return {
            'mass_error': mass_error,
            'momentum_error': momentum_error,
            'energy_error': energy_error,
            'mass_current': current['mass'],
            'momentum_current': current['momentum'],
            'energy_current': current['energy']
        }
    
    def dovis(self):
        """Visualization for Lagrangian simulation."""
        plt.clf()
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle(f'Lagrangian Simulation t = {self.cc_data.t:.4f}')
        
        # Plot 1: Density
        axes[0,0].plot(self.grid.x_center, self.gas_state.density, 'b-', linewidth=2)
        axes[0,0].set_xlabel('Position [m]')
        axes[0,0].set_ylabel('Density [kg/m³]')
        axes[0,0].set_title('Density')
        axes[0,0].grid(True, alpha=0.3)
        
        # Plot 2: Velocity
        axes[0,1].plot(self.grid.x_center, self.gas_state.velocity, 'r-', linewidth=2)
        axes[0,1].set_xlabel('Position [m]')
        axes[0,1].set_ylabel('Velocity [m/s]')
        axes[0,1].set_title('Velocity')
        axes[0,1].grid(True, alpha=0.3)
        
        # Plot 3: Pressure
        axes[1,0].plot(self.grid.x_center, self.gas_state.pressure, 'g-', linewidth=2)
        axes[1,0].set_xlabel('Position [m]')
        axes[1,0].set_ylabel('Pressure [Pa]')
        axes[1,0].set_title('Pressure')
        axes[1,0].grid(True, alpha=0.3)
        
        # Plot 4: Grid and pistons
        axes[1,1].plot(self.grid.x_face, np.zeros_like(self.grid.x_face), 'ko-', markersize=4)
        axes[1,1].set_xlabel('Position [m]')
        axes[1,1].set_ylabel('Grid Points')
        axes[1,1].set_title('Grid and Pistons')
        axes[1,1].grid(True, alpha=0.3)
        
        # Mark piston positions
        for i, piston in enumerate(self.pistons):
            color = 'red' if i == 0 else 'blue'
            axes[1,1].axvline(piston.position, color=color, linewidth=3, 
                             label=f'Piston {i+1}')
        
        if self.pistons:
            axes[1,1].legend()
        
        plt.tight_layout()
        plt.draw()
        
    def get_diagnostics(self) -> Dict[str, Any]:
        """Get diagnostic information."""
        conservation = self.check_conservation()
        grid_quality = self.grid.check_grid_quality()
        gas_validity = self.gas_state.check_physical_validity()
        
        diagnostics = {
            'time': self.cc_data.t,
            'timestep': self.dt,
            'conservation': conservation,
            'grid_quality': grid_quality,
            'gas_validity': gas_validity,
        }
        
        # Add piston diagnostics
        if self.pistons:
            piston_diag = {}
            for i, piston in enumerate(self.pistons):
                piston_diag[f'piston_{i}'] = {
                    'position': piston.position,
                    'velocity': piston.velocity,
                    'acceleration': piston.acceleration,
                    'pressure_front': piston.pressure_front,
                    'pressure_back': piston.pressure_back
                }
            diagnostics['pistons'] = piston_diag
        
        return diagnostics