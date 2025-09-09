"""
Adaptive time stepping algorithms for Lagrangian compressible solver.

This module implements sophisticated time step control algorithms that
consider multiple stability constraints including acoustic CFL, piston
dynamics, grid deformation, and solution accuracy requirements.

Based on the L1d4 solver from GDTk repository.
"""

import numpy as np
from numba import njit
from typing import Dict, Any, Optional, List
import warnings


class AdaptiveTimestepper:
    """
    Adaptive time step controller for Lagrangian compressible flow.
    
    Manages multiple time step constraints to ensure stability and accuracy:
    - Acoustic CFL condition (sound wave propagation)
    - Piston dynamics stability  
    - Grid deformation limits
    - Solution accuracy requirements
    - User-specified limits
    
    Theory:
    Multiple time step constraints must be satisfied simultaneously:
    1. Acoustic: dt ≤ CFL * min(dx / (|u| + c))
    2. Piston: dt ≤ safety * min(dx / |v_piston|)  
    3. Grid: dt ≤ max_deformation * min(dx / |u_face|)
    4. Accuracy: dt adjusted based on solution changes
    
    The final time step is the minimum of all constraints.
    
    Attributes:
        cfl_acoustic (float): CFL number for acoustic waves
        cfl_piston (float): CFL number for piston motion
        max_deformation_ratio (float): Maximum grid deformation per step
        accuracy_tolerance (float): Solution accuracy tolerance
        dt_min (float): Minimum allowed time step
        dt_max (float): Maximum allowed time step
        dt_growth_factor (float): Maximum growth rate for time step
        dt_reduction_factor (float): Reduction factor when constraints violated
        
        # History tracking
        dt_history (List[float]): Time step history
        constraint_history (List[str]): Limiting constraint history
        
        # Stability monitoring
        unstable_count (int): Count of unstable steps
        max_unstable_steps (int): Maximum allowed unstable steps
    """
    
    def __init__(self, cfl_acoustic: float = 0.5, cfl_piston: float = 0.1,
                 max_deformation_ratio: float = 0.1):
        """
        Initialize adaptive timestepper.
        
        Args:
            cfl_acoustic: CFL number for acoustic stability
            cfl_piston: CFL number for piston motion stability
            max_deformation_ratio: Maximum grid deformation per timestep
        """
        # Primary constraints
        self.cfl_acoustic = cfl_acoustic
        self.cfl_piston = cfl_piston
        self.max_deformation_ratio = max_deformation_ratio
        
        # Accuracy control
        self.accuracy_tolerance = 1e-3
        self.enable_accuracy_control = False
        
        # Time step limits
        self.dt_min = 1e-12
        self.dt_max = np.inf
        
        # Adaptation parameters
        self.dt_growth_factor = 1.1
        self.dt_reduction_factor = 0.5
        self.safety_factor = 0.9
        
        # History tracking
        self.dt_history = []
        self.constraint_history = []
        
        # Stability monitoring
        self.unstable_count = 0
        self.max_unstable_steps = 5
        
        # Previous time step for growth control
        self.dt_previous = 0.0
    
    def compute_acoustic_timestep(self, grid, gas_state) -> float:
        """
        Compute time step based on acoustic CFL condition.
        
        Args:
            grid: LagrangianGrid1d object
            gas_state: GasState object
            
        Returns:
            Maximum stable acoustic time step
        """
        # Compute wave speeds in each cell
        wave_speeds = np.abs(gas_state.velocity) + gas_state.sound_speed
        
        # CFL condition: dt <= CFL * min(dx / wave_speed)
        dt_acoustic = self.cfl_acoustic * np.min(grid.dx / wave_speeds)
        
        return dt_acoustic
    
    def compute_piston_timestep(self, grid, pistons) -> float:
        """
        Compute time step based on piston motion constraints.
        
        Args:
            grid: LagrangianGrid1d object  
            pistons: List of Piston objects
            
        Returns:
            Maximum stable piston time step
        """
        if not pistons:
            return np.inf
        
        dt_piston = np.inf
        dx_min = np.min(grid.dx)
        
        for piston in pistons:
            if abs(piston.velocity) > 1e-10:
                # Limit piston motion to fraction of grid spacing
                dt_p = self.cfl_piston * dx_min / abs(piston.velocity)
                dt_piston = min(dt_piston, dt_p)
        
        return dt_piston
    
    def compute_deformation_timestep(self, grid, u_face) -> float:
        """
        Compute time step based on grid deformation limits.
        
        Args:
            grid: LagrangianGrid1d object
            u_face: Face velocities
            
        Returns:
            Maximum time step to limit grid deformation
        """
        # Compute velocity differences across faces
        du_face = np.abs(u_face[1:] - u_face[:-1])
        
        # Limit grid deformation: |du * dt| < max_ratio * dx
        max_du = np.max(du_face)
        if max_du > 1e-10:
            dt_deformation = self.max_deformation_ratio * np.min(grid.dx) / max_du
        else:
            dt_deformation = np.inf
        
        return dt_deformation
    
    def compute_accuracy_timestep(self, grid, gas_state, rhs) -> float:
        """
        Compute time step based on solution accuracy requirements.
        
        Uses estimate of solution changes to maintain accuracy.
        
        Args:
            grid: LagrangianGrid1d object
            gas_state: GasState object  
            rhs: Right-hand side from previous step
            
        Returns:
            Time step for accuracy control
        """
        if not self.enable_accuracy_control or rhs is None:
            return np.inf
        
        # Estimate relative changes in primitive variables
        rho_change = np.abs(rhs[:, 0]) / (gas_state.density + 1e-12)
        mom_change = np.abs(rhs[:, 1]) / (np.abs(gas_state.momentum) + 1e-12)
        energy_change = np.abs(rhs[:, 2]) / (gas_state.energy + 1e-12)
        
        max_change = np.max([np.max(rho_change), np.max(mom_change), np.max(energy_change)])
        
        if max_change > 1e-12:
            dt_accuracy = self.accuracy_tolerance / max_change
        else:
            dt_accuracy = np.inf
        
        return dt_accuracy
    
    def compute_timestep(self, grid, gas_state, pistons=None, u_face=None, 
                        rhs=None) -> Dict[str, Any]:
        """
        Compute adaptive time step considering all constraints.
        
        Args:
            grid: LagrangianGrid1d object
            gas_state: GasState object
            pistons: List of piston objects (optional)
            u_face: Face velocities (optional)
            rhs: Right-hand side from previous step (optional)
            
        Returns:
            Dictionary with time step and diagnostic information
        """
        # Compute individual time step constraints
        dt_acoustic = self.compute_acoustic_timestep(grid, gas_state)
        dt_piston = self.compute_piston_timestep(grid, pistons or [])
        
        # Grid deformation constraint
        if u_face is not None:
            dt_deformation = self.compute_deformation_timestep(grid, u_face)
        else:
            dt_deformation = np.inf
        
        # Accuracy constraint
        dt_accuracy = self.compute_accuracy_timestep(grid, gas_state, rhs)
        
        # Find limiting constraint
        constraints = {
            'acoustic': dt_acoustic,
            'piston': dt_piston,
            'deformation': dt_deformation,
            'accuracy': dt_accuracy,
            'max_limit': self.dt_max
        }
        
        limiting_constraint = min(constraints.keys(), key=lambda k: constraints[k])
        dt_new = constraints[limiting_constraint]
        
        # Apply safety factor
        dt_new *= self.safety_factor
        
        # Apply growth rate limit
        if self.dt_previous > 0:
            dt_max_growth = self.dt_previous * self.dt_growth_factor
            if dt_new > dt_max_growth:
                dt_new = dt_max_growth
                limiting_constraint = 'growth_limit'
        
        # Apply minimum time step
        if dt_new < self.dt_min:
            dt_new = self.dt_min
            limiting_constraint = 'min_limit'
            warnings.warn(f"Time step hit minimum limit: {self.dt_min}")
        
        # Store history
        self.dt_history.append(dt_new)
        self.constraint_history.append(limiting_constraint)
        self.dt_previous = dt_new
        
        return {
            'dt': dt_new,
            'limiting_constraint': limiting_constraint,
            'constraints': constraints.copy(),
            'safety_factor': self.safety_factor
        }
    
    def check_stability(self, grid, gas_state) -> Dict[str, Any]:
        """
        Check solution stability and grid quality.
        
        Args:
            grid: LagrangianGrid1d object
            gas_state: GasState object
            
        Returns:
            Dictionary with stability diagnostics
        """
        stability_info = {
            'stable': True,
            'issues': []
        }
        
        # Check for negative densities or pressures
        if np.any(gas_state.density <= 0):
            stability_info['stable'] = False
            stability_info['issues'].append('negative_density')
        
        if np.any(gas_state.pressure <= 0):
            stability_info['stable'] = False
            stability_info['issues'].append('negative_pressure')
        
        # Check for NaN or Inf values
        arrays_to_check = [gas_state.density, gas_state.velocity, gas_state.pressure]
        for i, arr in enumerate(['density', 'velocity', 'pressure']):
            if np.any(~np.isfinite(arrays_to_check[i])):
                stability_info['stable'] = False
                stability_info['issues'].append(f'non_finite_{arr}')
        
        # Check grid quality
        grid_quality = grid.check_grid_quality()
        if grid_quality['is_tangled']:
            stability_info['stable'] = False
            stability_info['issues'].append('grid_tangling')
        
        if grid_quality['aspect_ratio'] > 1000:
            stability_info['stable'] = False
            stability_info['issues'].append('extreme_aspect_ratio')
        
        return stability_info
    
    def adapt_parameters(self, stability_info: Dict[str, Any]):
        """
        Adapt timestepper parameters based on stability.
        
        Args:
            stability_info: Stability diagnostic information
        """
        if not stability_info['stable']:
            self.unstable_count += 1
            
            # Reduce time step more aggressively
            self.dt_reduction_factor *= 0.8
            self.safety_factor *= 0.9
            
            if self.unstable_count >= self.max_unstable_steps:
                raise RuntimeError(f"Simulation unstable for {self.unstable_count} steps")
        
        else:
            # Reset counters on stable step
            self.unstable_count = 0
            
            # Gradually recover parameters
            self.dt_reduction_factor = min(0.5, self.dt_reduction_factor * 1.01)
            self.safety_factor = min(0.9, self.safety_factor * 1.001)
    
    def get_diagnostics(self) -> Dict[str, Any]:
        """Get timestepper diagnostic information."""
        if self.dt_history:
            avg_dt = np.mean(self.dt_history[-100:])  # Average of last 100 steps
            min_dt = np.min(self.dt_history)
            max_dt = np.max(self.dt_history)
        else:
            avg_dt = min_dt = max_dt = 0.0
        
        # Count constraint occurrences
        constraint_counts = {}
        for constraint in self.constraint_history[-100:]:  # Last 100 steps
            constraint_counts[constraint] = constraint_counts.get(constraint, 0) + 1
        
        return {
            'current_dt': self.dt_previous,
            'average_dt': avg_dt,
            'min_dt': min_dt,
            'max_dt': max_dt,
            'unstable_count': self.unstable_count,
            'total_steps': len(self.dt_history),
            'constraint_counts': constraint_counts,
            'parameters': {
                'cfl_acoustic': self.cfl_acoustic,
                'cfl_piston': self.cfl_piston,
                'safety_factor': self.safety_factor,
                'dt_reduction_factor': self.dt_reduction_factor
            }
        }


@njit
def compute_cfl_timestep_numba(dx: np.ndarray, velocity: np.ndarray, 
                              sound_speed: np.ndarray, cfl: float) -> float:
    """
    Numba-accelerated CFL time step calculation.
    
    Args:
        dx: Cell sizes
        velocity: Cell velocities
        sound_speed: Sound speeds
        cfl: CFL number
        
    Returns:
        Maximum stable time step
    """
    n = len(dx)
    dt_min = np.inf
    
    for i in range(n):
        wave_speed = abs(velocity[i]) + sound_speed[i]
        dt_cell = cfl * dx[i] / wave_speed
        if dt_cell < dt_min:
            dt_min = dt_cell
    
    return dt_min


def create_timestepper_from_params(runtime_params) -> AdaptiveTimestepper:
    """
    Create timestepper from runtime parameters.
    
    Args:
        runtime_params: RuntimeParameters object
        
    Returns:
        Configured AdaptiveTimestepper
    """
    timestepper = AdaptiveTimestepper(
        cfl_acoustic=runtime_params.get_param("driver.cfl"),
        cfl_piston=runtime_params.get_param("compressible_lagrangian.cfl_piston"),
        max_deformation_ratio=runtime_params.get_param("compressible_lagrangian.max_deformation")
    )
    
    # Set optional parameters
    timestepper.dt_max = runtime_params.get_param("driver.dt_max")
    timestepper.enable_accuracy_control = runtime_params.get_param("compressible_lagrangian.accuracy_control")
    
    return timestepper