"""
Gas slug thermodynamic state management for Lagrangian solver.

This module implements the GasSlug class that handles thermodynamic state
arrays, chemical composition tracking, and interfaces with equation of state
models. It manages the conversion between conservative and primitive variables
and computes thermodynamic properties.

Based on the L1d4 solver from GDTk repository.
"""

import numpy as np
from numba import njit
from typing import Optional, Dict, Any
import warnings


class GasState:
    """
    Thermodynamic state management for gas slugs in Lagrangian solver.
    
    Manages arrays of thermodynamic properties for gas cells, handles
    conversions between conservative and primitive variables, and provides
    interface to equation of state calculations.
    
    Theory:
    In Lagrangian formulation, mass is conserved within each cell, so we
    track density, velocity, and specific energy. The primitive variables
    (rho, u, p, T) are related to conservative variables (rho, rho*u, rho*E)
    through thermodynamic relations and equation of state.
    
    For ideal gas: p = rho * R * T, E = e + 0.5 * u^2, e = Cv * T
    
    Attributes:
        n_cells (int): Number of gas cells
        gamma (float): Ratio of specific heats
        gas_constant (float): Specific gas constant [J/(kg·K)]
        
        # Conservative variables
        density (np.ndarray): Density [kg/m³]
        momentum (np.ndarray): Momentum density [kg/(m²·s)]  
        energy (np.ndarray): Total energy density [J/m³]
        
        # Primitive variables
        velocity (np.ndarray): Velocity [m/s]
        pressure (np.ndarray): Pressure [Pa]
        temperature (np.ndarray): Temperature [K]
        sound_speed (np.ndarray): Sound speed [m/s]
        
        # Thermodynamic properties
        internal_energy (np.ndarray): Specific internal energy [J/kg]
        enthalpy (np.ndarray): Specific enthalpy [J/kg]
        entropy (np.ndarray): Specific entropy [J/(kg·K)]
    """
    
    def __init__(self, n_cells: int, gamma: float = 1.4, 
                 gas_constant: float = 287.0):
        """
        Initialize gas state arrays.
        
        Args:
            n_cells: Number of computational cells
            gamma: Ratio of specific heats (default 1.4 for air)
            gas_constant: Specific gas constant [J/(kg·K)] (default 287 for air)
        """
        self.n_cells = n_cells
        self.gamma = gamma
        self.gas_constant = gas_constant
        
        # Derived constants
        self.cv = gas_constant / (gamma - 1.0)  # Specific heat at constant volume
        self.cp = gamma * self.cv               # Specific heat at constant pressure
        
        # Conservative variables
        self.density = np.zeros(n_cells)
        self.momentum = np.zeros(n_cells)
        self.energy = np.zeros(n_cells)
        
        # Primitive variables  
        self.velocity = np.zeros(n_cells)
        self.pressure = np.zeros(n_cells)
        self.temperature = np.zeros(n_cells)
        self.sound_speed = np.zeros(n_cells)
        
        # Thermodynamic properties
        self.internal_energy = np.zeros(n_cells)
        self.enthalpy = np.zeros(n_cells)
        self.entropy = np.zeros(n_cells)
    
    def set_initial_conditions(self, density: np.ndarray, velocity: np.ndarray,
                             pressure: np.ndarray):
        """
        Set initial conditions from primitive variables.
        
        Args:
            density: Initial density distribution [kg/m³]
            velocity: Initial velocity distribution [m/s]  
            pressure: Initial pressure distribution [Pa]
        """
        self.density[:] = density
        self.velocity[:] = velocity  
        self.pressure[:] = pressure
        
        # Compute derived quantities
        self.prim_to_cons()
        self.update_thermodynamics()
    
    def prim_to_cons(self):
        """Convert primitive to conservative variables."""
        self.momentum[:] = self.density * self.velocity
        
        # Compute internal energy from ideal gas law
        self.internal_energy[:] = self.pressure / (self.density * (self.gamma - 1.0))
        
        # Total energy = internal + kinetic
        self.energy[:] = self.density * (self.internal_energy + 
                                       0.5 * self.velocity**2)
    
    def cons_to_prim(self):
        """
        Convert conservative to primitive variables.
        
        Updates velocity, internal energy, pressure, and temperature from
        conservative variables (density, momentum, energy).
        """
        # Check for positive density
        if np.any(self.density <= 0):
            min_density = np.min(self.density)
            raise ValueError(f"Non-positive density detected: min = {min_density}")
        
        # Compute velocity
        self.velocity[:] = self.momentum / self.density
        
        # Compute specific internal energy  
        kinetic_energy = 0.5 * self.velocity**2
        total_specific_energy = self.energy / self.density
        self.internal_energy[:] = total_specific_energy - kinetic_energy
        
        # Check for positive internal energy
        if np.any(self.internal_energy <= 0):
            min_e = np.min(self.internal_energy)
            warnings.warn(f"Non-positive internal energy detected: min = {min_e}")
            # Clamp to small positive value
            self.internal_energy[self.internal_energy <= 0] = 1e-10
        
        # Compute pressure from ideal gas law
        self.pressure[:] = self.density * self.internal_energy * (self.gamma - 1.0)
        
        # Compute temperature
        self.temperature[:] = self.pressure / (self.density * self.gas_constant)
    
    def update_thermodynamics(self):
        """Update all thermodynamic properties from current state."""
        # Sound speed
        self.sound_speed[:] = np.sqrt(self.gamma * self.pressure / self.density)
        
        # Specific enthalpy  
        self.enthalpy[:] = self.internal_energy + self.pressure / self.density
        
        # Specific entropy (relative to reference state)
        # s - s_ref = Cv * ln(T/T_ref) + R * ln(rho_ref/rho)
        T_ref = 273.15  # Reference temperature [K]
        rho_ref = 1.225  # Reference density [kg/m³]
        self.entropy[:] = (self.cv * np.log(self.temperature / T_ref) + 
                          self.gas_constant * np.log(rho_ref / self.density))
    
    def compute_pressure_forces(self, grid) -> np.ndarray:
        """
        Compute pressure forces at cell faces.
        
        The pressure force on a face is F = p * A, where A is the
        cross-sectional area. For momentum conservation, this becomes
        a source term in the momentum equation.
        
        Args:
            grid: LagrangianGrid1d object
            
        Returns:
            Pressure forces at faces [n_faces]
        """
        # Interpolate pressure to faces
        p_face = np.zeros(grid.n_faces)
        p_face[1:-1] = 0.5 * (self.pressure[:-1] + self.pressure[1:])
        
        # Boundary faces - use cell-centered values
        p_face[0] = self.pressure[0]
        p_face[-1] = self.pressure[-1]
        
        # Force = pressure * area
        force = p_face * grid.area
        
        return force
    
    def get_max_wave_speeds(self) -> tuple:
        """
        Get maximum wave speeds for CFL calculation.
        
        Returns:
            Tuple of (max_velocity, max_sound_speed, max_total_speed)
        """
        max_u = np.max(np.abs(self.velocity))
        max_c = np.max(self.sound_speed)
        max_total = np.max(np.abs(self.velocity) + self.sound_speed)
        
        return max_u, max_c, max_total
    
    def check_physical_validity(self) -> Dict[str, Any]:
        """
        Check physical validity of current state.
        
        Returns:
            Dictionary with validity metrics and flags
        """
        valid = True
        issues = []
        
        # Check density
        if np.any(self.density <= 0):
            valid = False
            issues.append(f"Non-positive density: min = {np.min(self.density)}")
            
        # Check pressure
        if np.any(self.pressure <= 0):
            valid = False
            issues.append(f"Non-positive pressure: min = {np.min(self.pressure)}")
            
        # Check temperature  
        if np.any(self.temperature <= 0):
            valid = False
            issues.append(f"Non-positive temperature: min = {np.min(self.temperature)}")
            
        # Check for NaN or Inf
        all_arrays = [self.density, self.velocity, self.pressure, self.temperature]
        for i, arr in enumerate(['density', 'velocity', 'pressure', 'temperature']):
            if np.any(~np.isfinite(all_arrays[i])):
                valid = False
                issues.append(f"Non-finite values in {arr}")
        
        return {
            'valid': valid,
            'issues': issues,
            'min_density': np.min(self.density),
            'max_density': np.max(self.density),
            'min_pressure': np.min(self.pressure),
            'max_pressure': np.max(self.pressure),
            'min_temperature': np.min(self.temperature),
            'max_temperature': np.max(self.temperature),
            'velocity_range': (np.min(self.velocity), np.max(self.velocity))
        }
    
    def compute_conserved_totals(self, grid) -> Dict[str, float]:
        """
        Compute total conserved quantities.
        
        Args:
            grid: LagrangianGrid1d object
            
        Returns:
            Dictionary with total mass, momentum, and energy
        """
        total_mass = np.sum(self.density * grid.volume)
        total_momentum = np.sum(self.momentum * grid.volume) 
        total_energy = np.sum(self.energy * grid.volume)
        
        return {
            'mass': total_mass,
            'momentum': total_momentum, 
            'energy': total_energy
        }
    
    def get_state(self) -> Dict[str, Any]:
        """Get complete state for output/restart."""
        return {
            'density': self.density.copy(),
            'momentum': self.momentum.copy(),
            'energy': self.energy.copy(),
            'velocity': self.velocity.copy(),
            'pressure': self.pressure.copy(), 
            'temperature': self.temperature.copy(),
            'sound_speed': self.sound_speed.copy(),
            'internal_energy': self.internal_energy.copy(),
            'enthalpy': self.enthalpy.copy(),
            'entropy': self.entropy.copy(),
            'gamma': self.gamma,
            'gas_constant': self.gas_constant
        }
    
    def set_state(self, state: Dict[str, Any]):
        """Restore state from saved data."""
        self.density = state['density'].copy()
        self.momentum = state['momentum'].copy()
        self.energy = state['energy'].copy()
        self.velocity = state['velocity'].copy()
        self.pressure = state['pressure'].copy()
        self.temperature = state['temperature'].copy()
        self.sound_speed = state['sound_speed'].copy()
        self.internal_energy = state['internal_energy'].copy()
        self.enthalpy = state['enthalpy'].copy()
        self.entropy = state['entropy'].copy()
        self.gamma = state['gamma']
        self.gas_constant = state['gas_constant']


@njit
def compute_eos_properties_numba(density: np.ndarray, internal_energy: np.ndarray,
                               gamma: float, gas_constant: float,
                               pressure: np.ndarray, temperature: np.ndarray,
                               sound_speed: np.ndarray):
    """
    Numba-accelerated equation of state calculations.
    
    Args:
        density: Density array [kg/m³]
        internal_energy: Specific internal energy [J/kg]
        gamma: Ratio of specific heats
        gas_constant: Specific gas constant [J/(kg·K)]
        pressure: Output pressure array [Pa]
        temperature: Output temperature array [K]
        sound_speed: Output sound speed array [m/s]
    """
    n = len(density)
    
    for i in range(n):
        # Ideal gas pressure
        pressure[i] = density[i] * internal_energy[i] * (gamma - 1.0)
        
        # Temperature from ideal gas law
        temperature[i] = pressure[i] / (density[i] * gas_constant)
        
        # Sound speed
        sound_speed[i] = np.sqrt(gamma * pressure[i] / density[i])


@njit
def prim_to_cons_numba(density: np.ndarray, velocity: np.ndarray, 
                      pressure: np.ndarray, gamma: float,
                      momentum: np.ndarray, energy: np.ndarray):
    """
    Numba-accelerated primitive to conservative conversion.
    
    Args:
        density: Density [kg/m³]
        velocity: Velocity [m/s]
        pressure: Pressure [Pa]
        gamma: Ratio of specific heats
        momentum: Output momentum density [kg/(m²·s)]
        energy: Output total energy density [J/m³]
    """
    n = len(density)
    
    for i in range(n):
        momentum[i] = density[i] * velocity[i]
        
        # Internal energy from ideal gas
        internal_e = pressure[i] / (density[i] * (gamma - 1.0))
        
        # Total energy = internal + kinetic
        energy[i] = density[i] * (internal_e + 0.5 * velocity[i]**2)


@njit  
def cons_to_prim_numba(density: np.ndarray, momentum: np.ndarray,
                      energy: np.ndarray, gamma: float,
                      velocity: np.ndarray, pressure: np.ndarray,
                      internal_energy: np.ndarray):
    """
    Numba-accelerated conservative to primitive conversion.
    
    Args:
        density: Density [kg/m³]
        momentum: Momentum density [kg/(m²·s)]
        energy: Total energy density [J/m³]
        gamma: Ratio of specific heats
        velocity: Output velocity [m/s]
        pressure: Output pressure [Pa]
        internal_energy: Output specific internal energy [J/kg]
    """
    n = len(density)
    
    for i in range(n):
        if density[i] <= 0:
            continue
            
        velocity[i] = momentum[i] / density[i]
        
        # Specific total energy
        total_e = energy[i] / density[i]
        
        # Specific internal energy
        internal_energy[i] = total_e - 0.5 * velocity[i]**2
        
        if internal_energy[i] <= 0:
            internal_energy[i] = 1e-10
            
        # Pressure from ideal gas
        pressure[i] = density[i] * internal_energy[i] * (gamma - 1.0)