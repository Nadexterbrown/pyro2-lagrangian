"""
Piston dynamics and moving boundary implementation for Lagrangian solver.

This module implements piston dynamics including Newton's laws integration,
pressure force calculation, friction models, and various mechanical systems
like springs, dampers, brakes, and buffers. Provides two-way coupling between
piston motion and gas dynamics.

Based on the L1d4 solver from GDTk repository.
"""

import numpy as np
from numba import njit
from typing import Dict, Any, Optional, Callable
import warnings


class Piston:
    """
    Piston dynamics with mechanical systems and gas coupling.
    
    Implements a complete piston model including:
    - Newton's laws of motion (F = ma)
    - Pressure forces from gas
    - Friction models (viscous, Coulomb)
    - Spring-damper systems
    - Brake mechanisms
    - Buffer/stopper systems
    - External forcing functions
    
    Theory:
    The piston motion is governed by Newton's second law:
    m * dv/dt = F_pressure + F_friction + F_spring + F_damper + F_external
    
    where each force component can be modeled with various levels of complexity.
    Position integration uses: dx/dt = v
    
    Attributes:
        mass (float): Piston mass [kg]
        area (float): Piston face area [m²]
        position (float): Current position [m]
        velocity (float): Current velocity [m/s]
        acceleration (float): Current acceleration [m/s²]
        
        # Mechanical systems
        friction_model (str): Type of friction model
        friction_coeff (float): Friction coefficient
        spring_constant (float): Spring constant [N/m]
        spring_preload (float): Spring preload force [N]
        spring_ref_position (float): Spring reference position [m]
        damper_constant (float): Damper constant [N·s/m]
        
        # Limits and constraints
        position_limits (tuple): (min_pos, max_pos)
        velocity_limits (tuple): (min_vel, max_vel)
        brake_engaged (bool): Brake system state
        buffer_positions (list): Buffer/stopper positions
        buffer_stiffness (float): Buffer spring constant
        
        # History tracking
        time_history (list): Time points
        position_history (list): Position history
        velocity_history (list): Velocity history
        force_history (list): Total force history
    """
    
    def __init__(self, mass: float, area: float, x0: float = 0.0, v0: float = 0.0):
        """
        Initialize piston with basic properties.
        
        Args:
            mass: Piston mass [kg]
            area: Piston face area [m²]
            x0: Initial position [m]
            v0: Initial velocity [m/s]
        """
        self.mass = mass
        self.area = area
        self.position = x0
        self.velocity = v0
        self.acceleration = 0.0
        
        # Mechanical systems (all optional)
        self.friction_model = 'none'  # 'none', 'viscous', 'coulomb', 'combined'
        self.friction_coeff = 0.0
        self.static_friction_coeff = 0.0
        
        self.spring_constant = 0.0
        self.spring_preload = 0.0
        self.spring_ref_position = 0.0
        
        self.damper_constant = 0.0
        
        # Limits and constraints
        self.position_limits = (-np.inf, np.inf)
        self.velocity_limits = (-np.inf, np.inf)
        self.brake_engaged = False
        self.brake_force = 0.0
        
        # Buffer system
        self.buffer_positions = []
        self.buffer_stiffness = 1e6  # Stiff buffer springs
        self.buffer_damping = 1e3
        
        # External forces
        self.external_force_function = None
        
        # History tracking
        self.time_history = []
        self.position_history = []
        self.velocity_history = []  
        self.force_history = []
        
        # Gas pressures (set by solver)
        self.pressure_front = 0.0  # Pressure on front face
        self.pressure_back = 0.0   # Pressure on back face
    
    def set_friction_model(self, model: str, viscous_coeff: float = 0.0,
                          coulomb_coeff: float = 0.0, static_coeff: float = None):
        """
        Set friction model parameters.
        
        Args:
            model: 'none', 'viscous', 'coulomb', 'combined'
            viscous_coeff: Viscous friction coefficient [N·s/m]
            coulomb_coeff: Coulomb friction coefficient (dimensionless)
            static_coeff: Static friction coefficient (defaults to coulomb_coeff)
        """
        self.friction_model = model
        self.friction_coeff = viscous_coeff if model == 'viscous' else coulomb_coeff
        self.static_friction_coeff = static_coeff or coulomb_coeff
    
    def set_spring_system(self, spring_constant: float, preload: float = 0.0,
                         ref_position: float = 0.0):
        """
        Set spring system parameters.
        
        Args:
            spring_constant: Spring constant [N/m]
            preload: Preload force [N] (positive = compression)
            ref_position: Reference position for zero spring force [m]
        """
        self.spring_constant = spring_constant
        self.spring_preload = preload
        self.spring_ref_position = ref_position
    
    def set_damper_system(self, damper_constant: float):
        """
        Set damper system parameters.
        
        Args:
            damper_constant: Damper constant [N·s/m]
        """
        self.damper_constant = damper_constant
    
    def set_limits(self, position_limits: tuple = None, velocity_limits: tuple = None):
        """
        Set position and velocity limits.
        
        Args:
            position_limits: (min_pos, max_pos) tuple
            velocity_limits: (min_vel, max_vel) tuple
        """
        if position_limits:
            self.position_limits = position_limits
        if velocity_limits:
            self.velocity_limits = velocity_limits
    
    def set_buffers(self, positions: list, stiffness: float = 1e6,
                   damping: float = 1e3):
        """
        Set buffer/stopper positions and properties.
        
        Args:
            positions: List of buffer positions [m]
            stiffness: Buffer spring stiffness [N/m]
            damping: Buffer damping [N·s/m]
        """
        self.buffer_positions = sorted(positions)
        self.buffer_stiffness = stiffness
        self.buffer_damping = damping
    
    def set_external_force(self, force_function: Callable[[float, float, float], float]):
        """
        Set external force function.
        
        Args:
            force_function: Function F(t, x, v) returning force [N]
        """
        self.external_force_function = force_function
    
    def compute_pressure_force(self) -> float:
        """
        Compute net pressure force on piston.
        
        Returns:
            Net pressure force [N] (positive = rightward)
        """
        # Net pressure force = (P_front - P_back) * Area
        # Front pressure pushes rightward (+), back pressure pushes leftward (-)
        return (self.pressure_front - self.pressure_back) * self.area
    
    def compute_friction_force(self) -> float:
        """
        Compute friction force based on selected model.
        
        Returns:
            Friction force [N] (always opposes motion)
        """
        if self.friction_model == 'none':
            return 0.0
        
        elif self.friction_model == 'viscous':
            # F_friction = -b * v
            return -self.friction_coeff * self.velocity
        
        elif self.friction_model == 'coulomb':
            # F_friction = -mu * N * sign(v)
            # For simplicity, assume normal force = weight or constant
            if abs(self.velocity) < 1e-10:
                # Static friction - resists other forces up to limit
                return 0.0  # Will be handled by static friction check
            else:
                return -self.friction_coeff * np.sign(self.velocity)
        
        elif self.friction_model == 'combined':
            # Combined viscous + Coulomb friction
            viscous = -self.friction_coeff * self.velocity
            if abs(self.velocity) < 1e-10:
                coulomb = 0.0
            else:
                coulomb = -self.static_friction_coeff * np.sign(self.velocity)
            return viscous + coulomb
        
        return 0.0
    
    def compute_spring_force(self) -> float:
        """
        Compute spring force.
        
        Returns:
            Spring force [N]
        """
        if self.spring_constant == 0.0:
            return 0.0
        
        # F_spring = -k * (x - x_ref) + F_preload
        displacement = self.position - self.spring_ref_position
        return -self.spring_constant * displacement + self.spring_preload
    
    def compute_damper_force(self) -> float:
        """
        Compute damper force.
        
        Returns:
            Damper force [N]
        """
        return -self.damper_constant * self.velocity
    
    def compute_buffer_force(self) -> float:
        """
        Compute buffer/stopper forces.
        
        Returns:
            Buffer force [N]
        """
        total_force = 0.0
        
        for buffer_pos in self.buffer_positions:
            penetration = self.position - buffer_pos
            
            if penetration > 0:  # Piston has hit buffer
                # Spring force proportional to penetration
                spring_force = -self.buffer_stiffness * penetration
                
                # Damping force proportional to velocity
                damping_force = -self.buffer_damping * self.velocity
                
                total_force += spring_force + damping_force
        
        return total_force
    
    def compute_external_force(self, time: float) -> float:
        """
        Compute external applied force.
        
        Args:
            time: Current simulation time [s]
            
        Returns:
            External force [N]
        """
        if self.external_force_function is None:
            return 0.0
        
        return self.external_force_function(time, self.position, self.velocity)
    
    def compute_total_force(self, time: float) -> float:
        """
        Compute total force on piston.
        
        Args:
            time: Current simulation time [s]
            
        Returns:
            Total force [N]
        """
        forces = {
            'pressure': self.compute_pressure_force(),
            'friction': self.compute_friction_force(), 
            'spring': self.compute_spring_force(),
            'damper': self.compute_damper_force(),
            'buffer': self.compute_buffer_force(),
            'external': self.compute_external_force(time)
        }
        
        total_force = sum(forces.values())
        
        # Apply brake if engaged
        if self.brake_engaged:
            # Simple brake model - opposes motion
            brake_force = -self.brake_force * np.sign(self.velocity) if self.velocity != 0 else 0
            total_force += brake_force
        
        return total_force
    
    def compute_acceleration(self, time: float) -> float:
        """
        Compute acceleration from Newton's second law.
        
        Args:
            time: Current simulation time [s]
            
        Returns:
            Acceleration [m/s²]
        """
        total_force = self.compute_total_force(time)
        return total_force / self.mass
    
    def update_state(self, dt: float, time: float):
        """
        Update piston position and velocity using integration.
        
        Uses second-order accurate integration (Verlet-like method).
        
        Args:
            dt: Time step [s]
            time: Current simulation time [s]
        """
        # Compute current acceleration
        self.acceleration = self.compute_acceleration(time)
        
        # Update velocity (midpoint method for better accuracy)
        v_old = self.velocity
        self.velocity += self.acceleration * dt
        
        # Apply velocity limits
        self.velocity = np.clip(self.velocity, *self.velocity_limits)
        
        # Update position using average velocity
        v_avg = 0.5 * (v_old + self.velocity)
        self.position += v_avg * dt
        
        # Apply position limits
        if self.position < self.position_limits[0]:
            self.position = self.position_limits[0]
            if self.velocity < 0:
                self.velocity = 0  # Stop at limit
        elif self.position > self.position_limits[1]:
            self.position = self.position_limits[1]
            if self.velocity > 0:
                self.velocity = 0  # Stop at limit
        
        # Store history
        self.time_history.append(time + dt)
        self.position_history.append(self.position)
        self.velocity_history.append(self.velocity)
        self.force_history.append(self.compute_total_force(time + dt))
    
    def get_cfl_timestep(self, dx_min: float, safety_factor: float = 0.1) -> float:
        """
        Compute maximum stable timestep for piston dynamics.
        
        Based on ensuring piston doesn't move too far in one timestep.
        
        Args:
            dx_min: Minimum grid spacing [m]
            safety_factor: Safety factor for stability
            
        Returns:
            Maximum stable timestep [s]
        """
        if abs(self.velocity) < 1e-10:
            return np.inf
        
        # Limit piston motion to fraction of minimum grid spacing
        dt_max = safety_factor * dx_min / abs(self.velocity)
        
        return dt_max
    
    def engage_brake(self, brake_force: float):
        """
        Engage brake system.
        
        Args:
            brake_force: Maximum brake force [N]
        """
        self.brake_engaged = True
        self.brake_force = brake_force
    
    def release_brake(self):
        """Release brake system."""
        self.brake_engaged = False
        self.brake_force = 0.0
    
    def get_state(self) -> Dict[str, Any]:
        """Get complete piston state for output/restart."""
        return {
            'mass': self.mass,
            'area': self.area,
            'position': self.position,
            'velocity': self.velocity,
            'acceleration': self.acceleration,
            'pressure_front': self.pressure_front,
            'pressure_back': self.pressure_back,
            'friction_model': self.friction_model,
            'friction_coeff': self.friction_coeff,
            'spring_constant': self.spring_constant,
            'spring_preload': self.spring_preload,
            'spring_ref_position': self.spring_ref_position,
            'damper_constant': self.damper_constant,
            'position_limits': self.position_limits,
            'velocity_limits': self.velocity_limits,
            'brake_engaged': self.brake_engaged,
            'brake_force': self.brake_force,
            'buffer_positions': self.buffer_positions.copy(),
            'buffer_stiffness': self.buffer_stiffness,
            'buffer_damping': self.buffer_damping,
            'history': {
                'time': self.time_history.copy(),
                'position': self.position_history.copy(),
                'velocity': self.velocity_history.copy(),
                'force': self.force_history.copy()
            }
        }
    
    def set_state(self, state: Dict[str, Any]):
        """Restore piston state from saved data."""
        self.mass = state['mass']
        self.area = state['area']
        self.position = state['position']
        self.velocity = state['velocity']
        self.acceleration = state['acceleration']
        self.pressure_front = state['pressure_front']
        self.pressure_back = state['pressure_back']
        self.friction_model = state['friction_model']
        self.friction_coeff = state['friction_coeff']
        self.spring_constant = state['spring_constant']
        self.spring_preload = state['spring_preload']
        self.spring_ref_position = state['spring_ref_position']
        self.damper_constant = state['damper_constant']
        self.position_limits = state['position_limits']
        self.velocity_limits = state['velocity_limits']
        self.brake_engaged = state['brake_engaged']
        self.brake_force = state['brake_force']
        self.buffer_positions = state['buffer_positions'].copy()
        self.buffer_stiffness = state['buffer_stiffness']
        self.buffer_damping = state['buffer_damping']
        
        history = state.get('history', {})
        self.time_history = history.get('time', [])
        self.position_history = history.get('position', [])
        self.velocity_history = history.get('velocity', [])
        self.force_history = history.get('force', [])


class PistonGroup:
    """
    Group of coupled pistons for multi-piston systems.
    
    Handles interactions between multiple pistons, including:
    - Mechanical coupling (rods, gears)
    - Shared gas chambers  
    - Synchronized motion constraints
    """
    
    def __init__(self, pistons: list):
        """
        Initialize piston group.
        
        Args:
            pistons: List of Piston objects
        """
        self.pistons = pistons
        self.n_pistons = len(pistons)
        self.coupling_matrix = np.eye(self.n_pistons)  # No coupling by default
        self.shared_chambers = []  # List of chamber indices that are shared
    
    def set_mechanical_coupling(self, coupling_matrix: np.ndarray):
        """
        Set mechanical coupling between pistons.
        
        Args:
            coupling_matrix: N×N matrix defining coupling coefficients
        """
        self.coupling_matrix = coupling_matrix
    
    def add_shared_chamber(self, piston_indices: list, chamber_type: str = 'front'):
        """
        Add shared gas chamber between pistons.
        
        Args:
            piston_indices: List of piston indices sharing the chamber
            chamber_type: 'front' or 'back' chamber
        """
        self.shared_chambers.append({
            'pistons': piston_indices,
            'type': chamber_type
        })
    
    def update_coupled_motion(self, dt: float, time: float):
        """
        Update motion of all coupled pistons.
        
        Args:
            dt: Time step [s]
            time: Current simulation time [s]
        """
        # Compute forces for all pistons
        forces = np.array([piston.compute_total_force(time) for piston in self.pistons])
        masses = np.array([piston.mass for piston in self.pistons])
        
        # Apply coupling
        coupled_forces = self.coupling_matrix @ forces
        
        # Update each piston
        for i, piston in enumerate(self.pistons):
            piston.acceleration = coupled_forces[i] / masses[i]
            
            # Update kinematics
            v_old = piston.velocity
            piston.velocity += piston.acceleration * dt
            piston.velocity = np.clip(piston.velocity, *piston.velocity_limits)
            
            v_avg = 0.5 * (v_old + piston.velocity)
            piston.position += v_avg * dt
            
            # Apply position limits
            if piston.position < piston.position_limits[0]:
                piston.position = piston.position_limits[0]
                if piston.velocity < 0:
                    piston.velocity = 0
            elif piston.position > piston.position_limits[1]:
                piston.position = piston.position_limits[1]
                if piston.velocity > 0:
                    piston.velocity = 0


@njit
def update_piston_numba(position: float, velocity: float, mass: float,
                       pressure_front: float, pressure_back: float, area: float,
                       friction_coeff: float, spring_k: float, spring_x0: float,
                       damper_c: float, dt: float, time: float) -> tuple:
    """
    Numba-accelerated piston update for basic systems.
    
    Args:
        position: Current position [m]
        velocity: Current velocity [m/s]
        mass: Piston mass [kg]
        pressure_front: Front pressure [Pa]
        pressure_back: Back pressure [Pa]
        area: Piston area [m²]
        friction_coeff: Friction coefficient
        spring_k: Spring constant [N/m]
        spring_x0: Spring reference position [m]
        damper_c: Damper constant [N·s/m]
        dt: Time step [s]
        time: Current time [s]
        
    Returns:
        Tuple of (new_position, new_velocity, acceleration)
    """
    # Pressure force
    F_pressure = (pressure_front - pressure_back) * area
    
    # Friction force (simple viscous)
    F_friction = -friction_coeff * velocity
    
    # Spring force
    F_spring = -spring_k * (position - spring_x0)
    
    # Damper force
    F_damper = -damper_c * velocity
    
    # Total force
    F_total = F_pressure + F_friction + F_spring + F_damper
    
    # Acceleration
    acceleration = F_total / mass
    
    # Update kinematics
    v_old = velocity
    velocity_new = velocity + acceleration * dt
    position_new = position + 0.5 * (v_old + velocity_new) * dt
    
    return position_new, velocity_new, acceleration