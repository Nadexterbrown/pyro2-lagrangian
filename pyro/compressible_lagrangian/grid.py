"""
Lagrangian grid implementation for 1D compressible flow.

This module implements a 1D Lagrangian grid where cell boundaries move with
the fluid velocity, ensuring no mass transfer between cells. It handles
geometric conservation laws and provides methods for grid motion updates.

Based on the L1d4 solver from GDTk repository.
"""

import numpy as np
from numba import njit
from typing import Optional


class LagrangianGrid1d:
    """
    1D Lagrangian grid with moving cell boundaries.
    
    In a Lagrangian grid, cell boundaries move with the fluid velocity,
    ensuring that no mass crosses cell boundaries. This implementation
    handles the geometric conservation law and provides efficient updates
    for boundary positions and cell volumes.
    
    Theory:
    In Lagrangian coordinates, the grid velocity at cell faces equals the
    fluid velocity. The geometric conservation law ensures that the time
    rate of change of cell volume equals the net flux of volume through
    the cell faces. For 1D: dV/dt = A * (u_right - u_left) where A is 
    the cross-sectional area.
    
    Attributes:
        n_cells (int): Number of computational cells
        n_faces (int): Number of cell faces (n_cells + 1)
        x_face (np.ndarray): Face positions [n_faces]
        x_center (np.ndarray): Cell center positions [n_cells]  
        dx (np.ndarray): Cell widths [n_cells]
        volume (np.ndarray): Cell volumes [n_cells]
        area (float): Cross-sectional area (constant for 1D)
        u_face (np.ndarray): Face velocities [n_faces]
    """
    
    def __init__(self, n_cells: int, x_left: float, x_right: float, 
                 area: float = 1.0):
        """
        Initialize a uniform 1D Lagrangian grid.
        
        Args:
            n_cells: Number of computational cells
            x_left: Left boundary position
            x_right: Right boundary position  
            area: Cross-sectional area (default 1.0 for 1D)
        """
        self.n_cells = n_cells
        self.n_faces = n_cells + 1
        self.area = area
        
        # Initialize uniform grid
        self.x_face = np.linspace(x_left, x_right, self.n_faces)
        self.x_center = np.zeros(n_cells)
        self.dx = np.zeros(n_cells)
        self.volume = np.zeros(n_cells)
        self.u_face = np.zeros(self.n_faces)
        
        self._compute_geometry()
        
    def _compute_geometry(self):
        """Compute cell centers, widths, and volumes from face positions."""
        self.dx[:] = self.x_face[1:] - self.x_face[:-1]
        self.x_center[:] = 0.5 * (self.x_face[1:] + self.x_face[:-1])
        self.volume[:] = self.area * self.dx
        
    def update_positions(self, u_face: np.ndarray, dt: float):
        """
        Update face positions based on face velocities.
        
        Implements the Lagrangian grid motion where face positions are
        updated according to: x_new = x_old + u_face * dt
        
        Args:
            u_face: Face velocities [n_faces]
            dt: Time step
        """
        # Store face velocities
        self.u_face[:] = u_face
        
        # Update face positions - boundaries may be fixed depending on BC
        self.x_face[:] += u_face * dt
        
        # Recompute geometry
        self._compute_geometry()
        
        # Check for grid tangling
        if np.any(self.dx <= 0):
            min_dx = np.min(self.dx)
            raise RuntimeError(f"Grid tangling detected: min(dx) = {min_dx}")
    
    def compute_face_velocities(self, u_cell: np.ndarray, 
                              boundary_conditions: Optional[dict] = None) -> np.ndarray:
        """
        Compute face velocities from cell-centered velocities.
        
        Uses linear interpolation for interior faces. Boundary faces are
        handled according to specified boundary conditions.
        
        Args:
            u_cell: Cell-centered velocities [n_cells]
            boundary_conditions: Dict specifying BC type and values
            
        Returns:
            Face velocities [n_faces]
        """
        u_face = np.zeros(self.n_faces)
        
        # Interior faces: linear interpolation
        u_face[1:-1] = 0.5 * (u_cell[:-1] + u_cell[1:])
        
        # Boundary faces
        if boundary_conditions is None:
            # Default: extrapolate from nearest cell
            u_face[0] = u_cell[0]
            u_face[-1] = u_cell[-1]
        else:
            # Handle specific boundary conditions
            self._apply_face_boundary_conditions(u_face, u_cell, boundary_conditions)
            
        return u_face
    
    def _apply_face_boundary_conditions(self, u_face: np.ndarray, u_cell: np.ndarray,
                                      boundary_conditions: dict):
        """Apply boundary conditions for face velocities."""
        
        # Left boundary
        left_bc = boundary_conditions.get('left', {'type': 'extrapolate'})
        if left_bc['type'] == 'fixed':
            u_face[0] = left_bc['value']
        elif left_bc['type'] == 'reflecting':
            u_face[0] = 0.0
        elif left_bc['type'] == 'extrapolate':
            u_face[0] = u_cell[0]
        elif left_bc['type'] == 'piston':
            # Will be set by piston dynamics
            u_face[0] = left_bc.get('velocity', 0.0)
            
        # Right boundary  
        right_bc = boundary_conditions.get('right', {'type': 'extrapolate'})
        if right_bc['type'] == 'fixed':
            u_face[-1] = right_bc['value']
        elif right_bc['type'] == 'reflecting':
            u_face[-1] = 0.0
        elif right_bc['type'] == 'extrapolate':
            u_face[-1] = u_cell[-1]
        elif right_bc['type'] == 'piston':
            # Will be set by piston dynamics
            u_face[-1] = right_bc.get('velocity', 0.0)
    
    def compute_volume_change_rate(self) -> np.ndarray:
        """
        Compute the rate of volume change for each cell.
        
        Implements the geometric conservation law:
        dV/dt = A * (u_right - u_left)
        
        Returns:
            Volume change rates [n_cells]
        """
        return self.area * (self.u_face[1:] - self.u_face[:-1])
    
    def get_cfl_timestep(self, sound_speed: np.ndarray, cfl_factor: float = 0.5) -> float:
        """
        Compute maximum stable time step based on acoustic CFL condition.
        
        For Lagrangian grids, the CFL condition is:
        dt <= CFL * min(dx / (|u| + c))
        
        Args:
            sound_speed: Sound speed in each cell [n_cells]
            cfl_factor: CFL safety factor (default 0.5)
            
        Returns:
            Maximum stable time step
        """
        # Compute cell-centered velocities from faces
        u_center = 0.5 * (self.u_face[1:] + self.u_face[:-1])
        
        # CFL condition: dt <= dx / (|u| + c)
        wave_speed = np.abs(u_center) + sound_speed
        dt_cfl = cfl_factor * np.min(self.dx / wave_speed)
        
        return dt_cfl
    
    def check_grid_quality(self) -> dict:
        """
        Check grid quality metrics.
        
        Returns:
            Dictionary with grid quality metrics
        """
        min_dx = np.min(self.dx)
        max_dx = np.max(self.dx)
        aspect_ratio = max_dx / min_dx if min_dx > 0 else np.inf
        
        return {
            'min_dx': min_dx,
            'max_dx': max_dx,
            'aspect_ratio': aspect_ratio,
            'total_length': self.x_face[-1] - self.x_face[0],
            'total_volume': np.sum(self.volume),
            'is_tangled': np.any(self.dx <= 0)
        }
    
    def get_state(self) -> dict:
        """Get complete grid state for output/restart."""
        return {
            'x_face': self.x_face.copy(),
            'x_center': self.x_center.copy(),
            'dx': self.dx.copy(),
            'volume': self.volume.copy(),
            'u_face': self.u_face.copy(),
            'area': self.area,
            'n_cells': self.n_cells
        }
    
    def set_state(self, state: dict):
        """Restore grid state from saved data."""
        self.x_face = state['x_face'].copy()
        self.x_center = state['x_center'].copy() 
        self.dx = state['dx'].copy()
        self.volume = state['volume'].copy()
        self.u_face = state['u_face'].copy()
        self.area = state['area']
        self.n_cells = state['n_cells']
        self.n_faces = self.n_cells + 1


@njit
def compute_face_velocities_numba(u_cell: np.ndarray, n_faces: int) -> np.ndarray:
    """
    Numba-accelerated computation of face velocities from cell velocities.
    
    Uses linear interpolation for interior faces and extrapolation for boundaries.
    
    Args:
        u_cell: Cell-centered velocities [n_cells]
        n_faces: Number of faces
        
    Returns:
        Face velocities [n_faces]
    """
    u_face = np.zeros(n_faces)
    
    # Interior faces
    for i in range(1, n_faces - 1):
        u_face[i] = 0.5 * (u_cell[i-1] + u_cell[i])
    
    # Boundary faces
    u_face[0] = u_cell[0]
    u_face[-1] = u_cell[-1]
    
    return u_face


@njit 
def update_geometry_numba(x_face: np.ndarray, dx: np.ndarray, 
                         x_center: np.ndarray, volume: np.ndarray, area: float):
    """
    Numba-accelerated geometry computation.
    
    Args:
        x_face: Face positions [n_faces] (input)
        dx: Cell widths [n_cells] (output)
        x_center: Cell centers [n_cells] (output)  
        volume: Cell volumes [n_cells] (output)
        area: Cross-sectional area
    """
    n_cells = len(dx)
    
    for i in range(n_cells):
        dx[i] = x_face[i+1] - x_face[i]
        x_center[i] = 0.5 * (x_face[i+1] + x_face[i])
        volume[i] = area * dx[i]