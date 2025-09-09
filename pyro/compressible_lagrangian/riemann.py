"""
Riemann solvers for Lagrangian compressible flow interfaces.

This module implements exact and approximate Riemann solvers suitable for
Lagrangian formulations where interface velocities are determined by the 
solution rather than prescribed. Includes HLLC and exact solvers with 
proper handling of contact discontinuities.

Based on the L1d4 solver from GDTk repository.
"""

import numpy as np
from numba import njit
from typing import Tuple, Optional
import warnings


class LagrangianRiemannSolver:
    """
    Riemann solver for Lagrangian interfaces.
    
    In Lagrangian formulation, the interface moves at the contact velocity
    determined by the Riemann solution. This class provides both exact and
    approximate solvers for the 1D Euler equations.
    
    Theory:
    The Riemann problem consists of finding the solution to:
    ∂U/∂t + ∂F(U)/∂x = 0
    with initial conditions U_L for x < 0 and U_R for x > 0.
    
    The solution consists of waves (shock, rarefaction, contact) and 
    constant states between them. For Lagrangian interfaces, we need
    the contact velocity and interface fluxes.
    
    Attributes:
        gamma (float): Ratio of specific heats
        tolerance (float): Convergence tolerance for iterative solvers
        max_iterations (int): Maximum iterations for iterative methods
    """
    
    def __init__(self, gamma: float = 1.4, tolerance: float = 1e-10,
                 max_iterations: int = 100):
        """
        Initialize Riemann solver.
        
        Args:
            gamma: Ratio of specific heats
            tolerance: Convergence tolerance
            max_iterations: Maximum iterations for pressure iteration
        """
        self.gamma = gamma
        self.tolerance = tolerance
        self.max_iterations = max_iterations
    
    def solve_interface(self, rho_L: float, u_L: float, p_L: float,
                       rho_R: float, u_R: float, p_R: float) -> Tuple[float, float, float]:
        """
        Solve Riemann problem at interface.
        
        Args:
            rho_L, u_L, p_L: Left state (density, velocity, pressure)
            rho_R, u_R, p_R: Right state (density, velocity, pressure)
            
        Returns:
            Tuple of (interface_velocity, interface_pressure, interface_density)
        """
        # Check for vacuum formation
        if self._check_vacuum(rho_L, u_L, p_L, rho_R, u_R, p_R):
            raise RuntimeError("Vacuum formation detected in Riemann problem")
        
        # Solve for star state (interface conditions)
        p_star, u_star = self._solve_star_state(rho_L, u_L, p_L, rho_R, u_R, p_R)
        
        # Compute interface density (average of star densities)
        rho_star_L = self._compute_star_density(rho_L, p_L, p_star, 'L')
        rho_star_R = self._compute_star_density(rho_R, p_R, p_star, 'R')
        rho_star = 0.5 * (rho_star_L + rho_star_R)
        
        return u_star, p_star, rho_star
    
    def _solve_star_state(self, rho_L: float, u_L: float, p_L: float,
                         rho_R: float, u_R: float, p_R: float) -> Tuple[float, float]:
        """
        Solve for pressure and velocity in star region.
        
        Uses Newton-Raphson iteration to solve:
        f_L(p) + f_R(p) + (u_R - u_L) = 0
        
        where f_L and f_R are the Riemann invariants.
        """
        # Initial guess for pressure
        c_L = np.sqrt(self.gamma * p_L / rho_L)
        c_R = np.sqrt(self.gamma * p_R / rho_R)
        
        p_guess = max(0.5 * (p_L + p_R), 0.1 * min(p_L, p_R))
        
        # Newton-Raphson iteration
        p_star = p_guess
        for iteration in range(self.max_iterations):
            # Compute f and df/dp for left and right states
            f_L, df_L = self._riemann_invariant(rho_L, u_L, p_L, c_L, p_star)
            f_R, df_R = self._riemann_invariant(rho_R, u_R, p_R, c_R, p_star)
            
            # Newton-Raphson update
            f_total = f_L + f_R + (u_R - u_L)
            df_total = df_L + df_R
            
            if abs(df_total) < 1e-15:
                break
                
            p_new = p_star - f_total / df_total
            
            # Ensure positive pressure
            p_new = max(p_new, 0.01 * min(p_L, p_R))
            
            # Check convergence
            if abs(p_new - p_star) / p_star < self.tolerance:
                p_star = p_new
                break
                
            p_star = p_new
        else:
            warnings.warn("Riemann solver failed to converge")
        
        # Compute star velocity
        f_L, _ = self._riemann_invariant(rho_L, u_L, p_L, c_L, p_star)
        u_star = u_L + f_L
        
        return p_star, u_star
    
    def _riemann_invariant(self, rho: float, u: float, p: float, c: float,
                          p_star: float) -> Tuple[float, float]:
        """
        Compute Riemann invariant and its derivative.
        
        Args:
            rho, u, p, c: State variables and sound speed
            p_star: Star region pressure
            
        Returns:
            Tuple of (f, df/dp) where f is the Riemann invariant
        """
        if p_star > p:
            # Shock wave
            A = 2.0 / ((self.gamma + 1) * rho)
            B = (self.gamma - 1) * p / (self.gamma + 1)
            
            term = A / (p_star + B)
            f = (p_star - p) * np.sqrt(term)
            df = np.sqrt(term) * (1 - 0.5 * (p_star - p) / (p_star + B))
        else:
            # Rarefaction wave
            gamma_ratio = (self.gamma - 1) / (2 * self.gamma)
            pressure_ratio = p_star / p
            
            f = (2 * c / (self.gamma - 1)) * (pressure_ratio**gamma_ratio - 1)
            df = (c / (self.gamma * p)) * pressure_ratio**gamma_ratio
        
        return f, df
    
    def _compute_star_density(self, rho: float, p: float, p_star: float,
                            side: str) -> float:
        """
        Compute density in star region.
        
        Args:
            rho, p: Initial state
            p_star: Star region pressure  
            side: 'L' or 'R' for left or right side
            
        Returns:
            Star region density
        """
        if p_star > p:
            # Shock wave - Rankine-Hugoniot relations
            pressure_ratio = p_star / p
            numerator = (self.gamma + 1) * pressure_ratio + (self.gamma - 1)
            denominator = (self.gamma - 1) * pressure_ratio + (self.gamma + 1)
            rho_star = rho * numerator / denominator
        else:
            # Rarefaction wave - isentropic relations
            pressure_ratio = p_star / p
            rho_star = rho * pressure_ratio**(1.0 / self.gamma)
        
        return rho_star
    
    def _check_vacuum(self, rho_L: float, u_L: float, p_L: float,
                     rho_R: float, u_R: float, p_R: float) -> bool:
        """
        Check if vacuum formation occurs.
        
        Vacuum forms when the rarefaction waves from left and right
        separate, creating a region with zero density.
        """
        c_L = np.sqrt(self.gamma * p_L / rho_L)
        c_R = np.sqrt(self.gamma * p_R / rho_R)
        
        # Critical velocity difference for vacuum formation
        critical_du = 2 * (c_L + c_R) / (self.gamma - 1)
        actual_du = u_R - u_L
        
        return actual_du >= critical_du
    
    def solve_array(self, rho_L: np.ndarray, u_L: np.ndarray, p_L: np.ndarray,
                   rho_R: np.ndarray, u_R: np.ndarray, p_R: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Solve Riemann problems for arrays of interface states.
        
        Args:
            rho_L, u_L, p_L: Left state arrays
            rho_R, u_R, p_R: Right state arrays
            
        Returns:
            Tuple of (interface_velocities, interface_pressures, interface_densities)
        """
        n_interfaces = len(rho_L)
        u_star = np.zeros(n_interfaces)
        p_star = np.zeros(n_interfaces)
        rho_star = np.zeros(n_interfaces)
        
        for i in range(n_interfaces):
            try:
                u_star[i], p_star[i], rho_star[i] = self.solve_interface(
                    rho_L[i], u_L[i], p_L[i], rho_R[i], u_R[i], p_R[i])
            except RuntimeError as e:
                # Handle vacuum or other failures
                warnings.warn(f"Riemann solver failed at interface {i}: {str(e)}")
                # Use simple average as fallback
                u_star[i] = 0.5 * (u_L[i] + u_R[i])
                p_star[i] = 0.5 * (p_L[i] + p_R[i])
                rho_star[i] = 0.5 * (rho_L[i] + rho_R[i])
        
        return u_star, p_star, rho_star


class HLLCRiemannSolver:
    """
    HLLC (Harten-Lax-van Leer-Contact) approximate Riemann solver.
    
    Faster approximate solver that captures contact discontinuities.
    Suitable for cases where computational efficiency is important.
    
    Theory:
    HLLC divides the solution into regions separated by three waves:
    - Left wave (speed S_L)
    - Contact wave (speed S_M)  
    - Right wave (speed S_R)
    
    The contact speed S_M gives the interface velocity in Lagrangian formulation.
    """
    
    def __init__(self, gamma: float = 1.4):
        """
        Initialize HLLC solver.
        
        Args:
            gamma: Ratio of specific heats
        """
        self.gamma = gamma
    
    def solve_interface(self, rho_L: float, u_L: float, p_L: float,
                       rho_R: float, u_R: float, p_R: float) -> Tuple[float, float, float]:
        """
        Solve interface using HLLC approximation.
        
        Returns:
            Tuple of (contact_velocity, interface_pressure, interface_density)
        """
        # Compute sound speeds
        c_L = np.sqrt(self.gamma * p_L / rho_L)
        c_R = np.sqrt(self.gamma * p_R / rho_R)
        
        # Estimate wave speeds (Davis, 1988)
        S_L = min(u_L - c_L, u_R - c_R)
        S_R = max(u_L + c_L, u_R + c_R)
        
        # Contact wave speed (HLLC)
        numerator = (p_R - p_L + rho_L * u_L * (S_L - u_L) - 
                    rho_R * u_R * (S_R - u_R))
        denominator = rho_L * (S_L - u_L) - rho_R * (S_R - u_R)
        
        if abs(denominator) < 1e-15:
            S_M = 0.5 * (u_L + u_R)
        else:
            S_M = numerator / denominator
        
        # Star region pressure
        p_star = (rho_R * u_R * (S_R - u_R) - rho_L * u_L * (S_L - u_L) + 
                 p_L - p_R) / (rho_R * (S_R - u_R) - rho_L * (S_L - u_L))
        
        # Star region densities
        rho_star_L = rho_L * (S_L - u_L) / (S_L - S_M)
        rho_star_R = rho_R * (S_R - u_R) / (S_R - S_M)
        rho_star = 0.5 * (rho_star_L + rho_star_R)
        
        return S_M, p_star, rho_star
    
    def solve_array(self, rho_L: np.ndarray, u_L: np.ndarray, p_L: np.ndarray,
                   rho_R: np.ndarray, u_R: np.ndarray, p_R: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Solve arrays using HLLC method."""
        return hllc_solve_numba(rho_L, u_L, p_L, rho_R, u_R, p_R, self.gamma)


@njit
def hllc_solve_numba(rho_L: np.ndarray, u_L: np.ndarray, p_L: np.ndarray,
                    rho_R: np.ndarray, u_R: np.ndarray, p_R: np.ndarray,
                    gamma: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Numba-accelerated HLLC solver for arrays.
    
    Args:
        rho_L, u_L, p_L: Left state arrays
        rho_R, u_R, p_R: Right state arrays  
        gamma: Ratio of specific heats
        
    Returns:
        Tuple of (contact_velocities, interface_pressures, interface_densities)
    """
    n = len(rho_L)
    S_M = np.zeros(n)
    p_star = np.zeros(n)
    rho_star = np.zeros(n)
    
    for i in range(n):
        # Sound speeds
        c_L = np.sqrt(gamma * p_L[i] / rho_L[i])
        c_R = np.sqrt(gamma * p_R[i] / rho_R[i])
        
        # Wave speeds
        S_L = min(u_L[i] - c_L, u_R[i] - c_R)
        S_R = max(u_L[i] + c_L, u_R[i] + c_R)
        
        # Contact speed
        numerator = (p_R[i] - p_L[i] + rho_L[i] * u_L[i] * (S_L - u_L[i]) - 
                    rho_R[i] * u_R[i] * (S_R - u_R[i]))
        denominator = rho_L[i] * (S_L - u_L[i]) - rho_R[i] * (S_R - u_R[i])
        
        if abs(denominator) < 1e-15:
            S_M[i] = 0.5 * (u_L[i] + u_R[i])
        else:
            S_M[i] = numerator / denominator
        
        # Star pressure
        p_star[i] = (rho_R[i] * u_R[i] * (S_R - u_R[i]) - 
                    rho_L[i] * u_L[i] * (S_L - u_L[i]) + 
                    p_L[i] - p_R[i]) / (rho_R[i] * (S_R - u_R[i]) - 
                                       rho_L[i] * (S_L - u_L[i]))
        
        # Star densities
        rho_star_L = rho_L[i] * (S_L - u_L[i]) / (S_L - S_M[i])
        rho_star_R = rho_R[i] * (S_R - u_R[i]) / (S_R - S_M[i])
        rho_star[i] = 0.5 * (rho_star_L + rho_star_R)
    
    return S_M, p_star, rho_star


@njit
def exact_riemann_numba(rho_L: float, u_L: float, p_L: float,
                       rho_R: float, u_R: float, p_R: float,
                       gamma: float, tolerance: float = 1e-10,
                       max_iter: int = 100) -> Tuple[float, float, float]:
    """
    Numba-accelerated exact Riemann solver.
    
    Solves the exact Riemann problem using Newton-Raphson iteration.
    
    Returns:
        Tuple of (star_velocity, star_pressure, star_density)
    """
    # Sound speeds
    c_L = np.sqrt(gamma * p_L / rho_L)
    c_R = np.sqrt(gamma * p_R / rho_R)
    
    # Initial pressure guess
    p_guess = max(0.5 * (p_L + p_R), 0.1 * min(p_L, p_R))
    
    # Newton-Raphson iteration
    p_star = p_guess
    for iteration in range(max_iter):
        # Compute Riemann invariants
        if p_star > p_L:
            # Shock
            A_L = 2.0 / ((gamma + 1) * rho_L)
            B_L = (gamma - 1) * p_L / (gamma + 1)
            term_L = A_L / (p_star + B_L)
            f_L = (p_star - p_L) * np.sqrt(term_L)
            df_L = np.sqrt(term_L) * (1 - 0.5 * (p_star - p_L) / (p_star + B_L))
        else:
            # Rarefaction  
            gamma_ratio = (gamma - 1) / (2 * gamma)
            pressure_ratio = p_star / p_L
            f_L = (2 * c_L / (gamma - 1)) * (pressure_ratio**gamma_ratio - 1)
            df_L = (c_L / (gamma * p_L)) * pressure_ratio**gamma_ratio
        
        if p_star > p_R:
            # Shock
            A_R = 2.0 / ((gamma + 1) * rho_R)
            B_R = (gamma - 1) * p_R / (gamma + 1)
            term_R = A_R / (p_star + B_R)
            f_R = (p_star - p_R) * np.sqrt(term_R)
            df_R = np.sqrt(term_R) * (1 - 0.5 * (p_star - p_R) / (p_star + B_R))
        else:
            # Rarefaction
            gamma_ratio = (gamma - 1) / (2 * gamma)
            pressure_ratio = p_star / p_R
            f_R = (2 * c_R / (gamma - 1)) * (pressure_ratio**gamma_ratio - 1)
            df_R = (c_R / (gamma * p_R)) * pressure_ratio**gamma_ratio
        
        # Newton update
        f_total = f_L + f_R + (u_R - u_L)
        df_total = df_L + df_R
        
        if abs(df_total) < 1e-15:
            break
            
        p_new = p_star - f_total / df_total
        p_new = max(p_new, 0.01 * min(p_L, p_R))
        
        if abs(p_new - p_star) / p_star < tolerance:
            p_star = p_new
            break
            
        p_star = p_new
    
    # Star velocity
    u_star = u_L + f_L
    
    # Star density (average)
    if p_star > p_L:
        pressure_ratio = p_star / p_L
        numerator = (gamma + 1) * pressure_ratio + (gamma - 1)
        denominator = (gamma - 1) * pressure_ratio + (gamma + 1)
        rho_star_L = rho_L * numerator / denominator
    else:
        pressure_ratio = p_star / p_L
        rho_star_L = rho_L * pressure_ratio**(1.0 / gamma)
    
    if p_star > p_R:
        pressure_ratio = p_star / p_R
        numerator = (gamma + 1) * pressure_ratio + (gamma - 1)
        denominator = (gamma - 1) * pressure_ratio + (gamma + 1)
        rho_star_R = rho_R * numerator / denominator
    else:
        pressure_ratio = p_star / p_R
        rho_star_R = rho_R * pressure_ratio**(1.0 / gamma)
    
    rho_star = 0.5 * (rho_star_L + rho_star_R)
    
    return u_star, p_star, rho_star