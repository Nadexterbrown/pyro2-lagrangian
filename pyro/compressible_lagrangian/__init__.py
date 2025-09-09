"""
The pyro compressible Lagrangian hydrodynamics solver. This implements a 
1D Lagrangian solver for the Euler equations with moving boundaries and
piston dynamics, based on the L1d4 solver from GDTk.

Key features:
- Lagrangian grid with cell boundaries moving at fluid velocity
- Gas slug thermodynamic state management  
- Piston dynamics with two-way coupling
- Conservative Riemann solvers at interfaces
- Adaptive time stepping
- Support for moving boundaries and pistons

"""

__all__ = ["simulation"]

from .simulation import Simulation, Variables