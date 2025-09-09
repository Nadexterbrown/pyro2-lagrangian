"""
Sod shock tube problem for Lagrangian compressible solver.

This implements the classic Sod shock tube problem in Lagrangian coordinates.
The problem consists of high pressure gas on the left separated from low
pressure gas on the right by a diaphragm that breaks at t=0.

Initial conditions:
Left state (x < 0.5): ρ=1.0, u=0.0, p=1.0
Right state (x > 0.5): ρ=0.125, u=0.0, p=0.1

The analytical solution consists of a leftward rarefaction, contact
discontinuity, and rightward shock.
"""

import numpy as np
from pyro.util import msg


def init_data(my_data, rp):
    """
    Initialize the Sod shock tube problem.
    
    Args:
        my_data: The CellCenterData2d object (not used for Lagrangian)
        rp: The RuntimeParameters object
    """
    msg.bold("initializing the Sod problem...")
    
    # This function is called by pyro2 framework but actual initialization
    # is handled by the Lagrangian simulation class
    pass


def finalize():
    """Finalize the Sod problem (clean up any resources)."""
    pass