from utils import torqueGlobalToLocal
from utils import torqueLocalToGlobal
from utils import normalize
import numpy as np

# Functions to calculate the torques based on time
def torque_gg(vec_nadir,cst):
    #function takes the inertia tensor, nadir vector and constants
    #Returns the torque vector numpy array
    t = 3*(cst.mu/(cst.h + cst.R)**3)*np.cross(vec_nadir,cst.I@vec_nadir)
    return t


def solar_torque(sun_vector_global, quaternion):
    """
    Calculates the solar radiation torque in the spacecraft's local frame.

    Parameters
    ----------
    sun_vector_global : np.array(3)
        Vector pointing from the spacecraft to the Sun (in the global frame).
    quaternion : np.array(4)
        Attitude quaternion (w, x, y, z) mapping body→global.
    Returns
    -------
    torque_local : np.array(3)
        Torque vector in the local (body) frame [N·m].
    """
    # Vector from CoG to center of pressure in local frame
    vec_cp_local = np.array([0.5638, -1.0327, -0.1815])   # [m]

    # Constants
    rho = 0.6          # reflectivity (from SMAD)
    p_s = 4.8e-5       # solar radiation pressure [N/m^2]
    s = 1.8**2         # exposed area [m^2]

    # Normalize sun vector
    sun_unit_global = normalize(sun_vector_global)

    # Transform Sun vector to local frame (using quaternion)
    sun_unit_local = torqueGlobalToLocal(sun_unit_global, quaternion)

    # Compute solar force (acts opposite Sun direction)
    solar_force_local = -(1 + rho) * p_s * s * sun_unit_local

    # Torque = r × F (in local frame)
    torque_local = np.cross(vec_cp_local, solar_force_local)
    return torque_local

