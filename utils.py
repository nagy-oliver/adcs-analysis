# Any util functions that might be needed
import numpy as np

def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm

def torqueGlobalToLocal(torque_global, quaternion):
    import scipy.spatial.transform as ro
    rotation = ro.Rotation.from_quat(quaternion)
    return rotation.inv().apply(torque_global)

def torqueLocalToGlobal(torque_local, quaternion):
    import scipy.spatial.transform as ro
    rotation = ro.Rotation.from_quat(quaternion)
    return rotation.apply(torque_local)