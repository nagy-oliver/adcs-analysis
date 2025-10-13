# Functions to calculate the torques based on time
def torque_gg(vec_nadir,cst):
    #function takes the inertia tensor, nadir vector and constants
    #Returns the torque vector numpy array
    t = 3*(cst.mu/(cst.h + cst.R)**3)*np.cross(vec_nadir,cst.I@vec_nadir)
    return t

