# Runs the simulation, integrates through time
import torques as tq
import constants as cst
import utils as ut
def physics(torques,cst):
    dt = 0.001 #1ms
    alpha = cst.alpha_0
    omega = cst.omega_0
    angles = cst.angles_0
    torques_sum = sum(torques)
    alpha += np.linalg.inv(cst.I)@(torques_sum-np.cross(cst.I@omega))
    omega += alpha*dt
    angles += omega*dt
    return omega, angles

import numpy as np

class Spacecraft:
    def __init__(self, mass, inertia, initialEulerAngles, initialAngularVelocity):
        self.mass = mass
        self.inertia = inertia
        self.eulerAngles = initialEulerAngles  # [roll, pitch, yaw]
        self.angularVelocity = initialAngularVelocity  # [wx, wy, wz]


    def update(self, torques, dt):
        # Update angular velocity
        angularAcceleration = np.linalg.inv(self.inertia) @ (torques - np.cross(self.angularVelocity, self.inertia @ self.angularVelocity))
        self.angularVelocity += angularAcceleration * dt
        
        # Update Euler angles
        phi, theta, psi = self.eulerAngles
        wx, wy, wz = self.angularVelocity
        
        dphi = wx + (wy * np.sin(phi) + wz * np.cos(phi)) * np.tan(theta)
        dtheta = wy * np.cos(phi) - wz * np.sin(phi)
        dpsi = (wy * np.sin(phi) + wz * np.cos(phi)) / np.cos(theta)

        self.eulerAngles += np.array([dphi, dtheta, dpsi]) * dt


    
        
        

object=Spacecraft(mass=1.0, inertia=cst.I, initialEulerAngles=np.array([0.0,0.0,0.0]), initialAngularVelocity=np.array([0.0,0.0,0.0]))

for t in range(0,100):
    object.update()
    

print("Final Euler Angles:", object.eulerAngles)