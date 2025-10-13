# Runs the simulation, integrates through time
import torques as tq
import constants as cst
import utils as ut
import numpy as np
import matplotlib.pyplot as plt

import scipy.spatial.transform as ro
        

class Spacecraft:
    def __init__(self, mass, inertia, initialEulerAnglesDeg, initialAngularVelocity):
        self.mass = mass
        self.inertia = inertia
        self.quaternion = ro.Rotation.from_euler('xyz', initialEulerAnglesDeg, degrees=True).as_quat()
        self.angularVelocity = initialAngularVelocity

    def update(self, torques, dt):
        angularAcceleration = np.linalg.inv(self.inertia) @ (torques - np.cross(self.angularVelocity, self.inertia @ self.angularVelocity))
        self.angularVelocity += angularAcceleration * dt
        
        deltaQuat = ro.Rotation.from_rotvec(self.angularVelocity * dt).as_quat()
        self.quaternion = ro.Rotation.from_quat(self.quaternion) * ro.Rotation.from_quat(deltaQuat)
        self.quaternion = self.quaternion.as_quat()

    def getEulerAnglesDeg(self):
        return ro.Rotation.from_quat(self.quaternion).as_euler('xyz', degrees=True)




object=Spacecraft(mass=1.0, inertia=cst.I, initialEulerAnglesDeg=np.array([0.0,0.0,0.0]), initialAngularVelocity=np.array([0.0,0.0,0.0]))


eulerAngleX = []

for i in range(1000):
    torque = np.array([10, 100, 100])
    object.update(torque, dt=0.001)
    eulerAngleX.append(object.getEulerAnglesDeg()[0])

plt.plot(eulerAngleX)
plt.xlabel('Time step')
plt.ylabel('Euler Angle X (deg)')
plt.title('Evolution of Euler Angle X over Time')
plt.grid()
plt.show()
