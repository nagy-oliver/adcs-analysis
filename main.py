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
eulerAngleY = []
eulerAngleZ = []

angularVelocityX = []
angularVelocityY = []
angularVelocityZ = []

magnetic_torques = np.array([0.00000002,0.00000002,0.00000002])
internal_torques = np.array([0.0,0.0,0.0])

n = 76257

for i in range(n):
    eulerAnglesGlobalWRTSolarDeg = np.array([0.0,-360*i/n,0.0])
    quaternionGlobalWRTSolar = ro.Rotation.from_euler('xyz', eulerAnglesGlobalWRTSolarDeg, degrees=True).as_quat()
    sunVectorSolar = np.array([0.4,0.7,0.5])
    sunVectorGlobal = ut.torqueGlobalToLocal(sunVectorSolar, quaternionGlobalWRTSolar)
    torque = tq.solar_torque(sun_vector_global=sunVectorGlobal, quaternion=object.quaternion)+magnetic_torques+internal_torques+tq.torque_gg(vec_nadir=cst.vec_nadir_0,cst=cst)
    object.update(torque, dt=1)
    eulerAngleX.append(object.getEulerAnglesDeg()[0])
    eulerAngleY.append(object.getEulerAnglesDeg()[1])
    eulerAngleZ.append(object.getEulerAnglesDeg()[2])

    angularVelocityX.append(object.angularVelocity[0])
    angularVelocityY.append(object.angularVelocity[1])
    angularVelocityZ.append(object.angularVelocity[2])

    
plt.figure(figsize=(12, 8))
plt.subplot(2, 1, 1)
plt.plot(angularVelocityX)
plt.plot(angularVelocityY)
plt.plot(angularVelocityZ)
plt.xlabel('Time step')
plt.ylabel('Angular Velocity (rad/s)')
plt.title('Evolution of Angular Velocities over Time')
plt.grid()
plt.legend(['X', 'Y', 'Z'])
plt.subplot(2, 1, 2)
plt.plot(eulerAngleX)
plt.plot(eulerAngleY)
plt.plot(eulerAngleZ)
plt.xlabel('Time step')
plt.ylabel('Euler Angle (deg)')
plt.title('Evolution of Euler Angles over Time')
plt.grid()
plt.legend(['X', 'Y', 'Z'])
plt.show()
