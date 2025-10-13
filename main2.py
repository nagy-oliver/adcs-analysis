import numpy as np
import matplotlib.pyplot as plt
import constants as cst

# ---------- Utility Functions ----------

def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm

# Quaternion multiplication q1 * q2
def quat_mult(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])

# Convert quaternion to rotation matrix (body→inertial DCM)
def quat_to_dcm(q):
    q = normalize(q)
    w, x, y, z = q
    return np.array([
        [1 - 2*(y**2 + z**2),     2*(x*y - z*w),       2*(x*z + y*w)],
        [2*(x*y + z*w),           1 - 2*(x**2 + z**2), 2*(y*z - x*w)],
        [2*(x*z - y*w),           2*(y*z + x*w),       1 - 2*(x**2 + y**2)]
    ])

# Quaternion derivative: dq/dt = 0.5 * Ω(ω) * q
def quat_derivative(q, omega):
    w, x, y, z = q
    wx, wy, wz = omega
    Omega = np.array([
        [0.0, -wx, -wy, -wz],
        [wx,  0.0,  wz, -wy],
        [wy, -wz,  0.0,  wx],
        [wz,  wy, -wx,  0.0]
    ])
    return 0.5 * Omega @ q

# ---------- Frame transforms ----------

def transform_local_to_global(q, vector):
    """Body → Inertial"""
    DCM = quat_to_dcm(q)
    return DCM @ vector

def transform_global_to_local(q, vector):
    """Inertial → Body"""
    DCM = quat_to_dcm(q)
    return DCM.T @ vector

# ---------- Torque Functions ----------

def torque_gg(q, cst):
    """Gravity-gradient torque in body frame"""
    vec_nadir_body = transform_global_to_local(q, cst.vec_nadir_global)
    vec_nadir_body = normalize(vec_nadir_body)
    r = cst.h + cst.R
    factor = 3.0 * (cst.mu / (r**3))
    return factor * np.cross(vec_nadir_body, cst.I @ vec_nadir_body)

def torque_solar(q):
    """
    Solar radiation torque in body frame.
    Parameters
    ----------
    q : np.array(4)
        Quaternion (w, x, y, z) mapping body→global
    sun_vector_global : np.array(3)
        Vector from spacecraft to Sun (in global/inertial frame)
    Returns
    -------
    torque_local : np.array(3)
        Solar radiation torque [N·m] in the body frame
    """
    # --- Constants (example values, tune as needed) ---
    rho = 0.6           # surface reflectivity
    p_s = 4.8e-5        # solar radiation pressure [N/m²]
    s = 1.8**2          # exposed area [m²]
    vec_cp_local = np.array([0.0, -1.0, 0.0])  # CoG → center of pressure [m]

    # --- Transform Sun vector into body frame ---
    sun_unit_local = transform_global_to_local(q, cst.sun_unit_global)

    # --- Force in body frame (acts opposite to Sun direction) ---
    solar_force_local = -(1 + rho) * p_s * s * sun_unit_local

    # --- Torque in body frame ---
    torque_local = np.cross(vec_cp_local, solar_force_local)
    return torque_local

# Example constant torques
magnetic_torques = np.array([2e-8, 2e-8, 2e-8])
internal_torques = np.array([0.0, 0.0, 0.0])

# ---------- Physics Integration ----------

I_inv = np.linalg.inv(cst.I)

def physics(torques, cst, state, dt):
    t, alpha, omega, q = state
    torque_sum = np.sum(torques, axis=0)
    alpha = I_inv @ (torque_sum - np.cross(omega, cst.I @ omega))
    omega = omega + alpha * dt
    dq = quat_derivative(q, omega)
    q = normalize(q + dq * dt)
    t = t + dt
    return t, alpha, omega, q

# ---------- Initialization ----------

dt = 0.1  # seconds
state = (0.0, np.zeros(3), np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]))
t, alpha, omega, q = state
tlist, roll_list, pitch_list, yaw_list = [], [], [], []


# ---------- Simulation Loop ----------

while t <= 7000:
    t_solar = torque_solar(q)
    t_gg = torque_gg(q, cst)
    torques = [t_solar, t_gg, magnetic_torques, internal_torques]
    state = physics(torques, cst, state, dt)
    t, alpha, omega, q = state

    # Convert quaternion to Euler angles for plotting
    w, x, y, z = q
    sinp = 2*(w*y - z*x)
    sinp = np.clip(sinp, -1.0, 1.0)
    roll = np.arctan2(2*(w*x + y*z), 1 - 2*(x**2 + y**2))
    pitch = np.arcsin(sinp)
    yaw = np.arctan2(2*(w*z + x*y), 1 - 2*(y**2 + z**2))

    tlist.append(t)
    roll_list.append(roll)
    pitch_list.append(pitch)
    yaw_list.append(yaw)

# ---------- Plot Results ----------

plt.figure(figsize=(10, 6))
plt.plot(tlist, roll_list, label='Roll (ϕ)')
plt.plot(tlist, pitch_list, label='Pitch (θ)')
plt.plot(tlist, yaw_list, label='Yaw (ψ)')
plt.xlabel('Time [s]')
plt.ylabel('Angle [rad]')
plt.title('Attitude Evolution with Solar and Gravity-Gradient Torques')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
