import numpy as np
import matplotlib.pyplot as plt
import constants as cst
import os

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

def quat_conjugate(q):
    """Conjugate of quaternion q = [w, x, y, z]."""
    return np.array([q[0], -q[1], -q[2], -q[3]])

def rotate_vector_by_quat(v, q):
    """Rotate vector v by quaternion q (body→inertial)."""
    q_conj = quat_conjugate(q)
    v_quat = np.array([0.0, *v])
    v_rot = quat_mult(quat_mult(q, v_quat), q_conj)
    return v_rot[1:]  # extract vector part

# ---------- Frame transforms ----------

def transform_local_to_global(q, vector):
    """Body → Inertial using quaternion multiplication."""
    return rotate_vector_by_quat(vector, q)

def transform_global_to_local(q, vector):
    """Inertial → Body using quaternion multiplication."""
    q_conj = quat_conjugate(q)
    return rotate_vector_by_quat(vector, q_conj)

# ---------- Conversion Function ----------

def euler_deg_to_quat(euler_deg):
    roll, pitch, yaw = np.radians(euler_deg)
    cr = np.cos(roll / 2)
    sr = np.sin(roll / 2)
    cp = np.cos(pitch / 2)
    sp = np.sin(pitch / 2)
    cy = np.cos(yaw / 2)
    sy = np.sin(yaw / 2)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return normalize(np.array([w, x, y, z]))

# ---------- Torque Functions ----------

def torque_gg(q, cst):
    """Gravity-gradient torque in body frame."""
    vec_nadir_body = transform_global_to_local(q, cst.vec_nadir_global)
    vec_nadir_body = normalize(vec_nadir_body)
    r = cst.h + cst.R
    factor = 3.0 * (cst.mu / (r**3))
    return factor * np.cross(vec_nadir_body, cst.I @ vec_nadir_body)

def torque_solar(q,t_eclipse,t,q_solar):
    """Solar radiation torque in body frame."""
    rho = 0.6
    p_s = 4.8e-5
    s = 1.8**2
    vec_cp_local = np.array([0.0, -1.0, 0.0])

    sunVectorGlobal = transform_global_to_local(q_solar, cst.sun_unit_solar)
    sun_unit_local = transform_global_to_local(q, sunVectorGlobal)
    solar_force_local = -(1 + rho) * p_s * s * sun_unit_local
    torque_local = np.zeros(3)
    if (t % cst.T) >= t_eclipse:
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
    omega += alpha * dt
    dq = quat_derivative(q, omega)
    q = normalize(q + dq * dt)
    t += dt
    return t, alpha, omega, q

# ---------- Initialization ----------

dt = 0.1  # seconds
state = (0.0, np.zeros(3), np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]))
t, alpha, omega, q = state
data = {
    't': [],
    'roll' : [],
    'pitch' : [],
    'yaw' : [],
    'omega roll' : [],
    'omega pitch' : [],
    'omega yaw' : [],
    'alpha roll': [],
    'alpha pitch': [],
    'alpha yaw': [],
    'torque gg': []
}

# ---------- Simulation Loop ----------

t_threshold = None

while t <= cst.T * 10:
    eulerAnglesGlobalWRTSolarDeg = np.array([0.0,-360*t/cst.T,0.0])
    quaternionGlobalWRTSolar = euler_deg_to_quat(eulerAnglesGlobalWRTSolarDeg)

    t_solar = torque_solar(q, cst.t_eclipse, t, quaternionGlobalWRTSolar)
    t_gg = torque_gg(q, cst)
    torques = [t_solar, t_gg, magnetic_torques, internal_torques]
    state = physics(torques, cst, state, dt)
    t, alpha, omega, q = state

    # Convert quaternion to Euler angles for plotting
    w, x, y, z = q
    sinp = 2*(w*y - z*x)
    sinp = np.clip(sinp, -1.0, 1.0)
    roll = np.arctan2(2*(w*x + y*z), 1 - 2*(x**2 + y**2))*180.0/np.pi
    pitch = np.arcsin(sinp)*180.0/np.pi
    yaw = np.arctan2(2*(w*z + x*y), 1 - 2*(y**2 + z**2))*180.0/np.pi

    if ((abs(roll) >= cst.threshold) or (abs(pitch) >= cst.threshold) or (abs(yaw) >= cst.threshold)) and (t_threshold is None):
        t_threshold = t
        i_threshold = int(t/dt)

    data['t'].append(t)
    data['roll'].append(roll)
    data['pitch'].append(pitch)
    data['yaw'].append(yaw)
    data['omega roll'].append(omega[0])
    data['omega pitch'].append(omega[1])
    data['omega yaw'].append(omega[2])
    data['alpha roll'].append(alpha[0])
    data['alpha pitch'].append(alpha[1])
    data['alpha yaw'].append(alpha[2])
    data['torque gg'].append(t_gg)




# ---------- Plot Results ----------

# Define plot configurations
plots = [
    {
        'title': 'Euler angles',
        'y_label': 'Angle [deg]',
        'series': [('roll', 'Roll'), ('pitch', 'Pitch'), ('yaw', 'Yaw')],
        'filename': 'euler_angles.pdf'
    },
    {
        'title': 'Angular velocity vs time',
        'y_label': 'Angular velocity [rad/s]',
        'series': [('omega roll', 'Roll'), ('omega pitch', 'Pitch'), ('omega yaw', 'Yaw')],
        'filename': 'angular_velocity.pdf'
    },
    {
        'title': 'Angular acceleration vs time',
        'y_label': 'Angular acceleration [rad/s²]',
        'series': [('alpha roll', 'Roll'), ('alpha pitch', 'Pitch'), ('alpha yaw', 'Yaw')],
        'filename': 'angular_acceleration.pdf'
    },
    {
        'title': 'Gravity-gradient torque vs time',
        'y_label': 'Torque [Nm]',
        'series': [('torque gg', 'Torque')],
        'filename': 'torque_gg.pdf'
    }
]

plt.figure(figsize=(12, 6))
angles = np.array(data['roll'])
diff = np.abs(np.diff(angles))
threshold = 350  # degrees
angles_plot = angles.copy()
angles_plot[1:][diff > threshold] = np.nan  # break line
plt.plot(data['t'],angles_plot,label='Roll')
angles = np.array(data['pitch'])
diff = np.abs(np.diff(angles))
threshold = 350  # degrees
angles_plot = angles.copy()
angles_plot[1:][diff > threshold] = np.nan  # break line
plt.plot(data['t'],angles_plot,label='Pitch')
angles = np.array(data['yaw'])
diff = np.abs(np.diff(angles))
threshold = 350  # degrees
angles_plot = angles.copy()
angles_plot[1:][diff > threshold] = np.nan  # break line
plt.plot(data['t'],angles_plot,label='Yaw')

# Generate and save each figure
for plot in plots:
    if plot['title'] == 'Euler angles':
        pass
    else:
        plt.figure(figsize=(12, 6))
        for key, label in plot['series']:
            plt.plot(data['t'], data[key], label=label)
    plt.title(plot['title'])
    plt.xlabel('Time (s)')
    plt.ylabel(plot['y_label'])
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    #plt.savefig(os.path.join('plots', plot['filename']), format='pdf')
plt.show()

print(f'Exceeded pointing accuracy threshold at time {t_threshold:.2f} s.')

