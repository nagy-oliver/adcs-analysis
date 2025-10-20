import numpy as np
import matplotlib.pyplot as plt
import os
import constants as cst

# ---------- Utility Functions ----------
def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm

def quat_mult(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])

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
    return np.array([q[0], -q[1], -q[2], -q[3]])

def rotate_vector_by_quat(v, q):
    q_conj = quat_conjugate(q)
    v_quat = np.array([0.0, *v])
    v_rot = quat_mult(quat_mult(q, v_quat), q_conj)
    return v_rot[1:]

def transform_local_to_global(q, vector):
    return rotate_vector_by_quat(vector, q)

def transform_global_to_local(q, vector):
    q_conj = quat_conjugate(q)
    return rotate_vector_by_quat(vector, q_conj)

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
    vec_nadir_body = transform_global_to_local(q, cst.vec_nadir_global)
    vec_nadir_body = normalize(vec_nadir_body)
    r = cst.h + cst.R
    factor = 3.0 * (cst.mu / (r**3))
    return factor * np.cross(vec_nadir_body, cst.I @ vec_nadir_body)

def torque_solar(q,t_eclipse,t,q_solar):
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

# ---------- Actuator / control parameters ----------
rw_max_torque = 0.06     # [N·m]  (for logging / if you later want to apply torque over dt)
rw_max_momentum = 0.5    # [N·m·s] momentum capacity
rcs_max_torque = 1.98    # [N·m]
rw_momentum = np.zeros(3)          # reaction wheel stored momentum per axis

# New requested behavior parameters:
angle_threshold_deg = 0.1          # ±0.1 degree deadband
omega_set_neg = -0.00001          # rad/s (value to set when RW acts) — chosen between -0.01 and -0.001

# PD gains (kept for optional use; not needed for instant RW intervention)
Kp = np.deg2rad(0.02)
Kd = 0.2

q_desired = np.array([1.0, 0.0, 0.0, 0.0])  # desired attitude

def quat_error(q_desired, q_current):
    return quat_mult(quat_conjugate(q_current), q_desired)

# ---------- Physics Integration ----------
I = cst.I
I_inv = np.linalg.inv(I)
I_diag = np.diag(I)  # used for per-axis inertia in simple momentum calculations

def physics(torques, cst, state, dt):
    # Integrate dynamics using torques (disturbances only here, because RW instant action is applied directly to omega)
    t, alpha, omega, q = state
    torque_sum = np.sum(torques, axis=0)
    alpha = I_inv @ (torque_sum - np.cross(omega, I @ omega))
    omega += alpha * dt
    dq = quat_derivative(q, omega)
    q = normalize(q + dq * dt)
    t += dt
    return t, alpha, omega, q

# ---------- Initialization ----------
dt = 0.1
state = (0.0, np.zeros(3), np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]))
t, alpha, omega, q = state

data = {key: [] for key in [
    't','roll','pitch','yaw',
    'omega roll','omega pitch','omega yaw',
    'alpha roll','alpha pitch','alpha yaw',
    'torque gg','torque total','torque solar',
    'rw torque_req','rcs torque_req','rw momentum'
]}

magnetic_torques = np.array([2e-8, 2e-8, 2e-8])
internal_torques = np.zeros(3)

# ---------- Simulation Loop ----------
while t <= 10 * cst.T:
    # Disturbance torques
    eulerAnglesGlobalWRTSolarDeg = np.array([0.0, -360 * t / cst.T, 0.0])
    quaternionGlobalWRTSolar = euler_deg_to_quat(eulerAnglesGlobalWRTSolarDeg)
    t_solar = torque_solar(q, cst.t_eclipse, t, quaternionGlobalWRTSolar)
    t_gg = torque_gg(q, cst)

    # --- compute current Euler angles from quaternion (for checking threshold) ---
    wq, xq, yq, zq = q
    sinp = 2*(wq*yq - zq*xq)
    sinp = np.clip(sinp, -1.0, 1.0)
    roll  = np.degrees(np.arctan2(2*(wq*xq + yq*zq), 1 - 2*(xq**2 + yq**2)))
    pitch = np.degrees(np.arcsin(sinp))
    yaw   = np.degrees(np.arctan2(2*(wq*zq + xq*yq), 1 - 2*(yq**2 + zq**2)))
    euler_deg = np.array([roll, pitch, yaw])

    # --- New RW behavior (requested) ---
    # We'll check each axis independently. If the absolute angle > threshold AND the current
    # angular velocity would increase that angle (sign(angle)*omega > 0), the RW instantly
    # sets omega to a small negative value (omega_set_neg) and we update reaction wheel momentum.
    t_rw_req = np.zeros(3)   # torque *required* by RW to produce delta (for logging only)
    t_rcs_req = np.zeros(3)  # torque *required* by RCS if RW saturates (for logging only)

    # Record previous omega to compute delta
    omega_prev = omega.copy()

    # RCS parameters
    rcs_impulse_unit = 0.1    # N·m·s  (discrete impulse step)
    rcs_moment_arm = 0.9      # m
    if 'rcs_total_dv' not in locals():
        rcs_total_dv = 0.0    # initialize total delta-v tracker

    for i in range(3):
        angle_i = euler_deg[i]
        omega_i = omega[i]

        # Condition: angle outside deadband and angular velocity is pushing it further away
        if (abs(angle_i) > angle_threshold_deg) and (np.sign(angle_i) * omega_i > 0):
            # RW attempts to reverse motion by setting omega opposite to current direction
            new_omega_i = omega_set_neg if angle_i > 0 else -omega_set_neg
            delta_omega = new_omega_i - omega_i

            # Change in spacecraft angular momentum
            delta_L_sc = I_diag[i] * delta_omega  # [N·m·s]

            # Reaction wheel momentum must change by -ΔL_sc
            new_rw_momentum_i = rw_momentum[i] - delta_L_sc

            # Check if RW saturates
            if abs(new_rw_momentum_i) > rw_max_momentum:
                # Wheel would saturate → use RCS in discrete 0.1 N·m·s impulses
                required_change = abs(delta_L_sc)
                # Round up to nearest multiple of 0.1
                impulse_used = np.ceil(required_change / rcs_impulse_unit) * rcs_impulse_unit

                # Apply impulse direction (same as delta_L_sc direction)
                delta_L_rcs = np.sign(delta_L_sc) * impulse_used

                # Apply the corresponding Δv (p = FΔt = mΔv → Δv = impulse / (moment_arm * m))
                # Here we don't know mass directly, so we log Δv per unit mass:
                # τ = r × F → F = τ / r  → impulse (N·m·s) / r (m) = linear impulse (N·s)
                # → Δv = linear_impulse / m → we log per unit mass, so Δv = impulse/r
                delta_v_per_mass = impulse_used / rcs_moment_arm
                rcs_total_dv += abs(delta_v_per_mass)

                # Transfer the discrete impulse to the reaction wheel momentum
                rw_momentum[i] += delta_L_rcs

                # Apply spacecraft angular velocity change instantly (per your request)
                omega[i] = new_omega_i

                # Log RCS and RW effects
                t_rcs_req[i] = delta_L_rcs / dt
                t_rw_req[i] = 0.0  # RW did not produce torque directly
            else:
                # RW can handle it: apply the change
                omega[i] = new_omega_i
                rw_momentum[i] = new_rw_momentum_i
                t_rw_req[i] = -delta_L_sc / dt
                t_rcs_req[i] = 0.0
        # else: no action


    # Combine disturbance torques only (we do NOT add the instantaneous RW torque into physics()
    # because we've already directly modified omega to represent the RW action in this step).
    # If you later prefer to apply actuators as torques over dt instead of direct velocity changes,
    # change this logic so the RW/RCS torques are included in the torques list and do NOT directly
    # change omega.
    torques = [t_solar, t_gg, magnetic_torques, internal_torques]
    torque_total = np.sum(torques, axis=0)

    # Integrate physics (disturbances only — RW already acted instantaneously)
    state = physics(torques, cst, (t, alpha, omega, q), dt)
    t, alpha, omega, q = state

    # Recompute Euler angles after physics integration for logging
    wq, xq, yq, zq = q
    sinp = 2*(wq*yq - zq*xq)
    sinp = np.clip(sinp, -1.0, 1.0)
    roll  = np.degrees(np.arctan2(2*(wq*xq + yq*zq), 1 - 2*(xq**2 + yq**2)))
    pitch = np.degrees(np.arcsin(sinp))
    yaw   = np.degrees(np.arctan2(2*(wq*zq + xq*yq), 1 - 2*(yq**2 + zq**2)))

    # Store data
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
    data['torque total'].append(torque_total)
    data['torque solar'].append(t_solar)
    data['rw torque_req'].append(t_rw_req.copy())
    data['rcs torque_req'].append(t_rcs_req.copy())
    data['rw momentum'].append(rw_momentum.copy())

# ---------- Plotting ----------
os.makedirs('plots', exist_ok=True)

plots = [
    {'title': 'Euler angles', 'y_label': 'Angle [deg]', 'series': [('roll','Roll'),('pitch','Pitch'),('yaw','Yaw')], 'filename': 'euler_angles.pdf'},
    {'title': 'Angular velocity vs time', 'y_label': 'Angular velocity [rad/s]', 'series': [('omega roll','Roll'),('omega pitch','Pitch'),('omega yaw','Yaw')], 'filename': 'angular_velocity.pdf'},
    {'title': 'Angular acceleration vs time', 'y_label': 'Angular acceleration [rad/s²]', 'series': [('alpha roll','Roll'),('alpha pitch','Pitch'),('alpha yaw','Yaw')], 'filename': 'angular_acceleration.pdf'},
    {'title': 'RW required torque (logged)', 'y_label': 'Torque [N·m]', 'series': [('rw torque_req','RW Torque (req)')], 'filename': 'rw_torque_req.pdf'},
    {'title': 'RCS required torque (logged)', 'y_label': 'Torque [N·m]', 'series': [('rcs torque_req','RCS Torque (req)')], 'filename': 'rcs_torque_req.pdf'},
    {'title': 'Reaction wheel momentum', 'y_label': 'Momentum [N·m·s]', 'series': [('rw momentum','RW Momentum')], 'filename': 'rw_momentum.pdf'},
]

for plot in plots:
    plt.figure(figsize=(8,3))
    if plot['title'] == 'Euler angles':
        for key, label in plot['series']:
            angles = np.array(data[key])
            diff = np.abs(np.diff(angles))
            threshold = 350
            angles_plot = angles.copy()
            angles_plot[np.where(diff > threshold)[0] + 1] = np.nan
            plt.plot(data['t'], angles_plot, label=label)
    elif plot['title'] in ['RW required torque (logged)','RCS required torque (logged)','Reaction wheel momentum']:
        key = plot['series'][0][0]
        val = np.vstack(data[key])
        plt.plot(data['t'], val[:,0], label='X')
        plt.plot(data['t'], val[:,1], label='Y')
        plt.plot(data['t'], val[:,2], label='Z')
    else:
        for key, label in plot['series']:
            plt.plot(data['t'], data[key], label=label)
    plt.title(plot['title'])
    plt.xlabel('Time (s)')
    plt.ylabel(plot['y_label'])
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join('plots', plot['filename']), format='pdf')


print(f"Total RCS delta-v per unit mass used: {rcs_total_dv:.4f} m/s")
plt.show()
