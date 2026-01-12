import numpy as np
import os, sys
import matplotlib.pyplot as plt

#   Arrange path to working directory
script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
os.chdir(script_dir)
IMU_DIR = "./IMU/"

def read_csv(file_path: str) -> np.array:
    return np.loadtxt(file_path, delimiter=',', skiprows=1)

def wrap_to_pi(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

def get_IMU_data_full(tot_time: float):
    # Reads: time, gx, gy, gz, ax, ay, az
    imu_data = read_csv(IMU_DIR + "normalized_imu.csv")
    imu_time = imu_data[:,0]
    imu_vals = imu_data[:, 1:7] 

    end_index = np.searchsorted(imu_time, tot_time, side='left')
    imu_time = imu_time[:end_index+1]
    imu_vals = imu_vals[:end_index+1,:]

    time_column = imu_time.reshape(-1, 1)
    stacked_matrix = np.hstack((time_column, imu_vals))
    return stacked_matrix

"""
    Hydrodynamic Model for 2D Motion (Surge & Sway)
    Based on equations (1) and (2) from the Randeni et al. paper.
    Constraint: Heave (w) and z_dot are forced to 0.
"""
def hydrodynamic_model_2D(state, rates, accels, u_imu, dt):
    # Unpack Rates (p=Roll Rate, q=Pitch Rate, r=Yaw Rate)
    p, q, r = rates[0], rates[1], rates[2]
    
    # Derivatives
    p_dot = (p - state['p_prev']) / dt
    q_dot = (q - state['q_prev']) / dt
    r_dot = (r - state['r_prev']) / dt
    
    # Orientation
    phi = state['phi']      # Roll
    theta = state['theta']  # Pitch
    
    # Inputs
    N_prop = state['prop_rpm']
    
    # --- MODEL PARAMETERS (Placeholder Values) ---
    # Coefficients for Surge (u)
    # alpha = [a1...a9]
    alpha = [0.0, 0.05, 0.05, 0.0, 0.0, 0.0, 0.0, 0.05, 0.002]
    
    # Coefficients for Sway (v)
    # beta = [b1...b8]
    beta = [0.0, 0.0, 0.0, -1.0, 0.0, 0.0, -0.6, 1.0]

    # Eq (1): Surge (u)
    # Terms involving z_dot are removed (0)
    u_model = (alpha[0]*q_dot + alpha[1]*r_dot + alpha[2]*u_imu + 
               0.0 + alpha[4]*(q**2) + alpha[5]*(r**2) + 
               alpha[6]*p*r + alpha[7]*np.sin(theta) + alpha[8]*N_prop)
               
    # Eq (2): Sway (v)
    v_model = (beta[0]*p_dot + beta[1]*r_dot + 0.0 + 
               beta[3]*u_imu*r + beta[4]*q*r + beta[5]*p*q + 
               beta[6]*r*abs(r) + beta[7]*np.cos(theta)*np.sin(phi))

    return u_model, v_model


if __name__ == "__main__":
    total_time = 231
    IMU_data = get_IMU_data_full(total_time)
    
    dt = 0.05
    time_steps = np.linspace(0, total_time, int((total_time + dt) / dt))

    # --- STATE VECTOR DEFINITION ---
    # Size: 5x1
    # [0]: x (Global Position X)
    # [1]: y (Global Position Y)
    # [2]: theta (Heading/Yaw) -> Corresponds to your "z" request
    # [3]: u (Body Surge Velocity)
    # [4]: v (Body Sway Velocity)
    state_vector = np.zeros((5, 1))
    
    # Internal model state
    model_internal = {
        'p_prev': 0.0, 'q_prev': 0.0, 'r_prev': 0.0,
        'phi': 0.0, 'theta': 0.0,
        'prop_rpm': 100.0
    }
    
    current_u_imu_naive = 0.0
    IMU_idx = 0
    
    history = []
    time_history = []
    last_t = -1

    print("Processing 2D Hydrodynamic Model (x, y, theta, u, v)...")
    
    for t in time_steps:
        # Sync IMU Data
        while IMU_idx < len(IMU_data) and IMU_data[IMU_idx,0] < t:
            IMU_idx += 1
        
        if IMU_idx > 0 and IMU_data[IMU_idx,0] > t and abs(t - last_t) > 1e-4:
            IMU_idx -= 1
            
            # Measurements: [p, q, r] and [ax, ay, az]
            gyro_meas = IMU_data[IMU_idx, 1:4]
            accel_meas = IMU_data[IMU_idx, 4:7]
            
            # 1. Integrate Naive u_IMU input (Forward Acceleration)
            current_u_imu_naive += accel_meas[0] * dt
            current_u_imu_naive *= 0.98
            
            # 2. Update Angles (Roll, Pitch)
            model_internal['phi'] += gyro_meas[0] * dt
            model_internal['theta'] += gyro_meas[1] * dt
            
            # 3. COMPUTE U and V (Eq 1 & 2) 
            u_body, v_body = hydrodynamic_model_2D(model_internal, gyro_meas, accel_meas, current_u_imu_naive, dt)
            
            # Update derivatives history
            model_internal['p_prev'] = gyro_meas[0]
            model_internal['q_prev'] = gyro_meas[1]
            model_internal['r_prev'] = gyro_meas[2]
            
            # 4. Update State Vector
            # Update Heading (Theta/Z) using Yaw rate (r)
            state_vector[2,0] += gyro_meas[2] * dt
            state_vector[2,0] = wrap_to_pi(state_vector[2,0])
            
            # Rotate Body Velocities to Global Frame
            # [x_dot]   [cos(th)  -sin(th)] [u]
            # [y_dot] = [sin(th)   cos(th)] [v]
            psi = state_vector[2,0]
            x_dot = u_body * np.cos(psi) - v_body * np.sin(psi)
            y_dot = u_body * np.sin(psi) + v_body * np.cos(psi)
            
            # Integrate Position
            state_vector[0,0] += x_dot * dt
            state_vector[1,0] += y_dot * dt
            
            # Store Velocities in state
            state_vector[3,0] = u_body
            state_vector[4,0] = v_body

            last_t = t
            
            history.append(state_vector.flatten().copy())
            time_history.append(t)

    # --- PLOTTING ---
    history = np.array(history)
    time_history = np.array(time_history)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('2D Hydrodynamic Model Localization', fontsize=16)

    # XY Trajectory
    ax1.plot(history[:, 0], history[:, 1], 'b-', label='Model Path')
    ax1.plot(history[0, 0], history[0, 1], 'go', label='Start')
    ax1.plot(history[-1, 0], history[-1, 1], 'ro', label='End')
    ax1.set_xlabel('X Position (m)')
    ax1.set_ylabel('Y Position (m)')
    ax1.axis('equal')
    ax1.grid(True)
    ax1.legend()

    # Velocity Profile (u, v)
    ax2.plot(time_history, history[:, 3], label='Surge (u)')
    ax2.plot(time_history, history[:, 4], label='Sway (v)')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Velocity (m/s)')
    ax2.set_title('Body-Frame Velocities')
    ax2.grid(True)
    ax2.legend()

    plt.show()