import numpy as np
from feature_matching import SonarFeatureMatcher
from image_processing import SonarImageProcessor
import cv2, os, sys, math, matplotlib.pyplot as plt


#   Set random seed for reproducibility
np.random.seed(50)


#   Arrange path to working directory
script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
os.chdir(script_dir)
SONAR_DIR = "./sonar/"
IMU_DIR = "./IMU/"
GPS_DIR = "./GPS/"


"""
    Reads a CSV file and returns its contents as a NumPy array.
    
    Args:
        file_path (str): The path to the CSV file.
"""
def read_csv(file_path: str) -> np.array:
    return np.loadtxt(file_path, delimiter=',', skiprows=1)


"""
    Wraps an angle in rad to the range (-pi, pi]
"""
def wrap_to_pi(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


"""
    USV model class implementing a particle filter for localization using IMU and Sonar data.
"""
class USV_Model:
    def __init__(self, num_particles: int):
        self.n_particles = num_particles
        
        #   Process covariance matrices
        self.Q_IMU = np.diag([0.5, 0.5, 0.5])
        self.dt = 0.05
        self.sonar_dt = 0.0667

        #   Measurement noise covariances for Sonar
        self.R = np.diag([0.5, 0.5, 0.5]) 
        self.R_inv = np.linalg.inv(self.R)

        #   Precompute Cholesky decomposition for IMU noise sampling
        self.L = np.linalg.cholesky(self.Q_IMU)

        #   Hydrodynamic Coefficients (Tuned values)
        self.alpha = [-0.05, 0.05, 0.005]       #   alpha[2], alpha[7], alpha[8] tuned
        self.beta = [-1.0, -0.6, 1.0]           #   beta[3], beta[6], beta[7] tuned
        self.prop_rpm = 100.0


    """
        Hydrodynamic Model for 2D Motion (Surge & Sway)
        Based on equations (1) and (2) from the Randeni et al. paper.
        The parameters alpha,beta employed are the dominant ones for which the estimation is sensitive.
    """
    def hydrodynamic_model_2D(self, r, phi, theta, u_imu):
        #   Surge (u)
        u_model = (self.alpha[0]*u_imu + self.alpha[1]*np.sin(theta) + self.alpha[2]*self.prop_rpm)
                   
        #   Sway (v)
        v_model = (self.beta[0]*u_imu*r + self.beta[1]*r*abs(r) + self.beta[2]*np.cos(theta)*np.sin(phi))

        return u_model, v_model
    

    """
        Computes real-world coordinates from GPS and heading data.
        Returns an array of shape (N, 5) where each row contains:
        [x, y, theta, vx, vy]
    """
    def compute_real_coords(self) -> np.array:
        heading = read_csv(GPS_DIR + "normalized_heading.csv")          #   In degrees
        coordinates = read_csv(GPS_DIR + "normalized_fix.csv")          #   lat, long, alt

        #   Convert time instants to integers (s)
        time = heading[:,0]
        heading = heading[:,1] * (np.pi/180)                    #   Convert to rad
        coordinates = coordinates[:,1:3] * (np.pi/180)          #   Remove time column

        #   Real positions = [x, y, theta]
        real_positions = np.zeros((len(time),5))

        y_0 = coordinates[0,0]
        x_0 = coordinates[0,1]
        theta_0 = heading[0]
        R_avg = 6371000                                         #   Radius of the Earth in meters

        real_positions[:,1] = (coordinates[:,0] - y_0) * R_avg
        real_positions[:,0] = (coordinates[:,1] - x_0) * R_avg * np.cos((coordinates[:,0] + y_0)/2)
        real_positions[:,2] = wrap_to_pi(heading - theta_0)

        #   Compute velocities (vx, vy, omega)
        real_positions[0,3] = 0.0
        real_positions[0,4] = 0.0

        for i in range(len(time) - 1):
            delta_t = time[i+1] - time[i]
            if delta_t > 0:
                real_positions[i+1,3] = (real_positions[i+1,0] - real_positions[i,0]) / delta_t
                real_positions[i+1,4] = (real_positions[i+1,1] - real_positions[i,1]) / delta_t

        return  real_positions
    

    """
        Retrieves IMU data up to a specified total time.
    """
    def get_IMU_data(self, tot_time: float):
        imu_data = read_csv(IMU_DIR + "normalized_imu.csv")
        imu_time = imu_data[:,0]

        # We need all 6 axes for the hydro model (gx, gy, gz, ax, ay, az)
        imu_vals = imu_data[:, 1:7] 
        end_index = np.searchsorted(imu_time, tot_time, side='left')

        return np.hstack((imu_time[:end_index+1].reshape(-1,1), imu_vals[:end_index+1,:]))
    

    """
        Retrieves Sonar data up to a specified total time.
    """
    def get_sonar_data(self, tot_time: float):
        sonar_data = read_csv("normalized_sonar_times.csv")     #   img_idx, time
        sonar_time = sonar_data[:,1]
        sonar_images = sonar_data[:,0].astype(int)

        end_index = np.searchsorted(sonar_time, tot_time, side='left')
        sonar_time = sonar_time[:end_index+1]
        sonar_images = sonar_images[:end_index+1]
        
        time_column = sonar_time.reshape(-1, 1)
        image_column = sonar_images.reshape(-1, 1)
        
        stacked_matrix = np.hstack((time_column, image_column))
        return stacked_matrix
    

    """
        Computes the transformation between two sonar images using feature matching.
        Returns a numpy array [dx, dy, dtheta_image].
    """
    def get_transformation(self, img1: cv2.typing.MatLike, img2: cv2.typing.MatLike) -> np.array:
        match = SonarFeatureMatcher()
        filter = SonarImageProcessor()

        if img1 is None or img2 is None:
            print("Error: One or more images failed to load. Check file paths.")
            return np.zeros(3)
        else:
            img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
            img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

            img1P = filter.process_image(img1_rgb)
            img2P = filter.process_image(img2_rgb)

            result, _ , _, _  = match.process_sonar_image_pair(img1P, img2P)
            T = result.get('transformation')
            dR = T[0:2,0:2]
            dp = T[0:2,2]
            dx = dp[1]                                      # right
            dy = dp[0]                                      # forward
            dtheta_image = np.arctan2(dR[1,0], dR[0,0])     # positive ccw from x axis

            return np.array([dx,dy,dtheta_image])
    

    """
        Computes the sonar difference between two images given their indices.
        Returns a numpy array [dx, dy, dtheta_image].
    """
    def compute_sonar_diff(self, img1: int, img2: int) -> np.array:
        img1_path = SONAR_DIR + str(img1) + ".png"
        img2_path = SONAR_DIR + str(img2) + ".png"

        img1 = cv2.imread(str(img1_path))
        img2 = cv2.imread(str(img2_path))
        arr = self.get_transformation(img1,img2)
        return arr
    

    """
        Particle Filter implementation for USV localization using IMU and Sonar data.
        Returns maximum error, error array, real coordinates, and estimated coordinates.
    """
    def particle_filter(self):
        real_coords = self.compute_real_coords()
        T = len(real_coords) - 1
        time_steps = np.linspace(0, T, int((T+self.dt)/self.dt))
        
        #   --- Initialization ---
        current_N = self.n_particles
        
        #   State: [x, y, theta, vx, vy]  Shape: (5, N)
        particles = np.zeros((5, current_N))
        weights = np.ones(current_N) / current_N
        omega_noisy = np.zeros(current_N)
        
        #   Initialize around first ground truth with some noise
        particles[0,:] = real_coords[0,0]
        particles[1,:] = real_coords[0,1]
        particles[2,:] = real_coords[0,2]

        jitter_N = self.L @ np.random.randn(3,current_N)

        particles[0,:] += jitter_N[0,:]
        particles[1,:] += jitter_N[1,:]
        particles[2,:] = wrap_to_pi(particles[2,:] + jitter_N[2,:])

        prev_particles = particles.copy()
        imu_data = self.get_IMU_data(T)
        sonar_data = self.get_sonar_data(T)

        #   Internal states for hydro model (per particle)
        p_prev = np.zeros(current_N)
        q_prev = np.zeros(current_N)
        r_prev = np.zeros(current_N)
        phi = np.zeros(current_N)
        theta_internal = np.zeros(current_N)
        u_imu_naive = np.zeros(current_N)
        
        #   Tracking variables
        estimated_coords = []
        error = []
        max_error = [0.0, 0.0, 0.0]
        R_corr = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])

        #   Estimated coordinates needed for resampling
        est_x = 0.0
        est_y = 0.0
        est_theta = 0.0

        imu_idx = 0
        sonar_idx = 0
        prev_sonar_idx = -1
        prev_imu_idx = -1
        count_particle = 0


        for _, t in enumerate(time_steps):
            
            #   --- DATA SYNC ------------------------------------------------------------------
            while imu_data[imu_idx,0] < t:
                imu_idx += 1
            while sonar_data[sonar_idx,0] < t:
                sonar_idx += 1

            if imu_idx > 0 and imu_data[imu_idx,0] > t:         imu_idx -= 1
            if sonar_idx > 0 and sonar_data[sonar_idx,0] > t:   sonar_idx -= 1


            #   IMU measurements at current time
            imu_measures = imu_data[imu_idx, 1:7]


            #   --- PROPAGATION WITH HYDRODYNAMIC MODEL (IMU) ----------------------------------
            if imu_idx > 0 and imu_idx != prev_imu_idx:
                #   This assures to only propagate when new IMU data is available
                prev_imu_idx = imu_idx

                for i in range(current_N):
                    #   Add noise to measurements per particle
                    noise = self.L @ np.random.randn(3)
                    gx, gy, gz = imu_measures[0]+noise[0], imu_measures[1]+noise[1], imu_measures[2]+noise[2]
                    omega_noisy[i] = gz
                    ax = imu_measures[3]
                    
                    #   Update internal hydro states
                    u_imu_naive[i] += ax * self.dt
                    u_imu_naive[i] *= 0.98
                    phi[i] += gx * self.dt
                    theta_internal[i] += gy * self.dt
                    
                    #   Compute Body Velocities (u, v)
                    u_b, v_b = self.hydrodynamic_model_2D(gz, phi[i], theta_internal[i], u_imu_naive[i])
                    
                    #   Update global position using heading
                    particles[2,i] = wrap_to_pi(particles[2,i] + gz * self.dt)
                    psi = particles[2,i]
                    
                    x_dot = u_b * np.cos(psi) - v_b * np.sin(psi)
                    y_dot = u_b * np.sin(psi) + v_b * np.cos(psi)
                    
                    particles[0,i] += x_dot * self.dt
                    particles[1,i] += y_dot * self.dt
                    particles[3,i], particles[4,i] = u_b, v_b
                    
                    #   Update history for derivatives
                    p_prev[i], q_prev[i], r_prev[i] = gx, gy, gz

            
            #   --- MEASUREMENT UPDATE (SONAR) ------------------------------------------------
            if sonar_idx > 1 and sonar_idx != prev_sonar_idx:
                #   Only update for NEW sonar frames
                prev_sonar_idx = sonar_idx
                
                #   Expensive image processing
                sonar_raw = self.compute_sonar_diff(     
                    int(sonar_data[sonar_idx-1,1]),
                    int(sonar_data[sonar_idx,1]))
                sonar_body = R_corr @ sonar_raw
                
                #   Mean estimate theta for the transformation
                est_theta = np.arctan2(np.sum(np.sin(particles[2,:]) * weights), 
                                       np.sum(np.cos(particles[2,:]) * weights))

                #   Calculate the Likelihood for each particle
                for i in range(current_N):
                    #   Get particle heading
                    psi = particles[2, i]
                    
                    #   Rotate the sonar_body displacement into the particle's GLOBAL frame
                    R_part = np.array([[np.cos(psi), -np.sin(psi), 0.0],
                                    [np.sin(psi),  np.cos(psi), 0.0],
                                    [0.0,            0.0,           1.0]])
                    
                    #   Observed global displacement according to particle
                    meas_global = (R_part @ sonar_body).reshape(3,1)
                    
                    #   Predicted global displacement according to the hydro model
                    pred_global = np.array([
                        particles[0,i] - prev_particles[0,i],
                        particles[1,i] - prev_particles[1,i],
                        wrap_to_pi(particles[2,i] - prev_particles[2,i])]).reshape(3,1)
                    
                    #   Residual between Observed and Predicted
                    residual = meas_global - pred_global
                    residual[2] = wrap_to_pi(residual[2])
                    
                    #   Weights update (Mahalanobis distance)
                    mahalanobis = residual.T @ self.R_inv @ residual
                    weights[i] *= np.exp(-0.5 * mahalanobis)

                weights += 1.e-10
                weights /= np.sum(weights)
                N_eff = 1.0 / np.sum(weights**2)
                prev_particles = particles.copy()

                #   Re-center Resampling when N_eff is low and 1s has passed
                if N_eff < self.n_particles / 1.5 and count_particle > 20:
                    new_particles = np.zeros_like(particles)

                    #   Define mean velocities to carry over
                    part_vx = np.sum(particles[3, :] * weights)
                    part_vy = np.sum(particles[4, :] * weights)
                    
                    for j in range(current_N):
                        #   Add jitter/noise relative to the L matrix
                        jitter = self.L @ np.random.randn(3)
                        
                        new_particles[0, j] = est_x + jitter[0]
                        new_particles[1, j] = est_y + jitter[1]
                        new_particles[2, j] = wrap_to_pi(est_theta + jitter[2])
                        
                        #   Carry over the velocity so momentum isn't lost
                        new_particles[3, j] = part_vx
                        new_particles[4, j] = part_vy

                    particles = new_particles
                    weights = np.ones(current_N) / current_N
                    count_particle = 0
                    
                    print(f"Time {t:.2f}s | Mean-Centered Resample | N_eff: {N_eff:.1f}")
                else:
                    pass

            #   --- ESTIMATION -----------------------------------------------------------------
            est_x = np.sum(particles[0, :] * weights)
            est_y = np.sum(particles[1, :] * weights)
            
            # Weighted circular mean for theta
            sin_sum = np.sum(np.sin(particles[2, :]) * weights)
            cos_sum = np.sum(np.cos(particles[2, :]) * weights)
            est_theta = np.arctan2(sin_sum, cos_sum)
            count_particle += 1


            #   --- LOGGING --------------------------------------------------------------------
            if t % 1.0 < 1e-5:
                # Retrieve real GPS data
                GPS_time_idx = math.floor(t)
                if GPS_time_idx < len(real_coords):
                    real_act = real_coords[GPS_time_idx, :3]
                    
                    error_act = [(real_act[0] - est_x), (real_act[1] - est_y), wrap_to_pi(real_act[2] - est_theta)]
                    error.append(error_act)
                    estimated_coords.append([est_x, est_y, est_theta])

                    max_error[0] = max(max_error[0], abs(error_act[0]))
                    max_error[1] = max(max_error[1], abs(error_act[1]))
                    max_error[2] = max(max_error[2], abs(error_act[2]))

                    print(f"Time {t:.2f}s | Error: [{error_act[0]:.3f}, {error_act[1]:.3f}, {error_act[2]:.3f}]\n")
            elif t % 0.2 < 1e-5:
                print(f"Time {t:.2f}s | Estimated Position: [{est_x:.3f}, {est_y:.3f}, {est_theta:.3f}]\n")

        return np.array(max_error, dtype=np.float16), np.array(error), real_coords, np.array(estimated_coords)

    
    """
        Function to plot the results of the particle filter localization algorithm. It generates a single
        window with three subplots:
        - ax2 (Heading) and ax3 (Error) in the left column (stacked).
        - ax1 (Trajectory) in the right column, spanning both rows.
    """
    def plot_results(self, real_coords : np.array, estimated_coords: np.array):
        #   Real and Estimated positions are [x, y, theta]
        real_x, real_y, real_theta = real_coords[:, 0], real_coords[:, 1], real_coords[:, 2]
        estimated_x, estimated_y, estimated_theta = estimated_coords[:, 0], estimated_coords[:, 1], estimated_coords[:, 2]

        #   Time vector (seconds)
        N = len(real_coords)
        time = np.arange(N)
        valid_indices = np.where(np.any(estimated_coords[:, 0:3] != 0, axis=1))[0]
        
        if len(valid_indices) == 0 and N > 0:
             print("Warning: Using all data points for error calculation.")
             valid_indices = np.arange(N)

        #   Subset of real and estimated data points
        real_x_valid = real_x[valid_indices]
        real_y_valid = real_y[valid_indices]
        real_theta_valid = real_theta[valid_indices]
        estimated_x_valid = estimated_x[valid_indices]
        estimated_y_valid = estimated_y[valid_indices]
        estimated_theta_valid = estimated_theta[valid_indices]
        time_valid = time[valid_indices]
        
        #   Compute Errors for the valid points
        error_x = real_x_valid - estimated_x_valid
        error_y = real_y_valid - estimated_y_valid
        error_theta = wrap_to_pi(real_theta_valid - estimated_theta_valid)


        #   Create the plot layout using GridSpec (2 rows, 2 columns)
        fig = plt.figure(figsize=(12, 8))
        fig.suptitle('USV Particle Filter Localization Performance', fontsize=16)
        gs = fig.add_gridspec(2, 2, hspace=0.4, wspace=0.2)
        
        ax1 = fig.add_subplot(gs[:, 1])
        ax2 = fig.add_subplot(gs[0, 0])
        ax3 = fig.add_subplot(gs[1, 0])

        #   --- Subplot 1: Trajectory Comparison (X vs Y) ---
        
        #   Real Trajectory (Red)
        ax1.plot(real_x, real_y, label='True Trajectory', color='red', linestyle='-', linewidth=1.5)

        #   Estimated Trajectory (Blue)
        ax1.plot(estimated_x_valid, estimated_y_valid, label='Estimated Trajectory', color='blue', linestyle='--', linewidth=2)
        
        #   Estimated Start/End (Blue)
        ax1.plot(estimated_x_valid[0], estimated_y_valid[0], marker='*', markersize=12, color='blue', label='Est. Start', zorder=5)
        ax1.plot(   estimated_x_valid[-1], estimated_y_valid[-1], marker='*', markersize=12, markerfacecolor='none', \
                    markeredgecolor='blue', label='Est. End', zorder=5)
        
        #   Real Start/End (Red)
        ax1.plot(real_x_valid[0], real_y_valid[0], marker='*', markersize=12, color='red', label='True Start', zorder=5)
        ax1.plot(   real_x_valid[-1], real_y_valid[-1], marker='*', markersize=12, markerfacecolor='none', \
                    markeredgecolor='red', label='True End', zorder=5)
        
        ax1.set_xlabel('X Position (m)')
        ax1.set_ylabel('Y Position (m)')
        ax1.set_title('1. Estimated vs. True USV Trajectory (X vs Y)')
        ax1.legend(loc='upper left')
        ax1.grid(True, linestyle=':', alpha=0.6)
        ax1.set_xlim(left=-15,right=15)
        ax1.set_ylim(bottom=-30,top=40)
        
        #   --- Subplot 2: Heading Comparison (Theta over Time) ---
        
        ax2.plot(time, real_theta, label='True $\\theta$', color='red', linewidth=1.5)
        ax2.plot(time_valid, estimated_theta_valid, label='Estimated $\\theta$', color='blue', linestyle='--', linewidth=2)
        
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Heading $\\theta$ (rad)')
        ax2.set_title('2. Estimated vs. True USV Heading Over Time')
        ax2.legend(loc='best')
        ax2.grid(True, linestyle=':', alpha=0.6)
        ax2.set_xlim(left=0)

        #   --- Subplot 3: Coordinate Errors Over Time ---
        
        ax3.plot(time_valid, error_x, label='Error $e_x$', color='C1', linewidth=2)
        ax3.plot(time_valid, error_y, label='Error $e_y$', color='C2', linewidth=2)
        ax3.plot(time_valid, error_theta, label='Error $e_\\theta$', color='C3', linewidth=2)
        
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Error Magnitude (m,rad)')
        ax3.set_title('3. Estimation Errors for State Coordinates ($e_x, e_y, e_\\theta$)')
        ax3.legend(loc='best')
        ax3.grid(True, linestyle=':', alpha=0.6)
        ax3.set_xlim(left=0)
        plt.tight_layout(rect=[0, 0.05, 1, 0.98])
        plt.show()
    

#   -------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    usv_model = USV_Model(num_particles=100)
    max_error, error, real_coords, estimated_coords = usv_model.particle_filter()

    #   Plot maximum error
    print(f"Maximum registered error (x,y,theta): {max_error}\n")
    length_error = len(error)
    mean_error = np.sum(np.abs(error))/length_error
    std_deviation = np.std(error, ddof=1)

    print(f"Mean error (x,y,theta): {mean_error}\n")
    print(f"Standard deviation (x,y,theta): {std_deviation}\n")

    #   Plot results
    usv_model.plot_results(real_coords,estimated_coords)