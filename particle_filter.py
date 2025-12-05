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


def read_csv(file_path: str) -> np.array:
    """
    Reads a CSV file and returns its contents as a NumPy array.
    
    Args:
        file_path (str): The path to the CSV file.
    """
    return np.loadtxt(file_path, delimiter=',', skiprows=1)


def wrap_to_pi(angle):
    """Wraps an angle in rad to the range (-pi, pi]"""
    return (angle + np.pi) % (2 * np.pi) - np.pi


"""
    In this class, the F and H matrices must be correctly defined, and the approach to deal with this
    problem is the following (idealized):

    1. Compute positions with particle filter up to an M iteration
    2. Run Factor Graph trajectory optimization (over robot and landmarks' positions) - EKF based
    3. Resample particles with equal weight around final position of the FG optimized trajectory
    4. Repeat process (each M iterations)
"""
class USV_Model:
    def __init__(self, num_particles: int):
        self.n_particles = num_particles

        #   Factor Graph related parameters (second part related - mapping)
        #self.M_iterations = M_iterations
        #self.max_landmarks = max_landmarks
        
        #   Definition: EKF Matrices for the PF-FG Approach -----------------------------------------

        #   Covariance matrices for state estimation
        self.P_IMU = None
        self.P_sonar = None
        self.P = np.eye(6)

        #   Process covariance matrices
        self.Q_IMU = np.diag([2.0, 2.0, 2.0])
        self.Q_sonar = np.diag([1.5, 1.5, 1.5])
        self.dt = 0.1
        self.sonar_dt = 0.0667
        self.IMU_dt = 0.01

        ##  H is defined as an identity matrix for this case (both sonar and IMU)
        self.F = np.array([[1.0, 0.0, 0.0, self.dt, 0.0, 0.0],
             [0.0, 1.0, 0.0, 0.0, self.dt, 0.0],
             [0.0, 0.0, 1.0, 0.0, 0.0, self.dt],
             [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
             [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
             [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
        
        self.G_IMU = np.array([[0.5*self.dt**2, 0.0, 0.0],
                 [0.0, 0.5*self.dt**2, 0.0],
                 [0.0, 0.0, self.dt],
                 [self.dt, 0.0, 0.0],
                 [0.0, self.dt, 0.0],
                 [0.0, 0.0, 1.0]])
        
        self.G_sonar = np.array([[1.0, 0.0, 0.0],
                   [0.0, 1.0, 0.0],
                   [0.0, 0.0, 1.0],
                   [0.0, 0.0, 0.0],
                   [0.0, 0.0, 0.0],
                   [0.0, 0.0, 0.0]])

        #   Measurement noise covariances for GPS/INS
        self.R = np.eye(6)
        self.R_inv = np.linalg.inv(self.R)

        self.Q_combined =   self.G_IMU @ self.Q_IMU @ self.G_IMU.T + \
                            self.G_sonar @ self.Q_sonar @ self.G_sonar.T
        
        #   Enforce Symmetry (Crucial for numerical stability)
        self.Q_combined = 0.5 * (self.Q_combined + self.Q_combined.T) 
        
        #   Add a small epsilon to the diagonal to ensure strict Positive Definiteness
        epsilon = 1e-12 
        self.Q_combined += np.eye(self.Q_combined.shape[0]) * epsilon

        #   Get the lower triangular matrix L for noise sampling in PF (Cholesky decomposition)
        self.L_combined = np.linalg.cholesky(self.Q_combined)
        
        return
    

    """
        Function to compute real USV coordinates given from real testing. Those coordinates are given
        by GPS/INS system and will be used for particle filter evaluation.
        
        The normalized heading file defines the angle of the USV with respect to the North direction
        (N = 0 degrees, E = 90 degrees, S = 180 degrees, W = 270 degrees); and the normalized fix file
        gives the latitude, longitude and altitude (not used) of the USV at each time step.

        For this case, the ABSOLUTE approach was employed, which compares the current GPS measurements
        with the initial known position and therefore avoiding drift issues.
    """
    def compute_real_coords(self) -> np.array:
        heading = read_csv(GPS_DIR + "normalized_heading.csv")          #   In degrees
        coordinates = read_csv(GPS_DIR + "normalized_fix.csv")          #   lat, long, alt

        #   Convert time instants to integers (s)
        time = heading[:,0]
        heading = heading[:,1] * (np.pi/180)                    #   Convert to rad
        coordinates = coordinates[:,1:3] * (np.pi/180)          #   Remove time column

        #   Real positions = [x, y, theta]
        real_positions = np.zeros((len(time),6))

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
        real_positions[0,5] = 0.0

        # Convert to float for accurate division
        for i in range(len(time) - 1):
            delta_t = time[i+1] - time[i]

            # --- Compute velocities (vx, vy, omega) by dividing by delta_t ---
            real_positions[i+1,3] = (real_positions[i+1,0] - real_positions[i,0]) / delta_t
            real_positions[i+1,4] = (real_positions[i+1,1] - real_positions[i,1]) / delta_t
            real_positions[i+1,5] = (real_positions[i+1,2] - real_positions[i,2]) / delta_t

        return  real_positions
    

    """
        Function to get IMU data (angular velocities and linear accelerations) from CSV file.
        The values for [ax, ay, gz] are given in the IMU frame of reference.
    """
    def get_IMU_data(self, tot_time: float):
        imu_data = read_csv(IMU_DIR + "normalized_imu.csv")     #   time, gx, gy, gz, ax, ay, az
        imu_time = imu_data[:,0]
        imu_data = imu_data[:,3:6]                              #   gz, ax, ay

        #   Find index for when simulation finishes and slice data accordingly
        end_index = np.searchsorted(imu_time, tot_time, side='left')
        imu_time = imu_time[:end_index+1]
        imu_data = imu_data[:end_index+1,:]

        time_column = imu_time.reshape(-1, 1)
        stacked_matrix = np.hstack((time_column, imu_data))
        return stacked_matrix
    

    """
        Function to retrieve sonar data (time, dx, dy, d_theta) from sonar images.
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
        Retrieves the transformation between two images (dx,dy,d_theta)
    """
    def get_transformation(self, img1: cv2.typing.MatLike, img2: cv2.typing.MatLike) -> np.array:
        match = SonarFeatureMatcher()
        filter = SonarImageProcessor()

        #  Check if images loaded successfully
        if img1 is None or img2 is None:
            print("Error: One or more images failed to load. Check file paths.")
            return ValueError
        else:
            # Convert BGR to RGB for matplotlib display
            img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
            img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

            img1P = filter.process_image(img1_rgb)
            img2P = filter.process_image(img2_rgb)

            result, _ , _, _  = match.process_sonar_image_pair(img1P, img2P)
            T = result.get('transformation')
            dR = T[0:2,0:2]
            dp = T[0:2,2]
            # in the image frame
            dx = dp[0]                                      # forward
            dy = dp[1]                                      # right
            dtheta_image = np.arctan2(dR[1,0], dR[0,0])     # positive ccw from x axis

            return np.array([dx,dy,dtheta_image])
    

    """
        Function to compute sonar increments (dx, dy, d_theta) between sonar image pairs.
    """
    def compute_sonar_diff(self, img1: int, img2: int) -> np.array:
        img1_path = SONAR_DIR + str(img1) + ".png"
        img2_path = SONAR_DIR + str(img2) + ".png"

        img1 = cv2.imread(str(img1_path))
        img2 = cv2.imread(str(img2_path))

        #   Retrieve transforming
        arr = self.get_transformation(img1,img2)
        return arr
    

    """
        Function to compute IMU increments over a time interval dt.
    """
    def compute_imu_diff(self, imu_data) -> np.array:
        #   Returns [ax, ay, gz]
        return np.array([imu_data[1], imu_data[2], imu_data[0]])
    

    """
        Implements the Low Variance Resampling (LVR) algorithm.
        This method is preferred for its low variance and O(N) complexity.
        
        Args:
            weights (np.array): Normalized particle weights (sum to 1).
            
        Returns:
            np.array: Indices of the selected particles (n_particles long).
    """
    def low_variance_resampling(self, weights):
        N = self.n_particles
        indices = np.zeros(N, dtype=int)
        
        # 1. Choose a random starting point (r0)
        r0 = np.random.uniform(0, 1 / N)
        
        # 2. Build the cumulative sum of weights
        C = np.cumsum(weights)
        
        # 3. Perform the sampling sweep
        i = 0  # index of the particle to select from
        j = 0  # index of the resampled particle
        
        # U_j = r0 + (j - 1)/N (where j is 1-indexed)
        # Simplified: U_j = r0 + j/N
        while j < N:
            # Check if the current pointer (r0 + j/N) crosses the current particle's cumulative weight
            u_j = r0 + j / N
            if u_j <= C[i]:
                # If below the cumulative weight, select particle i
                indices[j] = i
                j += 1
            else:
                # If above, move to the next particle (i+1)
                i += 1
                
        return indices
    

    """
        Function to implement a particle filter localization algorithm for the USV based on the
        odometry (from sonar image and IMU data) and GPS/INS data (for evaluation).

        The fusion approach used for the corrected PF prediction step is Combined Stochastic Motion Model,
        which considers both IMU and Sonar data for motion updates in a Monte Carlo framework.
    """
    def particle_filter(self):
        real_coords = self.compute_real_coords()        #   Get real coordinates for evaluation
        estimated_coords = np.zeros_like(real_coords)   #   To store estimated positions
        
        #   NEW: Three lines
        actual_coords = real_coords[0]
        IMU_coords = actual_coords[:, np.newaxis]
        sonar_coords = actual_coords[:, np.newaxis]

        #   Simulation parameters
        T = len(real_coords) - 1
        time = np.linspace(0, T, int((T+self.dt)/self.dt))
        N = len(time)

        particle_states = np.zeros((len(real_coords[0]),self.n_particles))

        particle_weights = np.ones((self.n_particles,)) / self.n_particles
        particle_states[0,:] = real_coords[0,0] + np.random.randn(self.n_particles)
        particle_states[1,:] = real_coords[0,1] + np.random.randn(self.n_particles)
        particle_states[2,:] = real_coords[0,2] + np.random.randn(self.n_particles)
        particle_states[3:6,:] = 0.0

        #   Get data before starting the filter
        imu_data = self.get_IMU_data(T)
        sonar_data = self.get_sonar_data(T)

        imu_idx = 0
        sonar_idx = 0
        max_error = 0.0

        for i in range(N):
            t = time[i]

            #   Advance (IMU,Sonar) index until its time is >= current time t
            #   Then after, reduce 1 unit so as to obtain value before given time
            while imu_data[imu_idx,0] < t:
                imu_idx += 1
            while sonar_data[sonar_idx,0] < t:
                sonar_idx += 1

            if imu_idx > 0 and imu_data[imu_idx,0] > t:         imu_idx -= 1
            if sonar_idx > 0 and sonar_data[sonar_idx,0] > t:   sonar_idx -= 1
            
            #   ---------------------------------
            #   Obtain diffs for sonar and IMU
            if sonar_idx > 1:
                sonar_measures = self.compute_sonar_diff(                                   #   [dx, dy, d_theta]        
                    int(sonar_data[sonar_idx-1,1]),
                    int(sonar_data[sonar_idx,1])
                )
            else:
                sonar_measures = np.array([0.0, 0.0, 0.0])

            imu_measures = self.compute_imu_diff(imu_data[imu_idx,1:4])                     #   [ax, ay, gz]


            # --- DYNAMIC UPDATE ----------------------------------------------------------------------------

            IMU_coords_prev = IMU_coords.copy()
            sonar_coords_prev = sonar_coords.copy()

            IMU_coords = self.F @ IMU_coords + self.G_IMU @ imu_measures[:, np.newaxis]
            sonar_coords = self.F @ sonar_coords + self.G_sonar @ sonar_measures[:, np.newaxis]
            IMU_coords[2] = wrap_to_pi(IMU_coords[2])
            sonar_coords[2] = wrap_to_pi(sonar_coords[2])

            #   Covariance calculations (IMU & sonar)
            term = self.F @ self.P @ self.F.T
            self.P_IMU = term + self.G_IMU @ self.Q_IMU @ self.G_IMU.T
            self.P_sonar = term + self.G_sonar @ self.Q_sonar @ self.G_sonar.T

            #   FUSION (Combine IMU and Sonar estimates to get the "Actual" position)
            P_IMU_inv = np.linalg.pinv(self.P_IMU)
            P_sonar_inv = np.linalg.pinv(self.P_sonar)
            P_actual_inv = P_IMU_inv + P_sonar_inv
            self.P = np.linalg.pinv(P_actual_inv)
            
            #   Fused X = P_actual * (P_IMU_inv * IMU_coords + P_sonar_inv * sonar_coords)
            actual_coords = (   self.P @ 
                                (P_IMU_inv @ IMU_coords + 
                                P_sonar_inv @ sonar_coords)).flatten()
            
            actual_coords[2] = wrap_to_pi(actual_coords[2])
            

            # --- 1. PREDICTION STEP ------------------------------------------------------------------------

            #   Delta between current actual state and the previous average state (approximates the deterministic motion)
            delta_x = ((IMU_coords - IMU_coords_prev) + (sonar_coords - sonar_coords_prev)) / 2

            #   Apply motion and add process noise to particles
            delta_x_matrix = delta_x * np.ones((6, self.n_particles))
            particle_states += delta_x_matrix

            #   Add stochastic process noise (w_k^(i))
            standard_normal_noise = np.random.randn(particle_states.shape[0], self.n_particles)
            process_noise = self.L_combined @ standard_normal_noise

            #   Apply the noise to spread particles
            particle_states += process_noise 

            #   Heading Wrap (Always done after motion update)
            particle_states[2,:] = wrap_to_pi(particle_states[2,:])


            # --- 2. CORRECTION (WEIGHT UPDATE / GPS MEASUREMENT) -------------------------------------------
            
            #   Work with the GPS measurement at time t (or most recent)
            GPS_time_idx = math.floor(t)
            coords_GPS = real_coords[GPS_time_idx][:, np.newaxis]

            #   Measurement residual (error)
            innovation = coords_GPS - particle_states
            
            #   Calculate likelihood (Probability Density Function)
            #   Assuming a Gaussian (Normal) distribution for the measurement noise R
            mahalanobis_sq = innovation.T @ self.R_inv @ innovation
            exponent = -0.5 * np.diag(mahalanobis_sq)

            #   L = 1 / (sqrt(2*pi*|R|)) * exp(exponent)
            likelihood = np.exp(exponent)
            
            #   Update weight: w_new = w_old * likelihood
            particle_weights *= likelihood
        
            
            # --- 3. RESAMPLING (Only perform if needed) ----------------------------------------------------

            #   Check for zero weights before normalizing to avoid division by zero.
            sum_weights = np.sum(particle_weights)
            
            if sum_weights == 0:
                #   Filter lost track due to underflow. Re-initialize weights to uniform
                print(f"Warning: Particle weights sum to zero at time {t}. Re-initializing weights to uniform.")
                particle_weights.fill(1.0 / self.n_particles)
                N_eff = self.n_particles # Max N_eff when weights are uniform
            else:
                #   Normalize weights
                particle_weights /= sum_weights
                
                #   Calculate effective number of particles (N_eff)
                N_eff = 1.0 / np.sum(particle_weights**2)

            
            #   Resample only if the variance is too high (N_eff < threshold)
            if N_eff < self.n_particles / 1.4:
                #   Perform Low Variance Resampling (or any preferred method)
                indices = self.low_variance_resampling(particle_weights)
                
                #   Replace old particles with new set based on indices
                particle_states = particle_states[:, indices]
                particle_weights.fill(1.0 / self.n_particles)


            # --- 4. ESTIMATE STATE -------------------------------------------------------------------------
            
            #   The final state estimate is the weighted mean of all particles
            #   Change conditions based on self.dt value!!
            actual_coords = np.sum(particle_states * particle_weights, axis=1)
            actual_coords[2] = wrap_to_pi(actual_coords[2])

            IMU_coords = actual_coords[:,np.newaxis]
            sonar_coords = actual_coords[:,np.newaxis]
            error = actual_coords[:3] - real_coords[int(t), :3]
            max_error = np.maximum(max_error,np.max(np.abs(error)))
            if (i % int(1/self.dt) == 0):
                estimated_coords[int(t), :] = actual_coords
                print(f"Time {t:.2f}s: Added Position to estimation vector : position {int(t)} | Error: {error}\n")
            elif (i % int(0.2/self.dt) == 0):
                actual_coords = np.sum(particle_states * particle_weights, axis=1)
                print(f"Time {t:.2f}s: Estimated Pos = {actual_coords} | True Pos = {real_coords[GPS_time_idx]}\n")

        return max_error, real_coords, estimated_coords

    
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
        
        # Ax1: Trajectory (Spans all rows in the 2nd column)
        ax1 = fig.add_subplot(gs[:, 1])
        
        # Ax2: Heading (1st row, 1st column)
        ax2 = fig.add_subplot(gs[0, 0])
        
        # Ax3: Errors (2nd row, 1st column)
        ax3 = fig.add_subplot(gs[1, 0])


        # --- Subplot 1: Trajectory Comparison (X vs Y) ---
        
        # Real Trajectory (Red)
        ax1.plot(real_x, real_y, label='True Trajectory', color='red', linestyle='-', linewidth=1.5)
        # Estimated Trajectory (Blue)
        ax1.plot(estimated_x_valid, estimated_y_valid, label='Estimated Trajectory', color='blue', linestyle='--', linewidth=2)
        
        # Start/End Markers
        # Estimated Start/End (Blue)
        ax1.plot(estimated_x_valid[0], estimated_y_valid[0], marker='*', markersize=12, color='blue', label='Est. Start', zorder=5)
        ax1.plot(   estimated_x_valid[-1], estimated_y_valid[-1], marker='*', markersize=12, markerfacecolor='none', \
                    markeredgecolor='blue', label='Est. End', zorder=5)
        # Real Start/End (Red)
        ax1.plot(real_x_valid[0], real_y_valid[0], marker='*', markersize=12, color='red', label='True Start', zorder=5)
        ax1.plot(   real_x_valid[-1], real_y_valid[-1], marker='*', markersize=12, markerfacecolor='none', \
                    markeredgecolor='red', label='True End', zorder=5)
        
        ax1.set_xlabel('X Position (m)')
        ax1.set_ylabel('Y Position (m)')
        ax1.set_title('1. Estimated vs. True USV Trajectory (X vs Y)')
        ax1.legend(loc='upper left')
        ax1.grid(True, linestyle=':', alpha=0.6)
        ax1.set_xlim(left=-15,right=15)
        ax1.set_ylim(bottom=-20,top=40)
        
        
        # --- Subplot 2: Heading Comparison (Theta over Time) ---
        
        ax2.plot(time, real_theta, label='True $\\theta$', color='red', linewidth=1.5)
        ax2.plot(time_valid, estimated_theta_valid, label='Estimated $\\theta$', color='blue', linestyle='--', linewidth=2)
        
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Heading $\\theta$ (rad)')
        ax2.set_title('2. Estimated vs. True USV Heading Over Time')
        ax2.legend(loc='best')
        ax2.grid(True, linestyle=':', alpha=0.6)
        ax2.set_xlim(left=0)


        # --- Subplot 3: Coordinate Errors Over Time ---
        
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
    max_error, real_coords, estimated_coords = usv_model.particle_filter()

    #   Plot maximum error
    print(f"Maximum registered error (x,y,theta): {max_error}\n")

    #   Plot results
    usv_model.plot_results(real_coords,estimated_coords)