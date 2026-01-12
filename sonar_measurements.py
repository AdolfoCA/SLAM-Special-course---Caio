import numpy as np
import os, sys, cv2
import matplotlib.pyplot as plt
from feature_matching import SonarFeatureMatcher
from image_processing import SonarImageProcessor

#   Arrange path to working directory
script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
os.chdir(script_dir)
SONAR_DIR = "./sonar/"

def read_csv(file_path: str) -> np.array:
    return np.loadtxt(file_path, delimiter=',', skiprows=1)

def wrap_to_pi(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

def get_sonar_data(tot_time: float):
    sonar_data = read_csv("normalized_sonar_times.csv")
    sonar_time = sonar_data[:,1]
    sonar_images = sonar_data[:,0].astype(int)
    end_index = np.searchsorted(sonar_time, tot_time, side='left')
    sonar_time = sonar_time[:end_index+1]
    sonar_images = sonar_images[:end_index+1]
    return np.hstack((sonar_time.reshape(-1, 1), sonar_images.reshape(-1, 1)))

def get_transformation(img1, img2) -> np.array:
    match = SonarFeatureMatcher()
    filter = SonarImageProcessor()
    if img1 is None or img2 is None:
        return np.array([0.0, 0.0, 0.0])
    
    img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    img1P = filter.process_image(img1_rgb)
    img2P = filter.process_image(img2_rgb)

    result, _ , _, _  = match.process_sonar_image_pair(img1P, img2P)
    T = result.get('transformation')
    dR = T[0:2,0:2]
    dp = T[0:2,2]
    dx, dy = dp[0], dp[1]
    dtheta_image = np.arctan2(dR[1,0], dR[0,0])
    return np.array([dx, dy, dtheta_image])

def compute_sonar_diff(img1: int, img2: int) -> np.array:
    img1_path = SONAR_DIR + str(img1) + ".png"
    img2_path = SONAR_DIR + str(img2) + ".png"
    return get_transformation(cv2.imread(img1_path), cv2.imread(img2_path))

if __name__ == "__main__":
    total_time = 231
    sonar_data = get_sonar_data(total_time)
    time_steps = np.linspace(0, total_time, int((total_time + 0.05) / 0.05))

    sonar_position = np.zeros((3, 1))
    sonar_idx = 0
    
    # Storage for plotting
    history = []
    time_history = []
    R_corr = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
    last_idx = -1

    print("Processing Sonar Odometry...")
    for t in time_steps:
        while sonar_idx < len(sonar_data) and sonar_data[sonar_idx, 0] < t:
            sonar_idx += 1
        if sonar_idx > 0 and sonar_idx < len(sonar_data) and sonar_data[sonar_idx, 0] > t:
            sonar_idx -= 1
        
        if sonar_idx > 1 and sonar_idx != last_idx:
            sonar_measures = compute_sonar_diff(     
                int(sonar_data[sonar_idx-1,1]),
                int(sonar_data[sonar_idx,1])
            ).reshape(3,1)

            last_idx = sonar_idx

            # Rotation logic provided in your prompt
            theta = sonar_position[2,0]
            R_matrix = np.array([[np.cos(theta), -np.sin(theta), 0.0],
                                [np.sin(theta),  np.cos(theta), 0.0],
                                [0.0,            0.0,           1.0]])
            
            # Applying the Pi rad (X) and -Pi/2 rad (Z) correction matrix
            R = R_matrix @ R_corr
            
            sonar_position += R @ sonar_measures
            sonar_position[2,0] = wrap_to_pi(sonar_position[2,0])
            print(f"Time {t:.2f}s | Sonar Position: [{sonar_position[0,0]:.3f}, {sonar_position[1,0]:.3f}, {sonar_position[2,0]:.3f}]")
            
            history.append(sonar_position.flatten().copy())
            time_history.append(t)


    # --- PLOTTING SECTION ---
    history = np.array(history)
    time_history = np.array(time_history)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Sonar Odometry Dead Reckoning', fontsize=16)

    # Subplot 1: XY Trajectory
    ax1.plot(history[:, 0], history[:, 1], b'-', label='Sonar Path')
    ax1.plot(history[0, 0], history[0, 1], 'go', label='Start')
    ax1.plot(history[-1, 0], history[-1, 1], 'ro', label='End')
    ax1.set_xlabel('X Position (m)')
    ax1.set_ylabel('Y Position (m)')
    ax1.set_title('2D Trajectory')
    ax1.legend()
    ax1.grid(True)
    ax1.axis('equal')

    # Subplot 2: Heading (Theta) over time
    ax2.plot(time_history, history[:, 2], color='orange')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Heading (rad)')
    ax2.set_title('Heading Over Time')
    ax2.grid(True)

    plt.tight_layout()
    plt.show()