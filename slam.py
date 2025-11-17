import numpy as np
from feature_matching import SonarFeatureMatcher
from image_processing import SonarImageProcessor
import scipy.sparse as sp
import cv2


#   Function defined for a Factor Graph implementation 
def dense_2_sp_lists(M: np.array, tl_row : int, tl_col: int, row_vec=True):
    '''
    This function takes in a dense matrix (L) and turns it into a flat array.
    Corresponding to that array are row and column entries with
    tl_row and tl_col giving the top left of all the entries
    
    Inputs:
        M : The dense to matrix to convert (np.array)
        tl_row: the top row for the matrix (int)
        tl_col: the left-most column for the matrix (int)
        row_vec:  In the corner case where M is 1d, should 
                it be a row or column vector?
        
    Returns:  a tuple with 3 lists (np.array) in it
    '''
    data_list = M.flatten()
    if len(M.shape)==2:
        rows,cols = M.shape
    elif len(M.shape)==1:
        if row_vec:
            rows=1
            cols=len(M)
        else:
            cols=1
            rows=len(M)
    else:
        assert False, 'M must be 1d or 2d!'
    row_list = np.zeros(len(data_list))
    col_list = np.zeros(len(data_list))
    for i in range(rows):
        for j in range(cols):
            idx = i*cols + j
            row_list[idx] = i+tl_row
            col_list[idx] = j+tl_col
    return (data_list,row_list,col_list)


class USV_Model:
    #   Initialization method still to be implemented
    def __init__(self):
        return
    

    """
        In this class, the F and H matrices must be correctly defined, and the approach to deal with this
        problem is the following (idealized):

        1. Compute positions with particle filter up to an M iteration
        2. Run Factor Graph trajectory optimization (over robot and landmarks' positions) - EKF based
        3. Resample particles with equal weight around final position of the FG optimized trajectory
        4. Repeat process (each M iterations)
    """


    def create_L(self):
        '''
        This creates the big matrix (L) given the current state of the whole system
        '''
        # First, determine how many non zero entries (nnz_entries) will be in the
        # sparse matrix. Then create the 3 parallel arrays that will be used to
        # form this matrix
        H_size = 4
        F_size=16 # Should be state size**2
        nnz_entries = 2*F_size*(self.N-1) + H_size*self.N
        data_l = np.zeros(nnz_entries)
        row_l = np.zeros(nnz_entries,dtype=int)
        col_l = np.zeros(nnz_entries,dtype=int)
        t_e = 0 #total number of entries so far
        # Put all the dynamics entries into L
        for i in range(1,self.N):
            mat1 = self.S_Q_inv.dot(self.F_mat(self.states[i-1]))
            data_l[t_e:t_e+F_size], row_l[t_e:t_e+F_size], col_l[t_e:t_e+F_size] = \
                dense_2_sp_lists(mat1,self.dyn_idx(i),self.state_idx(i-1))
            mat2 = -self.S_Q_inv
            t_e +=F_size
            data_l[t_e:t_e+F_size], row_l[t_e:t_e+F_size], col_l[t_e:t_e+F_size] = \
                dense_2_sp_lists(mat2,self.dyn_idx(i),self.state_idx(i))
            t_e +=F_size
        
        # Now do the measurements
        for i in range(self.N):
            # for S_R_inv a scalar
            mat = self.S_R_inv*self.H_mat(self.states[i])
            data_l[t_e:t_e+H_size], row_l[t_e:t_e+H_size], col_l[t_e:t_e+H_size] = \
                dense_2_sp_lists(mat,self.meas_idx(i),self.state_idx(i))
            t_e += H_size
        
        return sp.csr_matrix((data_l,(row_l,col_l)))



#   -----------------------------------------------------------------------------------------

"""
    Retrieves the transformation between two images (dx,dy,d_theta)
"""
def get_transformation(img1: cv2.typing.MatLike, img2: cv2.typing.MatLike) -> np.array:
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

        result, kp1,kp2, matches  = match.process_sonar_image_pair(img1P, img2P)
        T = result.get('transformation')
        dR = T[0:2,0:2]
        dp = T[0:2,2]
        # in the image frame
        dx = dp[0]                                      # forward
        dy = dp[1]                                      # right
        dtheta_image = np.arctan2(dR[1,0], dR[0,0])     # positive ccw from x axis

        return np.array([dx,dy,dtheta_image])
    

DIR = "./sonar/"
total_images = 3488

for i in range(total_images):
    img1_path = DIR + str(i) + ".png"
    img2_path = DIR + str(i+1) + ".png"

    img1 = cv2.imread(str(img1_path))
    img2 = cv2.imread(str(img2_path))

    #   Retrieve transforming
    arr = get_transformation(img1,img2)
    print (f"dtheta: {arr[2]:.3f}")
    print (f"forward  (m): {arr[0]:.3f}")
    print (f"right   (m): {arr[1]:.3f}")
    print (f"---")