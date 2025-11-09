import numpy
from feature_matching import SonarFeatureMatcher
from image_processing import SonarImageProcessor


img1_path = DIR / "280.png"
img2_path = DIR / "281.png"

match = SonarFeatureMatcher()
filter = SonarImageProcessor()

img1 = cv2.imread(str(img1_path))
img2 = cv2.imread(str(img2_path))


#  Check if images loaded successfully
if img1 is None or img2 is None:
    print("Error: One or more images failed to load. Check file paths.")
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
    dx = dp[0] # forward
    dy = dp[1] # right
    dtheta_image = np.arctan2(dR[1,0], dR[0,0]) # positive ccw from x axis

    print (f"dtheta: {dtheta_image:.3f}")
    print (f"forward  (m): {dx:.3f}")
    print (f"right   (m): {dy:.3f}")
