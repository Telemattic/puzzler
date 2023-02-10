import glob
import numpy as np
import cv2 as cv

def pattern_points(pattern_size, pattern_kind):

    assert pattern_kind in ('chessboard', 'circles_grid', 'asymmetric_circles_grid')
    points = np.zeros((np.prod(pattern_size), 3), np.float32)
    points[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)

    if pattern_kind == 'asymmetric_circles_grid':
        points[:,0] += 0.5 * (1 - (points[:,1] % 2))
        points[:,1] *= .5

    return points

pattern_size = (4, 11)
pattern_kind = 'asymmetric_circles_grid'

print(pattern_points(pattern_size, pattern_kind))

objpoints = []
imgpoints = []

params = cv.SimpleBlobDetector_Params()

params.minThreshold = 1
params.maxThreshold = 255

# params.filterByConvexity = True
# params.minConvexity = 0.4

params.filterByArea = True
params.minArea = 5000
params.maxArea = 100000

params.filterByInertia = True
params.minInertiaRatio = 0.5

params.filterByCircularity = True
params.minCircularity = 0.8

params.minDistBetweenBlobs = 7
    
detector = cv.SimpleBlobDetector_create(params)

for i, path in enumerate(glob.glob(r"WIN_*.jpg")):

    print(path)
    img = cv.imread(path)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    keypoints = detector.detect(gray)
    keypoint_img = cv.drawKeypoints(img, keypoints, np.array([]), (0,0,255), cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    path = f"keypoint_{i}.jpg"

    print(f"{path}: {len(keypoints)} keypoints")
    for j, kp in enumerate(keypoints):
        x, y = kp.pt
        print(f" {j:2d}: pt={x:4.0f},{y:4.0f} size={kp.size:5.1f}")
        
    cv.imwrite(path, keypoint_img)
    
    h, w = img.shape[0], img.shape[1]
    flags = cv.CALIB_CB_ASYMMETRIC_GRID + cv.CALIB_CB_CLUSTERING
    found, corners = cv.findCirclesGrid(gray, pattern_size, flags=flags, blobDetector=detector)
    if found:
        print("  target found")
        objpoints.append(pattern_points(pattern_size, pattern_kind))
        imgpoints.append(corners)
    else:
        print("  target not found :(")
    
rms, camera_matrix, distortion_coefficients, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, (w,h), None, None)

print(f"{rms=}")
print(f"{camera_matrix=}")
print(f"{distortion_coefficients.ravel()=}")
print(f"{rvecs=}")
print(f"{tvecs=}")

for i, path in enumerate(glob.glob(r"WIN_*.jpg")):

    print(path)
    img = cv.imread(path)
    
    h, w = img.shape[0], img.shape[1]

    new_camera_matrix, roi = cv.getOptimalNewCameraMatrix(camera_matrix, distortion_coefficients, (w, h), 1, (w, h))
    undistorted_img = cv.undistort(img, camera_matrix, distortion_coefficients, None, new_camera_matrix)

    outpath = f"undistorted_{i}.jpg"

    print(f"  {roi=}")
    print(f"  {outpath=}")
    
    cv.imwrite(outpath, undistorted_img)

exit(0)
        

path = r"C:\temp\fnord.jpg"
img = cv.imread(path, 0)

pattern_size = (9, 9)
found, corners = cv.findChessboardCorners(img, pattern_size)

print(f"{found=} {np.squeeze(corners)=}")

if found:
    term = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_COUNT, 30, 0.1)
    corners1 = corners.copy()
    corners2 = cv.cornerSubPix(img, corners, (11, 11), (-1, -1), term)
    print(np.squeeze(corners1-corners2))
    corners = corners2
    
img = cv.imread(path)
cv.drawChessboardCorners(img, pattern_size, corners, found)
cv.imwrite(r"C:\temp\fnord2.jpg", img)
