import cv2 as cv
import math
import numpy as np

class CornerDetector:

    def __init__(self, charuco_board):
        self.charuco_board = charuco_board

    def detect_corners(self, input_image):
        charuco_dict = self.charuco_board.getDictionary()
        
        charuco_params = cv.aruco.CharucoParameters()
        charuco_params.minMarkers = 2

        detector_params = cv.aruco.DetectorParameters()
        detector_params.cornerRefinementWinSize = 10
        detector_params.relativeCornerRefinmentWinSize = 0.5
        
        detector = cv.aruco.CharucoDetector(
            self.charuco_board, charucoParams=charuco_params, detectorParams=detector_params)
        charuco_corners, charuco_ids, marker_corners, marker_ids = detector.detectBoard(input_image)

        if charuco_corners is None or len(charuco_corners) < 2:
            return None, None

        uv = np.squeeze(charuco_corners)
        ij = self.get_ij_for_ids(np.squeeze(charuco_ids))

        return uv, ij
    
    def effective_dpi(self, corners, corner_ids):
        
        lookup = {ij: xy for ij, xy in zip(corner_ids, corners)}
        retval = {}

        # conversion from [pixels/square] to DPI:
        # 25.4 [mm/in] / squareLength [mm/sq]
        scale = 25.4 / self.charuco_board.getSquareLength()
        
        for ij, xy0 in lookup.items():
            i, j = ij
            dists = []
            for neighbor in [(i-1,j), (i+1,j), (i,j-1), (i,j+1)]:
                xy1 = lookup.get(neighbor)
                if xy1 is not None:
                    dists.append(xy1-xy0)
            if dists:
                retval[ij] = np.median(np.linalg.norm(np.array(dists), axis=1)) * scale

        return retval

    def get_ij_for_ids(self, ids):
        n_cols, n_rows = self.charuco_board.getChessboardSize()
        return [(k % (n_cols-1), k // (n_cols-1)) for k in ids]

class CameraCalibrator:

    def __init__(self, charuco_board):
        self.charuco_board = charuco_board

    def get_euler_angles(self, object_points, image_points, camera_matrix, dist_coeffs):

        ret, rvec, tvec = cv.solvePnP(
            object_points, image_points, camera_matrix, dist_coeffs)

        if not ret:
            return None

        # get the rotation matrix
        m = cv.Rodrigues(rvec)[0]

        # convert to Euler angles
        pitch = math.atan2(-m[1,2], m[2,2])
        yaw = math.asin(m[0,2])
        roll = math.atan2(-m[0,1], m[0,0])

        return np.array((pitch, yaw, roll)) * (180./math.pi)

    def get_object_points_for_ij(self, obj_ij):
        obj_xy = np.array(obj_ij) * self.charuco_board.getSquareLength()
        obj_z = np.zeros((len(obj_xy), 1), obj_xy.dtype)
        obj_xyz = np.hstack((obj_xy, obj_z), dtype=np.float32)
        return obj_xyz

    def calibrate_camera(self, corners, ids, image):

        # print(f"effective_dpi={self.effective_dpi(corners, ids)}")

        img_xy = corners
        obj_ij = ids

        obj_xyz = self.get_object_points_for_ij(obj_ij)

        object_points = [obj_xyz]
        image_points = [img_xy]
        image_size = (image.shape[1], image.shape[0])
        # flags for fancier models
        # cv.CALIB_RATIONAL_MODEL
        # cv.CALIB_THIN_PRISM_MODEL
        # cv.CALIB_TILTED_MODEL
        flags = cv.CALIB_FIX_ASPECT_RATIO
        input_camera_matrix = np.eye(3)
        
        ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv.calibrateCamera(
            object_points, image_points, image_size, input_camera_matrix,
            None, flags=flags)

        with np.printoptions(precision=6):
            print(f"{ret=}")
            print(f"{camera_matrix=}")
            print(f"{dist_coeffs=}")
            print(f"{rvecs=}")
            print(f"{tvecs=}")

        # alpha: fraction of sensor to take, 0=only "good" pixels, 1=everything
        alpha = 0
        new_camera_matrix, roi = cv.getOptimalNewCameraMatrix(
            camera_matrix, dist_coeffs, image_size, alpha, image_size)

        with np.printoptions(precision=3):
            print(f"{new_camera_matrix=}")
            print(f"{roi=}")

        return CalibratedCamera(camera_matrix, dist_coeffs, new_camera_matrix, image_size, roi, rvecs[0], tvecs[0])

class CalibratedCamera:

    def __init__(self, camera_matrix, dist_coeffs, new_camera_matrix, image_size, roi, rvec, tvec):
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.new_camera_matrix = new_camera_matrix
        self.image_size = image_size
        self.roi = roi
        self.rvec = rvec
        self.tvec = tvec

        # see opencv-4.10.0/modules/core/include/opencv2/core/hal/interface.h
        CV_32FC1 = 5
        CV_16UC2 = 10
        CV_16SC2 = 11

        self.x_map, self.y_map = cv.initUndistortRectifyMap(
            self.camera_matrix, self.dist_coeffs, None, self.new_camera_matrix, self.image_size, CV_16SC2)

    def undistort_image(self, image, interpolation=cv.INTER_LINEAR):
        return cv.remap(image, self.x_map, self.y_map, interpolation, borderValue=(255,255,0))

    def undistort_points(self, img_xy):
        return np.squeeze(cv.undistortPoints(
            img_xy, self.camera_matrix, self.dist_coeffs, None, self.new_camera_matrix))

class PerspectiveComputer:

    def __init__(self, charuco_board):
        self.charuco_board = charuco_board

    def compute_homography(self, image):
        
        corners, corner_ids = CornerDetector(self.charuco_board).detect_corners(image)

        n_corners = 0 if corners is None else len(corners)
        if n_corners < 4:
            print(f"PerspectiveComputer: found {n_corners} corners, aborting.")
            return None
        
        src_points = corners

        image_size = (image.shape[1], image.shape[0])
        
        dists = np.linalg.norm(src_points - np.array(image_size) * .5, axis=1)
        index_center = np.argmin(dists)

        center_xy = src_points[index_center]
        center_ij = corner_ids[index_center]

        print(f"{center_ij=} {center_xy=}")

        lookup = {ij: xy for ij, xy in zip(corner_ids, src_points)}
        dists = []
        for (i, j), xy0 in lookup.items():
            xy1 = lookup.get((i-1,j))
            if xy1 is not None:
                dists.append(xy1-xy0)
            xy1 = lookup.get((i,j-1))
            if xy1 is not None:
                dists.append(xy1-xy0)

        dists = np.linalg.norm(np.array(dists), axis=1)

        d = np.median(dists)

        dpi = d * 25.4 / self.charuco_board.getSquareLength()
        print(f"median distance between neighbors: {d=:.1f} {dpi=:.1f}")
        
        dst_points = (np.array(corner_ids) - center_ij) * d + center_xy

        homography, mask = cv.findHomography(
            src_points, dst_points, method=cv.LMEDS)
        print(f"{homography=}")
        print(f"{np.count_nonzero(mask)}/{np.size(mask)} points used to compute homography")

        return {'homography': homography, 'image_size': image_size,
                'corners': corners, 'corner_ids': corner_ids, 'mask': mask}
        
class PerspectiveWarper:

    def __init__(self, homography, image_size):
        self.homography = homography
        self.image_size = image_size
        
    def warp_image(self, image):
        return cv.warpPerspective(image, self.homography, self.image_size)

    def warp_points(self, points):
        assert points.ndim == 2 and points.shape[1] == 2
        n = points.shape[0]
        xyw = np.hstack((points, np.ones((n,1), points.dtype)))

        xyw = xyw @ self.homography.T

        return xyw[:,:2] / xyw[:,2:]
