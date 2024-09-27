import cv2 as cv
import functools
import json
import math
import numpy as np
import scipy.interpolate

def draw_detected_corners(image, corners, ids = None, *, thickness=1, color=(0,255,255), size=3):
    if corners is None or len(corners) == 0:
        return
    # cv.aruco.drawDetectedCornersCharuco doesn't do subpixel precision for some reason
    shift = 4
    size = size << shift
    for x, y in np.array(corners * (1 << shift), dtype=np.int32):
        cv.rectangle(image, (x-size, y-size), (x+size, y+size), color, thickness=thickness, lineType=cv.LINE_AA, shift=shift)
    if ids is not None:
        for (x, y), id in zip(corners, ids):
            cv.putText(image, str(id), (int(x)+5, int(y)-5), cv.FONT_HERSHEY_SIMPLEX, 0.5, color)

class CornerDetector:

    def __init__(self, charuco_board):
        self.charuco_board = charuco_board
        
        self.charuco_params = cv.aruco.CharucoParameters()
        self.charuco_params.minMarkers = 2
        self.charuco_params.tryRefineMarkers = True
        
        self.detector_params = cv.aruco.DetectorParameters()
        self.detector_params.cornerRefinementMethod = cv.aruco.CORNER_REFINE_SUBPIX
        self.detector_params.cornerRefinementWinSize = 10
        self.detector_params.relativeCornerRefinmentWinSize = 0.5

        self.median_ksize = None

    def detect_corners(self, input_image):
        if self.median_ksize is None:
            return self.detect_corners_impl(input_image)
    
        gray_image = cv.cvtColor(input_image, cv.COLOR_BGR2GRAY)
        median_image = cv.medianBlur(input_image, self.median_ksize)
        corners, ids = self.detect_corners_impl(median_image)
        return self.refine_subpix(gray_image, corners), ids

    def refine_subpix(self, gray_image, corners):
        # this doesn't seem to change anything (i.e. corners ==
        # refined), even though we're using it on the source image
        # instead of the median image that we used for corner
        # detection initially
        winSize = 10
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_COUNT, 40, .000001)
        refined = cv.cornerSubPix(gray_image, corners, (winSize, winSize), (-1, -1), criteria)
        return refined
        
    def detect_corners_impl(self, input_image):
        
        detector = cv.aruco.CharucoDetector(
            self.charuco_board, charucoParams=self.charuco_params, detectorParams=self.detector_params)
        charuco_corners, charuco_ids, marker_corners, marker_ids = detector.detectBoard(input_image)

        if charuco_corners is None or len(charuco_corners) < 1:
            return np.zeros((0,2)), []

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

class BigHammerCalibrator:

    def __init__(self, charuco_board):
        self.charuco_board = charuco_board

    @staticmethod
    def maximum_rect(ij_pairs):
        
        min_i = min(i for i, _ in ij_pairs)
        max_i = max(i for i, _ in ij_pairs)
        min_j = min(j for _, j in ij_pairs)
        max_j = max(j for _, j in ij_pairs)

        min_i_by_j = dict()
        max_i_by_j = dict()
        min_j_by_i = dict()
        max_j_by_i = dict()
        for i, j in ij_pairs:
            if i <= min_i_by_j.get(j, max_i):
                min_i_by_j[j] = i
            if i >= max_i_by_j.get(j, min_i):
                max_i_by_j[j] = i
            if j <= min_j_by_i.get(i, max_j):
                min_j_by_i[i] = j
            if j >= max_j_by_i.get(i, min_j):
                max_j_by_i[i] = j

        A = np.zeros((max_i+1, max_j+1), dtype=np.int32)

        for j in range(min_j, max_j+1):
            for i in range(min_i, max_i+1):
                if min_i_by_j[j] <= i <= max_i_by_j[j] and min_j_by_i[i] <= j <= max_j_by_i[i]:
                    A[i,j] = 1

        S = A.cumsum(axis=0).cumsum(axis=1)

        best_area = -1
        best_rects = []

        def is_valid(i0, j0, i1, j1):

            if i0 < 0 or i1 > max_i or j0 < 0 or j1 > max_j:
                return False

            area1 = (i1-i0+1) * (j1-j0+1)
            if i0 > 0 and j0 > 0:
                area2 = S[i1,j1] + S[i0-1,j0-1] - S[i0-1,j1] - S[i1,j0-1]
            elif i0 > 0:
                area2 = S[i1,j1] - S[i0-1,j1]
            elif j0 > 0:
                area2 = S[i1,j1] - S[i1,j0-1]

            return area1 == area2

            pred = (all(min_i_by_j[j] <= i0 and i1 <= max_i_by_j[j] for j in range(j0,j1+1)) and
                    all(min_j_by_i[i] <= j0 and j1 <= max_j_by_i[i] for i in range(i0,i1+1)))
            assert (area1 == area2) == pred, f"{i0=} {j0=} {i1=} {j1=} {area1=} {area2=} {pred=}"
            return pred

        @functools.cache
        def search(i0, j0, i1, j1):

            nonlocal best_area, best_rects

            if not is_valid(i0, j0, i1, j1):
                return False

            area = (i1-i0+1) * (j1-j0+1)
            if area > best_area:
                best_area = area
                best_rects = []
            if area >= best_area:
                best_rects.append((i0, j0, i1, j1))

            # try to expand in all directions, rather than being
            # incremental
            if search(i0-1,j0-1,i1+1,j1+1):
                return True

            # the all directions expansion failed, so try each direction
            # singly
            search(i0-1, j0, i1, j1)
            search(i0, j0-1, i1, j1)
            search(i0, j0, i1+1, j1)
            search(i0, j0, i1, j1+1)

            return True

        i = (min_i + max_i) // 2
        j = (min_j + max_j) // 2
        search(i, j, i, j)

        if len(best_rects) != 1:
            raise ValueError(f"maximum_rect: expected exactly 1 rectangle, got {best_rects}")

        return best_rects[0]

    def calibrate(self, image):
        
        detector = CornerDetector(self.charuco_board)
        detector.median_ksize = 5
        
        uv, ij = detector.detect_corners(image)
        if uv is None:
            return None

        print(f"BigHammer: found {len(ij)} corners")

        image_size = np.array(image.shape[:2][::-1])

        dist = np.linalg.norm(uv - image_size/2, axis=1)
        central_corner_idx = np.argmin(dist)
        
        center_ij = ij[central_corner_idx]

        rect_ij = self.maximum_rect(ij)
        min_i, min_j, max_i, max_j = rect_ij

        grid_ij = [(i, j) for i in range(min_i-1, max_i+2) for j in range(min_j-1, max_j+2)]
        
        have_dict = dict(zip(ij,uv))
        need_ij = [ij for ij in grid_ij if ij not in have_dict]

        print(f"{center_ij=} {rect_ij=} {len(have_dict)=} {len(need_ij)=}")

        lerper = scipy.interpolate.RBFInterpolator(ij, uv)
        Z = lerper(need_ij)

        image_copy = cv.resize(image, None, fx=.25, fy=.25)
        draw_detected_corners(image_copy, uv*.25, color=(255,128,0))
        draw_detected_corners(image_copy, Z*.25, color=(0,128,255))
        cv.imwrite('hammer_rbf.png', image_copy)

        need_dict = dict(zip(need_ij, Z))

        # print(f"{have_dict=} {need_dict=}")

        grid_dict = have_dict | need_dict

        pixels_per_mm = 600. / 25.4
        ij_scale = pixels_per_mm * self.charuco_board.getSquareLength()

        # point closest to the center stays in place
        grid_points_u = (np.arange(min_i-1, max_i+2, 1) - center_ij[0]) * ij_scale + image_size[0]/2
        grid_points_v = (np.arange(min_j-1, max_j+2, 1) - center_ij[1]) * ij_scale + image_size[1]/2
        grid_values_u = np.zeros((max_i-min_i+3, max_j-min_j+3))
        grid_values_v = np.zeros((max_i-min_i+3, max_j-min_j+3))

        for j in range(min_j-1, max_j+2):
            for i in range(min_i-1, max_i+2):
                uv = grid_dict[(i,j)]
                grid_values_u[i-min_i+1,j-min_j+1] = uv[0]
                grid_values_v[i-min_i+1,j-min_j+1] = uv[1]

        ll, ur = (np.array([[min_i, min_j], [max_i, max_j]]) - center_ij) * ij_scale + image_size/2
        ll = np.asarray(np.ceil(ll), dtype=np.int32).tolist()
        ur = np.asarray(np.floor(ur), dtype=np.int32).tolist()
        roi = tuple(ll + ur)
        
        print(f"{roi=}")

        o = {'dpi':600.,
             'roi':roi,
             'points_u':grid_points_u.tolist(),
             'points_v':grid_points_v.tolist(),
             'values_u':grid_values_u.tolist(),
             'values_v':grid_values_v.tolist()}
        with open('camera-calibration.json', 'w') as f:
            json.dump(o, f)

        return BigHammerRemapper.from_calibration(o)

        use_roi = False

        if use_roi:
            u_range = np.arange(0, image_size[0], 1)
            v_range = np.arange(0, image_size[1], 1)
        else:
            x0, y0, x1, y1 = roi
            u_range = np.arange(x0, x1, 1)
            v_range = np.arange(y0, y1, 1)
            
        grid_u, grid_v = np.meshgrid(u_range, v_range)
        
        u_interp = scipy.interpolate.RegularGridInterpolator((grid_points_u, grid_points_v), grid_values_u, method='linear', bounds_error=False)
        u_map = u_interp((grid_u, grid_v))
        
        v_interp = scipy.interpolate.RegularGridInterpolator((grid_points_u, grid_points_v), grid_values_v, method='linear', bounds_error=False)
        v_map = v_interp((grid_u, grid_v))

        if use_roi:
            x0, y0, x1, y1 = roi
            u_map = u_map[y0:y1,x0:x1]
            v_map = v_map[y0:y1,x0:x1]

        return BigHammerRemapper(u_map, v_map)

class BigHammerRemapper:

    @staticmethod
    def from_calibration(calibration):
        roi = tuple(calibration['roi'])
        points_u = np.array(calibration['points_u'])
        points_v = np.array(calibration['points_v'])
        values_u = np.array(calibration['values_u'])
        values_v = np.array(calibration['values_v'])

        x0, y0, x1, y1 = roi
        u_range = np.arange(x0, x1, 1)
        v_range = np.arange(y0, y1, 1)
            
        grid_u, grid_v = np.meshgrid(u_range, v_range)
        
        u_interp = scipy.interpolate.RegularGridInterpolator((points_u, points_v), values_u, method='linear', bounds_error=False)
        u_map = u_interp((grid_u, grid_v))
        
        v_interp = scipy.interpolate.RegularGridInterpolator((points_u, points_v), values_v, method='linear', bounds_error=False)
        v_map = v_interp((grid_u, grid_v))

        return BigHammerRemapper(u_map, v_map)

    def __init__(self, u_map, v_map):
        self.u_map = np.asarray(u_map, dtype=np.float32)
        self.v_map = np.asarray(v_map, dtype=np.float32)

    def __call__(self, image):
        return self.undistort_image(image)

    def undistort_image(self, image):
        return cv.remap(image, self.u_map, self.v_map, cv.INTER_LINEAR, borderValue=(128,128,128))
