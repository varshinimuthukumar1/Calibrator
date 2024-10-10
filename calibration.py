import copy
import os
from typing import List
from constants import DISPLAY_SIZE, RESULTS_PATH, TEST_IMAGE_LEFT, TEST_IMAGE_RIGHT
from stereo_calibrator import stereo_calibrate
from utils import (
    get_correspondences,
    match_points_stereo,
    load_calibration_data,
    rotate_yaw_pitch_roll,
)

import numpy as np
import cv2

imageSize = (2280, 3648)


def vis_image(img: np.array, img_pts: List):
    img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    for point in img_pts:
        cv2.circle(img, (int(point[0][0]), int(point[0][1])), 25, (0, 0, 255), -1)
    print(img.shape)

    # resize the image
    img = cv2.resize(img, (int(2280 / 4), int(3648 / 4)))

    cv2.imshow("left", img)
    cv2.waitKey(0)
    return



def get_3dto2d_correspondences(
    img_pts: List, obj_pts: List, vis: bool = True, img_path: str = None
):

    # Convert dictionaries to lists of tuples for easier processing
    img_pts_list = list(img_pts.items())
    obj_pts_list = list(obj_pts.items())

    # Filter out points that don't have a correspondence in both sets
    common_keys = set(img_pts).intersection(obj_pts)

    # Initialize lists to hold the matched points
    matched_img_pts = []
    matched_obj_pts = []

    for key in common_keys:
        # Find the matching points and add them to the lists
        for point in img_pts_list:
            if point[0] == key:
                matched_img_pts.append(point[1])
        for point in obj_pts_list:
            if point[0] == key:
                matched_obj_pts.append(point[1])

    # Convert lists to numpy arrays with the correct shape
    matched_img_pts = np.array(matched_img_pts, dtype=np.float32).reshape(-1, 1, 2)
    matched_obj_pts = np.array(matched_obj_pts, dtype=np.float32).reshape(-1, 1, 3)
    if vis:
        img = cv2.imread(img_path)
        vis_image(img, matched_img_pts)
    return matched_img_pts, matched_obj_pts


def undistort(img: np.array, mtx: np.array, dist: np.array, vis: bool = True):
    img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    img = cv2.resize(img, (2280, 3648))

    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        mtx, dist, np.eye(3), mtx, imageSize, cv2.CV_16SC2
    )
    undistorted_img = cv2.remap(
        img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
    )

    if vis:
        undistorted_img_d = copy.deepcopy(undistorted_img)
        img = cv2.resize(img, DISPLAY_SIZE)
        undistorted_img_d = cv2.resize(undistorted_img_d, DISPLAY_SIZE)

        # Display the original and undistorted images for comparison
        cv2.imshow("Original Image", img)
        cv2.imshow("Undistorted Image", undistorted_img_d)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return undistorted_img


def get_intrinsics(img_pts, obj_pts, save: bool = True):

    N_OK = len(obj_pts)
    mtx = np.zeros((3, 3))
    dist = np.zeros((4, 1))

    
    calibration_flags = (
        cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC
        + cv2.fisheye.CALIB_CHECK_COND
        + cv2.fisheye.CALIB_FIX_SKEW
    )
    retval, mtx, dist, _, _ = cv2.fisheye.calibrate(
        obj_pts,
        img_pts,
        (2280, 3648),
        mtx,
        dist,
        None,
        None,
        calibration_flags,
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6),
    )
    if save:
        np.save(os.path.join(RESULTS_PATH, "camera_matrix.npy"), mtx)
        np.save(os.path.join(RESULTS_PATH, "distortion_coefficients.npy"), dist)

    return mtx, dist


def get_extrinsics(
    img_pts: List,
    obj_pts: List,
    mtx: np.array,
    dist: np.array,
    rvecs: np.array = None,
    tvecs: np.array = None,
):
    fx = mtx[0, 0]
    fy = mtx[1, 1]
    cx = mtx[0, 2]
    cy = mtx[1, 2]
    undistort_pts = cv2.fisheye.undistortPoints(img_pts, mtx, dist)

    for point in undistort_pts:
        point[0][0],point[0][1] = point[0][0] * fx + cx, point[0][1]* fy + cy
        _, rvecs, tvecs, _ = cv2.solvePnPRansac(
    obj_pts, undistort_pts, mtx, np.array([]), flags=cv2.SOLVEPNP_P3P
    )
    print("rvec : \n", rvecs)
    print("tvec : \n", tvecs)
    return rvecs, tvecs, undistort_pts


def calibrate_camera(vis: bool = True):

    ## Load the calibration data
    imgpoints_r, imgpoints_l, objpoints = load_calibration_data()
    matched_img_points_left, matched_obj_points_l = get_3dto2d_correspondences(
        imgpoints_l["annotated_points"], objpoints, img_path=TEST_IMAGE_LEFT
    )
    matched_img_points_right, matched_obj_points_r = get_3dto2d_correspondences(
        imgpoints_r["annotated_points"], objpoints, img_path=TEST_IMAGE_RIGHT
    )

    ## Get the intrinsics
    mtx, dist = get_intrinsics(
        [matched_img_points_left, matched_img_points_right],
        [matched_obj_points_l, matched_obj_points_r],
    )

    # Undistort the images
    img_undist = undistort(cv2.imread(TEST_IMAGE_LEFT), mtx, dist, vis=True)
    img_undist_test = copy.deepcopy(img_undist)
    fx = mtx[0, 0]
    fy = mtx[1, 1]
    cx = mtx[0, 2]
    cy = mtx[1, 2]
    ## Apply undistortion
    undistort_pts = cv2.fisheye.undistortPoints(matched_img_points_left, mtx, dist)
    # Matrix multiplication with intrinsic matrix
    for point in undistort_pts:
        point[0][0], point[0][1] = point[0][0] * fx + cx, point[0][1] * fy + cy
        if (
            point[0][0] > 0
            and point[0][0] < 2280
            and point[0][1] > 0
            and point[0][1] < 3648
        ):
            img_undist = cv2.circle(
                img_undist, (int(point[0][0]), int(point[0][1])), 25, (0, 255, 0), -1
            )
    img_undist = cv2.resize(img_undist, DISPLAY_SIZE)
    cv2.imshow("Remapped to undistorted image", img_undist)
    cv2.waitKey(0)

    ## Get the extrinsics
    rvec, tvec, pts_temp = get_extrinsics(
        matched_img_points_left, matched_obj_points_l, mtx, dist
    )
    rmat_left = cv2.Rodrigues(rvec)[0]

    ## Test the calibration
    remapped_pts = []
    objpts = [objpoints[key] for key in objpoints.keys()]
    for i in objpts:
        imgpoints2, _ = cv2.projectPoints(np.array(i), rvec, tvec, mtx, np.array([]))
        remapped_pts.append(imgpoints2[0])
    fx = mtx[0, 0]
    fy = mtx[1, 1]
    cx = mtx[0, 2]
    cy = mtx[1, 2]
    undistort_pts = cv2.fisheye.undistortPoints(np.array(remapped_pts), mtx, dist)
    for point in undistort_pts:
        point[0][0], point[0][1] = point[0][0] * fx + cx, point[0][1] * fy + cy
        if (
            point[0][0] > 0
            and point[0][0] < 2280
            and point[0][1] > 0
            and point[0][1] < 3648
        ):
            img_undist_test = cv2.circle(
                img_undist_test,
                (int(point[0][0]), int(point[0][1])),
                25,
                (255, 0, 0),
                -1,
            )
    img_undist_test = cv2.resize(img_undist_test, DISPLAY_SIZE)
    cv2.imshow("Test calibration for Left Camera", img_undist_test)
    cv2.waitKey(0)

    ## Calibrate the right camera with an additional rotation of 90 degrees in rvec
    R90 = rotate_yaw_pitch_roll(90, 0, 0)
    Rleft, _ = cv2.Rodrigues(rvec)
    # change rvec for a camera at 90 degree rotation from rvec_left
    Rright = np.dot(R90, Rleft)
    rvec_right, _ = cv2.Rodrigues(Rright)

    ## Test the calibration
    img_undist_right = undistort(cv2.imread(TEST_IMAGE_RIGHT), mtx, dist, vis=True)
    remapped_pts = []
    objpts = [objpoints[key] for key in objpoints.keys()]
    for i in objpts:
        imgpoints2, _ = cv2.projectPoints(
            np.array(i), rvec_right, tvec, mtx, np.array([])
        )
        remapped_pts.append(imgpoints2[0])
    fx = mtx[0, 0]
    fy = mtx[1, 1]
    cx = mtx[0, 2]
    cy = mtx[1, 2]
    undistort_pts = cv2.fisheye.undistortPoints(np.array(remapped_pts), mtx, dist)
    for point in undistort_pts:
        point[0][0], point[0][1] = point[0][0] * fx + cx, point[0][1] * fy + cy
        if (
            point[0][0] > 0
            and point[0][0] < 2280
            and point[0][1] > 0
            and point[0][1] < 3648
        ):
            img_undist_right = cv2.circle(
                img_undist_right,
                (int(point[0][0]), int(point[0][1])),
                25,
                (255, 0, 0),
                -1,
            )
    img_undist_right = cv2.resize(img_undist_right, DISPLAY_SIZE)
    cv2.imshow("Test calibration for Right Camera", img_undist_right)
    cv2.waitKey(0)
    return mtx, dist, rvec, tvec


if __name__ == "__main__":
    mtx, dist, rvec, tvec = calibrate_camera()
    print("calibrated")
