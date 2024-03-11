import numpy as np
import cv2




def stereo_calibrate(matched_obj_points, matched_img_points_right, matched_img_points_left, mtx, dist):
   # Assume you have already detected the landmarks in your images and have
    # arrays of corresponding points in the left and right images: imgpoints_l, imgpoints_r

    # Initialize camera matrices (K) and distortion coefficients (D) for both cameras
    K1, K2 = mtx,mtx
    D1, D2 = dist,dist
    u_matched_obj_points = cv2.fisheye.undistortPoints(matched_obj_points, mtx, dist)
    u_matched_img_points_right = cv2.fisheye.undistortPoints(matched_img_points_right, mtx, dist)
    u_matched_img_points_left = cv2.fisheye.undistortPoints(matched_img_points_left, mtx, dist)
    # Ensure there's an equal number of objectPoints, imagePointsLeft, and imagePointsRight
    assert len(u_matched_obj_points) == len(u_matched_img_points_left) == len(u_matched_img_points_right), "Mismatch in the number of views"

    # Ensure each set of points has a consistent number of points across left and right images
    for obj, img_l, img_r in zip(u_matched_obj_points, u_matched_img_points_left, u_matched_img_points_right):
        assert obj.shape[0] == img_l.shape[0] == img_r.shape[0], "Mismatch in the number of points within a view"

    # Define the size of your images
    imageSize = (2280, 3648)
    # Calibrate each camera individually (this might be adapted based on how you detect landmarks)
    # You may need to adapt this step to your specific method of detecting landmarks and estimating their 3D positions
    assert matched_obj_points.dtype == np.float32, "Object points must be of type np.float32"
    assert matched_obj_points.shape[1:] == (1, 3), "Object points must have a shape of (-1, 1, 3)"
    print(matched_obj_points.shape)
    print(matched_obj_points.dtype)
    print(matched_obj_points)
    print(matched_img_points_left)
    print(matched_img_points_right)
    # Stereo calibration for fisheye lenses
    retval, K1, D1, K2, D2, R, T, E, F = cv2.stereoCalibrate(
        [u_matched_obj_points], [u_matched_img_points_left], [u_matched_img_points_right], K1, D1, K2, D2, imageSize,
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6),
        flags=cv2.fisheye.CALIB_FIX_INTRINSIC)
   

    print("calibrated")        
    return R, T
