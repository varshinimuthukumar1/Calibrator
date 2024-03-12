import json
import cv2
from matplotlib import pyplot as plt
import numpy as np

def rotate_yaw_pitch_roll(yaw, pitch, roll):
    # Convert yaw, pitch, and roll to radians
    yaw = np.deg2rad(yaw)
    pitch = np.deg2rad(pitch)
    roll = np.deg2rad(roll)

    # Create the rotation matrix
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(pitch), -np.sin(pitch)],
                    [0, np.sin(pitch), np.cos(pitch)]])
    R_y = np.array([[np.cos(yaw), 0, np.sin(yaw)],
                    [0, 1, 0],
                    [-np.sin(yaw), 0, np.cos(yaw)]])
    R_z = np.array([[np.cos(roll), -np.sin(roll), 0],
                    [np.sin(roll), np.cos(roll), 0],
                    [0, 0, 1]])

    # Combine the rotation matrices
    R = np.dot(R_z, np.dot(R_y, R_x))
    return R

def get_correspondences(img_pts, obj_pts, vis=True, img_path=None):
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
                #matched_img_pts.append([2280-point[1][1], point[1][0]])
                matched_img_pts.append(point[1])
        for point in obj_pts_list:
            if point[0] == key:
                matched_obj_pts.append(point[1])

    print(matched_img_pts)

    # Convert lists to numpy arrays with the correct shape
    matched_img_pts = np.array(matched_img_pts, dtype=np.float32).reshape(-1, 1, 2)
    matched_obj_pts = np.array(matched_obj_pts, dtype=np.float32).reshape(-1, 1, 3)

    

    if vis:
        # plot the points on image
        img = cv2.imread(img_path)
        #cv2.imread('/data/kinexon/calibration_challenge/#sca-ot-prod-a0cd0015-7fe6-4e9a-8027-e15d532a585f_764979d8_4f9dd4ef_frame0000000_right.jpg')
        # rotate image 90 degree Anti-clockwise
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        for point in matched_img_pts:
            cv2.circle(img, (int(point[0][0]),int(point[0][1])), 25, (0, 0, 255), -1)
        print(img.shape)
        
        # resize the image
        img = cv2.resize(img, (int(2280/4),int(3648/4)))
        
        cv2.imshow('left', img)
        cv2.waitKey(0)
        
    return matched_img_pts, matched_obj_pts

def match_points_stereo(image_points_left, image_points_right, object_points):
    # Convert dictionaries to lists of tuples for easier processing
    image_points_left_list = list(image_points_left["annotated_points"].items())
    image_points_right_list = list(image_points_right["annotated_points"].items())
    object_points_list = list(object_points.items())

    # Filter out points that don't have a correspondence in all three sets
    common_keys = set(image_points_left["annotated_points"]).intersection(
                   image_points_right["annotated_points"], object_points)

    # Initialize lists to hold the matched points
    matched_img_points_left = []
    matched_img_points_right = []
    matched_obj_points = []

    for key in common_keys:
        # Find the matching points and add them to the lists
        for point in image_points_left_list:
            if point[0] == key:
                matched_img_points_left.append([3648-point[1][1], point[1][0]])
        for point in image_points_right_list:
            if point[0] == key:
                matched_img_points_right.append([3648-point[1][1], point[1][0]])
        for point in object_points_list:
            if point[0] == key:
                matched_obj_points.append(point[1])

    # Convert lists to numpy arrays with the correct shape
    matched_img_points_left = np.array(matched_img_points_left, dtype=np.float32).reshape(-1, 1, 2)
    matched_img_points_right = np.array(matched_img_points_right, dtype=np.float32).reshape(-1, 1, 2)
    matched_obj_points = np.array(matched_obj_points, dtype=np.float32).reshape(-1, 1, 3)

    
    # plot the points on image
    img = cv2.imread('/data/kinexon/calibration_challenge/sca-ot-prod-a0cd0015-7fe6-4e9a-8027-e15d532a585f_764979d8_893e5070_frame0000000_left.jpg')
    for point in matched_img_points_left:
        cv2.circle(img, (int(point[0][0]),int(point[0][1])), 25, (0, 0, 255), -1)
    img_right = cv2.imread('/data/kinexon/calibration_challenge/sca-ot-prod-a0cd0015-7fe6-4e9a-8027-e15d532a585f_764979d8_4f9dd4ef_frame0000000_right.jpg')
    for point in matched_img_points_right:
        cv2.circle(img_right, (int(point[0][0]),int(point[0][1])), 25, (0, 0, 255), -1)
    print(img.shape)
    
    # resize the image
    img = cv2.resize(img, (int(3648/4),int(2280/4)))
    # rotate image 90 degree Anti-clockwise
    img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    cv2.imshow('left', img)
    cv2.waitKey(0)
    img_right = cv2.resize(img_right, (int(3648/4),int(2280/4)))
    # rotate image 90 degree
    img_right = cv2.rotate(img_right, cv2.ROTATE_90_COUNTERCLOCKWISE)

    cv2.imshow('right', img_right)
    cv2.waitKey(0)
    
    return matched_img_points_left, matched_img_points_right, matched_obj_points

def read_json(file_path):
    # Open the JSON file
    with open(file_path, 'r') as file:
        # Load the JSON data
        data = json.load(file)
    return data

def load_calibration_data():
    
    # Specify the path to your JSON file
    file_path_left = '/data/kinexon/calibration_challenge/sca-ot-prod-a0cd0015-7fe6-4e9a-8027-e15d532a585f_764979d8_893e5070.calib.points_left.json'
    file_path_right = '/data/kinexon/calibration_challenge/sca-ot-prod-a0cd0015-7fe6-4e9a-8027-e15d532a585f_764979d8_4f9dd4ef.calib.points_right.json'
    landmarks = '/data/kinexon/calibration_challenge/landmarks.json'
    # Open the JSON file
    image_data_left = read_json(file_path_left)
    image_data_right = read_json(file_path_right)
    landmarks_data = read_json(landmarks)

    return image_data_right, image_data_left, landmarks_data
