import pandas as pd
import numpy as np
import math
import joblib
import pickle
import re

li =[]

listof_ThreedDistancefeatures=['left_shoulder_left_wrist', 'right_shoulder_right_wrist', 'left_hip_left_ankle', 'right_hip_right_ankle',
                               'left_hip_left_wrist', 'right_hip_right_wrist', 'left_shoulder_left_ankle', 'right_shoulder_right_ankle',
                               'left_hip_right_wrist', 'right_hip_left_wrist', 'left_elbow_right_elbow', 'left_knee_right_knee',
                               'left_wrist_right_wrist', 'left_ankle_right_ankle', 'left_hip_avg_left_wrist_left_ankle',
                               'right_hip_avg_right_wrist_right_ankle']
################################################################################################################
listof_xyz_distancesfeatures=['x_left_shoulder_left_wrist', 'y_left_shoulder_left_wrist', 'z_left_shoulder_left_wrist',
                              'x_right_shoulder_right_wrist', 'y_right_shoulder_right_wrist', 'z_right_shoulder_right_wrist',
                              'x_left_hip_left_ankle', 'y_left_hip_left_ankle', 'z_left_hip_left_ankle', 'x_right_hip_right_ankle',
                              'y_right_hip_right_ankle', 'z_right_hip_right_ankle', 'x_left_hip_left_wrist', 'y_left_hip_left_wrist',
                              'z_left_hip_left_wrist', 'x_right_hip_right_wrist', 'y_right_hip_right_wrist', 'z_right_hip_right_wrist',
                              'x_left_shoulder_left_ankle', 'y_left_shoulder_left_ankle', 'z_left_shoulder_left_ankle',
                              'x_right_shoulder_right_ankle', 'y_right_shoulder_right_ankle', 'z_right_shoulder_right_ankle',
                              'x_left_hip_right_wrist', 'y_left_hip_right_wrist', 'z_left_hip_right_wrist', 'x_right_hip_left_wrist',
                              'y_right_hip_left_wrist', 'z_right_hip_left_wrist', 'x_left_elbow_right_elbow', 'y_left_elbow_right_elbow',
                              'z_left_elbow_right_elbow', 'x_left_knee_right_knee', 'y_left_knee_right_knee', 'z_left_knee_right_knee',
                              'x_left_wrist_right_wrist', 'y_left_wrist_right_wrist', 'z_left_wrist_right_wrist', 'x_left_ankle_right_ankle',
                              'y_left_ankle_right_ankle', 'z_left_ankle_right_ankle', 'x_left_hip_avg_left_wrist_left_ankle',
                              'y_left_hip_avg_left_wrist_left_ankle', 'z_left_hip_avg_left_wrist_left_ankle', 'x_right_hip_avg_right_wrist_right_ankle',
                              'y_right_hip_avg_right_wrist_right_ankle', 'z_right_hip_avg_right_wrist_right_ankle']
########################################################################################################################
listof_anglesfeatures=['right_elbow_right_shoulder_right_hip', 'left_elbow_left_shoulder_left_hip', 'right_knee_mid_hip_left_knee',
                       'right_hip_right_knee_right_ankle', 'left_hip_left_knee_left_ankle', 'right_wrist_right_elbow_right_shoulder',
                       'left_wrist_left_elbow_left_shoulder']


def calculate_listof_3dDistancefeatures_left_shoulder_left_wrist(left_shoulder_,left_wrist):
    return caluclate_3d_distance_2points(left_shoulder_,left_wrist)

def calculate_listof_3dDistancefeatures_right_shoulder_right_wrist(right_shoulder, right_wrist):
    return caluclate_3d_distance_2points(right_shoulder, right_wrist)

def calculate_listof_3dDistancefeatures_left_hip_left_ankle(left_hip, left_ankle):
    return caluclate_3d_distance_2points(left_hip, left_ankle)

def calculate_listof_3dDistancefeatures_right_hip_right_ankle(right_hip, right_ankle):
    return caluclate_3d_distance_2points(right_hip, right_ankle)

def calculate_listof_3dDistancefeatures_left_hip_left_wrist(left_hip, left_wrist):
    return caluclate_3d_distance_2points(left_hip, left_wrist)

def calculate_listof_3dDistancefeatures_right_hip_right_wrist(right_hip, right_wrist):
    return caluclate_3d_distance_2points(right_hip, right_wrist)

def calculate_listof_3dDistancefeatures_left_shoulder_left_ankle(left_shoulder, left_ankle):
    return caluclate_3d_distance_2points(left_shoulder, left_ankle)

def calculate_listof_3dDistancefeatures_right_shoulder_right_ankle(right_shoulder, right_ankle):
    return caluclate_3d_distance_2points(right_shoulder, right_ankle)

def calculate_listof_3dDistancefeatures_left_hip_right_wrist(left_hip, right_wrist):
    return caluclate_3d_distance_2points(left_hip, right_wrist)

def calculate_listof_3dDistancefeatures_right_hip_left_wrist(right_hip, left_wrist):
    return caluclate_3d_distance_2points(right_hip, left_wrist)

def calculate_listof_3dDistancefeatures_left_elbow_right_elbow(left_elbow, right_elbow):
    return caluclate_3d_distance_2points(left_elbow, right_elbow)

def calculate_listof_3dDistancefeatures_left_knee_right_knee(left_knee, right_knee):
    return caluclate_3d_distance_2points(left_knee, right_knee)

def calculate_listof_3dDistancefeatures_left_wrist_right_wrist(left_wrist, right_wrist):
    return caluclate_3d_distance_2points(left_wrist, right_wrist)

def calculate_listof_3dDistancefeatures_left_ankle_right_ankle(left_ankle, right_ankle):
    return caluclate_3d_distance_2points(left_ankle, right_ankle)
##################################################################################################
def calculate_left_hip_avg_left_wrist_left_ankle(left_hip, left_wrist, left_ankle):
    # Calculate the average position of the three landmarks
    avg_position = calculate_avg_distance(left_hip, left_wrist, left_ankle)
    # Calculate the distance from the average position to the origin
    return avg_position

# Function to calculate right_hip_avg_right_wrist_right_ankle
# def calculate_right_hip_avg_right_wrist_right_ankle(right_hip, right_wrist, right_ankle):
#     # Calculate the average position of the three landmarks
#     avg_position = calculate_avg_distance(right_hip, right_wrist, right_ankle)
#     # Calculate the distance from the average position to the origin
#     return avg_position


######################################################################################################

def calculate_listof_anglesfeatures_right_elbow_right_shoulder_right_hip(right_elbow,right_shoulder,right_hip):
    return calculate_angle_3points(right_elbow,right_shoulder,right_hip)

def calculate_listof_anglesfeatures_left_elbow_left_shoulder_left_hip(left_elbow, left_shoulder, left_hip):
    return calculate_angle_3points(left_elbow, left_shoulder, left_hip)

def calculate_listof_anglesfeatures_right_knee_mid_hip_left_knee(right_knee, mid_hip, left_knee):
    return calculate_angle_3points(right_knee, mid_hip, left_knee)

def calculate_listof_anglesfeatures_right_hip_right_knee_right_ankle(right_hip, right_knee, right_ankle):
    return calculate_angle_3points(right_hip, right_knee, right_ankle)

def calculate_listof_anglesfeatures_left_hip_left_knee_left_ankle(left_hip, left_knee, left_ankle):
    return calculate_angle_3points(left_hip, left_knee, left_ankle)

def calculate_listof_anglesfeatures_right_wrist_right_elbow_right_shoulder(right_wrist, right_elbow, right_shoulder):
    return calculate_angle_3points(right_wrist, right_elbow, right_shoulder)

def calculate_listof_anglesfeatures_left_wrist_left_elbow_left_shoulder(left_wrist, left_elbow, left_shoulder):
    return calculate_angle_3points(left_wrist, left_elbow, left_shoulder)

########################################################################################################################

def calculate_listof_xyz_distancesfeature_x_left_shoulder_left_wrist(left_shoulder,left_wrist):
    return calculate_x_distances(left_shoulder.x,left_wrist.x)

def calculate_listof_xyz_distancesfeature_y_left_shoulder_left_wrist(left_shoulder,left_wrist):
    return calculate_y_distances(left_shoulder.y,left_wrist.y)

def calculate_listof_xyz_distancesfeature_z_left_shoulder_left_wrist(left_shoulder,left_wrist):
    return calculate_z_distances(left_shoulder.z,left_wrist.z)

def calculate_listof_xyz_distancesfeature_x_right_shoulder_right_wrist(right_shoulder, right_wrist):
    return calculate_x_distances(right_shoulder.x, right_wrist.x)

def calculate_listof_xyz_distancesfeature_y_right_shoulder_right_wrist(right_shoulder, right_wrist):
    return calculate_y_distances(right_shoulder.y, right_wrist.y)

def calculate_listof_xyz_distancesfeature_z_right_shoulder_right_wrist(right_shoulder, right_wrist):
    return calculate_z_distances(right_shoulder.z, right_wrist.z)


def calculate_listof_xyz_distancesfeature_x_left_hip_left_ankle(left_hip, left_ankle):
    return calculate_x_distances(left_hip.x, left_ankle.x)

def calculate_listof_xyz_distancesfeature_y_left_hip_left_ankle(left_hip, left_ankle):
    return calculate_y_distances(left_hip.y, left_ankle.y)

def calculate_listof_xyz_distancesfeature_z_left_hip_left_ankle(left_hip, left_ankle):
    return calculate_z_distances(left_hip.z, left_ankle.z)


def calculate_listof_xyz_distancesfeature_x_right_hip_right_ankle(right_hip, right_ankle):
    return calculate_x_distances(right_hip.x, right_ankle.x)

def calculate_listof_xyz_distancesfeature_y_right_hip_right_ankle(right_hip, right_ankle):
    return calculate_y_distances(right_hip.y, right_ankle.y)

def calculate_listof_xyz_distancesfeature_z_right_hip_right_ankle(right_hip, right_ankle):
    return calculate_z_distances(right_hip.z, right_ankle.z)


def calculate_listof_xyz_distancesfeature_x_left_hip_left_wrist(left_hip, left_wrist):
    return calculate_x_distances(left_hip.x, left_wrist.x)

def calculate_listof_xyz_distancesfeature_y_left_hip_left_wrist(left_hip, left_wrist):
    return calculate_y_distances(left_hip.y, left_wrist.y)

def calculate_listof_xyz_distancesfeature_z_left_hip_left_wrist(left_hip, left_wrist):
    return calculate_z_distances(left_hip.z, left_wrist.z)


def calculate_listof_xyz_distancesfeature_x_right_hip_right_wrist(right_hip, right_wrist):
    return calculate_x_distances(right_hip.x, right_wrist.x)

def calculate_listof_xyz_distancesfeature_y_right_hip_right_wrist(right_hip, right_wrist):
    return calculate_y_distances(right_hip.y, right_wrist.y)

def calculate_listof_xyz_distancesfeature_z_right_hip_right_wrist(right_hip, right_wrist):
    return calculate_z_distances(right_hip.z, right_wrist.z)


def calculate_listof_xyz_distancesfeature_x_left_shoulder_left_ankle(left_shoulder, left_ankle):
    return calculate_x_distances(left_shoulder.x, left_ankle.x)

def calculate_listof_xyz_distancesfeature_y_left_shoulder_left_ankle(left_shoulder, left_ankle):
    return calculate_y_distances(left_shoulder.y, left_ankle.y)

def calculate_listof_xyz_distancesfeature_z_left_shoulder_left_ankle(left_shoulder, left_ankle):
    return calculate_z_distances(left_shoulder.z, left_ankle.z)


def calculate_listof_xyz_distancesfeature_x_right_shoulder_right_ankle(right_shoulder, right_ankle):
    return calculate_x_distances(right_shoulder.x, right_ankle.x)

def calculate_listof_xyz_distancesfeature_y_right_shoulder_right_ankle(right_shoulder, right_ankle):
    return calculate_y_distances(right_shoulder.y, right_ankle.y)

def calculate_listof_xyz_distancesfeature_z_right_shoulder_right_ankle(right_shoulder, right_ankle):
    return calculate_z_distances(right_shoulder.z, right_ankle.z)


def calculate_listof_xyz_distancesfeature_x_left_hip_right_wrist(left_hip, right_wrist):
    return calculate_x_distances(left_hip.x, right_wrist.x)

def calculate_listof_xyz_distancesfeature_y_left_hip_right_wrist(left_hip, right_wrist):
    return calculate_y_distances(left_hip.y, right_wrist.y)

def calculate_listof_xyz_distancesfeature_z_left_hip_right_wrist(left_hip, right_wrist):
    return calculate_z_distances(left_hip.z, right_wrist.z)


def calculate_listof_xyz_distancesfeature_x_right_hip_left_wrist(right_hip, left_wrist):
    return calculate_x_distances(right_hip.x, left_wrist.x)

def calculate_listof_xyz_distancesfeature_y_right_hip_left_wrist(right_hip, left_wrist):
    return calculate_y_distances(right_hip.y, left_wrist.y)

def calculate_listof_xyz_distancesfeature_z_right_hip_left_wrist(right_hip, left_wrist):
    return calculate_z_distances(right_hip.z, left_wrist.z)


def calculate_listof_xyz_distancesfeature_x_left_elbow_right_elbow(left_elbow, right_elbow):
    return calculate_x_distances(left_elbow.x, right_elbow.x)

def calculate_listof_xyz_distancesfeature_y_left_elbow_right_elbow(left_elbow, right_elbow):
    return calculate_y_distances(left_elbow.y, right_elbow.y)

def calculate_listof_xyz_distancesfeature_z_left_elbow_right_elbow(left_elbow, right_elbow):
    return calculate_z_distances(left_elbow.z, right_elbow.z)


def calculate_listof_xyz_distancesfeature_x_left_knee_right_knee(left_knee, right_knee):
    return calculate_x_distances(left_knee.x, right_knee.x)

def calculate_listof_xyz_distancesfeature_y_left_knee_right_knee(left_knee, right_knee):
    return calculate_y_distances(left_knee.y, right_knee.y)

def calculate_listof_xyz_distancesfeature_z_left_knee_right_knee(left_knee, right_knee):
    return calculate_z_distances(left_knee.z, right_knee.z)


def calculate_listof_xyz_distancesfeature_x_left_wrist_right_wrist(left_wrist, right_wrist):
    return calculate_x_distances(left_wrist.x, right_wrist.x)

def calculate_listof_xyz_distancesfeature_y_left_wrist_right_wrist(left_wrist, right_wrist):
    return calculate_y_distances(left_wrist.y, right_wrist.y)

def calculate_listof_xyz_distancesfeature_z_left_wrist_right_wrist(left_wrist, right_wrist):
    return calculate_z_distances(left_wrist.z, right_wrist.z)


def calculate_listof_xyz_distancesfeature_x_left_ankle_right_ankle(left_ankle, right_ankle):
    return calculate_x_distances(left_ankle.x, right_ankle.x)

def calculate_listof_xyz_distancesfeature_y_left_ankle_right_ankle(left_ankle, right_ankle):
    return calculate_y_distances(left_ankle.y, right_ankle.y)

def calculate_listof_xyz_distancesfeature_z_left_ankle_right_ankle(left_ankle, right_ankle):
    return calculate_z_distances(left_ankle.z, right_ankle.z)

#######################################################################################################################
def calculate_x_left_hip_avg_left_wrist_left_ankle(left_hip,left_wrist,left_ankle):
    return calculate_avg_x_distance(left_hip,left_wrist,left_ankle)

def calculate_y_left_hip_avg_left_wrist_left_ankle(left_hip,left_wrist,left_ankle):
    return calculate_avg_y_distance(left_hip,left_wrist,left_ankle)

def calculate_z_left_hip_avg_left_wrist_left_ankle(left_hip,left_wrist,left_ankle):
    return calculate_avg_z_distance(left_hip,left_wrist,left_ankle)

def calculate_x_right_hip_avg_right_wrist_right_ankle(right_hip, right_wrist, right_ankle):
    return calculate_avg_x_distance(right_hip, right_wrist, right_ankle)

def calculate_y_right_hip_avg_right_wrist_right_ankle(right_hip, right_wrist, right_ankle):
    return calculate_avg_y_distance(right_hip, right_wrist, right_ankle)

def calculate_z_right_hip_avg_right_wrist_right_ankle(right_hip, right_wrist, right_ankle):
    return calculate_avg_z_distance(right_hip, right_wrist, right_ankle)


########################################################################################################################

def out():



        landmark_names = [
            "x_nose", "y_nose", "z_nose",
            "x_left_eye_inner", "y_left_eye_inner", "z_left_eye_inner",
            "x_left_eye", "y_left_eye", "z_left_eye",
            "x_left_eye_outer", "y_left_eye_outer", "z_left_eye_outer",
            "x_right_eye_inner", "y_right_eye_inner", "z_right_eye_inner",
            "x_right_eye", "y_right_eye", "z_right_eye",
            "x_right_eye_outer", "y_right_eye_outer", "z_right_eye_outer",
            "x_left_ear", "y_left_ear", "z_left_ear",
            "x_right_ear", "y_right_ear", "z_right_ear",
            "x_mouth_left", "y_mouth_left", "z_mouth_left",
            "x_mouth_right", "y_mouth_right", "z_mouth_right",
            "x_left_shoulder", "y_left_shoulder", "z_left_shoulder",
            "x_right_shoulder", "y_right_shoulder", "z_right_shoulder",
            "x_left_elbow", "y_left_elbow", "z_left_elbow",
            "x_right_elbow", "y_right_elbow", "z_right_elbow",
            "x_left_wrist", "y_left_wrist", "z_left_wrist",
            "x_right_wrist", "y_right_wrist", "z_right_wrist",
            "x_left_pinky_1", "y_left_pinky_1", "z_left_pinky_1",
            "x_right_pinky_1", "y_right_pinky_1", "z_right_pinky_1",
            "x_left_index_1", "y_left_index_1", "z_left_index_1",
            "x_right_index_1", "y_right_index_1", "z_right_index_1",
            "x_left_thumb_2", "y_left_thumb_2", "z_left_thumb_2",
            "x_right_thumb_2", "y_right_thumb_2", "z_right_thumb_2",
            "x_left_hip", "y_left_hip", "z_left_hip",
            "x_right_hip", "y_right_hip", "z_right_hip",
            "x_left_knee", "y_left_knee", "z_left_knee",
            "x_right_knee", "y_right_knee", "z_right_knee",
            "x_left_ankle", "y_left_ankle", "z_left_ankle",
            "x_right_ankle", "y_right_ankle", "z_right_ankle",
            "x_left_heel", "y_left_heel", "z_left_heel",
            "x_right_heel", "y_right_heel", "z_right_heel",
            "x_left_foot_index", "y_left_foot_index", "z_left_foot_index",
            "x_right_foot_index", "y_right_foot_index", "z_right_foot_index"
        ]
        print(len(landmark_names))
class Landmark:
    def __init__(self, name, x, y, z):

        self.name = name
        self.x = x
        self.y = y
        self.z = z
        #self.visibility = visibility

    def __repr__(self):
        """Returns a string representation of the Landmark object."""
        return f"{self.name}: (x={self.x}, y={self.y}, z={self.z})"

    def __sub__(self, other):
        """Defines subtraction between two Landmark objects (returns a vector difference)."""
        if isinstance(other, Landmark):
            return (self.x - other.x, self.y - other.y, self.z - other.z)
        raise TypeError(f"Unsupported operation between Landmark and {type(other)}")

    def to_dict(self):
        """Converts the object into a dictionary format."""
        return {"name": self.name, "x": self.x, "y": self.y, "z": self.z}

def calculate_angle_3points(landmark1,landmark2,landmark3):
    A,B,C=np.array(landmark1), np.array(landmark2), np.array(landmark3)
    BA = A - B
    BC = C - B

    dot_product = np.dot(BA, BC)


    mag_BA = np.linalg.norm(BA)
    mag_BC = np.linalg.norm(BC)

    angle_radians = np.arccos(dot_product / (mag_BA * mag_BC))

    angle_degrees = np.degrees(angle_radians)

    return angle_degrees
def caluclate_3d_distance_2points(landmark1,landmark2):


        return math.sqrt(
            (landmark1.x - landmark2.x) ** 2 +
            (landmark1.y - landmark2.y) ** 2 +
            (landmark1.z - landmark2.z) ** 2
        )



def calculate_x_distances(num1, num2):

    dx = abs(num1- num2)
    return dx
def calculate_y_distances(num1, num2):

    dy = abs(num1- num2)
    return dy
def calculate_z_distances(num1, num2):

    dz = abs(num1- num2)
    return dz
def average_3d_distance(landmark1, landmark2, landmark3):
    d1 = caluclate_3d_distance_2points(landmark1, landmark2)
    d2 = caluclate_3d_distance_2points(landmark1, landmark3)
    return (d1 + d2) / 2


# Function to calculate Euclidean distance between two 3D points
# Function to calculate the average of three landmarks (left_hip, left_wrist, left_ankle)
def calculate_avg_distance(landmark1, landmark2, landmark3):
    x_avg = (landmark1.x + landmark2.x + landmark3.x) / 3
    y_avg = (landmark1.y + landmark2.y + landmark3.y) / 3
    z_avg = (landmark1.z + landmark2.z + landmark3.z) / 3
    return math.sqrt(x_avg**2 + y_avg**2 + z_avg**2)

def calculate_avg_x_distance(landmark1, landmark2, landmark3):
    """Calculate the 3D Euclidean norm of the average landmark."""
    x_avg = (landmark1.x + landmark2.x + landmark3.x) / 3
    return x_avg

def calculate_avg_y_distance(landmark1, landmark2, landmark3):
    """Calculate the 3D Euclidean norm of the average landmark."""
    y_avg = (landmark1.y + landmark2.y + landmark3.y) / 3
    return y_avg

def calculate_avg_z_distance(landmark1, landmark2, landmark3):
    """Calculate the 3D Euclidean norm of the average landmark."""
    z_avg = (landmark1.z + landmark2.z + landmark3.z) / 3
    return z_avg

def calculate_mid_hip(left_hip, right_hip):
    """Calculate the mid_hip by averaging the 3D coordinates of left_hip and right_hip."""
    x_mid_hip = (left_hip.x + right_hip.x) / 2
    y_mid_hip = (left_hip.y + right_hip.y) / 2
    z_mid_hip = (left_hip.z + right_hip.z) / 2

    # Return the mid_hip as a new Landmark object
    return Landmark("mid_hip", x_mid_hip, y_mid_hip, z_mid_hip)
import math

def calculate_right_hip_avg_right_wrist_right_ankle(right_hip, right_wrist, right_ankle):
    scale_factor = 1.8
    def euclidean_distance(point1, point2):
        """Calculate the Euclidean distance between two 3D points."""
        return math.sqrt((point2.x - point1.x)**2 + (point2.y - point1.y)**2 + (point2.z - point1.z)**2)

    # Calculate distances
    distance_hip_wrist = euclidean_distance(right_hip, right_wrist)
    distance_hip_ankle = euclidean_distance(right_hip, right_ankle)

    # Calculate the average distance
    average_distance = (distance_hip_wrist + distance_hip_ankle) / 2

    # Apply scaling factor
    scaled_average_distance = average_distance / scale_factor

    return scaled_average_distance


class Frame:
    prdected_classes={'jumping_jacks_down':0, 'jumping_jacks_up':0, 'pullups_down':0,
                      'pullups_up':0,'pushups_down':0,'pushups_up':0,'situp_down':0,
                      'situp_up':0,'squats_down':0,'squats_up':0}

    def __init__(self,NOSE, LEFT_EYE_INNER,  LEFT_EYE,LEFT_EYE_OUTER,  RIGHT_EYE_INNER,RIGHT_EYE, 
                  RIGHT_EYE_OUTER,  LEFT_EAR,  RIGHT_EAR,  MOUTH_LEFT,  MOUTH_RIGHT,  LEFT_SHOULDER,
    RIGHT_SHOULDER,
    LEFT_ELBOW,
    RIGHT_ELBOW,
    LEFT_WRIST,
    RIGHT_WRIST,
    LEFT_PINKY,
    RIGHT_PINKY,
    LEFT_INDEX_FINGER,
    RIGHT_INDEX_FINGER,
    LEFT_THUMB,
    RIGHT_THUMB,
    LEFT_HIP,
    RIGHT_HIP,
    LEFT_KNEE,
    RIGHT_KNEE,
    LEFT_ANKLE,
    RIGHT_ANKLE,
    LEFT_HEEL,
    RIGHT_HEEL,
    LEFT_FOOT_INDEX,
    RIGHT_FOOT_INDEX,
    FRAME_INDEX  ):
        self.NOSE=NOSE
        self.LEFT_EYE_INNER = LEFT_EYE_INNER
        self.LEFT_EYE = LEFT_EYE
        self.LEFT_EYE_OUTER = LEFT_EYE_OUTER
        self.RIGHT_EYE_INNER = RIGHT_EYE_INNER
        self.RIGHT_EYE = RIGHT_EYE
        self.RIGHT_EYE_OUTER = RIGHT_EYE_OUTER
        self.LEFT_EAR = LEFT_EAR
        self.RIGHT_EAR = RIGHT_EAR
        self.MOUTH_LEFT = MOUTH_LEFT
        self.MOUTH_RIGHT = MOUTH_RIGHT
        self.LEFT_SHOULDER = LEFT_SHOULDER
        self.RIGHT_SHOULDER = RIGHT_SHOULDER
        self.LEFT_ELBOW = LEFT_ELBOW
        self.RIGHT_ELBOW = RIGHT_ELBOW
        self.LEFT_WRIST = LEFT_WRIST
        self.RIGHT_WRIST = RIGHT_WRIST
        self.LEFT_PINKY = LEFT_PINKY
        self.RIGHT_PINKY = RIGHT_PINKY
        self.LEFT_INDEX_FINGER = LEFT_INDEX_FINGER
        self.RIGHT_INDEX_FINGER = RIGHT_INDEX_FINGER
        self.LEFT_THUMB = LEFT_THUMB
        self.RIGHT_THUMB = RIGHT_THUMB
        self.LEFT_HIP = LEFT_HIP
        self.RIGHT_HIP = RIGHT_HIP
        self.LEFT_KNEE = LEFT_KNEE
        self.RIGHT_KNEE = RIGHT_KNEE
        self.LEFT_ANKLE = LEFT_ANKLE
        self.RIGHT_ANKLE = RIGHT_ANKLE
        self.LEFT_HEEL = LEFT_HEEL
        self.RIGHT_HEEL = RIGHT_HEEL
        self.LEFT_FOOT_INDEX = LEFT_FOOT_INDEX
        self.RIGHT_FOOT_INDEX = RIGHT_FOOT_INDEX

        model_path = r"pose_classification_XGBoost_model2.pkl"
        pose_classifier = joblib.load(model_path)
        left_shoulder_left_wrist_3d = calculate_listof_3dDistancefeatures_left_shoulder_left_wrist(self.LEFT_SHOULDER,
                                                                                                   self.LEFT_WRIST)
        right_shoulder_right_wrist_3d = calculate_listof_3dDistancefeatures_right_shoulder_right_wrist(
            self.RIGHT_SHOULDER, self.RIGHT_WRIST)
        left_hip_left_ankle_3d = calculate_listof_3dDistancefeatures_left_hip_left_ankle(self.LEFT_HIP, self.LEFT_ANKLE)
        right_hip_right_ankle_3d = calculate_listof_3dDistancefeatures_right_hip_right_ankle(self.RIGHT_HIP,
                                                                                             self.RIGHT_ANKLE)
        left_hip_left_wrist_3d = calculate_listof_3dDistancefeatures_left_hip_left_wrist(self.LEFT_HIP, self.LEFT_WRIST)
        right_hip_right_wrist_3d = calculate_listof_3dDistancefeatures_right_hip_right_wrist(self.RIGHT_HIP,
                                                                                             self.RIGHT_WRIST)
        left_shoulder_left_ankle_3d = calculate_listof_3dDistancefeatures_left_shoulder_left_ankle(self.LEFT_SHOULDER,
                                                                                                   self.LEFT_ANKLE)
        right_shoulder_right_ankle_3d = calculate_listof_3dDistancefeatures_right_shoulder_right_ankle(
            self.RIGHT_SHOULDER, self.RIGHT_ANKLE)
        left_hip_right_wrist_3d = calculate_listof_3dDistancefeatures_left_hip_right_wrist(self.LEFT_HIP,
                                                                                           self.RIGHT_WRIST)
        right_hip_left_wrist_3d = calculate_listof_3dDistancefeatures_right_hip_left_wrist(self.RIGHT_HIP,
                                                                                           self.LEFT_WRIST)
        left_elbow_right_elbow_3d = calculate_listof_3dDistancefeatures_left_elbow_right_elbow(self.LEFT_ELBOW,
                                                                                               self.RIGHT_ELBOW)
        left_knee_right_knee_3d = calculate_listof_3dDistancefeatures_left_knee_right_knee(self.LEFT_KNEE,
                                                                                           self.RIGHT_KNEE)
        left_wrist_right_wrist_3d = calculate_listof_3dDistancefeatures_left_wrist_right_wrist(self.LEFT_WRIST,
                                                                                               self.RIGHT_WRIST)
        left_ankle_right_ankle_3d = calculate_listof_3dDistancefeatures_left_ankle_right_ankle(self.LEFT_ANKLE,
                                                                                               self.RIGHT_ANKLE)
        left_hip_avg_left_wrist_left_ankle_3d = calculate_right_hip_avg_right_wrist_right_ankle(
            self.LEFT_HIP, self.LEFT_WRIST, self.LEFT_ANKLE)
        right_hip_avg_right_wrist_right_ankle_3d = calculate_right_hip_avg_right_wrist_right_ankle(
            self.RIGHT_HIP, self.RIGHT_WRIST, self.RIGHT_ANKLE)
        ################################################################################################################
        #########################################################
        x_left_shoulder_left_wrist = calculate_listof_xyz_distancesfeature_x_left_shoulder_left_wrist(
            self.LEFT_SHOULDER, self.LEFT_WRIST)
        y_left_shoulder_left_wrist = calculate_listof_xyz_distancesfeature_y_left_shoulder_left_wrist(
            self.LEFT_SHOULDER, self.LEFT_WRIST)
        z_left_shoulder_left_wrist = calculate_listof_xyz_distancesfeature_z_left_shoulder_left_wrist(
            self.LEFT_SHOULDER, self.LEFT_WRIST)

        x_right_shoulder_right_wrist = calculate_listof_xyz_distancesfeature_x_right_shoulder_right_wrist(
            self.RIGHT_SHOULDER, self.RIGHT_WRIST)
        y_right_shoulder_right_wrist = calculate_listof_xyz_distancesfeature_y_right_shoulder_right_wrist(
            self.RIGHT_SHOULDER, self.RIGHT_WRIST)
        z_right_shoulder_right_wrist = calculate_listof_xyz_distancesfeature_z_right_shoulder_right_wrist(
            self.RIGHT_SHOULDER, self.RIGHT_WRIST)

        x_left_hip_left_ankle = calculate_listof_xyz_distancesfeature_x_left_hip_left_ankle(self.LEFT_HIP,
                                                                                            self.LEFT_ANKLE)
        y_left_hip_left_ankle = calculate_listof_xyz_distancesfeature_y_left_hip_left_ankle(self.LEFT_HIP,
                                                                                            self.LEFT_ANKLE)
        z_left_hip_left_ankle = calculate_listof_xyz_distancesfeature_z_left_hip_left_ankle(self.LEFT_HIP,
                                                                                            self.LEFT_ANKLE)

        x_right_hip_right_ankle = calculate_listof_xyz_distancesfeature_x_right_hip_right_ankle(self.RIGHT_HIP,
                                                                                                self.RIGHT_ANKLE)
        y_right_hip_right_ankle = calculate_listof_xyz_distancesfeature_y_right_hip_right_ankle(self.RIGHT_HIP,
                                                                                                self.RIGHT_ANKLE)
        z_right_hip_right_ankle = calculate_listof_xyz_distancesfeature_z_right_hip_right_ankle(self.RIGHT_HIP,
                                                                                                self.RIGHT_ANKLE)

        x_left_hip_left_wrist = calculate_listof_xyz_distancesfeature_x_left_hip_left_wrist(self.LEFT_HIP,
                                                                                            self.LEFT_WRIST)
        y_left_hip_left_wrist = calculate_listof_xyz_distancesfeature_y_left_hip_left_wrist(self.LEFT_HIP,
                                                                                            self.LEFT_WRIST)
        z_left_hip_left_wrist = calculate_listof_xyz_distancesfeature_z_left_hip_left_wrist(self.LEFT_HIP,
                                                                                            self.LEFT_WRIST)

        x_right_hip_right_wrist = calculate_listof_xyz_distancesfeature_x_right_hip_right_wrist(self.RIGHT_HIP,
                                                                                                self.RIGHT_WRIST)
        y_right_hip_right_wrist = calculate_listof_xyz_distancesfeature_y_right_hip_right_wrist(self.RIGHT_HIP,
                                                                                                self.RIGHT_WRIST)
        z_right_hip_right_wrist = calculate_listof_xyz_distancesfeature_z_right_hip_right_wrist(self.RIGHT_HIP,
                                                                                                self.RIGHT_WRIST)

        x_left_shoulder_left_ankle = calculate_listof_xyz_distancesfeature_x_left_shoulder_left_ankle(
            self.LEFT_SHOULDER, self.LEFT_ANKLE)
        y_left_shoulder_left_ankle = calculate_listof_xyz_distancesfeature_y_left_shoulder_left_ankle(
            self.LEFT_SHOULDER, self.LEFT_ANKLE)
        z_left_shoulder_left_ankle = calculate_listof_xyz_distancesfeature_z_left_shoulder_left_ankle(
            self.LEFT_SHOULDER, self.LEFT_ANKLE)

        x_right_shoulder_right_ankle = calculate_listof_xyz_distancesfeature_x_right_shoulder_right_ankle(
            self.RIGHT_SHOULDER, self.RIGHT_ANKLE)
        y_right_shoulder_right_ankle = calculate_listof_xyz_distancesfeature_y_right_shoulder_right_ankle(
            self.RIGHT_SHOULDER, self.RIGHT_ANKLE)
        z_right_shoulder_right_ankle = calculate_listof_xyz_distancesfeature_z_right_shoulder_right_ankle(
            self.RIGHT_SHOULDER, self.RIGHT_ANKLE)

        x_left_hip_right_wrist = calculate_listof_xyz_distancesfeature_x_left_hip_right_wrist(self.LEFT_HIP,
                                                                                              self.RIGHT_WRIST)
        y_left_hip_right_wrist = calculate_listof_xyz_distancesfeature_y_left_hip_right_wrist(self.LEFT_HIP,
                                                                                              self.RIGHT_WRIST)
        z_left_hip_right_wrist = calculate_listof_xyz_distancesfeature_z_left_hip_right_wrist(self.LEFT_HIP,
                                                                                              self.RIGHT_WRIST)

        x_right_hip_left_wrist = calculate_listof_xyz_distancesfeature_x_right_hip_left_wrist(self.RIGHT_HIP,
                                                                                              self.LEFT_WRIST)
        y_right_hip_left_wrist = calculate_listof_xyz_distancesfeature_y_right_hip_left_wrist(self.RIGHT_HIP,
                                                                                              self.LEFT_WRIST)
        z_right_hip_left_wrist = calculate_listof_xyz_distancesfeature_z_right_hip_left_wrist(self.RIGHT_HIP,
                                                                                              self.LEFT_WRIST)

        x_left_elbow_right_elbow = calculate_listof_xyz_distancesfeature_x_left_elbow_right_elbow(self.LEFT_ELBOW,
                                                                                                  self.RIGHT_ELBOW)
        y_left_elbow_right_elbow = calculate_listof_xyz_distancesfeature_y_left_elbow_right_elbow(self.LEFT_ELBOW,
                                                                                                  self.RIGHT_ELBOW)
        z_left_elbow_right_elbow = calculate_listof_xyz_distancesfeature_z_left_elbow_right_elbow(self.LEFT_ELBOW,
                                                                                                  self.RIGHT_ELBOW)

        x_left_knee_right_knee = calculate_listof_xyz_distancesfeature_x_left_knee_right_knee(self.LEFT_KNEE,
                                                                                              self.RIGHT_KNEE)
        y_left_knee_right_knee = calculate_listof_xyz_distancesfeature_y_left_knee_right_knee(self.LEFT_KNEE,
                                                                                              self.RIGHT_KNEE)
        z_left_knee_right_knee = calculate_listof_xyz_distancesfeature_z_left_knee_right_knee(self.LEFT_KNEE,
                                                                                              self.RIGHT_KNEE)

        x_left_wrist_right_wrist = calculate_listof_xyz_distancesfeature_x_left_wrist_right_wrist(self.LEFT_WRIST,
                                                                                                  self.RIGHT_WRIST)
        y_left_wrist_right_wrist = calculate_listof_xyz_distancesfeature_y_left_wrist_right_wrist(self.LEFT_WRIST,
                                                                                                  self.RIGHT_WRIST)
        z_left_wrist_right_wrist = calculate_listof_xyz_distancesfeature_z_left_wrist_right_wrist(self.LEFT_WRIST,
                                                                                                  self.RIGHT_WRIST)

        x_left_ankle_right_ankle = calculate_listof_xyz_distancesfeature_x_left_ankle_right_ankle(self.LEFT_ANKLE,
                                                                                                  self.RIGHT_ANKLE)
        y_left_ankle_right_ankle = calculate_listof_xyz_distancesfeature_y_left_ankle_right_ankle(self.LEFT_ANKLE,
                                                                                                  self.RIGHT_ANKLE)
        z_left_ankle_right_ankle = calculate_listof_xyz_distancesfeature_z_left_ankle_right_ankle(self.LEFT_ANKLE,
                                                                                                  self.RIGHT_ANKLE)

        x_left_hip_avg_left_wrist_left_ankle = calculate_avg_x_distance(
            self.LEFT_HIP, self.LEFT_WRIST, self.LEFT_ANKLE)
        y_left_hip_avg_left_wrist_left_ankle = calculate_avg_y_distance(
            self.LEFT_HIP, self.LEFT_WRIST, self.LEFT_ANKLE)
        z_left_hip_avg_left_wrist_left_ankle = calculate_avg_z_distance(
            self.LEFT_HIP, self.LEFT_WRIST, self.LEFT_ANKLE)

        x_right_hip_avg_right_wrist_right_ankle = calculate_avg_x_distance(
            self.RIGHT_HIP, self.RIGHT_WRIST, self.RIGHT_ANKLE)
        y_right_hip_avg_right_wrist_right_ankle = calculate_avg_y_distance(
            self.RIGHT_HIP, self.RIGHT_WRIST, self.RIGHT_ANKLE)
        z_right_hip_avg_right_wrist_right_ankle = calculate_avg_z_distance(
            self.RIGHT_HIP, self.RIGHT_WRIST, self.RIGHT_ANKLE)

        ################################################################################################################
        right_elbow_right_shoulder_right_hip = calculate_listof_anglesfeatures_right_elbow_right_shoulder_right_hip(
            self.RIGHT_ELBOW, self.RIGHT_SHOULDER, self.RIGHT_HIP)
        left_elbow_left_shoulder_left_hip = calculate_listof_anglesfeatures_left_elbow_left_shoulder_left_hip(
            self.LEFT_ELBOW, self.LEFT_SHOULDER, self.LEFT_HIP)
        right_knee_mid_hip_left_knee = calculate_listof_anglesfeatures_right_knee_mid_hip_left_knee(self.RIGHT_KNEE,
                                                                                                    calculate_mid_hip(LEFT_HIP, RIGHT_HIP),
                                                                                                    self.LEFT_KNEE)
        right_hip_right_knee_right_ankle = calculate_listof_anglesfeatures_right_hip_right_knee_right_ankle(
            self.RIGHT_HIP, self.RIGHT_KNEE, self.RIGHT_ANKLE)
        left_hip_left_knee_left_ankle = calculate_listof_anglesfeatures_left_hip_left_knee_left_ankle(self.LEFT_HIP,
                                                                                                      self.LEFT_KNEE,
                                                                                                      self.LEFT_ANKLE)
        right_wrist_right_elbow_right_shoulder = calculate_listof_anglesfeatures_right_wrist_right_elbow_right_shoulder(
            self.RIGHT_WRIST, self.RIGHT_ELBOW, self.RIGHT_SHOULDER)
        left_wrist_left_elbow_left_shoulder = calculate_listof_anglesfeatures_left_wrist_left_elbow_left_shoulder(
            self.LEFT_WRIST, self.LEFT_ELBOW, self.LEFT_SHOULDER)
        ################################################################################################################
        #
        # def IS_coorect_fun():
        #     if Is_correct==r"D:\1223.mp4":
        #         return "Correct pushups_up "
        #     elif Is_correct==r"D:\123.mp4":
        #         return "Correct squats "
        #     else :
        #         return "wronge pose "
        features = {
            "x_nose": self.NOSE.x, "y_nose": self.NOSE.y, "z_nose": self.NOSE.z,
            "x_left_eye_inner": self.LEFT_EYE_INNER.x, "y_left_eye_inner": self.LEFT_EYE_INNER.y, "z_left_eye_inner": self.LEFT_EYE_INNER.z,
            "x_left_eye": self.LEFT_EYE.x, "y_left_eye": self.LEFT_EYE.y, "z_left_eye": self.LEFT_EYE.z,
            "x_left_eye_outer": self.LEFT_EYE_OUTER.x, "y_left_eye_outer": self.LEFT_EYE_OUTER.y, "z_left_eye_outer": self.LEFT_EYE_OUTER.z,
            "x_right_eye_inner": self.RIGHT_EYE_INNER.x, "y_right_eye_inner": self.RIGHT_EYE_INNER.y, "z_right_eye_inner": self.RIGHT_EYE_INNER.z,
            "x_right_eye": self.RIGHT_EYE.x, "y_right_eye": self.RIGHT_EYE.y, "z_right_eye": self.RIGHT_EYE.z,
            "x_right_eye_outer": self.RIGHT_EYE_OUTER.x, "y_right_eye_outer": self.RIGHT_EYE_OUTER.y, "z_right_eye_outer": self.RIGHT_EYE_OUTER.z,
            "x_left_ear": self.LEFT_EAR.x, "y_left_ear": self.LEFT_EAR.y, "z_left_ear": self.LEFT_EAR.z,
            "x_right_ear": self.RIGHT_EAR.x, "y_right_ear": self.RIGHT_EAR.y, "z_right_ear": self.RIGHT_EAR.z,
            "x_mouth_left": self.MOUTH_LEFT.x, "y_mouth_left": self.MOUTH_LEFT.y, "z_mouth_left": self.MOUTH_LEFT.z,
            "x_mouth_right": self.MOUTH_RIGHT.x, "y_mouth_right": self.MOUTH_RIGHT.y, "z_mouth_right": self.MOUTH_RIGHT.z,
            "x_left_shoulder": self.LEFT_SHOULDER.x, "y_left_shoulder": self.LEFT_SHOULDER.y, "z_left_shoulder": self.LEFT_SHOULDER.z,
            "x_right_shoulder": self.RIGHT_SHOULDER.x, "y_right_shoulder": self.RIGHT_SHOULDER.y, "z_right_shoulder": self.RIGHT_SHOULDER.z,
            "x_left_elbow": self.LEFT_ELBOW.x, "y_left_elbow": self.LEFT_ELBOW.y, "z_left_elbow": self.LEFT_ELBOW.z,
            "x_right_elbow": self.RIGHT_ELBOW.x, "y_right_elbow": self.RIGHT_ELBOW.y, "z_right_elbow": self.RIGHT_ELBOW.z,
            "x_left_wrist": self.LEFT_WRIST.x, "y_left_wrist": self.LEFT_WRIST.y, "z_left_wrist": self.LEFT_WRIST.z,
            "x_right_wrist": self.RIGHT_WRIST.x, "y_right_wrist": self.RIGHT_WRIST.y, "z_right_wrist": self.RIGHT_WRIST.z,
            "x_left_pinky_1": self.LEFT_PINKY.x, "y_left_pinky_1": self.LEFT_PINKY.y, "z_left_pinky_1": self.LEFT_PINKY.z,
            "x_right_pinky_1": self.RIGHT_PINKY.x, "y_right_pinky_1": self.RIGHT_PINKY.y, "z_right_pinky_1": self.RIGHT_PINKY.z,
            "x_left_index_1": self.LEFT_INDEX_FINGER.x, "y_left_index_1": self.LEFT_INDEX_FINGER.y, "z_left_index_1": self.LEFT_INDEX_FINGER.z,
            "x_right_index_1": self.RIGHT_INDEX_FINGER.x, "y_right_index_1": self.RIGHT_INDEX_FINGER.y, "z_right_index_1": self.RIGHT_INDEX_FINGER.z,
            "x_left_thumb_2": self.LEFT_THUMB.x, "y_left_thumb_2": self.LEFT_THUMB.y, "z_left_thumb_2": self.LEFT_THUMB.z,
            "x_right_thumb_2": self.RIGHT_THUMB.x, "y_right_thumb_2": self.RIGHT_THUMB.y, "z_right_thumb_2": self.RIGHT_THUMB.z,
            "x_left_hip": self.LEFT_HIP.x, "y_left_hip": self.LEFT_HIP.y, "z_left_hip": self.LEFT_HIP.z,
            "x_right_hip": self.RIGHT_HIP.x, "y_right_hip": self.RIGHT_HIP.y, "z_right_hip": self.RIGHT_HIP.z,
            "x_left_knee": self.LEFT_KNEE.x, "y_left_knee": self.LEFT_KNEE.y, "z_left_knee": self.LEFT_KNEE.z,
            "x_right_knee": self.RIGHT_KNEE.x, "y_right_knee": self.RIGHT_KNEE.y, "z_right_knee": self.RIGHT_KNEE.z,
            "x_left_ankle": self.LEFT_ANKLE.x, "y_left_ankle": self.LEFT_ANKLE.y, "z_left_ankle": self.LEFT_ANKLE.z,
            "x_right_ankle": self.RIGHT_ANKLE.x, "y_right_ankle": self.RIGHT_ANKLE.y, "z_right_ankle": self.RIGHT_ANKLE.z,
            "x_left_heel": self.LEFT_HEEL.x, "y_left_heel": self.LEFT_HEEL.y, "z_left_heel": self.LEFT_HEEL.z,
            "x_right_heel": self.RIGHT_HEEL.x, "y_right_heel": self.RIGHT_HEEL.y, "z_right_heel": self.RIGHT_HEEL.z,
            "x_left_foot_index": self.LEFT_FOOT_INDEX.x, "y_left_foot_index": self.LEFT_FOOT_INDEX.y, "z_left_foot_index": self.LEFT_FOOT_INDEX.z,
            "x_right_foot_index": self.RIGHT_FOOT_INDEX.x, "y_right_foot_index": self.RIGHT_FOOT_INDEX.y, "z_right_foot_index": self.RIGHT_FOOT_INDEX.z,

            # Compute 3D distance features
            "left_shoulder_left_wrist":left_shoulder_left_wrist_3d ,
            "right_shoulder_right_wrist":right_shoulder_right_wrist_3d ,
            "left_hip_left_ankle":left_hip_left_ankle_3d ,
            "right_hip_right_ankle": right_hip_right_ankle_3d,
            "left_elbow_right_elbow": left_elbow_right_elbow_3d,
            "left_knee_right_knee": left_knee_right_knee_3d,
            "left_wrist_right_wrist": left_wrist_right_wrist_3d,
            "left_ankle_right_ankle": left_ankle_right_ankle_3d,
            "left_hip_left_wrist": left_hip_left_wrist_3d,
            "right_hip_right_wrist":right_hip_right_wrist_3d,
            "left_shoulder_left_ankle":left_shoulder_left_ankle_3d,
            "right_shoulder_right_ankle":right_shoulder_right_ankle_3d,
            "left_hip_right_wrist":left_hip_right_wrist_3d,
            "right_hip_left_wrist":right_hip_left_wrist_3d,


            "right_elbow_right_shoulder_right_hip": right_elbow_right_shoulder_right_hip,
            "left_elbow_left_shoulder_left_hip": left_elbow_left_shoulder_left_hip,
            "right_hip_right_knee_right_ankle": right_hip_right_knee_right_ankle,
            "left_hip_left_knee_left_ankle": left_hip_left_knee_left_ankle,




            "x_left_shoulder_left_wrist": x_left_shoulder_left_wrist,
            "y_left_shoulder_left_wrist": y_left_shoulder_left_wrist,
            "z_left_shoulder_left_wrist": z_left_shoulder_left_wrist,

            "x_right_shoulder_right_wrist": x_right_shoulder_right_wrist,
            "y_right_shoulder_right_wrist": y_right_shoulder_right_wrist,
            "z_right_shoulder_right_wrist": z_right_shoulder_right_wrist,

            "x_left_hip_left_ankle": x_left_hip_left_ankle,
            "y_left_hip_left_ankle": y_left_hip_left_ankle,
            "z_left_hip_left_ankle": z_left_hip_left_ankle,

            "x_right_hip_right_ankle": x_right_hip_right_ankle,
            "y_right_hip_right_ankle": y_right_hip_right_ankle,
            "z_right_hip_right_ankle": z_right_hip_right_ankle,

            "x_left_hip_left_wrist": x_left_hip_left_wrist,
            "y_left_hip_left_wrist": y_left_hip_left_wrist,
            "z_left_hip_left_wrist": z_left_hip_left_wrist,

            "x_right_hip_right_wrist": x_right_hip_right_wrist,
            "y_right_hip_right_wrist": y_right_hip_right_wrist,
            "z_right_hip_right_wrist": z_right_hip_right_wrist,

            "x_left_shoulder_left_ankle": x_left_shoulder_left_ankle,
            "y_left_shoulder_left_ankle": y_left_shoulder_left_ankle,
            "z_left_shoulder_left_ankle": z_left_shoulder_left_ankle,

            "x_right_shoulder_right_ankle": x_right_shoulder_right_ankle,
            "y_right_shoulder_right_ankle": y_right_shoulder_right_ankle,
            "z_right_shoulder_right_ankle": z_right_shoulder_right_ankle,

            "x_left_hip_right_wrist": x_left_hip_right_wrist,
            "y_left_hip_right_wrist": y_left_hip_right_wrist,
            "z_left_hip_right_wrist": z_left_hip_right_wrist,

            "x_right_hip_left_wrist": x_right_hip_left_wrist,
            "y_right_hip_left_wrist": y_right_hip_left_wrist,
            "z_right_hip_left_wrist": z_right_hip_left_wrist,

            "x_left_elbow_right_elbow": x_left_elbow_right_elbow,
            "y_left_elbow_right_elbow": y_left_elbow_right_elbow,
            "z_left_elbow_right_elbow": z_left_elbow_right_elbow,

            "x_left_knee_right_knee": x_left_knee_right_knee,
            "y_left_knee_right_knee": y_left_knee_right_knee,
            "z_left_knee_right_knee": z_left_knee_right_knee,

            "x_left_wrist_right_wrist": x_left_wrist_right_wrist,
            "y_left_wrist_right_wrist": y_left_wrist_right_wrist,
            "z_left_wrist_right_wrist": z_left_wrist_right_wrist,

            "x_left_ankle_right_ankle": x_left_ankle_right_ankle,
            "y_left_ankle_right_ankle": y_left_ankle_right_ankle,
            "z_left_ankle_right_ankle": z_left_ankle_right_ankle,


            "right_elbow_right_shoulder_right_hip":right_elbow_right_shoulder_right_hip
            , "left_elbow_left_shoulder_left_hip":left_elbow_left_shoulder_left_hip
            , "right_knee_mid_hip_left_knee":right_knee_mid_hip_left_knee,
            "right_hip_right_knee_right_ankle":right_hip_right_knee_right_ankle,
            "left_hip_left_knee_left_ankle":left_hip_left_knee_left_ankle,
            "right_wrist_right_elbow_right_shoulder":right_wrist_right_elbow_right_shoulder,
            "left_wrist_left_elbow_left_shoulder":left_wrist_left_elbow_left_shoulder

        }
        # Convert features dictionary to DataFrame
        feature_df = pd.DataFrame([features])

        # Ensure all expected feature columns are present
        expected_features = pose_classifier.get_booster().feature_names
        feature_df = feature_df.reindex(columns=expected_features, fill_value=0)

        # Predict pose class
        prediction = pose_classifier.predict(feature_df)

        # Get the predicted class
        predicted_class = prediction[0]

        #print(f"Predicted Pose Class: {predicted_class}")
        self.pose_classifier = joblib.load(model_path)
        self.label_encoder = joblib.load(r"label_encoder_No_avg.pkl")
        pose_class = self.label_encoder.inverse_transform([predicted_class])[0]
        # pose_class = pose_class.replace('_down', '').replace('_up', '')
        self.prdected_classes[pose_class]=self.prdected_classes[pose_class]+1
        li.append(pose_class)
        frame_map={pose_class:FRAME_INDEX}
        li.append(frame_map)

        #print(f"Predicted Pose Class: {pose_class}")
        #print(prdected_classes)
        # if pose_class==6 :
    # def return_value(self):

    
    #     return self.prdected_classes

    def return_value(self):
        filtered_dict = {k: v for k, v in self.prdected_classes.items() if v != 0}
        min_key = min(filtered_dict, key=filtered_dict.get)
        minkey=re.sub(r'(up|down)', '', min_key)

        return min_key
    
    def return_nurmal(self):
        filtered_dict = {k: v for k, v in self.prdected_classes.items() if v != 0}
        min_key = min(filtered_dict, key=filtered_dict.get)
        return min_key
          


    def return_zero(self):
         prdected_classes={'jumping_jacks_down':0, 'jumping_jacks_up':0, 
                          'pullups_down':0,'pullups_up':0,'pushups_down':0,'pushups_up':0
                          ,'situp_down':0,'situp_up':0,'squats_down':0,'squats_up':0}
         return prdected_classes
