import cv2
import argparse # for CLI
import numpy as np

def parsing_CLI_arg():
    parser = argparse.ArgumentParser(description='DMS')

    # selection the camera number, default is 0 (webcam)
    parser.add_argument('-c', '--camera', type=int,
                        default=0, metavar='', help='Camera number, default is 0 (webcam)')

    # TODO: add option for choose if use camera matrix and dist coeffs

    # visualisation parameters
    parser.add_argument('--show_fps', type=bool, default=False,
                        metavar='', help='Show the actual FPS of the capture stream, default is true')
    parser.add_argument('--show_proc_time', type=bool, default=False,
                        metavar='', help='Show the processing time for a single frame, default is true')
    parser.add_argument('--show_eye_proc', type=bool, default=False,
                        metavar='', help='Show the eyes processing, deafult is false')
    parser.add_argument('--show_axis', type=bool, default=True,
                        metavar='', help='Show the head pose axis, default is true')
    parser.add_argument('--verbose', type=bool, default=False,
                        metavar='', help='Prints additional info, default is false')

    # Attention Scorer parameters (EAR, Gaze Score, Pose)
    parser.add_argument('--smooth_factor', type=float, default=0.5,
                        metavar='', help='Sets the smooth factor for the head pose estimation keypoint smoothing, default is 0.5')
    parser.add_argument('--ear_thresh', type=float, default=0.15,
                        metavar='', help='Sets the EAR threshold for the Attention Scorer, default is 0.15')
    parser.add_argument('--ear_time_thresh', type=float, default=2,
                        metavar='', help='Sets the EAR time (seconds) threshold for the Attention Scorer, default is 2 seconds')
    parser.add_argument('--gaze_thresh', type=float, default=0.015,
                        metavar='', help='Sets the Gaze Score threshold for the Attention Scorer, default is 0.2')
    parser.add_argument('--gaze_time_thresh', type=float, default=2, metavar='',
                        help='Sets the Gaze Score time (seconds) threshold for the Attention Scorer, default is 2. seconds')
    parser.add_argument('--pitch_thresh', type=float, default=20,
                        metavar='', help='Sets the PITCH threshold (degrees) for the Attention Scorer, default is 30 degrees')
    parser.add_argument('--yaw_thresh', type=float, default=20,
                        metavar='', help='Sets the YAW threshold (degrees) for the Attention Scorer, default is 20 degrees')
    parser.add_argument('--roll_thresh', type=float, default=20,
                        metavar='', help='Sets the ROLL threshold (degrees) for the Attention Scorer, default is 30 degrees')
    parser.add_argument('--pose_time_thresh', type=float, default=2.5,
                        metavar='', help='Sets the Pose time threshold (seconds) for the Attention Scorer, default is 2.5 seconds')

    # parse the arguments and store them in the args variable dictionary
    args = parser.parse_args()

    if args.verbose:
        print(f"Arguments and Parameters used:\n{args}\n")

    return args

def get_cam_matrix():
    # camera matrix obtained from the camera calibration script, using a 9x6 chessboard
    camera_matrix = np.array(
        [[899.12150372, 0., 644.26261492],
        [0., 899.45280671, 372.28009436],
        [0, 0,  1]], dtype="double")

    # distortion coefficients obtained from the camera calibration script, using a 9x6 chessboard
    dist_coeffs = np.array(
        [[-0.03792548, 0.09233237, 0.00419088, 0.00317323, -0.15804257]], dtype="double")

# Input: list of face mesh, Output: bounding box of largest face
def _get_landmarks(lms):
    surface = 0 # why tf surface is always 0
    for lms0 in lms:
        landmarks = [np.array([point.x, point.y, point.z]) \
                        for point in lms0.landmark]

        landmarks = np.array(landmarks)
        # Normalizes x, y to b.w 0-1
        landmarks[landmarks[:, 0] < 0., 0] = 0.
        landmarks[landmarks[:, 0] > 1., 0] = 1.
        landmarks[landmarks[:, 1] < 0., 1] = 0.
        landmarks[landmarks[:, 1] > 1., 1] = 1.

        dx = landmarks[:, 0].max() - landmarks[:, 0].min() # Get dimension
        dy = landmarks[:, 1].max() - landmarks[:, 1].min()

        new_surface = dx * dy # Calculate surface area 
        if new_surface > surface:
            biggest_face = landmarks

    return biggest_face

def resize(frame, scale_percent):
    """
    Resize the image maintaining the aspect ratio
    :param frame: opencv image/frame
    :param scale_percent: int
        scale factor for resizing the image
    :return:
    resized: rescaled opencv image/frame
    """
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    dim = (width, height)

    resized = cv2.resize(frame, dim, interpolation=cv2.INTER_LINEAR)
    return resized


def get_face_area(face):
    """
    Computes the area of the bounding box ROI of the face detected by the dlib face detector
    It's used to sort the detected faces by the box area

    :param face: dlib bounding box of a detected face in faces
    :return: area of the face bounding box
    """
    return abs((face.left() - face.right()) * (face.bottom() - face.top()))


def show_keypoints(keypoints, frame):
    """
    Draw circles on the opencv frame over the face keypoints predicted by the dlib predictor

    :param keypoints: dlib iterable 68 keypoints object
    :param frame: opencv frame
    :return: frame
        Returns the frame with all the 68 dlib face keypoints drawn
    """
    for n in range(0, 68):
        x = keypoints.part(n).x
        y = keypoints.part(n).y
        cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
        return frame


def midpoint(p1, p2):
    """
    Compute the midpoint between two dlib keypoints

    :param p1: dlib single keypoint
    :param p2: dlib single keypoint
    :return: array of x,y coordinated of the midpoint between p1 and p2
    """
    return np.array([int((p1.x + p2.x) / 2), int((p1.y + p2.y) / 2)])


def get_array_keypoints(landmarks, dtype="int", verbose: bool = False):
    """
    Converts all the iterable dlib 68 face keypoint in a numpy array of shape 68,2

    :param landmarks: dlib iterable 68 keypoints object
    :param dtype: dtype desired in output
    :param verbose: if set to True, prints array of keypoints (default is False)
    :return: points_array
        Numpy array containing all the 68 keypoints (x,y) coordinates
        The shape is 68,2
    """
    points_array = np.zeros((68, 2), dtype=dtype)
    for i in range(0, 68):
        points_array[i] = (landmarks.part(i).x, landmarks.part(i).y)

    if verbose:
        print(points_array)

    return points_array


def isRotationMatrix(R, precision=1e-4):
    """
    Checks if a matrix is a rotation matrix
    :param R: np.array matrix of 3 by 3
    :param precision: float
        precision to respect to accept a zero value in identity matrix check (default is 1e-4)
    :return: True or False
        Return True if a matrix is a rotation matrix, False if not
    """
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < precision


def rotationMatrixToEulerAngles(R, precision=1e-4):
    '''
    Computes the Tait–Bryan Euler (XYZ) angles from a Rotation Matrix.
    Also checks if there is a gymbal lock and eventually use an alternative formula
    :param R: np.array
        3 x 3 Rotation matrix
    :param precision: float
        precision to respect to accept a zero value in identity matrix check (default is 1e-4)
    :return: (yaw, pitch, roll) tuple of float numbers
        Euler angles in radians in the order of YAW, PITCH, ROLL
    '''

    # Calculates Tait–Bryan Euler angles from a Rotation Matrix
    assert (isRotationMatrix(R, precision))  # check if it's a Rmat

    # assert that sqrt(R11^2 + R21^2) != 0
    sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < precision

    if not singular:  # if not in a singularity, use the standard formula
        x = np.arctan2(R[2, 1], R[2, 2])  # atan2(R31, R33) -> YAW, angle PSI

        # atan2(-R31, sqrt(R11^2 + R21^2)) -> PITCH, angle delta
        y = np.arctan2(-R[2, 0], sy)

        z = np.arctan2(R[1, 0], R[0, 0])  # atan2(R21,R11) -> ROLL, angle phi

    else:  # if in gymbal lock, use different formula for yaw, pitch roll
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arctan2(-R[2, 0], sy)
        z = 0

    return np.array([x, y, z])  # returns YAW, PITCH, ROLL angles in radians


def draw_pose_info(frame, img_point, point_proj, roll=None, pitch=None, yaw=None):
    """
    Draw 3d orthogonal axis given a frame, a point in the frame, the projection point array.
    Also prints the information about the roll, pitch and yaw if passed

    :param frame: opencv image/frame
    :param img_point: tuple
        x,y position in the image/frame for the 3d axis for the projection
    :param point_proj: np.array
        Projected point along 3 axis obtained from the cv2.projectPoints function
    :param roll: float, optional
    :param pitch: float, optional
    :param yaw: float, optional
    :return: frame: opencv image/frame
        Frame with 3d axis drawn and, optionally, the roll,pitch and yaw values drawn
    """
    frame = cv2.line(frame, img_point, tuple(
        point_proj[0].ravel().astype(int)), (255, 0, 0), 3)
    frame = cv2.line(frame, img_point, tuple(
        point_proj[1].ravel().astype(int)), (0, 255, 0), 3)
    frame = cv2.line(frame, img_point, tuple(
        point_proj[2].ravel().astype(int)), (0, 0, 255), 3)

    if roll is not None and pitch is not None and yaw is not None:
        cv2.putText(frame, "Roll:" + str(round(roll, 0)), (500, 50),
                    cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, "Pitch:" + str(round(pitch, 0)), (500, 70),
                    cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, "Yaw:" + str(round(yaw, 0)), (500, 90),
                    cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, cv2.LINE_AA)

    return frame
