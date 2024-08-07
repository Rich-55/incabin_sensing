import cv2
import numpy as np
from numpy import linalg as LA


MOUTH_IDS = [291,  61,  13,  14]

class YawnDetector:

    def __init__(self, show_processing: bool = False):

        self.show_processing = show_processing

    def get_mouth_ratio(self, frame, landmarks):

        # numpy array for storing the keypoints positions of the left and right eyes
        mouth_pts = np.zeros(shape=(4, 2))

        # get the face mesh keypoints
        for i in range(len(MOUTH_IDS)):
            # array of x,y coordinates for the left eye reference point
            mouth_pts[i] = landmarks[MOUTH_IDS[i], :2]

        mouth_ratio =  LA.norm(mouth_pts[0] - mouth_pts[1]) / LA.norm(mouth_pts[2] - mouth_pts[3])
        # print(mouth_ratio)
        return mouth_ratio
    
    def show_mouth_keypoints(self, color_frame, landmarks, frame_size):
        for n in MOUTH_IDS:
            x = int(landmarks[n, 0] * frame_size[0])
            y = int(landmarks[n, 1] * frame_size[1])
            cv2.circle(color_frame, (x, y), 1, (0, 0, 255), -1)
        return
