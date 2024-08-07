import cv2
import time
import argparse # for CLI
import pickle
import numpy as np
import pandas as pd
import mediapipe as mp

from DMS_Eye_Dector_Module import EyeDetector as EyeDet
from DMS_Yawn_Detector_Module import YawnDetector as YawnDet
from DMS_Pose_Estimation_Module import HeadPoseEstimator as HeadPoseEst
from DMS_Attention_Scorer_Module import AttentionScorer as AttScorer

mp_facemesh = mp.solutions.face_mesh

def parsing_CLI_arg():
    parser = argparse.ArgumentParser(description='DMS')

    # selection the camera number, default is 0 (webcam)
    parser.add_argument('-c', '--camera', type=int,
                        default=0, metavar='', help='Camera number, default is 0 (webcam)')

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
    parser.add_argument('--ear_thresh', type=float, default=0.295,
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

def get_mediapipe_FaceMeshSolution(
    max_num_faces=2, # do we still need this?
    static_image_mode=False, # default: False for detect and tracking vid, True for constantly detect
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
):
    face_mesh = mp_facemesh.FaceMesh(
        max_num_faces=max_num_faces,
        static_image_mode=static_image_mode,
        refine_landmarks=refine_landmarks,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )
    return face_mesh

# Input: list of face mesh, Output: bounding box of largest face
def _get_landmarks(lms):
    surface = 0 # why tf surface is always 0?
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
            surface = new_surface
            
    return biggest_face

class DriverMonitoringFeatures: 
    def __init__(self):
        self.i = 0
        self.t0 = time.perf_counter()
        self.args = parsing_CLI_arg()
        self.face_mesh = get_mediapipe_FaceMeshSolution()
        
        # camera matrix obtained from the camera calibration script, using a 7x10 chessboard
        self.camera_matrix = np.array(
            [[1.35318579e+03, 0.00000000e+00, 9.55068470e+02],
            [0.00000000e+00, 1.35253631e+03, 5.49565483e+02],
            [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]], dtype="double")

        # distortion coefficients obtained from the camera calibration script, using a 7x10 chessboard
        self.dist_coeffs = np.array(
            [[ 0.07746648,  0.04698141,  0.00230451, -0.00210415, -0.65805834]], dtype="double")

        # instantiation of the eye detector, pose estimator, scorer objects
        self.Eye_det = EyeDet(show_processing=self.args.show_eye_proc)
        self.Yawn_det = YawnDet(show_processing=False)
        self.Head_pose = HeadPoseEst(show_axis=self.args.show_axis)
        self.Scorer = AttScorer(t_now=self.t0, ear_thresh=self.args.ear_thresh, 
                                ear_time_thresh=self.args.ear_time_thresh, verbose=self.args.verbose)

    def run(self, color_frame: np.array,  ir_frame: np.array, show_rgb):

        self.t_now = time.perf_counter()
        self.fps = self.i / (self.t_now - self.t0)
        if self.fps == 0:
            self.fps = 20

        # flip it so it looks like a mirror.
        color_frame = cv2.flip(color_frame, 2)
        ir_frame = cv2.flip(ir_frame, 2)

        # start the tick counter for computing the processing time for each frame
        self.e1 = cv2.getTickCount()

        # check what to show
        if show_rgb:
            show_frame = color_frame 
            # transform the BGR frame in grayscale
            gray = cv2.cvtColor(color_frame, cv2.COLOR_BGR2GRAY)
            # apply a bilateral filter to lower noise but keep frame details
            process_frame = np.expand_dims(cv2.bilateralFilter(gray, 5, 10, 10), axis=2)
            show = 'RGB'
        elif not show_rgb:
            show_frame = cv2.cvtColor(ir_frame, cv2.COLOR_GRAY2BGR)
            # apply a bilateral filter to lower noise but keep frame details
            process_frame = np.expand_dims(cv2.bilateralFilter(ir_frame, 5, 10, 10), axis=2)
            show = 'IR'

        frame_size = show_frame.shape[1], show_frame.shape[0]

        # Indicate what frame is shown
        height, width = frame_size[1], frame_size[0]
        x1, y1 = width - 40, height - 10
        cv2.putText(show_frame, show, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (203, 235, 247), 1, cv2.LINE_AA)

        process_frame = np.concatenate([process_frame, process_frame, process_frame], axis=2)
        process_frame.flags.writeable = False
        # find the faces using the face mesh model
        self.lms = self.face_mesh.process(process_frame).multi_face_landmarks
        process_frame.flags.writeable = True

        if self.lms:  # process the frame only if at least a face is found
            # getting face landmarks and then take only the bounding box of the biggest face
            landmarks = _get_landmarks(self.lms)

            # shows the eye keypoints (can be commented)
            self.Eye_det.show_eye_keypoints(color_frame=show_frame, landmarks=landmarks, frame_size=frame_size)

            # compute the EAR score of the eyes
            self.ear = self.Eye_det.get_EAR(frame=process_frame, landmarks=landmarks)

            # this is for yawnning detection
            self.Yawn_det.show_mouth_keypoints(color_frame=show_frame, landmarks=landmarks, frame_size=frame_size)
            self.yawn = self.Yawn_det.get_mouth_ratio(frame=process_frame, landmarks=landmarks)

            #### compute the YF score and state of fatigue 
            self.fatigue, self.yawn_count = self.Scorer.get_Yawn_Frequence(self.t_now, self.yawn)

            # compute the PERCLOS score and state of tiredness
            self.tired, self.perclos_score = self.Scorer.get_PERCLOS(self.t_now, self.fps, self.ear)

            # compute the Gaze Score
            self.gaze, self.left_gaze_score, self.right_gaze_score = self.Eye_det.get_Gaze_Score(frame=process_frame, landmarks=landmarks, frame_size=frame_size)

            # compute the head pose
            self.frame_det, self.roll, self.pitch, self.yaw = self.Head_pose.get_pose(
                frame=show_frame, landmarks=landmarks, frame_size=frame_size)

            self.asleep = self.Scorer.eval_scores(t_now=self.t_now,
                                             ear_score=self.ear)
            self.gaze_data = [self.roll[0], self.pitch[0], self.yaw[0], self.left_gaze_score, self.right_gaze_score]

            # if the head pose estimation is successful, show the results
            if self.frame_det is not None:
                show_frame = self.frame_det






            with open("saved_models/rf.pkl", 'rb') as f:
                model = pickle.load(f)

            self.x = pd.DataFrame([self.gaze_data])
            self.predictions = model.predict(self.x)[0]
            self.predic_proba = model.predict_proba(self.x)[0]

            self.probs = round(self.predic_proba[np.argmax(self.predic_proba)],2)

            # thres_frames = 2
            # if probs > 0.85:
            #     counter += 1
            # else:
            #     counter = 0
            # if counter >= thres_frames:
                # print(predictions, probs)
            # else:
            #     print('gg')


            text = f"Predictions: {self.predictions}     Probabilities: {self.probs}"
            x2, y2 = 10, height - 10
            cv2.putText(show_frame, text, (x2, y2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)






            # show the real-time EAR score
            if self.ear is not None:
                cv2.putText(show_frame, "EAR:" + str(round(self.ear, 3)), (10, 40),
                            cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 1, cv2.LINE_AA)

                    # show the real-time PERCLOS score
                cv2.putText(show_frame, "PERCLOS:" + str(round(self.perclos_score, 3)), (10, 70),
                            cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 1, cv2.LINE_AA)

            if (self.yawn_count is not None) and (self.yawn_count > 0):
                cv2.putText(show_frame, "YAWN:" + str(round(self.yawn_count, 3)), (10, 100),
                            cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 1, cv2.LINE_AA)

            if self.roll is not None:
                cv2.putText(show_frame, "Roll:"+str(self.roll.round(1)[0]), (480, 40),
                            cv2.FONT_HERSHEY_PLAIN, 1.5, (80, 127, 255), 1, cv2.LINE_AA)
            if self.pitch is not None:
                cv2.putText(show_frame, "Pitch:"+str(self.pitch.round(1)[0]), (480, 70),
                            cv2.FONT_HERSHEY_PLAIN, 1.5,  (80, 127, 255), 1, cv2.LINE_AA)
            if self.yaw is not None:
                cv2.putText(show_frame, "Yaw:"+str(self.yaw.round(1)[0]), (480, 100),
                            cv2.FONT_HERSHEY_PLAIN, 1.5,  (80, 127, 255), 1, cv2.LINE_AA)

            # if the state of attention of the driver is not normal, show an alert on screen
            if self.asleep:
                cv2.putText(show_frame, "ASLEEP!", (10, 260),
                            cv2.FONT_HERSHEY_PLAIN, 1.2, (0, 0, 255), 1, cv2.LINE_AA)
                
            # if the driver is tired, show and alert on screen
            if self.tired:
                cv2.putText(show_frame, "TIRED!", (10, 280),
                            cv2.FONT_HERSHEY_PLAIN, 1.2, (0, 0, 255), 1, cv2.LINE_AA)
                
            # if the state of attention of the driver is not normal, show an alert on screen
            if self.fatigue:
                cv2.putText(show_frame, "Fatigue!", (10, 300),
                            cv2.FONT_HERSHEY_PLAIN, 1.2, (0, 0, 255), 1, cv2.LINE_AA)

        # stop the tick counter for computing the processing time for each frame
        self.e2 = cv2.getTickCount()
        # processign time in milliseconds
        self.proc_time_frame_ms = ((self.e2 - self.e1) / cv2.getTickFrequency()) * 1000
        # print fps and processing time per frame on screen
        if self.args.show_fps:
            cv2.putText(show_frame, "FPS:" + str(round(self.fps)), (10, 400), cv2.FONT_HERSHEY_PLAIN, 2,
                        (255, 0, 255), 1)
        if self.args.show_proc_time:
            cv2.putText(show_frame, "PROC. TIME FRAME:" + str(round(self.proc_time_frame_ms, 0)) + 'ms', (10, 430), cv2.FONT_HERSHEY_PLAIN, 2,
                        (255, 0, 255), 1)
        
        self.i += 1
       
        return show_frame
