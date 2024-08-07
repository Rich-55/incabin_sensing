import cv2
import DMS_DriverMonitoring
from DMS_realsense_IR_config import *

dc = DepthCamera()
DMS_handler = DMS_DriverMonitoring.DriverMonitoringFeatures()
scale_factor = 1.6

def main():
    show_rgb = True # Flag to check color or ir
    try:
        if not cv2.useOptimized():
            try:
                cv2.setUseOptimized(True)  # set OpenCV optimization to True
            except:
                print("OpenCV optimization could not be set to True, the script may be slower than expected")

        while True:
            ret, color_frame, ir1_frame = dc.get_frame()
            if not ret:  # if a frame can't be read, exit the program
                print("Can't receive frame from camera/stream end")
                continue
            
            frame = DMS_handler.run(color_frame, ir1_frame, show_rgb)
            show_frame = cv2.resize(frame, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)

            cv2.imshow("DMS", show_frame)

            key = cv2.waitKey(1)
            if key == ord('s'):
                show_rgb = not show_rgb  # Switch between RGB and infrared
            elif key == 27:
                break
    finally:
        # Stop streaming
        dc.release()
        # Destroy all OpenCV windows
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
