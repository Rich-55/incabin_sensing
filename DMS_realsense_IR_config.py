import pyrealsense2 as rs
import numpy as np

class DepthCamera:
    def __init__(self):
        # Configure depth and color streams
        self.pipeline = rs.pipeline()
        config = rs.config()

        # Get device product line for setting a supporting resolution
        pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        pipeline_profile = config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()
        device_product_line = str(device.get_info(rs.camera_info.product_line))

        # Get the depth sensor
        depth_sensor = device.first_depth_sensor()
        # Check if the sensor supports the laser power option
        if depth_sensor.supports(rs.option.laser_power):
            # Turn off the laser
            depth_sensor.set_option(rs.option.laser_power, 0)
            # print("Laser turned off")
        else:
            print("Laser power option is not supported by this device")


        # nho set frame size cho dung
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 60)
        config.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8, 60)
        config.enable_stream(rs.stream.infrared, 2, 640, 480, rs.format.y8, 60)

        # Start streaming
        self.pipeline.start(config)

    def get_frame(self):
        frames = self.pipeline.wait_for_frames()

        color_frame = frames.get_color_frame()
        ir1_frame = frames.get_infrared_frame(1)
        ir2_frame = frames.get_infrared_frame(2)

        if not ir1_frame or not ir2_frame or not color_frame:
            return None, None, None, None

        color_image = np.asanyarray(color_frame.get_data())
        ir1_image = np.asanyarray(ir1_frame.get_data())
        ir2_image = np.asanyarray(ir2_frame.get_data())

        return True, color_image, ir1_image

    def release(self):
        self.pipeline.stop()