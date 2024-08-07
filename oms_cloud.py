import struct
import numpy as np

def parse_TLV_301(tlvData, tlvLength, pointCloud):
    pUnitStruct = 'ffffHH' # Units for the 5 results to decompress them
    pointStruct = 'hhhhBB' # x y z doppler snr noise
    pUnitSize = struct.calcsize(pUnitStruct)
    pointSize = struct.calcsize(pointStruct)

    # Parse the decompression factors
    try:
        point_unit = struct.unpack(pUnitStruct, tlvData[:pUnitSize])
    except:
            print('Error: Point Cloud TLV Parser Failed')
            return 0, pointCloud
    # Update data pointer
    tlvData = tlvData[pUnitSize:]

    # Parse each point
    numPoints = int((tlvLength-pUnitSize)/pointSize)
    for i in range(numPoints):
        try:
            x, y, z, doppler, snr, noise = struct.unpack(pointStruct, tlvData[:pointSize])
        except:
            numPoints = i
            print('Error: Point Cloud TLV Parser Failed')
            break
        
        tlvData = tlvData[pointSize:]
        # Decompress values
        pointCloud[0,i] = x * point_unit[0]          # x
        pointCloud[1,i] = y * point_unit[0]          # y
        pointCloud[2,i] = z * point_unit[0]          # z
        pointCloud[3,i] = doppler * point_unit[1]    # Doppler
        pointCloud[4,i] = snr * point_unit[2]        # SNR
        pointCloud[5,i] = noise * point_unit[3]      # Noise
    return numPoints, pointCloud        
def convert (radar_cloud,sensor_position):
    x = radar_cloud[0]
    y = radar_cloud[1]
    z = radar_cloud[2]
    snr = radar_cloud[4]

    # Convert rotation angles from degrees to radians
    yzRot = np.radians(sensor_position["yzRot"])
    xyRot = np.radians(sensor_position["xyRot"])
    xzRot = np.radians(sensor_position["xzRot"])

    # Rotation matrices
    R_xz = np.array([[np.cos(xzRot), np.sin(xzRot)], [-np.sin(xzRot), np.cos(xzRot)]])
    R_yz = np.array([[np.cos(yzRot), np.sin(yzRot)], [-np.sin(yzRot), np.cos(yzRot)]])
    R_xy = np.array([[np.cos(xyRot), np.sin(xyRot)], [-np.sin(xyRot), np.cos(xyRot)]])

    # Rotating the point cloud
    rotated_xz = np.dot(R_xz, np.array([x, z]))
    rotated_yz = np.dot(R_yz, np.array([y, rotated_xz[1]]))
    rotated_xy = np.dot(R_xy, np.array([rotated_xz[0], rotated_yz[0]]))

    # Adjusting the sensor position
    x_ = rotated_xy[0] + sensor_position["xOffset"]
    y_ = rotated_xy[1] + sensor_position["yOffset"]
    z_ = rotated_yz[1] + sensor_position["zOffset"]

    # Constructing the transformed point cloud
    transformed_cloud = np.vstack((x_, y_, z_, snr))
    return transformed_cloud

