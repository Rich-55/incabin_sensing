import struct
import numpy as np
import math
import oms_cloud
MMWDEMO_OUTPUT_EXT_MSG_DETECTED_POINTS = 301
MMWDEMO_OUTPUT_MSG_EXT_STATS = 306

#process frame header, TLV type, length
def h_parse(frame_data):
    # Constants for parsing frame header
    header_struct = 'Q8I'
    frame_header_len = struct.calcsize(header_struct)
    tlv_header_length = 8

    # Define the function's output structure and initialize error field to no error
    output_dict = {}
    output_dict['error'] = 0

    # A sum to track the frame packet length for verification for transmission integrity 
    total_len_check = 0   

    # Read in frame Header
    try:
        magic, version, total_packet_len, platform, frame_num, time_CPU_cycles, num_det_obj, num_TLV, sub_frame_num = struct.unpack(
            header_struct, frame_data[:frame_header_len])
    except:
        print('Error: Could not read frame header')
        output_dict['error'] = 1
    # print(  'Frame no.', frame_num, 
    #         'Detected points:', num_det_obj)
    # Move frameData pointer to start of 1st TLV   
    frame_data = frame_data[frame_header_len:]
    total_len_check += frame_header_len

    # Save frame number to output
    output_dict['frame_num'] = frame_num

    # Initialize the point cloud struct since it is modified by multiple TLV's
    # Each point has the following: X, Y, Z, Doppler, SNR, Noise, Track index
    output_dict['point_cloud'] = np.zeros((6,num_det_obj), np.float64)
    output_dict['num_det_points'] = 0
    # Find and parse all TLV's
    for i in range(num_TLV):
        try:
            tlv_type, tlv_length = tlv_header_decode(frame_data[:tlv_header_length])
            # print(f"TLV Type: {tlv_type}, Length:", tlv_length)
            frame_data = frame_data[tlv_header_length:]
            total_len_check += tlv_header_length
        except KeyboardInterrupt:
            data_port.close()        
        except:
            print('TLV Header Parsing Failure: Ignored frame due to parsing error')
            output_dict['error'] = 2
            return {}

        # Detected Points
        if(tlv_type == MMWDEMO_OUTPUT_EXT_MSG_DETECTED_POINTS): #301
            output_dict['num_det_points'], output_dict['point_cloud'] = oms_cloud.parse_TLV_301(frame_data[:tlv_length], tlv_length, output_dict['point_cloud'])
            # print(output_dict['point_cloud'][1])
        # Move to next TLV
        frame_data = frame_data[tlv_length:]
        total_len_check += tlv_length

    total_len_check = 32 * math.ceil(total_len_check / 32)

    # Verify the total packet length to detect transmission error that will cause subsequent frames to dropped
    if (total_len_check != total_packet_len):
        print('Warning: Frame packet length read is not equal to total_packet_len in frame header. Subsequent frames may be dropped.')
        output_dict['error'] = 3

    return output_dict

# Decode TLV Header
def tlv_header_decode(data):
    tlv_type, tlv_length = struct.unpack('2I', data)
    return tlv_type, tlv_length