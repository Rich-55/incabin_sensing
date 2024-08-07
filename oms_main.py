#code imports
from oms_header import *
from oms_config import *
from oms_cloud import *
from oms_zone_map import *

#define parameters
control_baudrate   = 115200
cfg_file      = 'cfg/cpd.cfg'  # Configuration File Address

def init_oms():
    params = parse(cfg_file)
    #find COM port
    data_port = uart_communication(cfg_file,control_baudrate)
    return params, data_port

def oms(params,data_port):
    #parse config file to extract parameters
    #receive binary data from COM port
    frame_data = read_magic(data_port)

    #*******************************Read UART data*********************************
    version_bytes = data_port.read(4)                                   # Read in version from the header
    frame_data += bytearray(version_bytes)

    length_bytes = data_port.read(4)                                    # Read in length from header
    frame_data += bytearray(length_bytes)
    frame_length = int.from_bytes(length_bytes, byteorder='little')
    frame_length -= 16 

    frame_data += bytearray(data_port.read(frame_length))               # Read rest of the frame

    #receive parsed output data
    output_dict = h_parse(frame_data)
    #***************************OMS main func *************************************
    car_cloud = convert(output_dict['point_cloud'], params['sensorPosition'])                                                #convert point cloud to car coords
    
    zone_map = assign(output_dict['num_det_points'], car_cloud, 
                        params['numZones'], params['zone'])           #filter points not in boundary

    params['tracker'] = occupancy_detection(params['numZones'], params['tracker'], 
                                            car_cloud, zone_map, params['stateMach'])     # yield occupancy status
    
    occ_state = []
    for zone_id in range(params['numZones']):
        occ_state.append(params['tracker'][zone_id]['state'])
    # print(occ_state)
    return occ_state

if __name__ == '__main__' :
    params, data_port = init_oms()
    while True:
        oms(params,data_port)
        # print(occ_state)

