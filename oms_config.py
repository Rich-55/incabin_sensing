import serial, time
import serial.tools.list_ports as ports_list

MAGIC_WORD = bytearray(b'\x02\x01\x04\x03\x06\x05\x08\x07')
unsupported_commands = {'dynamicRACfarCfg', 'staticRACfarCfg', 'dynamicRangeAngleCfg', 'dynamic2DAngleCfg',
                            'staticRangeAngleCfg', 'antGeometry0', 'antGeometry1', 'antPhaseRot', 'fovCfg',
                            'numZones', 'totNumRows', 'sensorPosition', 'occStateMach', 'interiorBounds',
                            'cuboidDef', 'zoneNeighDef', '%','#'}
def find_port():
    print('Searching for ports...')
    ports = list(ports_list.comports())
    CLI_port = None
    for p in ports:
        print('Found port:', p)
        if 'XDS110 Class Application/User UART' in str(p):
            CLI_port = str(p)[:4]
    return CLI_port
def read_magic(data_port):
    index = 0
    magic = data_port.read(1)
    frame_data = bytearray(b'')
    while (1):
        if (len(magic) < 1):
            print ("ERROR: No data detected on COM Port, read timed out")
            magic = data_port.read(1)

        # Found matching byte
        elif (magic[0] == MAGIC_WORD[index]):
            index += 1
            frame_data.append(magic[0])
            if (index == 8): # Found the full magic word
                # print('New message: ',end="")
                break
            magic = data_port.read(1)
        else:
            if (index == 0):
                magic = data_port.read(1)
            index = 0 # Reset index
            frame_data = bytearray(b'') # Reset current frame data
    return frame_data
def send(control_port,file_name,control_baudrate):
    try:
        serial_port = serial.Serial(port=control_port,baudrate=control_baudrate)
        print(f"Sending configuration from {file_name} file to AWRL6432 ...")

        with open(file_name, 'r') as cfgFile:
            for command in cfgFile:
                if command.strip() and not command.startswith('%'):
                    split_command = command.split()
                    if split_command and split_command[0] in unsupported_commands:
                        continue
                    print(command.strip())  # Print the command being sent                               
                    
                    serial_port.write((command.strip() + '\n').encode('utf-8'))
                    
                    if split_command[0] == 'baudRate':
                        serial_port.baudrate=int(split_command[1])
                        time.sleep(0.5)
                        continue
                    
                    for _ in range(8):
                        cc = serial_port.readline().decode('utf-8', errors='ignore').strip()
                        #print(cc)
                        if 'Done' in cc:
                            print(cc)  # Print response indicating success
                            break
                        elif 'not recognized as a CLI command' in cc or 'Debug:' in cc or 'Error' in cc:
                            print(cc)  # Print response indicating failure
                            return  # Return inside the loop should be indented properly
                        # # If neither success nor failure, continue waiting for response
                        time.sleep(0.1)
        print("Configuration sent successfully")
        return serial_port
    except serial.SerialException as e:
        print(f"Serial port error: {e}")
    ## Handle the error gracefully, possibly by logging it or retrying
    except Exception as e:
        print(f"Error: {e}") 

def parse(file_name):
    P = {
        'channelCfg': {
        'txChannelEn': None,
        'rxChannelEn': None
        },
        'dataPath': {
        'numTxAzimAnt': None,
        'numTxElevAnt': None,
        'numRxAnt': None,
        'numTxAnt': None,
        'numChirpsPerFrame': 0,
        'numDopplerBins': 0,
        'numRangeBins': 0,
        'rangeResolutionMeters': 0,
        'rangeIdxToMeters': 0,
        'dopplerResolutionMps': 0
        },
        'chirpComnCfg': {
        'DigOutputSampRate_Dcim': None,
        'DigOutputBitsSel': None,
        'DfeFirSel': None,
        'NumOfAdcSamples': None,
        'ChirpRampEndTime': None,
        'ChirpRxHpfSel': None
        },
        'chirpTimingCfg': {
        'ChirpIdleTime': None,
        'ChirpAdcSkipSamples': None,
        'ChirpTxStartTime': None,
        'ChirpRfFreqSlope': None,
        'ChirpRfFreqStart': None
        },
        'frameCfg': {
        'chirpStartIdx': None,
        'chirpEndIdx': None,
        'numLoops': None,
        'numFrames': None,
        'framePeriodicity': None
        },
        'guiMonitor': {
        'detectedObjects': None,
        'logMagRange': None,
        'rangeAzimuthHeatMap': None,
        'rangeDopplerHeatMap': None
        },
        'sensorPosition': {
        'xOffset': None,
        'yOffset': None,
        'zOffset': None,
        'yzRot': None,
        'xyRot': None,
        'xzRot': None,
        'azimuthFov': None,
        'elevationFov': None
        },
        'numZones': None,
        'totNumRows': None,
        'zone': {},  # Placeholder for a list of zones, populated later
        'tracker': {},  # Placeholder for a list of trackers, populated later
        'stateMach': {},  # Placeholder for a list of state machines, populated later
        'intBound': {
            'minX': None,
            'maxX': None,
            'minY': None,
            'maxY': None
        },
        'profileCfg':{},
        }
    with open(file_name, 'r') as cfgFile:
        for command in cfgFile:
            if command.strip() and not command.startswith('%'):
                C = command.split()
                if C[0] == 'channelCfg':
                    P['channelCfg']['txChannelEn'] = float(C[2])
                    P['channelCfg']['rxChannelEn'] = float(C[1])
                    tx_channel_en = int(P['channelCfg']['txChannelEn'])
                    rx_channel_en = int(P['channelCfg']['rxChannelEn'])
                    P['dataPath']['numTxAzimAnt'] = sum((tx_channel_en >> i) & 1 for i in range(3))
                    P['dataPath']['numRxAnt'] = sum((rx_channel_en >> i) & 1 for i in range(4))
                    P['dataPath']['numTxElevAnt'] = 0
                    P['dataPath']['numTxAnt'] = P['dataPath']['numTxElevAnt'] + P['dataPath']['numTxAzimAnt']
                    #print(C[0])
                elif C[0] == 'chirpComnCfg':
                    P['chirpComnCfg']['DigOutputSampRate_Dcim'] = float(C[1])
                    P['chirpComnCfg']['DigOutputBitsSel'] = float(C[2])
                    P['profileCfg']['digOutSampleRate'] = P['chirpComnCfg']['DigOutputBitsSel']
                    P['chirpComnCfg']['DfeFirSel'] = float(C[3])
                    P['chirpComnCfg']['NumOfAdcSamples'] = float(C[4])
                    P['profileCfg']['numAdcSamples'] = P['chirpComnCfg']['NumOfAdcSamples']
                    P['chirpComnCfg']['ChirpRampEndTime'] = float(C[6])
                    P['profileCfg']['rampEndTime'] = P['chirpComnCfg']['ChirpRampEndTime']
                    P['chirpComnCfg']['ChirpRxHpfSel'] = float(C[7])
                    #print(C[0])
                elif C[0] == 'chirpTimingCfg':
                    P['chirpTimingCfg']['ChirpIdleTime'] = float(C[1])
                    P['profileCfg']['idleTime'] = P['chirpTimingCfg']['ChirpIdleTime']
                    P['chirpTimingCfg']['ChirpAdcSkipSamples'] = float(C[2])
                    P['chirpTimingCfg']['ChirpTxStartTime'] = float(C[3])
                    P['chirpTimingCfg']['ChirpRfFreqSlope'] = float(C[4])
                    P['profileCfg']['freqSlopeConst'] = P['chirpTimingCfg']['ChirpRfFreqSlope']
                    P['chirpTimingCfg']['ChirpRfFreqStart'] = float(C[5])
                    P['profileCfg']['startFreq'] = P['chirpTimingCfg']['ChirpRfFreqStart']
                    #print(C[0])
                elif C[0] == 'frameCfg':
                    P['frameCfg']['chirpStartIdx'] = float(C[1])
                    P['frameCfg']['chirpEndIdx'] = float(C[2])
                    P['frameCfg']['numLoops'] = float(C[3])
                    P['frameCfg']['numFrames'] = float(C[4])
                    P['frameCfg']['framePeriodicity'] = float(C[5])
                    #print(C[0])
                elif C[0] == 'guiMonitor':
                    P['guiMonitor']['detectedObjects'] = float(C[1])
                    P['guiMonitor']['logMagRange'] = float(C[2])
                    P['guiMonitor']['rangeAzimuthHeatMap'] = float(C[3])
                    P['guiMonitor']['rangeDopplerHeatMap'] = float(C[4])
                    #print(C[0])
                elif C[0] == 'sensorPosition':
                    P['sensorPosition']['xOffset'] = float(C[1])  # x offset of the sensor from the center
                    P['sensorPosition']['yOffset'] = float(C[2])  # y offset of the sensor from the mirror of the car
                    P['sensorPosition']['zOffset'] = float(C[3])  # Height of the sensor above the floorboard
                    P['sensorPosition']['yzRot'] = float(C[4])  # 0.0 degrees = Rot in y-z plane
                    P['sensorPosition']['xyRot'] = float(C[5])  # 0.0 degrees = Rot in x-y plane
                    P['sensorPosition']['xzRot'] = float(C[6])  # 0.0 degrees = Rot in x-z plane
                    #print(C[0])
                elif C[0] == 'fovCfg':
                    P['sensorPosition']['azimuthFov'] = float(C[2])
                    P['sensorPosition']['elevationFov'] = float(C[3])
                    #print(C[0])
                elif C[0] == 'numZones':
                    P['numZones'] = int(C[1])
                    #print(C[0])
                elif C[0] == 'totNumRows':
                    P['totNumRows'] = float(C[1])
                    #print(C[0])
                elif C[0] == 'cuboidDef':
                    zoneIdx = int(C[1])-1       #offset for matlab indexing from 1
                    cubeIdx = int(C[2])        
                    if zoneIdx not in P['zone']:
                        P['zone'][zoneIdx] = {'cuboid': {}}
                    if zoneIdx > P['numZones']:
                        print(f'ERROR! numZones {P["numZones"]} is less than cuboid zone index {zoneIdx}!')
                        exit()
                    if zoneIdx <= P['numZones'] and cubeIdx <= 3:
                        P['zone'][zoneIdx]['numCuboids'] = cubeIdx      # relies on cuboids being defined in rank order
                        P['zone'][zoneIdx]['cuboid'][cubeIdx-1]={       #offset for matlab indexing from 1
                                    'x' : [float(C[3]), float(C[4])],  # left-right
                                    'y' : [float(C[5]), float(C[6])],  # back-front
                                    'z' : [float(C[7]), float(C[8])]   # floor-ceiling
                        }     
                    #print(C[0])
                elif C[0] == 'zoneNeighDef':
                    zoneIdx = int(C[1])-1 #offset for matlab indexing from 1
                    if zoneIdx > P['numZones']:
                        print('ERROR! numZones {} is less than cuboid zone index {}!'.format(P['numZones'], zoneIdx))
                        exit()
                    if zoneIdx <= P['numZones']:
                        P['tracker'][zoneIdx]= {
                            'zoneType': float(C[2]),
                            'neighbors':[],
                            'state': 0,
                            'freeze': 0,
                            'numEntryCount': 0,
                            'freezeSourceFlag': 0,
                            'detect2freeCount': 0,
                            'avgSnr': 0,
                            'numPoints':0
                            }
                        numNeigh = int(C[3])
                        for id in range(numNeigh):
                            P['tracker'][zoneIdx]['neighbors'].append(float(C[3+id]))  # Convert to float
                    #print(C[0])
                elif C[0] == 'occStateMach':
                    zoneType = float(C[1])
                    P['stateMach'][zoneType] = {
                        'numPointForEnterThreshold1': float(C[2]),
                        'avgSnrForEnterThreshold1': float(C[3]),
                        'numPointForEnterThreshold2': float(C[4]),
                        'avgSnrForEnterThreshold2': float(C[5]),
                        'numEntryThreshold': float(C[6]),
                        'numPointForStayThreshold': float(C[7]),
                        'avgSnrForStayThreshold': float(C[8]),
                        'forgetThreshold': float(C[9]),
                        'numPointToForget': float(C[10]),
                        'overloadThreshold': float(C[11])
                    }
                    #print(C[0])
                elif C[0] == 'interiorBounds':
                    P['intBound']['minX'] = float(C[1])
                    P['intBound']['maxX'] = float(C[2])
                    P['intBound']['minY'] = float(C[3])
                    P['intBound']['maxY'] = float(C[4])
                    #print(C[0])
    return P

def uart_communication(cfg_file,control_baudrate):
    #find COM port
    serial_port = find_port()
    #send config file       
    data_port = send(serial_port,cfg_file,control_baudrate)
    time.sleep(0.1)
    # data_port.flush()
    return data_port
if __name__=="__main__":
    find_port()
    read_magic()
    send()
    parse()
    uart_communication()