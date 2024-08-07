import numpy as np

def assign(numPoints, point3d, numZones, zone):
    #print(zone)
    if numPoints == 0:
        zoneMap = np.zeros((5, numZones), dtype=bool)
        xLoc, yLoc, zLoc = [], [], []
        return zoneMap

    xLoc, yLoc, zLoc = point3d[0], point3d[1], point3d[2]
    zoneMap = np.zeros((numPoints, numZones), dtype=bool)

    for idx in range(numZones):
        cZone = zone[idx]
        # print(zoneMap)

        for cIdx in range(int(cZone['numCuboids'])):
            ind = np.zeros_like(xLoc, dtype=bool)
            for i in range(len(xLoc)):
                ind_x = (xLoc[i] > cZone['cuboid'][cIdx]['x'][0]) & (xLoc[i] < cZone['cuboid'][cIdx]['x'][1])
                ind_y = (yLoc[i] > cZone['cuboid'][cIdx]['y'][0]) & (yLoc[i] < cZone['cuboid'][cIdx]['y'][1])
                ind_z = (zLoc[i] > cZone['cuboid'][cIdx]['z'][0]) & (zLoc[i] < cZone['cuboid'][cIdx]['z'][1])
                ind[i] = ind_x & ind_y & ind_z

            zoneMap[:, idx] |= ind

    return zoneMap

def occupancy_detection(num_zones, tracker, point_3d, zone_map, statemach_params):
    for idx in range(num_zones):
        # Find the number of detected points in this zone
        tracker[idx]['numPoints'] = sum(zone_map[:, idx])

        # Calculate the average SNR for detected points in this zone
        if tracker[idx]['numPoints'] > 0:
            detected_snr = [point_3d[3, i] for i in range(len(point_3d[3])) if zone_map[i, idx] == 1]
            tracker[idx]['avgSnr'] = (sum(detected_snr) / len(detected_snr))
        else:
            tracker[idx]['avgSnr'] = 0.0

    # Check for overload conditions (large movements in the row)
    for idx in range(num_zones):
        if tracker[idx]['avgSnr'] >= statemach_params[0]['overloadThreshold']:
            # If an overload occurs, freeze all zones as well.
            for idx2 in range(num_zones):
                tracker[idx2]['freeze'] = 2  # freeze for 2 frames
        elif tracker[idx]['freeze'] > 0:
            tracker[idx]['freeze'] -= 1

    # Update the occupancy state of each zone
    for idx in range(num_zones):
        zone_type = tracker[idx]['zoneType'] + 1
        tracker = update_state_machine(idx, tracker, statemach_params[zone_type - 1])

    return tracker

def update_state_machine(idx, tracker, statemach):
    trackerCur = tracker[idx]
    neighbors = tracker[idx]['neighbors']
    
    if trackerCur['freeze'] == 0:
        maxAvgSnr = statemach['avgSnrForEnterThreshold2']
        
        for tt in neighbors:
            avgSnrNeighbor = tracker[tt]['avgSnr']
            maxAvgSnr = max(maxAvgSnr, avgSnrNeighbor)
        
        if trackerCur['state'] == 0:  # NOT_OCCUPIED
            if ((trackerCur['numPoints'] > statemach['numPointForEnterThreshold1'] and 
                 trackerCur['avgSnr'] > statemach['avgSnrForEnterThreshold1']) or
                (trackerCur['numPoints'] > statemach['numPointForEnterThreshold2'] and 
                 trackerCur['avgSnr'] > maxAvgSnr)):
                
                trackerCur['numEntryCount'] += 1
            else:
                trackerCur['numEntryCount'] = 0
            
            if trackerCur['numEntryCount'] >= statemach['numEntryThreshold']:
                trackerCur['state'] = 1
                trackerCur['detect2freeCount'] = 0
                
        elif trackerCur['state'] == 1:  # OCCUPIED
            if (trackerCur['numPoints'] > statemach['numPointForStayThreshold'] and
                trackerCur['avgSnr'] > statemach['avgSnrForStayThreshold']):
                
                trackerCur['detect2freeCount'] = 0
            elif trackerCur['numPoints'] < statemach['numPointToForget']:
                if trackerCur['detect2freeCount'] > statemach['forgetThreshold']:
                    trackerCur['state'] = 0
                else:
                    trackerCur['detect2freeCount'] += 1
            else:
                trackerCur['detect2freeCount'] -= 1

    tracker[idx] = trackerCur
    return tracker
