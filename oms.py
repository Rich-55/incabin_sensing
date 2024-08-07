import cv2
from multiprocessing import Process
from oms_main import *

def dmsgui(): 
    cap = cv2.VideoCapture(0)
    while(True):
        ret, frame = cap.read()
        # vid_ratio = frame.shape[1] / frame.shape[0]
        if not ret:
            print("Error: Failed to capture frame.")
            break
        # if set_height != frame.shape[0]:
        #     frame_resized = cv2.resize(frame, (int(set_height * vid_ratio), set_height))     
        cv2.imshow("DMS", frame)  
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break      
    cap.release()
    cv2.destroyAllWindows()

def omsgui(set_height):
    image = cv2.imread('img/5seat.jpg')
    img_ratio = image.shape[1] / image.shape[0]
    seat_positions =    ((60, 170, 75, 110),    # Driver
                        (150, 170, 75, 110),   # Front passenger
                        (60, 290, 52, 110),    # Rear left passenger
                        (117, 290, 51, 110),   # Rear middle passenger
                        (173, 290, 52, 110),   # Rear right passenger
                        (60, 265, 165, 40))
    morphed_seat_positions = tuple((int(x / image.shape[1]*set_height*img_ratio), 
                                    int(y / image.shape[0]*set_height), 
                                    int(w / image.shape[1]*set_height*img_ratio), 
                                    int(h / image.shape[0]*set_height)) for x, y, w, h in seat_positions)
    params, data_port = init_oms()
    # print(morphed_seat_positions)

    while(True):
        image_resized = cv2.resize(image, (int(set_height * img_ratio), set_height))
        # print(image_resized.shape[1],image_resized.shape[0])
        occ_state = oms(params, data_port)
        # occ_state = [random.choice([0, 1]) for _ in range(6)]
        for i, presence_state in enumerate(occ_state): 
            x,y,w,h = morphed_seat_positions[i]  
            if presence_state == 1:            
                cv2.rectangle(image_resized, (x, y), (x+w, y+h), (0, 255, 0), 2)

        cv2.imshow('OMS',image_resized)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break     
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # p1= Process(target = dmsgui)
    # p2= Process(target = omsgui, args=(640,))
    # p1.start() 
    # p2.start()

    # p1.join()
    # p2.join()
    omsgui(int(480*1.2))
