from time import sleep
import cv2
import matplotlib.pyplot as plt
from math import pi
import numpy as np
import urx

# Bring photo from camera
cap = cv2.VideoCapture(0)
for i in range(30):
    cap.read()

for i in range(220,330):
    ret, frame = cap.read()
    cv2.imwrite('UR_with_iiwa/UR_IIWA_'+str(i)+'.png',frame)
    sleep(5)

# rob = urx.Robot("192.168.0.12")
# rob.set_tcp((0, 0, 0, 0, 0, 0))
# rob.set_payload(2, (0, 0, 0.1))
# sleep(0.2)  #leave some time to robot to process the setup commands
#
# js=np.load('joint_state.npy')
#
# for i in range(js.shape[0]):
#     rob.movej((js[i,0], js[i,1], js[i,2], js[i,3], js[i,4], js[i,5]), 0.6, 0.25, wait=False)
#     sleep(10)
#     ret, frame = cap.read()
#     cv2.imwrite('UR_with_environment/'+str(i)+'.png',frame)

#
# rob.movej((0, -pi/2, pi/180, -pi/2, pi/4, 0), 0.5, 0.25,wait=False)
# sleep(10)
# ret, frame = cap.read()
# plt.imshow(frame)
# plt.show()
# cap.release()