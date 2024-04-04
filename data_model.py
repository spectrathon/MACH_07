import mediapipe as mp 
import numpy as np 
import cv2 
import os

img_dir = 'DATASET/DOWNDOG'
def inFrame(lst):
	if lst[28].visibility > 0.6 and lst[27].visibility > 0.6 and lst[15].visibility>0.6 and lst[16].visibility>0.6:
		return True 
	return False

name = input("Enter the name of the Asana : ")

holistic = mp.solutions.pose
holis = holistic.Pose()
drawing = mp.solutions.drawing_utils

X = []
data_size = 0

for filename in os.listdir(img_dir):
    lst = []
    sample_image = cv2.imread(os.path.join(img_dir, filename))
    img_copy = sample_image.copy()
    frm = cv2.flip(img_copy, 1)
    res = holis.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))

    if res.pose_landmarks and inFrame(res.pose_landmarks.landmark):
        for i in res.pose_landmarks.landmark:
            lst.append(i.x - res.pose_landmarks.landmark[0].x)
            lst.append(i.y - res.pose_landmarks.landmark[0].y)

        X.append(lst)
        data_size = data_size+1

print(X)
#np.savetxt(f"{name}.csv", X, delimiter=",")
np.save(f"{name}.npy", np.array(X))
print(np.array(X).shape)