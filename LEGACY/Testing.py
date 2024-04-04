import mediapipe as mp 
import numpy as np 
import cv2  as cv
mp_drawing = mp.solutions.drawing_utils #drawing utility, visualition of poses
mp_pose = mp.solutions.pose #importing pose estimation model from mp

cap = cv.VideoCapture(0)
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        image = cv.cvtColor(frame, cv.COLOR_BGR2RGB) #Recolor Image
        image.flags.writeable = False
        results = pose.process(image)
        
        image.flags.writeable = True
        image = cv.cvtColor(frame, cv.COLOR_RGB2BGR) #Recoloring it back to BGR
        
        try:
            landmarks = results.pose_landmarks.landmarks
        except:
            pass
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS) #Drawing Landmarks
        
        cv.imshow("Yoga Detection System", image)
        if cv.waitKey(10) & 0xFF == ord('q'): #Checks if Q is pressed to end the stream
            break
    cap.release()
    cv.destroyAllWindows()