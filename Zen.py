import mediapipe as mp 
import numpy as np 
import cv2  as cv


mp_drawing = mp.solutions.drawing_utils #drawing utility, visualition of poses
mp_pose = mp.solutions.pose #importing pose estimation model from mp


# def calculate_angle(a,b,c):
#     a = np.array(a) # First
#     b = np.array(b) # Mid
#     c = np.array(c) # End
    
#     radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
#     angle = np.abs(radians*180.0/np.pi)
    
#     if angle > 180.0:
#         angle = 360-angle
        
#     return angle 


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
            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            
            # Calculate angle
            angle = calculate_angle(shoulder, elbow, wrist)
            
            # Visualize angle
            print(angle)
            #cv.putText(image, str(angle), tuple(np.multiply(elbow, [640, 480]).astype(int)), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv.LINE_AA)
        except:
            pass
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS) #Drawing Landmarks
        
        # landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].visibility
        # landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]
        # landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
        cv.imshow("Yoga Detection System", image)
        if cv.waitKey(10) & 0xFF == ord('q'): #Checks if Q is pressed to end the stream
            break
    cap.release()
    cv.destroyAllWindows()