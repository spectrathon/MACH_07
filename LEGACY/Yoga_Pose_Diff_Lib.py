import cv2 as cv #Computer Vision
import matplotlib.pyplot as plt #Pyplot for pose detection
import mediapipe as mp
import numpy as np
import tensorflow as tf
from tensorflow.python import keras
from sklearn.model_selection import train_test_split
from keras import layers, models
import os
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

#images = "dataset"

# images = []
# classNames = []
# myList = os.listdir(path)
# print(myList)
# for cl in myList:
#     curImg = cv.imread(f'{path}/{cl}')
#     images.append(curImg)
#     classNames.append(os.path.splitext(cl)[0])
# print(classNames)
net = cv.dnn.readNetFromTensorflow("graph_opt.pb")

inHeight = 368
inWidth = 368
thr = 0.2

BODY_PARTS = { "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
               "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
               "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
               "LEye": 15, "REar": 16, "LEar": 17, "Background": 18 }

POSE_PAIRS = [ ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
               ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
               ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
               ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
               ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"] ]

#img = cv.imread("image.jpg") #for image detection
###############################################LIVE VIDEO DETECTION###################################################################
cap = cv.VideoCapture(0)
cap.set(cv.CAP_PROP_FPS, 10)
cap.set(3, 800)
cap.set(4, 800)

while cv.waitKey(1) < 0:
    hasFrame, frame = cap.read()
    if not hasFrame:
        cv.waitKey()
        break

    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]
    net.setInput(cv.dnn.blobFromImage(frame, 1.0, (inWidth, inHeight), (127.5,127.5,127.5), swapRB = True, crop = False))
    out = net.forward()
    out = out[:, :19, :, :]  # MobileNet output [1, 57, -1, -1], we only need the first 19 elements
    assert(len(BODY_PARTS) == out.shape[1])

    points = []
    for i in range(len(BODY_PARTS)):
        # Slice heatmap of corresponging body's part.
        heatMap = out[0, i, :, :]

        # Originally, we try to find all the local maximums. To simplify a sample
        # we just find a global one. However only a single pose at the same time
        # could be detected this way.
        _, conf, _, point = cv.minMaxLoc(heatMap)
        x = (frameWidth * point[0]) / out.shape[3]
        y = (frameHeight * point[1]) / out.shape[2]
        # Add a point if it's confidence is higher than threshold.
        points.append((int(x), int(y)) if conf > thr else None)

    for pair in POSE_PAIRS:
        partFrom = pair[0]
        partTo = pair[1]
        assert(partFrom in BODY_PARTS)
        assert(partTo in BODY_PARTS)

        idFrom = BODY_PARTS[partFrom]
        idTo = BODY_PARTS[partTo]

        if points[idFrom] and points[idTo]:
            cv.line(frame, points[idFrom], points[idTo], (0, 255, 0), 3)
            cv.ellipse(frame, points[idFrom], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)
            cv.ellipse(frame, points[idTo], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)
            
    t, _ = net.getPerfProfile()
    freq = cv.getTickFrequency() / 1000
    
    cv.putText(frame, '%.2fms' % (t / freq), (10, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
    cv.imshow('OpenPose using OpenCV', frame)
    #print(points)

    # Visualize keypoints
    # for point in points:
    #     cv.circle(frame, point, 5, (0, 255, 0), -1)

    # # Display result
    # cv.imshow("Landmarks", frame)
    
    #return frame
    
########################################## Reading img #############################################################################    

BATCH_SIZE = 32
IMAGE_SIZE = (224, 224)

# Define the path to your dataset directory
dataset_dir = r"./dataset"

# Function to walk through a directory
def walk_through_dir(directory):
    for dirpath, dirnames, filenames in os.walk(directory):
        print(f"Found directory: {dirpath}")
        for file_name in filenames:
            print(f"\tFound file: {file_name}")

# Now, let's call the function with the correct directory path
walk_through_dir(dataset_dir)


## Setup mediapipe instance
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        
        # Recolor image to RGB
        image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        image.flags.writeable = False
      
        # Make detection
        results = pose.process(image)
    
        # Recolor back to BGR
        image.flags.writeable = True
        image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
        
        # Extract landmarks
        try:
            landmarks = results.pose_landmarks.landmark
            print(landmarks)
        except:
            pass
        
        
        # Render detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                 )               
        
        cv.imshow('Mediapipe Feed', image)

        if cv.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()






#################################################TRAINING MODEL####################################################################    
# def create_model(input_shape, num_keypoints):
#     model = models.Sequential()
    
#     # Convolutional layers
#     model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
#     model.add(layers.MaxPooling2D((2, 2)))
#     model.add(layers.Conv2D(64, (3, 3), activation='relu'))
#     model.add(layers.MaxPooling2D((2, 2)))
#     model.add(layers.Conv2D(128, (3, 3), activation='relu'))
#     model.add(layers.MaxPooling2D((2, 2)))
    
#     # Flatten layer
#     model.add(layers.Flatten())
    
#     # Dense layers
#     model.add(layers.Dense(128, activation='relu'))
#     model.add(layers.Dropout(0.5))
#     model.add(layers.Dense(num_keypoints * 2))  # Output layer with (x, y) coordinates for each keypoint
    
#     return model

# # Define data loading and preprocessing functions
# def load_data():
#     # Load and preprocess your dataset here
#     X_train, y_train = ...
#     X_val, y_val = ...
#     X_test, y_test = ...
#     return X_train, y_train, X_val, y_val, X_test, y_test

# def preprocess_data(X_train, X_val, X_test):
#     # Preprocess your input data here (e.g., normalization)
#     X_train_processed = ...
#     X_val_processed = ...
#     X_test_processed = ...
#     return X_train_processed, X_val_processed, X_test_processed

# # Define model training function
# def train_model(model, X_train, y_train, X_val, y_val, batch_size=32, epochs=10):
#     model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
#     history = model.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size=batch_size, epochs=epochs)
#     return history

# # Main function
# def main():
#     # Load and preprocess data
#     X_train, y_train, X_val, y_val, X_test, y_test = load_data()
#     X_train, X_val, X_test = preprocess_data(X_train, X_val, X_test)
    
#     # Define model parameters
#     input_shape = X_train.shape[1:]
#     num_keypoints = y_train.shape[1] // 2  # Divide by 2 because each keypoint has (x, y) coordinates
    
#     # Create and compile the model
#     model = create_model(input_shape, num_keypoints)
    
#     # Train the model
#     history = train_model(model, X_train, y_train, X_val, y_val)
    
#     # Evaluate the trained model
#     test_loss, test_accuracy = model.evaluate(X_test, y_test)
#     print("Test Loss:", test_loss)
#     print("Test Accuracy:", test_accuracy)

# if __name__ == "__main__":
#     main()
            

##############################################################IMAGE DETECTION############################################################
# def pose_checker(frame):
    
#     frameWidth = frame.shape[1]
#     frameHeight = frame.shape[0]
#     net.setInput(cv.dnn.blobFromImage(frame, 1.0, (inWidth, inHeight), (127.5,127.5,127.5), swapRB = True, crop = False))
#     out = net.forward()
#     out = out[:, :19, :, :]  # MobileNet output [1, 57, -1, -1], we only need the first 19 elements
#     assert(len(BODY_PARTS) == out.shape[1])

#     points = []
#     for i in range(len(BODY_PARTS)):
#         # Slice heatmap of corresponging body's part.
#         heatMap = out[0, i, :, :]

#         # Originally, we try to find all the local maximums. To simplify a sample
#         # we just find a global one. However only a single pose at the same time
#         # could be detected this way.
#         _, conf, _, point = cv.minMaxLoc(heatMap)
#         x = (frameWidth * point[0]) / out.shape[3]
#         y = (frameHeight * point[1]) / out.shape[2]
#         # Add a point if it's confidence is higher than threshold.
#         points.append((int(x), int(y)) if conf > thr else None)

#     for pair in POSE_PAIRS:
#         partFrom = pair[0]
#         partTo = pair[1]
#         assert(partFrom in BODY_PARTS)
#         assert(partTo in BODY_PARTS)

#         idFrom = BODY_PARTS[partFrom]
#         idTo = BODY_PARTS[partTo]

#         if points[idFrom] and points[idTo]:
#             cv.line(frame, points[idFrom], points[idTo], (0, 255, 0), 3)
#             cv.ellipse(frame, points[idFrom], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)
#             cv.ellipse(frame, points[idTo], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)
            
#     t, _ = net.getPerfProfile()
#     freq = cv.getTickFrequency() / 1000
#     cv.putText(frame, '%.2fms' % (t / freq), (10, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
#     return frame
            
# newpose = pose_checker(img)
# plt.imshow(newpose)
# plt.imshow(cv.cvtColor(newpose, cv.COLOR_BGR2RGB))
# plt.show() #Display's image