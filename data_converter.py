import os
import cv2
import matplotlib.pyplot as plt
import mediapipe as mp

# Initialize Mediapipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Directory containing images
img_dir = 'DATASET/GODDESS'

# Loop through images in the directory
for filename in os.listdir(img_dir):
    # Load image
    sample_image = cv2.imread(os.path.join(img_dir, filename))
    img_copy = sample_image.copy()

    # Perform pose detection after converting the image into RGB format.
    pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.3, model_complexity=2)
    results = pose.process(cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB))

    # Check if any landmarks are found.
    if results.pose_landmarks:
        # Draw Pose landmarks on the sample image.
        mp_drawing.draw_landmarks(image=img_copy, landmark_list=results.pose_landmarks, connections=mp_pose.POSE_CONNECTIONS)

        # Specify a size of the figure.
        fig = plt.figure(figsize=[10, 10])

        # Display the output image with the landmarks drawn (convert BGR to RGB for display).
        plt.title("Output")
        plt.axis('off')
        plt.imshow(img_copy[:, :, ::-1])

        # Save the figure with a dynamic filename (including the full path).
        suffix = ".png"  # Define the file format
        plt.savefig(os.path.join(img_dir, f"{filename.split('.')[0]}_output{suffix}"))
        plt.close(fig)  # Close the figure to release resources

# Release resources
pose.close()
    