
import numpy as np
import imutils
import cv2
# Load the object detection model
model = cv2.dnn.readNetFromCaffe(
'MobileNetSSD_deploy.prototxt',
'MobileNetSSD_deploy.caffemodel'
)
# Initialize the video capture object
#cap = cv2.VideoCapture(0) # for accessing the default camera
cap = cv2.VideoCapture('videoplayback .mp4') # for accessing the video file
cv2.namedWindow('Object Detection', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Object Detection', 1200, 800)
while True:
    # Read the frame
    ret, frame = cap.read()
    
    # Resize the frame
    frame = imutils.resize(frame, width=500)
    
    # Pass the frame to the object detection model
    blob = cv2.dnn.blobFromImage(frame, 0.007843, (500, 500), 127.5)
    model.setInput(blob)
    detections = model.forward()
    
    # Loop through the detections
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        
        # Filter out weak detections
        if confidence > 0.5:
            # Get the bounding box coordinates
            box = detections[0, 0, i, 3:7] * np.array([500, 500, 500, 500])
            (startX, startY, endX, endY) = box.astype('int')
            
            # Draw the bounding box and label on the frame
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
            label = '{:.2f}%'.format(confidence * 100)
            cv2.putText(frame, label, (startX, startY - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Show the frame
    cv2.imshow('Object Detection', frame)
    
    # Check for key presses
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
        
while True:
    # Read the frame
    ret, frame = cap.read()
    
    # Resize the frame
    frame = imutils.resize(frame, width=1000)
    
    # Pass the frame to the object detection model
    blob = cv2.dnn.blobFromImage(frame, 0.007843, (500, 500), 127.5)
    model.setInput(blob)
    detections = model.forward()
    
    # Loop through the detections
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        
        # Filter out weak detections
        if confidence > 0.5:
            # Get the bounding box coordinates
            box = detections[0, 0, i, 3:7] * np.array([500, 500, 500, 500])
            (startX, startY, endX, endY) = box.astype('int')
            
            # Draw the bounding box and label on the frame
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
            label = '{:.2f}%'.format(confidence * 100)
            cv2.putText(frame, label, (startX, startY - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Show the frame
    cv2.imshow('Object Detection', frame)
    
    # Check for key presses
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
        
# Release the video capture object and close all windows
cap.release()
