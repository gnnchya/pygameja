import pygame
import numpy as np
import onnxruntime as ort
import cv2

# load the ONNX model
session = ort.InferenceSession("yolov5s6.onnx")

# define the input names and shapes
input_names = session.get_inputs()[0].name
input_shapes = session.get_inputs()[0].shape
print(input_shapes)
input_shapes[0] = 3  # set the first dimension to 3

# initialize pygame
pygame.init()
pygame.font.init()
font = pygame.font.SysFont('Comic Sans MS', 30)

# initialize the camera
cap = cv2.VideoCapture(0)

while True:
    # capture frame-by-frame
    print("hi")
    ret, frame = cap.read()

    # preprocess the image
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (input_shapes[1], input_shapes[2]))
    # frame = np.transpose(frame, [2, 0, 1])
    frame = frame.astype(np.float32) / 255.0
    frame = np.expand_dims(frame, axis=0)
    

    # run inference using the ONNX model
    outputs = session.run(None, {input_names: frame})

    boxes = outputs[0]
    confs = outputs[1]
    class_ids = outputs[2]
    
    # Draw the bounding boxes on the image
    for i in range(len(boxes)):
        box = boxes[i]
        conf = confs[i]
        class_id = class_ids[i]
        
        if conf > 0.5:
            x1, y1, x2, y2 = box
            x1 = int(x1 * frame.shape[1])
            y1 = int(y1 * frame.shape[0])
            x2 = int(x2 * frame.shape[1])
            y2 = int(y2 * frame.shape[0])
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, str(class_id), (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    # Display the resulting image
    cv2.imshow('frame', frame)
    
    # Exit the loop if the user presses 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
