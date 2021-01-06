import cv2
import numpy as np
import time

# Read neural networks from yolo file
network = cv2.dnn.readNet("yolov2.weights","yolov2.cfg")

classes_names = []
# Reads defined class names in coco file
with open("coco","r") as f:
    classes = [line.strip() for line in f.readlines()]

# Reads the image
frame = cv2.imread('giraffe.jpg')
height, width, ch = frame.shape
#reduce frame to 608x608 pixel to feed NN
blob = cv2.dnn.blobFromImage(frame,0.00392,(608,608),(0,0,0),True,crop=False)    

# Sets the new input value for the network. 
network.setInput(blob)

outputs = network.forward() # array 1805X85

min_confidence=0.6

for i in range(outputs.shape[0]):
    prob_arr = outputs[i][5:] # Looks the prob value of each class for each grid cell
    class_index = np.argmax(prob_arr) # Gets the max prob
    confidence = prob_arr[class_index]
    if confidence > min_confidence: # Draws rectangle for only sheep
        print(confidence)
        x_center_box = outputs[i][0] * width # Calculates centerx of box in input image frame
        y_center_box = outputs[i][1] * height # Calculates centery of box in input image frame
        width_box = outputs[i][2] * width
        height_box = outputs[i][3] * height
        x1 = int(x_center_box - width_box * 0.5)  # Start X coordinate of bounding box
        y1 = int(y_center_box - height_box * 0.5)  # Start Y coordinate of bounding box
        x2 = int(x_center_box + width_box * 0.5)  # End X coordinate of bounding box
        y2 = int(y_center_box + height_box * 0.5)  # End y coordinate of bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 225), 2)
   


cv2.imshow("Image",frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
