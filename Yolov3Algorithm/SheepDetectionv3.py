import cv2
import numpy as np
import time
import tensorflow as tf
from keras import backend as K

def yolo_filter_boxes(box_confidence, boxes, box_class_probs, threshold = .6):
    """Filters YOLO boxes by thresholding on object and class confidence.
    
    Arguments:
    box_confidence -- tensor of shape (19, 19, 5, 1)
    boxes -- tensor of shape (19, 19, 5, 4)
    box_class_probs -- tensor of shape (19, 19, 5, 80)
    threshold -- real value, if [ highest class probability score < threshold], then get rid of the corresponding box
    
    Returns:
    scores -- tensor of shape (None,), containing the class probability score for selected boxes
    boxes -- tensor of shape (None, 4), containing (b_x, b_y, b_h, b_w) coordinates of selected boxes
    classes -- tensor of shape (None,), containing the index of the class detected by the selected boxes
    
    Note: "None" is here because you don't know the exact number of selected boxes, as it depends on the threshold. 
    For example, the actual output size of scores would be (10,) if there are 10 boxes.
    """
    
    # Step 1: Compute box scores
    ### START CODE HERE ### (≈ 1 line)
    box_scores = box_confidence * box_class_probs
    ### END CODE HERE ###
    
    # Step 2: Find the box_classes using the max box_scores, keep track of the corresponding score
    ### START CODE HERE ### (≈ 2 lines)
    box_classes = K.argmax(box_scores, axis = -1)
    box_class_scores = K.max(box_scores, axis = -1, keepdims=False)
    ### END CODE HERE ###
    
    # Step 3: Create a filtering mask based on "box_class_scores" by using "threshold". The mask should have the
    # same dimension as box_class_scores, and be True for the boxes you want to keep (with probability >= threshold)
    ### START CODE HERE ### (≈ 1 line)
    filtering_mask = (box_class_scores >= threshold)
    ### END CODE HERE ###
    
    # Step 4: Apply the mask to box_class_scores, boxes and box_classes
    ### START CODE HERE ### (≈ 3 lines)
    scores = tf.boolean_mask(box_class_scores, filtering_mask)
    boxes = tf.boolean_mask(boxes, filtering_mask)
    classes = tf.boolean_mask(box_classes, filtering_mask)
    ### END CODE HERE ###
    
    return scores, boxes, classes

# Read neural networks from yolo file
network = cv2.dnn.readNet("yolov2.weights","yolov2.cfg")

classes_names = []
# Reads defined class names in coco file
with open("coco","r") as f:
    classes = [line.strip() for line in f.readlines()]


# Reads the image
frame = cv2.imread('sheep1.jpeg')
height, width, channels = frame.shape

#reduce input image to 608x608 pix to feed NN
blob = cv2.dnn.blobFromImage(frame,0.00392,(608,608),(0,0,0),True,crop=False)    

# Sets the new input value for the network. 
network.setInput(blob)

outputs = network.forward() # array 1805X85

outputs = outputs.reshape([19,19,5,85]) # 85 values for each anchor boxes in each of grid cells
                                        # format : [bx,by,bh,bw,pc,c1,c2.....c80]
 
box_confidence = outputs[:,:,:,0:1] # probs(Pc) for each anchor boxes in each of grid cells

boxes = outputs[:,:,:,0:4] # coord(bx,by,bh,bw) of rectangle for each anchor boxes in each of grid cells

box_class_probs = outputs[:,:,:,5:85] #class probs(c1,c2...c80) for each anchor boxes in each of grid cells

scores, boxes, classes = yolo_filter_boxes(box_confidence,boxes,box_class_probs)


cv2.imshow("Image",frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
