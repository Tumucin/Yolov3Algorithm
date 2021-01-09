import cv2
import numpy as np
import time
import sys

class SheepDetection():
    def __init__(self):
        # Read deep learning network
        self.net = cv2.dnn.readNet('yolov3.weights','yolov3.cfg')

        # Read classes names
        with open('coco','r') as f:
            self.class_names = f.read().splitlines()
        
        # Get unconnected output layer names
        layer_names = self.net.getLayerNames()
        self.outputlayers = [layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]

        # Input ımage shape
        self.height = 0
        self.width = 0

        # Spatial size for output image 
        self.size = 608

        self.blob = None

        # Threshold for probabilities
        self.min_confidence = 0.3

        # Colors for bounding box
        self.colors= np.random.uniform(0,255,size=(len(self.class_names),3))

    def setImage(self,image:str):
        """ Gets and image and makes ready for feeding Neural Network
            args:
                image: Name of the file to be loaded (string)
            returns:
                frame: This method returns an image that is loaded from the specified file. 
        """
        frame  = cv2.imread(image)
        self.height, self.width, _ =frame.shape

        # Sets blob for Neural Network
        self.blob = cv2.dnn.blobFromImage(frame,0.00392,(self.size,self.size),(0,0,0),True,crop=False)
        cv2.imshow('Image',frame)
        cv2.waitKey(100)
        return frame

    def SetVideoFrame(self,frame):
        self.height, self.width, _ =frame.shape
        # Sets blob for Neural Network
        self.blob = cv2.dnn.blobFromImage(frame,0.00392,(self.size,self.size),(0,0,0),True,crop=False)

    def detectAndLocaliza(self,frame):
        """ This function detects the object in a given frame

            args:
                    frame: Given frame to the Neural Network for detecting objects
            returns:
                    boxes: a list which contains bounding box parameters for detected objects
                            format : [[x,y,w,h]
                                      [x2,y2,w2,h2]]
                                        x: Starting position x of possible bounding box in image frame
                                        y: Starting position y of possible bounding box in image frame
                                        w: Width of the possible bounding bounding boxes
                                        h: Height of the possible bounding bounding boxes
                    confidences: a list which contains the probability of the detected object
                    class_ids: a list which contains the index of max probabilities

        """
        self.net.setInput(self.blob)

        # Output of the Neural Network
        outs = self.net.forward(self.outputlayers)
        class_ids=[]
        confidences=[]
        boxes=[]
        font = cv2.FONT_HERSHEY_PLAIN
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                self.confidence = scores[class_id]
                if self.confidence > self.min_confidence and class_id==18:
                    #onject detected
                    center_x= int(detection[0]*self.width)
                    center_y= int(detection[1]*self.height)
                    w = int(detection[2]*self.width)
                    h = int(detection[3]*self.height)
                    x=int(center_x - w/2)
                    y=int(center_y - h/2)
                    boxes.append([x,y,w,h]) #put all rectangle areas
                    confidences.append(float(self.confidence)) #how self.confidence was that object detected and show that percentage
                    class_ids.append(class_id) #name of the object tha was detected

        return boxes, confidences, class_ids

    def drawRectangle(self, boxes:list, confidence:list, class_ids:list, indexes, frame):
        """ Draws rectangle using selected boxes to the frame 

            args:
                    boxes: a list which contains bounding box parameters for detected objects
                            format : [[x,y,w,h]
                                      [x2,y2,w2,h2]]
                                        x: Starting position x of possible bounding box in image frame
                                        y: Starting position y of possible bounding box in image frame
                                        w: Width of the possible bounding bounding boxes
                                        h: Height of the possible bounding bounding boxes
                    confidences: a list which contains the probability of the detected object
                    class_ids: a list which contains the index of max probabilities
                    indexes: the kept indices of bboxes after NMS. 
                    frame: Given frame to the Neural Network for detecting objects

        """
        for i in range(len(boxes)):
            if i in indexes:
                x,y,w,h = boxes[i]
                self.confidence= confidences[i]
                color = self.colors[class_ids[i]]
                cv2.rectangle(frame,(x,y),(x+w,y+h),color,2)
        cv2.imshow("Image",frame)
        cv2.waitKey(2000)
         

if __name__ == "__main__":
    try:
        argumentList = sys.argv[1:]
        sheepObj = SheepDetection()
        if argumentList[0] == '-1': # If it is image 
            # Set image for the Neural Networks
            frame = sheepObj.setImage(argumentList[1])

            # Detects the possible objects and boxes
            boxes, confidences, class_ids = sheepObj.detectAndLocaliza(frame)

            # Performs non maximum suppression given boxes and corresponding scores. 
            indexes = cv2.dnn.NMSBoxes(boxes,confidences,0.4,0.6)

            # Finally draw the bounding boxes 
            sheepObj.drawRectangle(boxes, confidences, class_ids,indexes,frame)
        else:
            # Video capture
            cap = cv2.VideoCapture(argumentList[1])
            
            while True:
                _,frame= cap.read()
                sheepObj.SetVideoFrame(frame)
                boxes, confidences, class_ids = sheepObj.detectAndLocaliza(frame)
                # Performs non maximum suppression given boxes and corresponding scores. 
                indexes = cv2.dnn.NMSBoxes(boxes,confidences,0.4,0.6)
                # Finally draw the bounding boxes 
                sheepObj.drawRectangle(boxes, confidences, class_ids,indexes,frame)

            cap.release()
            cv2.destroyAllWindows()
    except:
        pass