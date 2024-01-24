from tkinter import *
from tkinter import ttk
from tkinter import filedialog
from PIL import Image, ImageTk
from utils import *
from matplotlib import pyplot as plt
import os
from playsound import playsound
import subprocess
from gtts import gTTS
import cv2
import numpy as np
# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import cv2




class Root(Tk):
    def __init__(self):
        super(Root, self).__init__()
        self.title("Face Recognisation System")
        self.minsize(800, 400)
        
        self.labelFrame = ttk.LabelFrame(self, text = "Menu")
        self.labelFrame.grid(column = 0, row = 1, padx = 10, pady = 10)
        img = Image.open('obj.jpg')
     
        resize_image = img.resize((600, 400), Image.ANTIALIAS)
        img = ImageTk.PhotoImage(resize_image)

        self.label2 = Label(image=img)
        self.label2.image = img 
        self.label2.grid(column=2, row=1, padx = 10, pady = 10)
        self.button()
        
    def verifyPerson(self):
        
        filename = 'python attendance.py'
        os.system(filename) #Open file [Same as Right-click Open]
      
        
        
    def Web(self):
    
        CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
        	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
        	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
        	"sofa", "train", "tvmonitor"]
        COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
        
        # load our serialized model from disk
        print("[INFO] loading model...")
        net = cv2.dnn.readNetFromCaffe('MobileNetSSD_deploy.prototxt.txt', 'MobileNetSSD_deploy.caffemodel')
        
        # initialize the video stream, allow the cammera sensor to warmup,
        # and initialize the FPS counter
        print("[INFO] starting video stream...")
        vs = VideoStream(src=0).start()
        time.sleep(2.0)
        fps = FPS().start()
        
        # loop over the frames from the video stream
        while True:
        	# grab the frame from the threaded video stream and resize it
        	# to have a maximum width of 400 pixels
        	frame = vs.read()
        	frame = imutils.resize(frame, width=500)
        
        	# grab the frame dimensions and convert it to a blob
        	(h, w) = frame.shape[:2]
        	blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
        		0.007843, (300, 300), 127.5)
        
        	# pass the blob through the network and obtain the detections and
        	# predictions
        	net.setInput(blob)
        	detections = net.forward()
        
        	# loop over the detections
        	for i in np.arange(0, detections.shape[2]):
        		# extract the confidence (i.e., probability) associated with
        		# the prediction
        		confidence = detections[0, 0, i, 2]
        
        		# filter out weak detections by ensuring the `confidence` is
        		# greater than the minimum confidence
        		if confidence > 0.2:
        			# extract the index of the class label from the
        			# `detections`, then compute the (x, y)-coordinates of
        			# the bounding box for the object
        			idx = int(detections[0, 0, i, 1])
        			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        			(startX, startY, endX, endY) = box.astype("int")
        
        			# draw the prediction on the frame
        			label = "{}: {:.2f}%".format(CLASSES[idx],
        				confidence * 100)
        			cv2.rectangle(frame, (startX, startY), (endX, endY),
        				COLORS[idx], 2)
        			y = startY - 15 if startY - 15 > 15 else startY + 15
        			cv2.putText(frame, label, (startX, y),
        				cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
        
        	# show the output frame
        	cv2.imshow("Frame", frame)
        	key = cv2.waitKey(1) & 0xFF
        
        	# if the `q` key was pressed, break from the loop
        	if key == ord("q"):
        		break
        
        	# update the FPS counter
        	fps.update()
        
        # stop the timer and display FPS information
        fps.stop()
        print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
        print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
        
        # do a bit of cleanup
        cv2.destroyAllWindows()
        vs.stop()
        
 
        
    
    def video(self):
        # Make sure the video file is in the same directory as your code
        filename = 'videos/street.mp4'
        file_size = (1920,1080) # Assumes 1920x1080 mp4
 
        # We want to save the output to a video file
        output_filename = 'videos/streetSSD.mp4'
        output_frames_per_second = 20.0
 
        RESIZED_DIMENSIONS = (300, 300) # Dimensions that SSD was trained on. 
        IMG_NORM_RATIO = 0.007843 # In grayscale a pixel can range between 0 and 255
         
        # Load the pre-trained neural network
        neural_network = cv2.dnn.readNetFromCaffe('MobileNetSSD_deploy.prototxt.txt', 
                'MobileNetSSD_deploy.caffemodel')
         
        # List of categories and classes
        categories = { 0: 'background', 1: 'aeroplane', 2: 'bicycle', 3: 'bird', 
                       4: 'boat', 5: 'bottle', 6: 'bus', 7: 'car', 8: 'cat', 
                       9: 'chair', 10: 'cow', 11: 'diningtable', 12: 'dog', 
                      13: 'horse', 14: 'motorbike', 15: 'person', 
                      16: 'pottedplant', 17: 'sheep', 18: 'sofa', 
                      19: 'train', 20: 'tvmonitor'}
         
        classes =  ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", 
                    "bus", "car", "cat", "chair", "cow", 
                   "diningtable",  "dog", "horse", "motorbike", "person", 
                   "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
                              
        # Create the bounding boxes
        bbox_colors = np.random.uniform(255, 0, size=(len(categories), 3))
     
  
       
        # Load a video
        cap = cv2.VideoCapture(filename)
       
        # Create a VideoWriter object so we can save the video output
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        result = cv2.VideoWriter(output_filename,  
                                 fourcc, 
                                 output_frames_per_second, 
                                 file_size) 
           
        # Process the video
        while cap.isOpened():
               
          # Capture one frame at a time
          success, frame = cap.read() 
       
          # Do we have a video frame? If true, proceed.
          if success:
               
            # Capture the frame's height and width
            (h, w) = frame.shape[:2]
       
            # Create a blob. A blob is a group of connected pixels in a binary 
            # frame that share some common property (e.g. grayscale value)
            # Preprocess the frame to prepare it for deep learning classification
            frame_blob = cv2.dnn.blobFromImage(cv2.resize(frame, RESIZED_DIMENSIONS), 
                           IMG_NORM_RATIO, RESIZED_DIMENSIONS, 127.5)
           
            # Set the input for the neural network
            neural_network.setInput(frame_blob)
       
            # Predict the objects in the image
            neural_network_output = neural_network.forward()
       
            # Put the bounding boxes around the detected objects
            for i in np.arange(0, neural_network_output.shape[2]):
                   
              confidence = neural_network_output[0, 0, i, 2]
           
              # Confidence must be at least 30%       
              if confidence > 0.30:
                       
                idx = int(neural_network_output[0, 0, i, 1])
       
                bounding_box = neural_network_output[0, 0, i, 3:7] * np.array(
                  [w, h, w, h])
       
                (startX, startY, endX, endY) = bounding_box.astype("int")
       
                label = "{}: {:.2f}%".format(classes[idx], confidence * 100) 
               
                cv2.rectangle(frame, (startX, startY), (
                  endX, endY), bbox_colors[idx], 2)     
                               
                y = startY - 15 if startY - 15 > 15 else startY + 15    
       
                cv2.putText(frame, label, (startX, y),cv2.FONT_HERSHEY_SIMPLEX, 
                  0.5, bbox_colors[idx], 2)
            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1) & 0xFF
            # if the `q` key was pressed, break from the loop
            if key == ord("q"):
                cv2.destroyAllWindows()
                break
              
              
               
            # We now need to resize the frame so its dimensions
            # are equivalent to the dimensions of the original frame
            frame = cv2.resize(frame, file_size, interpolation=cv2.INTER_NEAREST)
       
                  # Write the frame to the output video file
            result.write(frame)
               
          # No more video frames left
          else:
              cv2.destroyAllWindows()
              break
            
                   
        # Stop when the video is finished
        cap.release()
           
        # Release the video recording
        result.release()
        
        
    def button(self):
                self.button = ttk.Button(self.labelFrame, text = "Image",command = self.fileDialog)
                self.button.grid(column = 1, row = 1, padx = 10, pady = 10)
                self.button1 = ttk.Button(self.labelFrame, text = "Live Cam",command = self.Web)
                self.button1.grid(column = 1, row = 2, padx = 10, pady = 10)
                self.button1 = ttk.Button(self.labelFrame, text = "Video Cam",command = self.video)
                self.button1.grid(column = 1, row = 3, padx = 10, pady = 10)
                self.button1 = ttk.Button(self.labelFrame, text = "Verify Person",command = self.verifyPerson)
                self.button1.grid(column = 1, row = 4, padx = 10, pady = 10)
                
    def fileDialog(self):

        self.filename = filedialog.askopenfilename(initialdir =  "/", title = "Select A File", filetype =
        (("jpeg files","*.jpg"),("all files","*.*")) )
        
        net = cv2.dnn.readNet("weights/yolov3.weights", "cfg/yolov3.cfg")
        classes = []
        with open("coco.names", "r") as f:
            classes = [line.strip() for line in f.readlines()]
        layer_names = net.getLayerNames()
        output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
        colors = np.random.uniform(0, 255, size=(len(classes), 3))
        
        # Loading image
        img = cv2.imread(self.filename)
        img = cv2.resize(img, None, fx=0.8, fy=0.7)
        height, width, channels = img.shape
        
        # Detecting objects
        blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        
        net.setInput(blob)
        outs = net.forward(output_layers)
        
        # Showing informations on the screen
        class_ids = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    # Object detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
        
                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
        
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        print(indexes)
        font = cv2.FONT_HERSHEY_PLAIN
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                color = colors[i]
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 1)
                cv2.putText(img, label, (x, y + 30), font, 3, color, 2)
        
        
        cv2.imshow("Image", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

root = Root()
root.mainloop()
