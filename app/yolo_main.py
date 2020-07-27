###  Importing the Necessary Libraries ###

import cv2
import numpy as np
import os
import argparse
from scipy.spatial import distance as dist
from imutils.object_detection import non_max_suppression
from imutils import paths
import imutils
from tensorflow.keras.models import model_from_json
import requests
import json
import matplotlib.pyplot as plt

### END ###

### Function definitons ###

def detect_people(frame, net, ln, personIdx=0):
    
  MIN_CONF = 0.3
  NMS_THRESH = 0.3
  # grab the dimensions of the frame and  initialize the list of
	# results
  (H, W) = frame.shape[:2]
  results = []
  # construct a blob from the input frame and then perform a forward
	# pass of the YOLO object detector, giving us our bounding boxes
	# and associated probabilities
  blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
		swapRB=True, crop=False)
  net.setInput(blob)
  layerOutputs = net.forward(ln)

	# initialize our lists of detected bounding boxes, centroids, and
	# confidences, respectively
  boxes = []
  centroids = []
  confidences = []

	# loop over each of the layer outputs
  for output in layerOutputs:
		# loop over each of the detections
    for detection in output:
			# extract the class ID and confidence (i.e., probability)
			# of the current object detection
      scores = detection[5:]
      classID = np.argmax(scores)
      confidence = scores[classID]

			# filter detections by (1) ensuring that the object
			# detected was a person and (2) that the minimum
			# confidence is met
      if classID == personIdx and confidence > MIN_CONF:
				# scale the bounding box coordinates back relative to
				# the size of the image, keeping in mind that YOLO
				# actually returns the center (x, y)-coordinates of
				# the bounding box followed by the boxes' width and
				# height
        box = detection[0:4] * np.array([W, H, W, H])
        (centerX, centerY, width, height) = box.astype("int")

				# use the center (x, y)-coordinates to derive the top
				# and and left corner of the bounding box
        x = int(centerX - (width / 2))
        y = int(centerY - (height / 2))

				# update our list of bounding box coordinates,
				# centroids, and confidences
        boxes.append([x, y, int(width), int(height)])
        centroids.append((centerX, centerY))
        confidences.append(float(confidence))

	# apply non-maxima suppression to suppress weak, overlapping
	# bounding boxes
  idxs = cv2.dnn.NMSBoxes(boxes, confidences, MIN_CONF, NMS_THRESH)

	# ensure at least one detection exists
  if len(idxs) > 0:
		# loop over the indexes we are keeping
    for i in idxs.flatten():
			# extract the bounding box coordinates
      (x, y) = (boxes[i][0], boxes[i][1])
      (w, h) = (boxes[i][2], boxes[i][3])

			# update our results list to consist of the person
			# prediction probability, bounding box coordinates,
			# and the centroid
      r = (confidences[i], (x, y, x + w, y + h), centroids[i])
      results.append(r)

	# return the list of results
  return results

def distvio(image):

    # base path to YOLO directory
    MODEL_PATH = '/content/drive/My Drive/Colab_Notebooks/yolo-coco'
    # load the COCO class labels our YOLO model was trained on 
    labelsPath = os.path.sep.join([MODEL_PATH, "coco.names"])
    LABELS = open(labelsPath).read().strip().split("\n")
    # initialize minimum probability to filter weak detections along with
    # the threshold when applying non-maxima suppression


    
    # boolean indicating if NVIDIA CUDA GPU should be used
    USE_GPU = True

    # define the minimum safe distance (in pixels) that two people can be
    # from each other
    MIN_DISTANCE = 50

	# derive the paths to the YOLO weights and model configuration
    weightsPath = os.path.sep.join([MODEL_PATH, "yolov3.weights"])
    configPath = os.path.sep.join([MODEL_PATH, "yolov3.cfg"])

	# load our YOLO object detector trained on COCO dataset (80 classes)
    print("[INFO] loading YOLO from disk...")
    net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

	# check if we are going to use GPU
    if USE_GPU:
	# set CUDA as the preferable backend and target
        print("[INFO] setting preferable backend and target to CUDA...")
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

	# determine only the *output* layer names that we need from YOLO
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

	# loop over the frames from the video stream
    frame = image
	# resize the frame and then detect people (and only people) in it
    frame = imutils.resize(frame, width=700)
    results = detect_people(frame, net, ln,
                            personIdx=LABELS.index("person"))
 
    bboxes = []
    for i in results:
        bboxes.append(i[1])	
	# ensure there are *at least* two people detections (required in
	# order to compute our pairwise distance maps)
    if len(results) >= 2:
	# extract all centroids from the results and compute the
	# Euclidean distances between all pairs of the centroids
        centroids = np.array([r[2] for r in results])
        D = dist.cdist(centroids, centroids, metric="euclidean")

		# loop over the upper triangular of the distance matrix
        violation = 0
        for i in range(0, D.shape[0]):
            for j in range(i + 1, D.shape[1]):
				# check to see if the distance between any two
				# centroid pairs is less than the configured number
				# of pixels
                if D[i, j] < MIN_DISTANCE:
					# update our violation set with the indexes of
					# the centroid pairs
                    violation += 1
    return violation, np.array(bboxes)

def MaskDetector(bboxes, image, tolerance):
    def prepImg(img):
      return cv2.resize(image,(224,224)).reshape(1,224,224,3)/255.0
    with open(os.path.join('/content/drive/My Drive/Colab_Notebooks/DATA/model_mask.json'), 'r') as f:
        loadthehason = f.read()
    model = model_from_json(loadthehason)
    # Here we can upload our weights manually through providing its path
    model.load_weights(os.path.join("/content/drive/My Drive/Colab_Notebooks/DATA/model_mask.h5"))
    maskcounter = 0
    nomaskcounter = 0
    for (x, y, w, h) in bboxes:
        slicedImg = image[(y-tolerance):(y+tolerance+h),(x-tolerance):(x+tolerance+w)]
        pred = model.predict(prepImg(slicedImg))
        pred = np.argmax(pred)
        if pred==0:
          maskcounter += 1
        else:
          nomaskcounter += 1
    return [maskcounter, nomaskcounter]

### END ###

### Main BODY ###

def main():

	# Here we can define a single image for test purposes, if you want to check it, you can uncomment below
  #image = cv2.imread('/content/drive/My Drive/Colab_Notebooks/test.jpg')
  urlden = 'https://a-team-mall-api.herokuapp.com/density'
  urlmask = 'https://a-team-mall-api.herokuapp.com/mask'
  urlvio = 'https://a-team-mall-api.herokuapp.com/violations'
  while True:
    # Here I provided a link from https://www.insecam.org/, which is a good spot from Colorado, USA
    cap = cv2.VideoCapture('http://14.34.45.49:5000/webcapture.jpg?command=snap&channel=1?1595819773#.Xx5CgNgoHtk.link')
    try:
      _, image = cap.read()
    except Exception as e:
      continue
    violation, facecount = distvio(image)
    if len(facecount)==0:
        facecountstreaming = {"x": 70, 
                              "y": 20, 
                              "count": 0
                              }
        maskstreaming = {'x': 70,
                         'y': 20,
                        'mask': 0,
                        'nomask': 0
                         }
        violationstreaming = {"x": 70, 
                            "y": 20, 
                            "count": 0
                              }
    else:
      facecountnumber = facecount.shape[0]
      facecountstreaming = {"x": 70, 
                            "y": 20, 
                            "count": facecountnumber
                            }
      # I used tolerance = 10 for the MaskDetector function, it can be adjusted	
      maskamount = MaskDetector(facecount, image, 10)
      maskstreaming = {'x': 70,
                       'y': 20,
                       'mask': maskamount[0],
                       'nomask': maskamount[1]
                       }
      violationstreaming = {"x": 70, 
                           "y": 20, 
                           "count": violation
                          }
    # Here I am just trying to check whether I can post the data to our app
    gg = requests.post(urlden, json = facecountstreaming)
    zz = requests.post(urlmask, json = maskstreaming)
    yy = requests.post(urlvio, json = violationstreaming)
    # print(gg.status_code)
    #if zz.status_code == requests.codes.ok:
    #  print('Mask Uploaded')
    # This print below is to show people count, amount of mask on and off manually on console, you can uncomment this part if you want to check values on console
    print('Density: ' + str(facecountstreaming['count']) 
    + ', Mask On: ' + str(maskstreaming['mask']) 
    + ', Mask Off: ' + str(maskstreaming['nomask']) + ', Social Dist Violation: ' + str(violation))
    k = cv2.waitKey(30) & 0xff
    if k==27:
      break
    # Release the VideoCapture object
  cap.release()
  cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
