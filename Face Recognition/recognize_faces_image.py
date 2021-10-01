# Author: Ajinkya Gholape
# python recognize_faces_image.py --encodings encodings.pickle --image input\ images/avg.jpg

import face_recognition
import argparse
import pickle
import cv2
import os
import datetime

path = "output/"
# This is the name of Image file that we will save 
currentDT = datetime.datetime.now() 
ap = argparse.ArgumentParser()
ap.add_argument("-e", "--encodings", required=True,
	help="path to serialized db of facial encodings")
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
ap.add_argument("-d", "--detection-method", type=str, default="cnn",
	help="face detection model to use: either `hog` or `cnn`")
args = vars(ap.parse_args())

print("[INFO] loading encodings...")
# Load pickle object as in read byte mode
data = pickle.loads(open(args["encodings"], "rb").read())

image = cv2.imread(args["image"])
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

print("[INFO] recognizing faces...")
boxes = face_recognition.face_locations(rgb,
	model=args["detection_method"])
encodings = face_recognition.face_encodings(rgb, boxes)

names = []

for encoding in encodings:
	try:
		matches = face_recognition.compare_faces(data["encodings"],
			encoding)
		name = "Unknown"
		if True in matches:
			matchedIdxs = [i for (i, b) in enumerate(matches) if b]
			counts = {}
			for i in matchedIdxs:
				name = data["names"][i]
				counts[name] = counts.get(name, 0) + 1
			name = max(counts, key=counts.get)
		names.append(name)
	except:
		print('something went wrong')
		return

for ((top, right, bottom, left), name) in zip(boxes, names):
	try:
		cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
		y = top - 15 if top - 15 > 15 else top + 15
		cv2.putText(image, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
			0.75, (0, 255, 0), 2)
	except:
		print('something went wrong')

cv2.imwrite(os.path.join(path , currentDT.strftime("%Y-%m-%d%_H:%M:%S.jpg")),image)
print("Image Saved")
#cv2.imshow("Image", image)
#cv2.waitKey(0)
