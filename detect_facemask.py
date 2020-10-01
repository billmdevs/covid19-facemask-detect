# For video feeds

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
import os


def detect_mask(frame, facenet, masknet):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
    facenet.setInput(blob)
    detections = facenet.forward()

    faces = []
    locs = []
    preds = []

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > args["confidence"]:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)

            faces.append(face)
            locs.append((startX, startY, endX, endY))
        

    if len(faces) > 0:
        faces = np.array(faces, dtype="float32")
        preds = masknet.predict(faces, batch_size=32)

    return (preds, locs)



parser = argparse.ArgumentParser()
parser.add_argument("-f", "--face", type=str, default="face_detector", help="path to face detector model")
parser.add_argument("-m", "--model", type=str, default="mask_detector.model", help="path to trained face mask model")
parser.add_argument("-c", "--confidence", type=float, default=0.5, help="minimum probability to filter weak detections")

args = vars(parser.parse_args())

print("[INFO] Loading face detector model")
prototxtpath = os.path.sep.join([args["face"], "deploy.prototxt"])
weightspath = os.path.sep.join([args["face"], "res10_300x300_ssd_iter_140000.caffemodel"])
facenet = cv2.dnn.readNet(prototxtpath, weightspath)

print("[INFO] Loading face mask detector model...")
masknet = load_model(args["model"])

print("[INFO] Starting video stream...")
vidstream = VideoStream(src=0).start()
time.sleep(2.0)

while True:
    frame = vidstream.read()
    frame = imutils.resize(frame, width=400)

    (preds, locs) = detect_mask(frame, facenet, masknet)

    for (pred, box) in zip(preds, locs):
        (startX, startY, endX, endY) = box
        (withmask, withoutmask) = pred

        if withmask > withoutmask:
            label = "Safe"
        else: 
            label = "Not Safe"
        
        if label == "Safe":
            color = (0, 255, 0)
        else:
            color = (0, 0, 255) 
        
        label = "{}: {:.2f}%".format(label, max(withmask, withoutmask) * 100)

        cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)


    cv2.imshow("Video", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
vidstream.stop()