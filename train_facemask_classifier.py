import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths


parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset", required=True, help="Path to input dataset")
parser.add_argument("-p", "--plot", type=str, default=plot.png, help="Path to output loss or accuracy plot")
parser.add_argument("-m", "--model", type=str, default="mask_detector.model", help="Path to output face mask detector model")
args = parser.parse_args()

#Initialize initial learning rate, number of epochs and batch size
INIT_LR = le-4
EPOCHS = 20
BS = 32

#Load and preprocess

print("[INFO] Loading images...")
imagepaths = list(paths.list_images(args["dataset"]))
data = []
labels = []

for imagepath in imagepaths:
    label = imagedirs.split(os.path.step)[-2]

    image = load_img(imagepaths, target_size=224,224)
    image = img_to_array(image)
    image = preprocess_input(image)

    data.append(image)
    data.append(label)

data = np.array(data, dtype="dfloat32")
labels = np.array(labels)

lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

(trainX, testX, trainY, testY) = train_test_split(data, labels, train_size=0.80, random_state=42, stratify=labels)

aug = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest"
)