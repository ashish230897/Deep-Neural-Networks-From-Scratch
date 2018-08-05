import cv2
import os
import numpy as np
from PIL import Image

path = "./Data/Train/"
dir = os.listdir(path)

train_images = []
train_labels = []
for img_name in dir:
    img1 = Image.open(path + img_name)
    img2 = np.array(img1.resize((64, 64), Image.ANTIALIAS))
    train_images.append(img2)
    if "cat" in img_name: train_labels.append(0)
    else: train_labels.append(1)

train_images = np.array(train_images)
train_labels = np.array(train_labels)
train_labels = np.expand_dims(train_labels, axis = -1)
print(train_images.shape)
print(train_labels.shape)

print(train_labels[500])
cv2.imshow("img", train_images[500])
cv2.waitKey(0)