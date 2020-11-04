import tensorflow
from tensorflow import keras

import operator as op
import numpy as np
import matplotlib.pyplot as plt

import os
import glob

import cv2
import imutils
from imutils.contours import sort_contours

# A neural network that can detect handwritten digits.

# Remove images from a past test.
capture = None
del_list = glob.glob('Digit Recogition Images/imgtest*.jpg')
for path in del_list:
   try:
      os.remove(path)
   except:
      raise OSError("Could not remove file " + path)

# Load images you've taken of hand-written digits.
images = ["Images/testthree.jpg", "Images/testfour.jpg", "Images/testeight.jpg"]
correct_label_list = [3, 4, 8]

def process_image(PATH):
   my_img = cv2.imread(PATH, cv2.IMREAD_GRAYSCALE)
   my_img = cv2.resize(my_img, (28, 28))
   ret, bw_img = cv2.threshold(my_img, 127, 255, cv2.THRESH_BINARY)
   bw_img = 255 - bw_img
   tensorflow.image.convert_image_dtype(bw_img, dtype = tensorflow.float32)
   bw_img = np.array([bw_img]).astype('float32') / 255.0
   return bw_img

LABELS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

# Dataset and training/test sets.
set = keras.datasets.mnist
(train_imgs, train_labels), (test_imgs, test_labels) = set.load_data()
train_imgs = op.__truediv__(train_imgs, 255.0); test_imgs = op.__truediv__(test_imgs, 255.0)

def show_example():
   # Shows the first 25 images (and labels) in the training sample.
   # Call the function below if you want to see it.
   plt.figure(figsize = (10, 10))
   for i in range(25):
      plt.subplot(5, 5, i + 1)
      plt.grid(False); plt.xticks([]); plt.yticks([])
      plt.imshow(train_imgs[i], cmap = plt.cm.binary)
      plt.xlabel(LABELS[train_labels[i]])
   plt.show()

# The Network
model = keras.models.Sequential([
   keras.layers.Flatten(input_shape = (28, 28)),
   keras.layers.Dense(128, activation = 'relu'),
   keras.layers.Dropout(0.2),
   keras.layers.Dense(10)])
model.compile(
   optimizer = 'adam',
   loss = keras.losses.SparseCategoricalCrossentropy(from_logits = True),
   metrics = ['accuracy'])
model.fit(train_imgs, train_labels, epochs = 5)
loss, accuracy = model.evaluate(test_imgs, test_labels, verbose = 1)

probability = keras.Sequential([model, keras.layers.Softmax()])
predictions = probability.predict(test_imgs)

# Network Accuracy
val = 0
for index, num in enumerate(predictions):
   if np.argmax(predictions[index]) == test_labels[index]:
      val += 1
print("\nTotal: {}/{} --> Accuracy: {}%".format(
   val, len(predictions), round(accuracy * 100, 4)))

# User-Input Images and their results.
print("\nUser-Inputted Images:\n____________________")
for index, path in enumerate(images):
   print("Original Image: {}, Prediction: {}".format(
         correct_label_list[index],
         np.argmax(probability.predict(process_image(path)))))
   
   
   
