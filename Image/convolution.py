from __future__ import absolute_import, division

import os
import sys

import cv2
import numpy as np

from skimage.exposure import rescale_intensity

def convolve(image, kernel):
   '''
   A basic convolution operation involving an image and a kernel.
   :param image: The input image.
   :param kernel: The input kernel.
   :return: The convolved image.
   '''

   # Image/Kernel Spatial Dimensions
   image_height, image_width = image.shape[:2]
   kernel_height, kernel_width = kernel.shape[:2]

   # Padding Image
   padding = (kernel_width - 1) // 2
   image = cv2.copyMakeBorder(image, padding, padding, padding, padding, cv2.BORDER_REPLICATE)

   output = np.zeros((image_height, image_width), dtype = 'float32')

   # Convolution
   for y in np.arange(padding, image_height + padding):
      for x in np.arange(padding, image_width + padding):
         reg = image[y - padding: y + padding + 1, x - padding: x + padding + 1]
         sum = (reg * kernel).sum()
         output[y - padding, x - padding] = sum

   # Bring Image to Correct Range
   output = rescale_intensity(output, in_range = (0, 255))
   output = (output * 255).astype(int)

   return output

# Blurring Kernels --> Blur or Smoothen
small = np.ones((7, 7), dtype = "float32") * (1.0 / (7 * 7))
large = np.ones((21, 21), dtype = "float32") * (1.0 / (21 * 21))

# Sharpening Kernel --> Sharpen Image Features
sharpen = np.array((
   [0, -1, 0],
   [-1, 5, -1],
   [0, -1, 0]), dtype = 'int32')

# Sobel Kernels --> Detect Horizontal/Vertical Changes
sobel_x = np.array((
   [-1, 0, 1],
   [-2, 0, 2],
   [-1, 0, 1]), dtype = 'int32')

sobel_y = np.array((
   [-1, -2, -1],
   [0, 0, 0],
   [1, 2, 1]), dtype = 'int32')

# Laplacian Kernel --> Edge Detection
laplacian = np.array((
   [0, 1, 0],
   [1, -4, 1],
   [0, 1, 0]), dtype = 'int32')

# Test on Images
imgdir = os.path.join(os.path.dirname(__file__), 'test_images')
test_images = [os.path.join(imgdir, 'travel-back-in-time-with-google-street-view-136396071608202601-150211172511.jpg')]

for img in test_images:
   image = cv2.imread(img)
   gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

   output = convolve(gray_image, laplacian)

   cv2.imwrite(os.path.join(imgdir, "test1.jpg"), output)









