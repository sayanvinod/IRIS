import kagglehub
import pandas as pd
import numpy as np
from glob import glob
import cv2
import matplotlib.pylab as plt

#Reading in images
#path = kagglehub.dataset_download("kellysanderson/myopia-image-dataset")
path = "C:/Users/sayan/.cache/kagglehub/datasets/kellysanderson/myopia-image-dataset/versions/1/IMAGES"

#print("Path to dataset files:", path)

myopia_files = glob(path + "/Myopia_images/*.png")
normal_files = glob(path + "/Normal_images/*.png")

img_mpl = plt.imread(myopia_files[20])
img_cv2 = cv2.imread(myopia_files[20])
img = plt.imread(normal_files[2])

#print(img_mpl.shape)
#print(img_cv2.shape)

#pd.Series(img_cv2.flatten()).plot(kind='hist', bins=50, title='Distribution of Pixel Values')
#plt.show()

'''
#Display images
fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(img_mpl)
ax.axis('off')
#plt.show()

#Display RGB Channels of image
fig, axs = plt.subplots(1, 3, figsize=(15,5))
axs[0].imshow(img_mpl[:,:,0], cmap='Reds')
axs[1].imshow(img_mpl[:,:,1], cmap='Greens')
axs[2].imshow(img_mpl[:,:,2], cmap='Blues')
axs[0].axis('off')
axs[1].axis('off')
axs[2].axis('off')
axs[0].set_title('Red Channel')
axs[1].set_title('Green Channel')
axs[2].set_title('Blue Channel')
#plt.show()

# Mpl vs cv2 numpy arrays
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].imshow(img_cv2)
axs[1].imshow(img_mpl)
axs[0].axis('off')
axs[1].axis('off')
axs[0].set_title('CV Image')
axs[1].set_title('MPL Image')
#plt.show()

# Converting from BGR to RGB
img_cv2_rgb = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
fig, ax = plt.subplots()
ax.imshow(img_cv2_rgb)
ax.axis('off')
#plt.show()

# Image Manipulation
fig, ax = plt.subplots()
ax.imshow(img)
ax.axis('off')
#plt.show()


img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
fig, ax = plt.subplots()
ax.imshow(img_gray, cmap='Greys')
ax.axis('off')
plt.show()


# Resizing and Scaling
img_resized = cv2.resize(img, None, fx=0.25, fy=0.25)
fig, ax = plt.subplots()
ax.imshow(img_resized)
ax.axis('off')
plt.show()

img_resize = cv2.resize(img, (100,200))
fig, ax = plt.subplots()
ax.imshow(img_resize)
ax.axis('off')
plt.show()

img_resizes = cv2.resize(img, (5000, 5000), interpolation = cv2.INTER_CUBIC)
fig, ax = plt.subplots()
ax.imshow(img_resizes)
ax.axis('off')
plt.show()
'''