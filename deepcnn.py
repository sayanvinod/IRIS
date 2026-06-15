import tensorflow as tf
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.callbacks import TensorBoard
import os
import cv2
import imghdr
from matplotlib import pyplot as plt
import numpy as np

# Prevent OOM errors by setting GPU memoory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

data_dir = 'C:/Users/sayan/.cache/kagglehub/datasets/kellysanderson/myopia-image-dataset/versions/1/IMAGES'

image_exts = ['jpeg', 'jpg', 'bmp', 'png', 'tiff', 'gif', 'webp']

# CLEAN DATA
for image_class in os.listdir(data_dir):
    for image in os.listdir(os.path.join(data_dir, image_class)):
        image_path = os.path.join(data_dir, image_class, image)
        try:
            img = cv2.imread(image_path)
            tip = imghdr.what(image_path)
            if tip not in image_exts:
                print('Image not in ext list {}'.format(image_path))
                os.remove(image_path)
        except Exception as e:
            print('Issue with image {}'.format(image_path))

# LOAD DATA
# Load data with limited samples per class for testing
data = image_dataset_from_directory(data_dir, batch_size=32, shuffle=True)

# Limit the dataset size for faster testing (approximately 100 images per class)
# Assuming 2 classes, this will give roughly 100 images per class
data = data.take(200)  # Adjust this number based on your number of classes

print(type(data))

data_iterator = data.as_numpy_iterator()

batch = data_iterator.next()

fig, ax = plt.subplots(ncols = 4, figsize = (20, 20))
for idx, img in enumerate(batch[0][:4]):
    ax[idx].imshow(img.astype(int))
    ax[idx].title.set_text(batch[1][idx])
    plt.show()

# PREPROCESS DATA
data = data.map(lambda x, y: (x/255, y))
scaled_iterator = data.as_numpy_iterator()
scaled_batch = scaled_iterator.next()

print("Data length: ", len(data))

train_size = int(len(data)*.7)
val_size = int(len(data)*.2)
test_size = int(len(data)*.1)

print("Train size: ", train_size)
print("Validation size: ", val_size)
print("Test size: ", test_size)

train = data.take(train_size)
val = data.skip(train_size).take(val_size)
test = data.skip(train_size + val_size).take(test_size)

# MODEL BUILDING
model = Sequential()

model.add(Conv2D(16, (3, 3), 1, activation = 'relu', input_shape = (256, 256, 3)))
model.add(MaxPooling2D())

model.add(Conv2D(32, (3, 3), 1, activation = 'relu'))
model.add(MaxPooling2D())

model.add(Conv2D(16, (3, 3), 1, activation = 'relu'))
model.add(MaxPooling2D())

model.add(Flatten())

model.add(Dense(256, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))

model.compile('adam', loss = BinaryCrossentropy(), metrics = ['accuracy'])
print(model.summary())

logdir = 'logs'
tensorboard_callback = TensorBoard(log_dir = logdir)
hist = model.fit(train, epochs = 20, validation_data = val, callbacks = [tensorboard_callback])

fig = plt.figure()
plt.plot(hist.history['loss'], color = 'teal', label = 'loss')
plt.plot(hist.history['val_loss'], color = 'orange', label = 'val_loss')
fig.suptitle('Loss', fontsize = 20)
plt.legend(loc = 'upper left')
plt.show()