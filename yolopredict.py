from ultralytics import YOLO
import numpy as np

model = YOLO(r"C:\Users\sayan\OneDrive\Cursor-Files\runs\classify\train3\weights\best.pt")

results = model(r"C:\Users\sayan\.cache\kagglehub\datasets\kellysanderson\myopia-image-dataset\versions\1\IMAGES\Myopia_images\myopia47247.png")

names_dict = results[0].names

probs = results[0].probs.data
print(type(probs))
probs_list = probs.tolist() if hasattr(probs, 'tolist') else list(probs)
print(probs_list)
print(names_dict)
print(names_dict[np.argmax(probs_list)])