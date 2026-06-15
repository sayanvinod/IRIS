from ultralytics import YOLO

DATASET_PATH = r"C:\Users\sayan\.cache\kagglehub\datasets\yolo_dataset\myopia-image-dataset\versions\1"

# Use a slightly stronger classification model
model = YOLO("yolov8s-cls.pt")

results = model.train(

    data=DATASET_PATH,

    # Training length
    epochs=100,
    patience=15,

    # Image resolution (much better than 64)
    imgsz=224,

    # Batch size (adjust if you get memory errors)
    batch=32,

    # Performance improvements
    pretrained=True,
    cache=True,
    workers=4,

    # Save experiment nicely
    project="myopia_training",
    name="myopia_v1",
    exist_ok=True,

    # Mild augmentation
    degrees=5,
    translate=0.05,
    scale=0.10,
    fliplr=0.0
)

# Validate model
metrics = model.val()
print(metrics)

# Export best model to CoreML for Xcode
model.export(format="coreml")