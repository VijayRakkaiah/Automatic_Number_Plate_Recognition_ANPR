from ultralytics import YOLO

# Load a model
model = YOLO("yolov8m.pt")  # build a new model from scratch

# Use the model
results = model.train(data="config.yaml",
                      epochs=5,
                      batch=16
                      )  # train the model