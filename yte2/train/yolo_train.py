from ultralytics import YOLO

# Load a model
model = YOLO("/home/ubuntu/ductq/yte2/train/yolo11m.pt")  # load a pretrained model (recommended for training)

# Train the model
results = model.train(data="/home/ubuntu/ductq/yte2/train/data_yolo/data.yaml", \
                      epochs=70, imgsz=512, cls=1.0, box=8.5, dfl=2.5, \
                      name="/home/ubuntu/ductq/yte2/train/runs/train")
