from ultralytics import YOLO

# Load a model
model = YOLO("yolo11s.pt")  # load a pretrained model (recommended for training)

# Train the model
results = model.train(data="/home/ubuntu/ductq/data_yolo_c1/data.yaml", epochs=50, imgsz=512, cls=1.0, box=9.5, dfl=2.5)


# results = model.train(data="/home/ubuntu/ductq/data_yolo_c2/data.yaml", epochs=50, imgsz=512, cls=1.0, box=5.5, dfl=3.5)


# results = model.train(data="/home/ubuntu/ductq/data_yolo_c3/data.yaml", epochs=50, imgsz=512, cls=1.0, box=5.5, dfl=3.5)


# results = model.train(data="/home/ubuntu/ductq/data_yolo_c4/data.yaml", epochs=50, imgsz=512, cls=1.0, box=5.5, dfl=3.5)


