from ultralytics import YOLO
import torch

model = YOLO("/home/ubuntu/ductq/yte2/train/runs/train/weights/last.pt")

model.export(format="torchscript",imgsz=512, nms=True)

# 1, 5, 5376


# model = YOLO("/home/ubuntu/ductq/yte2/train/runs/train/weights/last.pt").eval().cuda()

# traced_model = torch.jit.trace(model, torch.randn(1, 3, 512, 512).cuda())
# torch.jit.save(traced_model, "model.pt")