import onnx
import onnxruntime as ort
import numpy as np


# Load the ONNX model
model = onnx.load("/home/ubuntu/ductq/yte2/train/runs/train/weights/last.onnx")
for input_tensor in model.graph.input:
    print(input_tensor.name)
for output_tensor in model.graph.output:
    print(output_tensor.name)

# Rename inputs
for input_tensor in model.graph.input:
    if input_tensor.name == "inputs":
        input_tensor.name = "images"

# Rename outputs
for output_tensor in model.graph.output:
    if output_tensor.name == "output0":
        output_tensor.name = "output1"

# Update any node references to the renamed inputs/outputs
for node in model.graph.node:
    node.input[:] = [("images" if name == "inputs" else name) for name in node.input]
    node.output[:] = [("output1" if name == "output0" else name) for name in node.output]

# Save the updated model
onnx.save(model, "/home/ubuntu/ductq/yte2/train/runs/train/weights/last.onnx")
print(f"Updated model saved")


for input_tensor in model.graph.input:
    print(input_tensor.name)
for output_tensor in model.graph.output:
    print(output_tensor.name)