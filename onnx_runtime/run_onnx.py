import onnx
import onnxruntime as ort
import numpy as np


# ort_sess = ort.InferenceSession("/home/ubuntu/ductq/triton_pipeline/models/detection_model/1/model.onnx")
# x = np.random.rand(1,3,512,512).astype(np.float32)
# outputs = ort_sess.run(None, {'images': x})
# predicted = outputs
# print(predicted)


# Load the ONNX model
model = onnx.load("/home/ubuntu/ductq/triton_pipeline/models/detection_model_3/1/model.onnx")
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
    if output_tensor.name == "output2":
        output_tensor.name = "output3"

# Update any node references to the renamed inputs/outputs
for node in model.graph.node:
    node.input[:] = [("images" if name == "inputs" else name) for name in node.input]
    node.output[:] = [("output3" if name == "output2" else name) for name in node.output]

# Save the updated model
onnx.save(model, "/home/ubuntu/ductq/triton_pipeline/models/detection_model_3/1/model.onnx")
print(f"Updated model saved")


for input_tensor in model.graph.input:
    print(input_tensor.name)
for output_tensor in model.graph.output:
    print(output_tensor.name)


