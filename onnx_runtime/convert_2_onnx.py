import torch
import timm

def convert(model, input, path, input_names, output_names):
    """
    Convert a PyTorch model to ONNX format.

    model: model
    input: input shape
    path: save path
    input_names: ['input']
    output_names: ['output']
    """

    torch.onnx.export(model, input, path, input_names, output_names)

    print("Done")




# model = timm.create_model("tf_efficientnet_b0", num_classes=1)

# state_dict = torch.load("/home/ubuntu/ductq/yte/results/ckpt/v5/last.ckpt")["state_dict"]
# state_dict = {k.replace('model.', ''):v for k, v in state_dict.items()}

# model.load_state_dict(state_dict)
# model.cuda()

# input = torch.rand(1, 3, 512, 512).cuda()
# path = "/home/ubuntu/ductq/onnx_runtime/save_model/tf_efficientnet_b0.onnx"
# input_names = ['input']
# output_names = ['output']


# convert(model=model, input=input, path=path, input_names=input_names, output_names=output_names)



from ultralytics import YOLO

model = YOLO("/home/ubuntu/ductq/yte/runs/detect/train/weights/last.pt")
model.export(format="torchscript", dynamic = True)

#.onnx 
#.engine (tensorrt)
# .torchscript (.pt)