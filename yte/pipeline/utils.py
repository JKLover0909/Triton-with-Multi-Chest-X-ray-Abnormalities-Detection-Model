import torch
import yaml
import timm
import numpy as np
import cv2
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
from ultralytics import YOLO
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

def get_configs(config):
    with open(config, 'r',encoding='utf-8') as f:
        configs = yaml.safe_load(f)
    return configs


def get_image(input, configs):
    
    dicom = pydicom.dcmread(input)
    
    # VOI LUT (if available by DICOM device) is used to transform raw DICOM data to "human-friendly" view
    if configs["data"]["voi_lut"]:
        data = apply_voi_lut(dicom.pixel_array, dicom)
    else:
        data = dicom.pixel_array
                
        # depending on this value, X-ray may look inverted - fix that:
    if configs["data"]["fix_monochrome"] and dicom.PhotometricInterpretation == "MONOCHROME1":
        data = np.amax(data) - data
            
    data = data - np.min(data)
    data = data / np.max(data)
    data = (data * 255).astype(np.uint8)

    image = cv2.cvtColor(data, cv2.COLOR_GRAY2RGB)
    image = cv2.resize(image, (configs["data"]["shape"], configs["data"]["shape"]))

    image_cls = A.Compose([A.Normalize(),ToTensorV2()])(image=image)["image"]
    image_cls = image_cls.unsqueeze(0)

    image_detect = torch.from_numpy((image/255.0)).permute(2,0,1).unsqueeze(0)

    return image_cls, image_detect
    


def get_classification_model(config):
    model = timm.create_model(config["classification"]["model"]["name"], num_classes=config["classification"]["model"]["classes"])

    state_dict = torch.load(config["classification"]["model"]["path"])["state_dict"]
    state_dict = {k.replace('model.', ''):v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)

    return model


def get_class(output, threshold):
    
    logit = torch.sigmoid(output)
    
    return 1 if logit > threshold else 0



def get_detection_model(configs):

    model = YOLO(configs["detection"]["model"]["path"])

    return model
