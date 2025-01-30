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



def normalize(image):

    mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
    std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)

    return (image - mean*255) / (std*255)



def get_image(input, voi_lut=True, fix_monochrome=True):
    
    dicom = pydicom.dcmread(input)
    
    # VOI LUT (if available by DICOM device) is used to transform raw DICOM data to "human-friendly" view
    if voi_lut == True:
        data = apply_voi_lut(dicom.pixel_array, dicom)
    else:
        data = dicom.pixel_array
                
        # depending on this value, X-ray may look inverted - fix that:
    if fix_monochrome == True and dicom.PhotometricInterpretation == "MONOCHROME1":
        data = np.amax(data) - data
            
    data = data - np.min(data)
    data = data / np.max(data)
    data = (data * 255).astype(np.uint8)

    image = cv2.cvtColor(data, cv2.COLOR_GRAY2RGB)
    image = cv2.resize(image, (512, 512))


    image_cls = normalize(image)
    image_cls = np.expand_dims(image_cls.transpose(2,0,1), axis=0)

    image_det = np.expand_dims(image.transpose(2,0,1), axis=0)

    return image_cls.astype(np.float32), image_det.astype(np.float32)


    
def sigmoid(x):

    s = 1/(1+np.exp(-x))

    return s

