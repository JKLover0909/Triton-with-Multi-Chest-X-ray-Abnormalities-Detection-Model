import torch
import sys
sys.path.append('/home/ubuntu/ductq/yte')
from pipeline.utils import get_classification_model, get_image, get_class



class XRayClassification:
    def __init__(self, configs, device):
        self.configs = configs
        self.device = device
        self.model = get_classification_model(self.configs).to(self.device)
        self.model.eval()

    def class_predict(self, input, pipeline=None):

        if pipeline == True:
            image = input
        else:
            image,_ = get_image(input, self.configs)

        with torch.no_grad():
            output = self.model(image)
            predict = get_class(output, self.configs["classification"]["predict"]["threshold"])

        return predict
