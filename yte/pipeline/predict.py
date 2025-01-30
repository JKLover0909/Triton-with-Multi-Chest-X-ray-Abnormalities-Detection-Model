from classification import XRayClassification
from detection import XRayDetection
import sys
sys.path.append('/home/ubuntu/ductq/yte')
from pipeline.utils import get_configs, get_image


class Prediction:
    def __init__(self, configs, device):
        self.configs = get_configs(configs)
        self.device = device
        self.model_classification = XRayClassification(self.configs, self.device)
        self.model_detection = XRayDetection(self.configs, self.device)


    def prediction(self, input):

        image_cls, image_detect = get_image(input, self.configs)

        output_class = self.model_classification.class_predict(image_cls.to(self.device), pipeline=True)

        if output_class == 1: 
            output_detect = self.model_detection.object_predict(image_detect.to(self.device), pipeline=True)
            return output_detect

        else:
            print("No finding")
