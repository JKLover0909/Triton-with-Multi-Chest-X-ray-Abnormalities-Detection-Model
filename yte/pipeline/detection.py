import sys
sys.path.append('/home/ubuntu/ductq/yte')
from pipeline.utils import get_detection_model, get_image

class XRayDetection:
    def __init__(self, configs, device):
        
        self.configs = configs
        self.device = device
        self.model = get_detection_model(self.configs).to(self.device)

    def object_predict(self, input, pipeline=None):

        if pipeline == True:
            image = input
        else:
            _,image = get_image(input, self.configs)

        output = self.model(image, save=True, imgsz=self.configs["data"]["shape"],\
                            conf=self.configs["detection"]["predict"]["conf"], iou=self.configs["detection"]["predict"]["iou"])
        
        return output
