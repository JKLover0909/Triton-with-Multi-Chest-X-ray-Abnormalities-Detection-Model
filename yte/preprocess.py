import cv2
import os, glob
from utils import parallel_iterate
import numpy as np
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut



def process(idx, paths, voi_lut = True, fix_monochrome = True):

    try: 
        dicom = pydicom.dcmread(paths[idx])
    
    # VOI LUT (if available by DICOM device) is used to transform raw DICOM data to "human-friendly" view
        if voi_lut:
            data = apply_voi_lut(dicom.pixel_array, dicom)
        else:
            data = dicom.pixel_array
                
        # depending on this value, X-ray may look inverted - fix that:
        if fix_monochrome and dicom.PhotometricInterpretation == "MONOCHROME1":
            data = np.amax(data) - data
            
        data = data - np.min(data)
        data = data / np.max(data)
        data = (data * 255).astype(np.uint8)

        image = cv2.cvtColor(data, cv2.COLOR_GRAY2RGB)
        image = cv2.resize(image, (512, 512))

        output_folder = "/home/ubuntu/ductq/yte/data"
        
        image_name = paths[idx].split("/")[-1].replace(".dicom", ".jpg")
        cv2.imwrite(os.path.join(output_folder,image_name), image)

        return data
    
    except:
        pass



image_files = list(glob.glob("/home/ubuntu/duynd/AIyte/secondnd/input/train/*.dicom"))[10000:]
parallel_iterate(range(len(image_files)), process, paths=image_files, workers=12)