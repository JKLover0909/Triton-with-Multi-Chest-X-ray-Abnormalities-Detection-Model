import numpy as np
import cv2
import tritonclient.grpc as grpcclient
import sys
import argparse
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut




def get_triton_client(url: str = 'localhost:8004'):
    try:
        keepalive_options = grpcclient.KeepAliveOptions(
            keepalive_time_ms=2**31 - 1,
            keepalive_timeout_ms=20000,
            keepalive_permit_without_calls=False,
            http2_max_pings_without_data=2
        )
        triton_client = grpcclient.InferenceServerClient(
            url=url,
            verbose=False,
            keepalive_options=keepalive_options)
    except Exception as e:
        print("channel creation failed: " + str(e))
        sys.exit()
    return triton_client


def get_image(input, voi_lut=True, fix_monochrome=True):
    
    dicom = pydicom.dcmread(input)
    
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
    image = cv2.resize(image, (512,512))
    image = np.expand_dims(image.swapaxes(2,0),axis=0)

    return image.astype(np.float32)



def run_inference(model_name: str, input_image: np.ndarray,
                  triton_client: grpcclient.InferenceServerClient):
    inputs = []
    outputs = []
    inputs.append(grpcclient.InferInput('images', input_image.shape, "FP32"))
    # Initialize the data
    inputs[0].set_data_from_numpy(input_image)

    outputs.append(grpcclient.InferRequestedOutput('output0'))


    # Test with outputs
    results = triton_client.infer(model_name=model_name,
                                  inputs=inputs,
                                  outputs=outputs)

    score = results.as_numpy('output0')

    return score


def main(image_path, model_name, url):
    triton_client = get_triton_client(url)
    input_image= get_image(image_path)
    score = run_inference(model_name, input_image, triton_client)

    print(score)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, default='/home/ubuntu/ductq/yte/1.2.392.200046.100.14.3358867221307220014091186453111417222.dcm')
    parser.add_argument('--model_name', type=str, default='classification_model')
    parser.add_argument('--url', type=str, default='localhost:8004')
    args = parser.parse_args()
    main(args.image_path, args.model_name, args.url)
