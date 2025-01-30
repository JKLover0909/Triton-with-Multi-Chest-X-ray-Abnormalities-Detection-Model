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


def draw_bounding_box(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    label = f'({class_id}: {confidence:.2f})'
    color = (255, 0, )
    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
    cv2.putText(img, label, (x - 10, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)



def normalize(image):

    mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)

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

    scale = [data.shape[0]/512, data.shape[1]/512]

    image = cv2.cvtColor(data, cv2.COLOR_GRAY2RGB)
    image = cv2.resize(image, (512, 512))

    image_trs = image.transpose(2,0,1)

    image_cls = normalize(image_trs)
    image_cls = np.expand_dims(image_cls, axis=0)

    image_det = np.expand_dims(image_trs, axis=0)

    return image_cls.astype(np.float32), image_det.astype(np.float32), data, scale


    
def sigmoid(x):

    s = 1/(1+np.exp(-x))

    return s



def run_classification(model_name: str, input_image: np.ndarray,
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

    scores = results.as_numpy('output0')

    return scores



def run_detection(model_name: str, input_image: np.ndarray,
                  triton_client: grpcclient.InferenceServerClient, num=None):
    inputs = []
    outputs = []
    inputs.append(grpcclient.InferInput('images', input_image.shape, "FP32"))
    # Initialize the data
    inputs[0].set_data_from_numpy(input_image)

    outputs.append(grpcclient.InferRequestedOutput(f"num_detections_1"))
    outputs.append(grpcclient.InferRequestedOutput(f"detection_boxes_1"))
    outputs.append(grpcclient.InferRequestedOutput(f"detection_scores_1"))
    outputs.append(grpcclient.InferRequestedOutput(f"detection_classes_1"))

    # Test with outputs
    results = triton_client.infer(model_name=model_name, inputs=inputs, outputs=outputs)
    
    num_detections = results.as_numpy(f"num_detections_1")
    detection_boxes = results.as_numpy(f"detection_boxes_1")
    detection_scores = results.as_numpy(f"detection_scores_1")
    detection_classes = results.as_numpy(f"detection_classes_1")

    
    return num_detections, detection_boxes, detection_scores, detection_classes


def main(image_path, model_name, url):
    triton_client = get_triton_client(url)
    image_cls, image_det, orig_image, scale = get_image(image_path)

    score = run_classification(model_name, image_cls, triton_client)

    prob = sigmoid(score)


    if prob > 0.3:
        print(prob)
        
    
        num_detections, detection_boxes, detection_scores, detection_classes = \
                run_detection(model_name, image_det, triton_client)
        
        print(num_detections)
        for index in range(num_detections):
            box = detection_boxes[index]

            draw_bounding_box(orig_image,
                          detection_classes[index], 
                          detection_scores[index],
                          round(box[0] * scale[1]),
                          round(box[1] * scale[0]),
                          round((box[0] + box[2]) * scale[1]),
                          round((box[1] + box[3]) * scale[0]))

        cv2.imwrite('output.jpg', orig_image)

    else:
        print("No finding")
        cv2.imwrite('output.jpg', orig_image)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, default='/home/ubuntu/duynd/AIyte/secondnd/input/train/0007d316f756b3fa0baea2ff514ce945.dicom')
    parser.add_argument('--model_name', type=str, default='ensemble_model')
    parser.add_argument('--url', type=str, default='localhost:8004')
    args = parser.parse_args()
    main(args.image_path, args.model_name, args.url)
