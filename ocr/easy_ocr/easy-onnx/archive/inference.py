import onnxruntime
import numpy as np
import cv2
import easyocr
from skimage import io
from timeit import default_timer as timer

# Specify the path to the quantized ONNX model
onnx_model_path = "./models/detector_quantized.onnx"

def loadImage(img_file):
    img = io.imread(img_file)  # RGB order
    if img.shape[0] == 2:
        img = img[0]
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    if img.shape[2] == 4:
        img = img[:, :, :3]
    return img

def normalizeMeanVariance(in_img, mean=(0.485, 0.456, 0.406), variance=(0.229, 0.224, 0.225)):
    img = in_img.astype(np.float32)
    img -= np.array([mean[0] * 255.0, mean[1] * 255.0, mean[2] * 255.0], dtype=np.float32)
    img /= np.array([variance[0] * 255.0, variance[1] * 255.0, variance[2] * 255.0], dtype=np.float32)
    return img

def load_and_preprocess_image(image_path):
    img = loadImage(image_path)
    img = cv2.resize(img, (608, 608))  # Resize as per model requirements
    img = normalizeMeanVariance(img)
    img = img.transpose(2, 0, 1)  # Change data layout to C, H, W
    return img[np.newaxis, :]  # Add batch dimension

def to_numpy(tensor):
    if tensor.requires_grad:
        return tensor.detach().cpu().numpy()
    return tensor.cpu().numpy()

def onnx_detector(image_path):
    input_data = load_and_preprocess_image(image_path)
    ort_inputs = {input_name: input_data}
    ort_outs = detector_session.run(None, ort_inputs)
    return ort_outs[0] 

# Initialize the EasyOCR reader without loading the default detector
reader = easyocr.Reader(['en'])

reader.detector = onnx_model_path

# Create an inference session using ONNX Runtime
cpu_options = onnxruntime.SessionOptions()
detector_session = onnxruntime.InferenceSession(onnx_model_path, providers=['CPUExecutionProvider'], sess_options=cpu_options)
input_name = detector_session.get_inputs()[0].name

# Process images and perform inference
for i in range(1, 8):
    start = timer()
    image_path = "./images/test_image_" + str(i) + ".png"
    input_data = load_and_preprocess_image(image_path)
    ort_inputs = {input_name: input_data}

    # Run the ONNX detector
    detected_boxes = detector_session.run(None, ort_inputs)

    # Now, use the detected boxes in the EasyOCR Reader
    # This step assumes you need to further process detected_boxes to fit EasyOCR's expected input format
    results = reader.recognize(image=input_data, text_detection_boxes=detected_boxes, detail=0, paragraph=True)
    
    print(results)
    print(f"NPU Total Time: {timer() - start}")




 

# Initialize the EasyOCR reader
reader = easyocr.Reader(['en'], detector=False)  # We replace the default detector

# Use the ONNX model as the detector
reader.detector = onnx_detector

# Processing images and extracting text
for i in range(1, 8):
    start = timer()
    image_path = "./images/test_image_" + str(i) + ".png"
    result = reader.readtext(image_path, detail=0, paragraph=True)
    print(result)
    print(f"NPU Total Time: {timer() - start}")
