import onnxruntime
import numpy as np
import cv2
from skimage import io
from timeit import default_timer as timer

# Specify the path to the quantized ONNX model
onnx_model_path = "./models/detector.onnx"

def loadImage(img_file):
    img = io.imread(img_file)  # RGB order
    if img.shape[0] == 2: img = img[0]
    if len(img.shape) == 2: img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    if img.shape[2] == 4: img = img[:, :, :3]
    return img

def normalizeMeanVariance(in_img, mean=(0.485, 0.456, 0.406), variance=(0.229, 0.224, 0.225)):
    img = in_img.astype(np.float32)
    img -= np.array([mean[0] * 255.0, mean[1] * 255.0, mean[2] * 255.0], dtype=np.float32)
    img /= np.array([variance[0] * 255.0, variance[1] * 255.0, variance[2] * 255.0], dtype=np.float32)
    return img

# Load and preprocess an image
def load_and_preprocess_image(image_path):
    img = loadImage(image_path)
    img = cv2.resize(img, (608, 800))  # Adjust this size if your model expects a different input
    img = normalizeMeanVariance(img)
    img = img.transpose(2, 0, 1)  # Change data layout to C, H, W
    return img[np.newaxis, :]  # Add batch dimension

# Prepare the image
image_path = 'path_to_your_image.jpg'  # Provide the path to your image file
input_data = load_and_preprocess_image(image_path)

# Create an inference session using ONNX Runtime
cpu_options = onnxruntime.SessionOptions()
detector_session = onnxruntime.InferenceSession(
    onnx_model_path,
    providers=['CPUExecutionProvider'],
    sess_options=cpu_options
)

# Prepare inputs for the model
input_name = detector_session.get_inputs()[0].name
ort_inputs = {input_name: input_data}

# Measure inference time
start = timer()
cpu_results = detector_session.run(None, ort_inputs)
cpu_total = timer() - start

# Print the inference time
print(f"Inference time: {cpu_total} seconds")


# import easyocr
# import torch
# from timeit import default_timer as timer

# #Load the quantized models
# detector_onnx = a

# # Initialize the EasyOCR reader
# reader = easyocr.Reader(['en'])  # Initialize with the desired language

# # Replace the models in the reader object
# reader.detector = q_detector

# for i in range(1,8):
#     start = timer()
#     image_path = "./images/test_image_"+str(i)+".png"
#     result = reader.readtext(image_path, detail = 0, paragraph=True)
#     print(result)
#     #End Timer
#     # print(f"CPU Total Time: {timer() - start}")
#     print(f"NPU Total Time: {timer() - start}")
