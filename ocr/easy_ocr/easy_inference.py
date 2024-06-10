import easyocr
import matplotlib.pyplot as plt
import torch
import qlinear
from utils import Utils
from timeit import default_timer as timer

# Check if CUDA is available and set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

#Load the quantized models
q_detector = torch.load('./models/quantized_detection_model.pt')
q_recognizer = torch.load('./models/quantized_recognition_model.pt')

#Transform to AIE
# Replace the quantized linear layers with QLinear in both encoder and decoder
node_args = ()
node_kwargs = {'device': 'aie'}
Utils.replace_node(q_detector, torch.ao.nn.quantized.dynamic.modules.linear.Linear, qlinear.QLinear, node_args, node_kwargs)
Utils.replace_node(q_recognizer, torch.ao.nn.quantized.dynamic.modules.linear.Linear, qlinear.QLinear, node_args, node_kwargs)

print(q_detector)
print(q_recognizer)

# Initialize the EasyOCR reader
reader = easyocr.Reader(['en'])  # Initialize with the desired language

# # Initialize the EasyOCR reader with GPU support if available
# reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())

# Replace the models in the reader object
reader.detector = q_detector
reader.recognizer = q_recognizer

for i in range(1,8):
    start = timer()
    image_path = "./images/test_image_"+str(i)+".png"
    result = reader.readtext(image_path, detail = 0, paragraph=True)
    print(result)
    #End Timer
    # print(f"CPU Total Time: {timer() - start}")
    print(f"NPU Total Time: {timer() - start}")



