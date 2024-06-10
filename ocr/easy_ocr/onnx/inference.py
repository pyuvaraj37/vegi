import easyocr
import matplotlib.pyplot as plt
import torch
import qlinear
from utils import Utils
from timeit import default_timer as timer


#Load the quantized models
q_detector = torch.load('./models/quantized_detection_model.pt')
# q_recognizer = torch.load('./models/quantized_recognition_model.pt')


print(q_detector)

# Initialize the EasyOCR reader
reader = easyocr.Reader(['en'])  # Initialize with the desired language

# Replace the models in the reader object
reader.detector = q_detector
# reader.recognizer = q_recognizer

for i in range(1,8):
    start = timer()
    image_path = "./images/test_image_"+str(i)+".png"
    result = reader.readtext(image_path, detail = 0, paragraph=True)
    print(result)
    #End Timer
    # print(f"CPU Total Time: {timer() - start}")
    print(f"NPU Total Time: {timer() - start}")



