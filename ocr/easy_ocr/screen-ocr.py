import sys
import os
import torch
import qlinear
from PIL import ImageGrab
from timeit import default_timer as timer
from utils import Utils
import easyocr


def screenGrab(rect):
    """ Given a rectangle, return a PIL Image of that part of the screen.
        Uses ImageGrab on Windows. """
    x, y, width, height = rect
    image = ImageGrab.grab(bbox=(x, y, x + width, y + height))
    return image

# Custom map location function to handle 'easyocr' module not found
def custom_load(path):
    return torch.load(path, map_location=lambda storage, loc: storage)

# Load the quantized models
q_detector = custom_load('./models/quantized_detection_model.pt')
q_recognizer = custom_load('./models/quantized_recognition_model.pt')

# Replace the quantized linear layers with QLinear in both encoder and decoder
node_args = ()
node_kwargs = {'device': 'aie'}
Utils.replace_node(q_detector, torch.ao.nn.quantized.dynamic.modules.linear.Linear, qlinear.QLinear, node_args, node_kwargs)
Utils.replace_node(q_recognizer, torch.ao.nn.quantized.dynamic.modules.linear.Linear, qlinear.QLinear, node_args, node_kwargs)

# Initialize the EasyOCR reader
reader = easyocr.Reader(['en'])  # Initialize with the desired language

# Replace the models in the reader object
reader.detector = q_detector
reader.recognizer = q_recognizer

if __name__ == "__main__":
    EXE = sys.argv[0]
    del(sys.argv[0])

    # Catch zero-args
    if len(sys.argv) != 4 or sys.argv[0] in ('--help', '-h', '-?', '/?'):
        sys.stderr.write(EXE + ": monitors section of screen for text\n")
        sys.stderr.write(EXE + ": Give x, y, width, height as arguments\n")
        sys.exit(1)

    # Get the screen coordinates from command line arguments
    x, y, width, height = map(int, sys.argv[:4])
    screen_rect = [x, y, width, height]
    print(EXE + ": watching " + str(screen_rect))

    # Loop to monitor the specified rectangle of the screen
    while True:
        start = timer()
        image = screenGrab(screen_rect)  # Grab the area of the screen
        image_path = "./temp_screengrab.png"
        image.save(image_path)  # Save the image to a temporary file
        
        result = reader.readtext(image_path, detail=0, paragraph=True)  # OCR the image
        
        # Output the OCR results
        if result:
            print(result)
        
        # print(f"Processing Time: {timer() - start}")
