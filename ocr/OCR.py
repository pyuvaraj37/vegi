
import os
import torch
from PIL import ImageGrab
from timeit import default_timer as timer
import easyocr

class OCR:
    def __init__(self, x, y, width, height, settings):
        print("Intializing OCR-based Gamer Capture")
        self.settings = settings
        self.screen_rect = [x, y, width, height]
        self.reader = easyocr.Reader(['en'])  # Initialize with the desired language
        self.settings = settings
        if not os.path.isfile('./models/ocr/quantized_detection_model.pt') and not os.path.isfile('./models/ocr/quantized_recognition_model.pt'): 
            print("Game capture models not downloaded. Installing models...")
            # Initialize the EasyOCR reader
            detection_model = self.reader.detector
            recognition_model = self.reader.recognizer
            # Dynamically quantize the full EasyOCR model
            quantized_reader = torch.ao.quantization.quantize_dynamic(
                detection_model, {torch.nn.Linear}, dtype=torch.qint8, inplace=True
            )
            quantized_recognition_model = torch.ao.quantization.quantize_dynamic(
                recognition_model, {torch.nn.Linear}, dtype=torch.qint8, inplace=True
            )
            # Save the quantized models separately
            os.mkdir("./models/ocr")
            torch.save(quantized_reader, './models/ocr/quantized_detection_model.pt')
            torch.save(quantized_recognition_model, './models/ocr/quantized_recognition_model.pt')

        print("Loading models...")
        self.reader.detector = torch.load('./models/ocr/quantized_detection_model.pt')
        self.reader.recognizer = torch.load('./models/ocr/quantized_recognition_model.pt')
        node_args = ()
        node_kwargs = {'device': 'aie'}
        print("Converting models...")
        if (self.settings[0] == 'npu'):
            import qlinear 
            from utils import Utils
            Utils.replace_node(self.reader.detector, torch.ao.nn.quantized.dynamic.modules.linear.Linear, qlinear.QLinear, node_args, node_kwargs)
            Utils.replace_node(self.reader.recognizer, torch.ao.nn.quantized.dynamic.modules.linear.Linear, qlinear.QLinear, node_args, node_kwargs)
        #Add gpu support
        self.image_path = "./temp/dialogue_capture.png"   

    def screenGrab(self, rect):
        """ Given a rectangle, return a PIL Image of that part of the screen.
            Uses ImageGrab on Windows. """
        x, y, width, height = rect
        image = ImageGrab.grab(bbox=(x, y, x + width, y + height))
        return image

    def run(self):
        print("Extracting dialogue...")
        image = self.screenGrab(self.screen_rect)  # Grab the area of the screen
        image.save(self.image_path)   # Save the image to a temporary file
        result = self.reader.readtext(self.image_path, detail=0, paragraph=True)  # OCR the image
        return result
