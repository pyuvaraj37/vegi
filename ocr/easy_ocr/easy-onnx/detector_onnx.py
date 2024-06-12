import torch
from EasyOCR import easyocr
import onnx

ocr_reader = easyocr.Reader(['en'],
                                gpu=False,
                                detector=True,
                                recognizer=True,
                                quantize=True,
                                model_storage_directory=None,
                                user_network_directory=None,
                                download_enabled=True)

#Inputs
# in_shape = [1, 3, 608, 800]
# dummy_input = torch.rand(in_shape)
batch_size_1 = 500
batch_size_2 = 500
in_shape=[1, 3, batch_size_1, batch_size_2]
dummy_input = torch.rand(in_shape)

detector_onnx_save_path = "./models/detector.onnx"

with torch.no_grad():
    torch.onnx.export(ocr_reader.detector,
                        dummy_input,
                        detector_onnx_save_path,
                        export_params=True,
                        do_constant_folding=True,
                        opset_version=12,
                        # model's input names
                        input_names=['input'],
                        # model's output names, ignore the 2nd output
                        output_names=['output'],
                        # variable length axes
                        # dynamic_axes= {'input' : {0 : 'batch_size', 3: 'width'},  # Allow dynamic batch size and width
                        # 'output': {0: 'batch_size'}},
                        # verbose=False
                        dynamic_axes={'input' : {2 : 'batch_size_1', 3: 'batch_size_2'}})

onnx_model = onnx.load("./models/detector.onnx")
try:
    onnx.checker.check_model(onnx_model)
except onnx.checker.ValidationError as e:
    print('The model is invalid: %s' % e)
else:
    print('The model is valid!')