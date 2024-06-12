import torch
from EasyOCR import easyocr
import onnx

ocr_reader = easyocr.Reader(['en'])

#Inputs
batch_size_1_1 = 500
in_shape_1=[1, 1, 64, batch_size_1_1]
dummy_input_1 = torch.rand(in_shape_1)
dummy_input_1 = dummy_input_1

batch_size_2_1 = 50
in_shape_2=[1, batch_size_2_1]
dummy_input_2 = torch.rand(in_shape_2)
dummy_input_2 = dummy_input_2

dummy_input = (dummy_input_1, dummy_input_2)

recognizer_onnx_save_path = "./models/recognizer.onnx"

with torch.no_grad():
    torch.onnx.export(ocr_reader.recognizer,
                        dummy_input,
                        recognizer_onnx_save_path,
                        export_params=True,
                        opset_version=11,
                        # model's input names
                        input_names = ['input1','input2'],
                        output_names = ['output'],
                        dynamic_axes={'input1' : {3 : 'batch_size_1_1'}})

onnx_model = onnx.load("./models/recognizer.onnx")
try:
    onnx.checker.check_model(onnx_model)
except onnx.checker.ValidationError as e:
    print('The model is invalid: %s' % e)
else:
    print('The model is valid!')