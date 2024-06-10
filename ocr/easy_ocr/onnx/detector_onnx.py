import torch
import easyocr

ocr_reader = easyocr.Reader(['en'],
                                gpu=False,
                                detector=True,
                                recognizer=True,
                                quantize=True,
                                model_storage_directory=None,
                                user_network_directory=None,
                                download_enabled=True)

#Inputs
in_shape = [1, 3, 608, 800]
dummy_input = torch.rand(in_shape)

detector_onnx_save_path = "./models/detector.onnx"

with torch.no_grad():
    y_torch_out, feature_torch_out = ocr_reader.detector(dummy_input)
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
                        dynamic_axes= {'input' : {0 : 'batch_size', 3: 'width'},  # Allow dynamic batch size and width
                        'output': {0: 'batch_size'}},
                        verbose=False)