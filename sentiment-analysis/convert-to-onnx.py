# import numpy as np
# import torch 
# #from datasets import load_dataset, load_metric
from transformers import (AutoTokenizer, 
                          AutoModelForSequenceClassification, 
                          TrainingArguments, 
                          Trainer)


# model_name = './model'
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=6, local_files_only=True)

# vai_q_onnx.quantize_static(
#     model_name,
#     output_model_path,
#     dr,
#     activation_type=vai_q_onnx.QuantType.QUInt8,
#     calibrate_method=vai_q_onnx.PowerOfTwoMethod.MinMSE,
#     enable_dpu=True,
#     extra_options={
#         'ActivationSymmetric': True,
#     })
from optimum.onnxruntime import ORTModelForSequenceClassification
import torch


model = AutoModelForSequenceClassification.from_pretrained('./model')
torch.ao.quantization.quantize_dynamic(
                model, {torch.nn.Linear}, dtype=torch.qint8, inplace=True )
torch.save(model, "./model_onnx_quantized/model.pt")
# vai_q_onnx.quantize_static(
#     './model_onnx/model.onnx',
#     './model_onnx_quantized/model.onnx',
#     calibration_data_reader=None,
#     quant_format=vai_q_onnx.QuantFormat.QDQ,
#     calibrate_method=vai_q_onnx.CalibrationMethod.MinMax,
#     activation_type=vai_q_onnx.QuantType.QInt8,
#     weight_type=vai_q_onnx.QuantType.QInt8,
# )

print('hello')

