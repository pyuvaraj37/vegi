# import numpy as np
# import torch 
# #from datasets import load_dataset, load_metric
# from transformers import (AutoTokenizer, 
#                           AutoModelForSequenceClassification, 
#                           TrainingArguments, 
#                           Trainer)


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

model = ORTModelForSequenceClassification.from_pretrained('./model',from_transformers=True)
model.save_pretrained('./model_onnx')
print('hello')

