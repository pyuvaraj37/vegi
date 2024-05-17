#!/bin/python3
import threading
import argparse
import numpy as np
import cv2
import torch
import onnx
import onnxruntime as ort
import numpy as np
from PIL import Image
from pathlib import Path

from transformers import (AutoTokenizer, 
                          AutoModelForSequenceClassification)

sentiment_analysis_quantized_model_path = r'./model_onnx/model.onnx'
sentiment_analysis = onnx.load(sentiment_analysis_quantized_model_path)

providers = ['VitisAIExecutionProvider']
cache_dir = Path(__file__).parent.resolve()
provider_options = [{
            'config_file': 'vaip_config.json',
            'cacheDir': str(cache_dir),
            'cacheKey': 'modelcachekey'
            }]

session = ort.InferenceSession(sentiment_analysis.SerializeToString(), providers=providers,
                               provider_options=provider_options)
sentence = "I am feeling sad today"
tokenizer = AutoTokenizer.from_pretrained('./model')
inputs = tokenizer(sentence, return_tensors="pt", padding="max_length", truncation=True, max_length=128)

#print(inputs['attention_mask'])

outputs = session.run(None, 
    {'input_ids': inputs['input_ids'].numpy(), 
     'attention_mask' : inputs['attention_mask'].numpy(), 
     'token_type_ids' : inputs['token_type_ids'].numpy()})


#print(outputs)

#logits = outputs.logits
probabilities = torch.nn.functional.softmax(torch.from_numpy(np.array(outputs)), dim=-1)
predicted_class_index = probabilities.argmax().item()


sentiment_labels = ['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise']
    
print(f"Predicted sentiment: {sentiment_labels[predicted_class_index]}")
print('fin')