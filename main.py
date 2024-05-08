#!/bin/python3
import threading
import argparse
import numpy as np
import cv2
import onnx
import onnxruntime as ort
import numpy as np
from PIL import Image
from pathlib import Path


def unpickle(file):
    import pickle
    with open(file,'rb') as fo:
        dict = pickle.load(fo, encoding='latin1')
    return dict

def run_model(session): 
    datafile = r'./data/cifar-10-batches-py/test_batch'
    metafile = r'./data/cifar-10-batches-py/batches.meta'

    data_batch_1 = unpickle(datafile) 
    metadata = unpickle(metafile)

    images = data_batch_1['data']
    labels = data_batch_1['labels']
    images = np.reshape(images,(10000, 3, 32, 32))


    dirname = 'images'
    import os
    if not os.path.exists(dirname):
        os.mkdir(dirname)


    #Extract and dump first 10 images 
    for i in range (0,10): 
        im = images[i]
        im  = im.transpose(1,2,0)
        im = cv2.cvtColor(im,cv2.COLOR_RGB2BGR)
        im_name = f'./images/image_{i}.png'
        cv2.imwrite(im_name, im)

    #Pick dumped images and predict
    for i in range (0,10): 
        image_name = f'./images/image_{i}.png'
        image = Image.open(image_name).convert('RGB')
        # Resize the image to match the input size expected by the model
        image = image.resize((32, 32))  
        image_array = np.array(image).astype(np.float32)
        image_array = image_array/255

        # Reshape the array to match the input shape expected by the model
        image_array = np.transpose(image_array, (2, 0, 1))  

        # Add a batch dimension to the input image
        input_data = np.expand_dims(image_array, axis=0)


        # Run the model
        outputs = session.run(None, {'input': input_data})


        # Process the outputs
        output_array = outputs[0]
        predicted_class = np.argmax(output_array)
        predicted_label = metadata['label_names'][predicted_class]
        label = metadata['label_names'][labels[i]]
        print(f'Image {i}: Actual Label {label}, Predicted Label {predicted_label}')




print('Start')
ocr_quantized_model_path = r'./ocr/models/<ocr_model>.onnx'
sentiment_analysis_quantized_model_path = r'./sentiment-analysis/models/<sentiment_analysis_model>.onnx'
tts_quantized_model_path = r'./tts/models/<tts_model>.onnx'

ocr = onnx.load(ocr_quantized_model_path)
sentiment_analysis = onnx.load(sentiment_analysis_quantized_model_path)
tts = onnx.load(tts_quantized_model_path)

parser = argparse.ArgumentParser()
parser.add_argument('--ep', type=str, default ='cpu',choices = ['cpu','ipu'], help='EP backend selection')
opt = parser.parse_args()

# Can be changed if we have more sub-models for 
num_procs = 3; 

session = {}
t = {}
# if opt.ep == 'ipu':

providers = ['VitisAIExecutionProvider']
cache_dir = Path(__file__).parent.resolve()
provider_options = [{
            'config_file': 'vaip_config.json',
            'cacheDir': str(cache_dir),
            'cacheKey': 'modelcachekey'
            }]

print('INIT MODELS')
for i in range(0,num_procs): 
    session[i] = ort.InferenceSession(model.SerializeToString(), providers=providers,
                               provider_options=provider_options)

print('INIT THREAD')
for i in range(0,num_procs):
    t[i] = threading.Thread(target=run_model, args=(session[i],))


print('START THREAD')
for i in range(0,num_procs):
    t[i].start()


print('JOIN THREAD')
for i in range(0,num_procs):
    t[i].join()




