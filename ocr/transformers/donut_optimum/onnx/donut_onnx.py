import re
import onnxruntime
from transformers import DonutProcessor, AutoTokenizer
from optimum.onnxruntime import ORTModelForVision2Seq
from PIL import Image
from timeit import default_timer as timer

#Config file path
config_file_path = "C:\\Users\\mikuv\\Desktop\\ryzen-ai-sw-1.1\\RyzenAI-SW\\donut_optimum\\archives\\vaip_config.json"
aie_options = onnxruntime.SessionOptions()

#Onnx Model Path
model_path = "./onnx_models"

# #Quantized Model Path
model_path = "./quantized_models"

#start timer
start = timer()

# Load processor, tokenizer, and model
processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")
tokenizer = AutoTokenizer.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")
model = ORTModelForVision2Seq.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2", export=True)


#save as .onnx
#Make sure to enable export=True to save model
save_directory = "./models"
model.save_pretrained(save_directory)
tokenizer.save_pretrained(save_directory)
