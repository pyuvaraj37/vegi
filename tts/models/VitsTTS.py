import torch
from transformers import VitsTokenizer, VitsModel, set_seed
import soundfile as sf
import qlinear
from utils import Utils
from timeit import default_timer as timer
tokenizer = VitsTokenizer.from_pretrained("facebook/mms-tts-eng")
model = VitsModel.from_pretrained("facebook/mms-tts-eng")
def quantize_model(model):
    return torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8, inplace=True)

quantized_model=quantize_model(model)
node_args = ()
node_kwargs = {'device': 'aie'}
Utils.replace_node(quantized_model, torch.ao.nn.quantized.dynamic.modules.linear.Linear, qlinear.QLinear, node_args, node_kwargs)
inputs = tokenizer(text="Hello I am not, able to, see if there is some inference and also not able to see if the inference speed is quick or not", return_tensors="pt")
# quantized_model.speaking_rate = 1
# quantized_model.noise_scale = 1
set_seed(555)  # make deterministic


start = timer()



with torch.no_grad():
   outputs = quantized_model(**inputs)

cpu_total = timer() - start
print(f"Totoal time:{cpu_total}")
waveform = outputs.waveform[0]

sf.write("synthesized_speech.wav", waveform, samplerate=16000)