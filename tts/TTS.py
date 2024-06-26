
import torch
import os
from transformers import VitsTokenizer, VitsModel, set_seed
import soundfile as sf

class TTS:
    def __init__(self, settings):
        print("Intializing TTS Speaker")
        print("Loading models...")
        self.settings = settings
        self.tokenizer = VitsTokenizer.from_pretrained("facebook/mms-tts-eng")
        self.model = VitsModel.from_pretrained("facebook/mms-tts-eng")
        self.model = torch.quantization.quantize_dynamic(self.model, {torch.nn.Linear}, dtype=torch.qint8, inplace=True)
        if (self.settings[0] == 'npu'):
            import qlinear
            from utils import Utils
            node_args = ()
            node_kwargs = {'device': 'aie'}
            Utils.replace_node(self.model, torch.ao.nn.quantized.dynamic.modules.linear.Linear, qlinear.QLinear, node_args, node_kwargs )
        #Add gpu support
        
    def run(self, dialogue, path):
        inputs = self.tokenizer(text=dialogue, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
        waveform = outputs.waveform[0]
        sf.write(path, waveform, samplerate=16000)
        return path