
import torch
import os
from transformers import VitsTokenizer, VitsModel, set_seed
import soundfile as sf
from rvc_python.infer import infer_file

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

        # result = infer_file(
        #     input_path=str(path),
        #     model_path="./models/rvc/aatrox/model.pth",
        #     index_path="./models/rvc/aatrox/model.index",  # Optional: specify path to index file if available
        #     device="cpu:0", # Use cpu or cuda
        #     f0method="harvest",  # Choose between 'harvest', 'crepe', 'rmvpe', 'pm'
        #     f0up_key=0,  # Transpose setting
        #     opt_path="./temp/converted_dialogue.wav",  # Output file path
        #     index_rate=0.5,
        #     filter_radius=3,
        #     resample_sr=0,  # Set to desired sample rate or 0 for no resampling.
        #     rms_mix_rate=0.25,
        #     protect=0.33,
        #     version="v2"
        # )

        return path