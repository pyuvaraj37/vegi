from transformers import AutoProcessor, BarkModel
import scipy
import qlinear
import torch
from utils import Utils
from timeit import default_timer as timer
# from optimum.bettertransformer import BetterTransformer

processor = AutoProcessor.from_pretrained("suno/bark")
model = BarkModel.from_pretrained("suno/bark-small")
print(model)
# model = BetterTransformer.transform(model, keep_original_model=False)
# model.enable_cpu_offload()
def quantize_model(model):
    return torch.quantization.quantize_dynamic(model, {torch.nn.Linear,torch.nn.LSTM}, dtype=torch.qint8, inplace=True)

quantized_model=quantize_model(model)
node_args = ()
node_kwargs = {'device': 'aie'}
Utils.replace_node(quantized_model, torch.ao.nn.quantized.dynamic.modules.linear.Linear, qlinear.QLinear, node_args, node_kwargs)
# print(quantized_model)



voice_preset = "v2/en_speaker_7"

inputs = processor("[clears throat] i am doing very well aryan [laughs]", voice_preset=voice_preset)
inputs['attention_mask'] = inputs['attention_mask']  # Ensure attention mask is included
# inputs['pad_token_id'] = processor.pad_token_id 


attention_mask = inputs["attention_mask"]
start=timer()
audio_array = quantized_model.generate(
  input_ids=inputs["input_ids"], 
  attention_mask=attention_mask,
  
  # add pad token id here
  pad_token_id=processor.tokenizer.pad_token_id
)
cpu_total=timer()-start
# audio_array = quantized_model.generate(**inputs)
audio_array = audio_array.cpu().numpy().squeeze()
print(cpu_total)


sample_rate = quantized_model.generation_config.sample_rate
scipy.io.wavfile.write("bark_out.wav", rate=sample_rate, data=audio_array)

#######END OF BARK QUANTIZE CODE

# device = torch.device('gpu' if torch.npu.is_available() else 'cpu')
# quantized_model.to(device)

# # Prepare input and generate audio
# voice_preset = "v2/en_speaker_7"
# inputs = processor("I am doing very well Aryan", voice_preset=voice_preset)

# # Move inputs to NPU
# inputs = {k: v.to(device) for k, v in inputs.items()}

# # Generate audio
# with torch.no_grad():
#     audio_array = quantized_model.generate(**inputs)

# # Move the result back to CPU
# audio_array = audio_array.cpu().numpy().squeeze()

# # Save the audio output
# sample_rate = quantized_model.generation_config.sample_rate
# scipy.io.wavfile.write("bark_out.wav", rate=sample_rate, data=audio_array)


# import torch
# from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech
# from utils import Utils
# import qlinear
# # Load the processor and model
# processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
# model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")

# # Quantize the model
# def quantize_model(model):
#     return torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8, inplace=False)

# quantized_model = quantize_model(model)


# node_args = ()
# node_kwargs = {'device': 'aie'}
# Utils.replace_node(quantized_model, torch.ao.nn.quantized.dynamic.modules.linear.Linear, qlinear.QLinear, node_args, node_kwargs)



# import torch
# import numpy as np
# import soundfile as sf
# from timeit import default_timer as timer
# from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech,SpeechT5HifiGan
# from datasets import load_dataset
# # Load the processor and the quantized model
# processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
# # quantized_model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
# # quantized_model.load_state_dict(torch.load('./quantized_speecht5_model.pth'))
# # model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
# # quantized_model.eval()
# vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
# # Define the input text
# input_text = "I am a student at u c r"
# embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
# speaker_embeddings = torch.tensor(embeddings_dataset[6005]["xvector"]).unsqueeze(0)
# # Tokenize the input text
# inputs = processor(text=input_text, return_tensors="pt")

# # Run inference
# with torch.no_grad():
#     start = timer()
#     speech = quantized_model.generate_speech(inputs["input_ids"],speaker_embeddings=speaker_embeddings,vocoder=vocoder)
#     cpu_total = timer() - start

# print(f"Total inference time on CPU: {cpu_total:.6f} seconds")

# # Save the generated speech
# sf.write("output_audio.wav", speech.squeeze().numpy(), samplerate=16000)
