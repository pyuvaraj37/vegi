import torch
from transformers import DonutProcessor, VisionEncoderDecoderModel

# Load the processor and model
processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")
model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")

# Extract the encoder and decoder
encoder = model.encoder
decoder = model.decoder

# Quantize the encoder and decoder
quantized_encoder = torch.ao.quantization.quantize_dynamic(
    encoder, {torch.nn.Linear}, dtype=torch.qint8, inplace=True
)
quantized_decoder = torch.ao.quantization.quantize_dynamic(
    decoder, {torch.nn.Linear}, dtype=torch.qint8, inplace=True
)

# Save the quantized models
torch.save(quantized_encoder, "./torch_q_models/quantized_encoder.pt")
torch.save(quantized_decoder, "./torch_q_models/quantized_decoder.pt")

print("Quantization complete and models saved.")
