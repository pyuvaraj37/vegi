import easyocr
import torch

# Initialize the EasyOCR reader
reader = easyocr.Reader(['en'])  

detection_model = reader.detector
recognition_model = reader.recognizer

# Dynamically quantize the full EasyOCR model
quantized_reader = torch.ao.quantization.quantize_dynamic(
    detection_model, {torch.nn.Linear}, dtype=torch.qint8, inplace=True
)

quantized_recognition_model = torch.ao.quantization.quantize_dynamic(
    recognition_model, {torch.nn.Linear}, dtype=torch.qint8, inplace=True
)

# Save the quantized models separately
torch.save(quantized_reader, './models/quantized_detection_model.pt')
torch.save(quantized_recognition_model, './models/quantized_recognition_model.pt')

print("Quantization complete and models saved.")
