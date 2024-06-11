import torch
import easyocr

# Create the OCR reader
ocr_reader = easyocr.Reader(['en'], gpu=False, recognizer=True)

# Create dummy inputs for the image and text
dummy_image = torch.rand(1, 3, 608, 800)  # Your existing dummy image
# Convert the dummy image to grayscale by averaging the channels
dummy_image = dummy_image.mean(1, keepdim=True)

dummy_text = torch.tensor([0])  # Example dummy text input, adjust according to actual requirements

# Define the save path for the ONNX model
onnx_save_path = "./models/recognizer.onnx"

# Export the model to ONNX
with torch.no_grad():
    torch.onnx.export(
        ocr_reader.recognizer,
        (dummy_image, dummy_text),  # Pass both image and text inputs
        onnx_save_path,
        export_params=True,
        do_constant_folding=True,
        opset_version=12,
        input_names=['input_image', 'input_text'],  # Name inputs appropriately
        output_names=['output'],
        dynamic_axes={
            'input_image': {0: 'batch_size', 3: 'width'},  # Dynamic axes for image
            'input_text': {0: 'batch_size'},  # Dynamic axes for text
            'output': {0: 'batch_size'}
        },
        verbose=True
    )
