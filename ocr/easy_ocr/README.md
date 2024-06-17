# Easy OCR on NPU

## Steps to run screen-ocr.py

1. Syntax to run:
"""
python screen-ocr.py x-coordinate y-coordinate width height
"""
## Steps to run easy_inference.py

"""
conda activate ryzenai-transformers
"""

1. Make sure all requirements are installed (could find it in requirements.txt)
Note: easyocr should also be installed using pip.
2. Run setup.bat from \ryzen-ai-sw-1.1\RyzenAI-SW\examples\transformers\setup.bat (Example: C:\Users\mikuv\Desktop\ryzen-ai-sw-1.1\RyzenAI-SW\example\transformers\setup.bat)
3. Perform dynamic quantization by running easy_quantize.py
4. Inference the model by running easy_inference.py  