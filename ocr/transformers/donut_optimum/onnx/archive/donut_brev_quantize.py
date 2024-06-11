from optimum_amd.optimum.amd import BrevitasQuantizationConfig, BrevitasQuantizer
from transformers import AutoTokenizer
from optimum_amd.optimum.amd.brevitas.export import onnx_export_from_quantized_model

# Prepare the quantizer, specifying its configuration and loading the model.
# qconfig = BrevitasQuantizationConfig(
#     is_static=False,
#     apply_gptq=False,
#     apply_weight_equalization=False,
#     activations_equalization=False,
#     weights_symmetric=True,
#     activations_symmetric=False,
# )
qconfig = BrevitasQuantizationConfig(activations_equalization=None)

quantizer = BrevitasQuantizer.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")
print("Model retrived")

model = quantizer.quantize(qconfig)

save_path = "./brevitas_onnx"
onnx_export_from_quantized_model(model, save_path)