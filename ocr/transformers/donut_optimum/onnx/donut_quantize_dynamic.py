from onnxruntime.quantization import quantize_dynamic, registry
import os
import shutil

def quantize(src_folder, dst_folder):
    # Quantize a float ONNX model to an int8 ONNX model

    # Conv layer quantization is not supported with dynamic quantization
    all_op = registry.CommonOpsRegistry
    all_op.update(registry.IntegerOpsRegistry)
    all_op.update(registry.QDQRegistry)
    all_op.update(registry.QLinearOpsRegistry)
    all_op_but_conv = {k: v for k, v in all_op.items() if k != "Conv"}

    if dst_folder is None:
        dst_folder = src_folder + "_quant"

    shutil.copytree(src_folder, dst_folder, dirs_exist_ok=True)

    for file in ["encoder_model", "decoder_model", "decoder_with_past_model"]:
        print("Quantizing " + file + "...")
        src = os.path.join(dst_folder, file + '.onnx')
        dst = os.path.join(dst_folder, file + '_quantized.onnx')
        # Skipping quant_pre_process step due to symbolic shape inference issues
        quantize_dynamic(src, dst, op_types_to_quantize=list(all_op_but_conv.keys()))
        print("Done.")

# Define the source folder containing the ONNX models
src_folder = "./models"
dst_folder = "./quantized_models"

# Quantize the models
quantize(src_folder, dst_folder)

print('Quantized models saved in:', dst_folder)
