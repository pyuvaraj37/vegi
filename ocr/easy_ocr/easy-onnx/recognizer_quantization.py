import vai_q_onnx
from onnxruntime.quantization import QuantFormat, QuantType

# `input_model_path` is the path to the original, unquantized ONNX model.
input_model_path = "./models/recognizer.onnx"

# `output_model_path` is the path where the quantized model will be saved.
output_model_path = "./models/recognizer_quantized.onnx"

vai_q_onnx.quantize_static(
    input_model_path,
    output_model_path,
    calibration_data_reader=None,
    quant_format=vai_q_onnx.QuantFormat.QDQ,
    calibrate_method=vai_q_onnx.PowerOfTwoMethod.MinMSE,
    activation_type=vai_q_onnx.QuantType.QInt8,
    weight_type=vai_q_onnx.QuantType.QInt8,
    enable_ipu_cnn=True,
    # enable_ipu_transformer=True,
    enable_dpu=True,
    extra_options={'ActivationSymmetric': True}
)

print('Calibrated and quantized model saved at:', output_model_path)