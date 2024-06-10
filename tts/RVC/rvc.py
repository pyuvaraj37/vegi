# from pathlib import Path

# from dotenv import load_dotenv
# from scipy.io import wavfile

# from rvc.modules.vc.modules import VC


# def main():
#       vc = VC()
#       vc.get_vc("{G32k.pth}")
#       tgt_sr, audio_opt, times, _ = vc.vc_inference(
#             1, Path("{synthesized_speech.wav}")
#       )
#       wavfile.write("{rvc.wav}", tgt_sr, audio_opt)


# if __name__ == "__main__":
#       load_dotenv("{onnx1}")
#       main()

from rvc_python.infer import infer_file, infer_files

# To process a single file:
result = infer_file(
    input_path="bark_out.wav",
    model_path="model.pth",
    index_path="",  # Optional: specify path to index file if available
    device="cpu", # Use cpu or cuda
    f0method="harvest",  # Choose between 'harvest', 'crepe', 'rmvpe', 'pm'
    index_rate=0.5,
    filter_radius=3,
    resample_sr=0,  # Set to desired sample rate or 0 for no resampling.
    rms_mix_rate=0.25,
    protect=0.33,
    version="v2"
)

print("Inference completed. Output saved to:", result)
