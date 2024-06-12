import re
import onnxruntime
from transformers import DonutProcessor, AutoTokenizer
from optimum.onnxruntime import ORTModelForVision2Seq
from PIL import Image
from timeit import default_timer as timer

#Config file path
config_file_path = "./vaip_config.json"
# config_file_path = "./vaip_config_transformers.json"
aie_options = onnxruntime.SessionOptions()

# Onnx Model Path
model_path = "./models"

# Quantized Model Path
model_path = "./quantized_models"

#start timer
start = timer()

# Load processor, tokenizer, and model
processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")
tokenizer = AutoTokenizer.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")

#CPU Run
# model = ORTModelForVision2Seq.from_pretrained(model_path, export=False, session_options = aie_options)

#Un comment the following to run on IPU
model = ORTModelForVision2Seq.from_pretrained(model_path, 
                                              export=False, 
                                              provider="VitisAIExecutionProvider",  
                                              session_options = aie_options,
                                              provider_options = {'config_file': config_file_path}       
                                              )


for i in range(1,6):
    #Start Timer:
    start = timer()

    # Load the image from a local file
    image_path = ".\\images\\test_image_"+str(i)+".png"  # Update this to the path of your image
    image = Image.open(image_path).convert('RGB')

    # Prepare decoder inputs
    task_prompt = "<s_synthdog>"
    decoder_input_ids = tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt").input_ids

    # Preprocess the image
    pixel_values = processor(image, return_tensors="pt").pixel_values

    # Generate output using ORTModel (Assuming use_cache and other parameters are supported by ORTModelForVision2Seq)
    outputs = model.generate(
        pixel_values=pixel_values,
        decoder_input_ids=decoder_input_ids,
        # max_length=model.decoder.config.max_position_embeddings,
        max_length=100,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=True,
        bad_words_ids=[[tokenizer.unk_token_id]],
        return_dict_in_generate=True,
    )

    # Decode the output
    # sequence = tokenizer.batch_decode(outputs.sequences)[0]
    # sequence = sequence.replace(tokenizer.eos_token, "").replace(tokenizer.pad_token, "")
    # sequence = re.sub(r"<.*?>", "", sequence, count=1).strip()  # remove first task start token

    # # Convert the cleaned sequence to JSON, but only extract the 'text_sequence' field
    # cleaned_sequence = processor.token2json(sequence)['text_sequence']

    # # Remove any remaining unwanted tokens or HTML-like tags from the JSON output
    # cleaned_sequence = re.sub(r"</?.*?>", "", cleaned_sequence).strip()

    # # Print the final cleaned text
    # print(cleaned_sequence)
    # print(f" Total Time for Quantized ONNX-CPU: {timer() - start}")
    sequence = processor.batch_decode(outputs.sequences)[0]
    sequence = sequence.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
    sequence = re.sub(r"<.*?>", "", sequence, count=1).strip()  # remove first task start token
    print(processor.token2json(sequence))
    print(f"Onnx-CPU Total Time: {timer() - start}")