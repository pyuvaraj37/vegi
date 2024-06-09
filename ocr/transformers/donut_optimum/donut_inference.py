import re
from transformers import DonutProcessor, AutoTokenizer
from PIL import Image
import torch
import qlinear 
from utils import Utils
from timeit import default_timer as timer


# Load the quantized encoder and decoder models
quantized_encoder = torch.load("./torch_q_models/quantized_encoder.pt")
quantized_decoder = torch.load("./torch_q_models/quantized_decoder.pt")
# quantized_decoder = torch.load("./torch_q_models/transformed_quantized_decoder.pt")
# decoder_state_dict = torch.load_state_dict("./torch_q_models/transformed_quantized_decoder_state_dict.pt")
# quantized_decoder.load_state_dict(decoder_state_dict)

# Replace the quantized linear layers with QLinear in both encoder and decoder
node_args = ()
node_kwargs = {'device': 'aie'}

# Utils.replace_node(quantized_encoder, torch.ao.nn.quantized.dynamic.modules.linear.Linear, qlinear.QLinear, node_args, node_kwargs)
#Was unsuccessfull as the following error was taking place:
# File "C:\Users\mikuv\miniconda3\envs\ryzenai-transformers\lib\site-packages\transformers\models\donut\modeling_donut_swin.py", line 403, in forward
#attention_scores = attention_scores + relative_position_bias.unsqueeze(0)
#RuntimeError: The size of tensor a (768) must match the size of tensor b (100) at non-singleton dimension 3 


Utils.replace_node(quantized_decoder, torch.ao.nn.quantized.dynamic.modules.linear.Linear, qlinear.QLinear, node_args, node_kwargs)
# print(quantized_encoder)
print(quantized_decoder)

# Load processor and tokenizer
processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")
tokenizer = AutoTokenizer.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")

for i in range(1, 6):
    #Start Timer:
    start = timer()

    # Load the image from a local file
    image_path = ".\\images\\test_image_"+str(i)+".png"  # Update this to the path of your image
    image = Image.open(image_path).convert('RGB')

    # Prepare decoder inputs
    task_prompt = "<s_synthdog>"
    decoder_input_ids = tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt").input_ids

    # Preprocess the image
    pixel_values = processor(image, return_tensors="pt")
    # print(pixel_values.shape)

    # Perform encoding with debug statements
    with torch.no_grad():
        encoder_outputs = quantized_encoder(**pixel_values)
        # print("Encoder outputs shape:", encoder_outputs.last_hidden_state.shape)

    # Initialize the decoder input for the first time step
    generated_ids = decoder_input_ids

    # Perform decoding step-by-step
    max_length = 512  # Maximum length of the output sequence
    for _ in range(max_length):
        # print(i)
        with torch.no_grad():
            outputs = quantized_decoder(input_ids=generated_ids, encoder_hidden_states=encoder_outputs.last_hidden_state)
            # print("Decoder outputs shape:", outputs.logits.shape)
        next_token_logits = outputs.logits[:, -1, :]
        next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        generated_ids = torch.cat([generated_ids, next_token_id], dim=-1)
        if next_token_id.item() == tokenizer.eos_token_id:
            break

    # Decode the generated token IDs to get the output text
    sequence = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    sequence = re.sub(r"<.*?>", "", sequence, count=1).strip()  # Remove first task start token

    # Convert the sequence to JSON
    output_json = processor.token2json(sequence)
    print(output_json)
    #End Timer
    # print(f"CPU Total Time: {timer() - start}")
    print(f"NPU Total Time: {timer() - start}")
