import re
import sys
import os
from transformers import DonutProcessor, AutoTokenizer
import torch
import qlinear
from PIL import ImageGrab, Image
from timeit import default_timer as timer
from utils import Utils


def screenGrab(rect):
    """ Given a rectangle, return a PIL Image of that part of the screen.
        Uses ImageGrab on Windows. """
    x, y, width, height = rect
    image = ImageGrab.grab(bbox=(x, y, x + width, y + height))
    return image

# Load the quantized encoder and decoder models
quantized_encoder = torch.load("./torch_q_models/quantized_encoder.pt")
quantized_decoder = torch.load("./torch_q_models/quantized_decoder.pt")

# Replace the quantized linear layers with QLinear in both encoder and decoder
node_args = ()
node_kwargs = {'device': 'aie'}
Utils.replace_node(quantized_decoder, torch.ao.nn.quantized.dynamic.modules.linear.Linear, qlinear.QLinear, node_args, node_kwargs)

# Load processor and tokenizer
processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")
tokenizer = AutoTokenizer.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")


if __name__ == "__main__":
    EXE = sys.argv[0]
    del(sys.argv[0])

    # Catch zero-args
    if len(sys.argv) != 4 or sys.argv[0] in ('--help', '-h', '-?', '/?'):
        sys.stderr.write(EXE + ": monitors section of screen for text\n")
        sys.stderr.write(EXE + ": Give x, y, width, height as arguments\n")
        sys.exit(1)

    # Get the screen coordinates from command line arguments
    x, y, width, height = map(int, sys.argv[:4])
    screen_rect = [x, y, width, height]
    print(EXE + ": watching " + str(screen_rect))

    # Loop to monitor the specified rectangle of the screen
    while True:
        start = timer()
        image = screenGrab(screen_rect)  # Grab the area of the screen
        image_path = "./temp_screengrab.png"
        image.save(image_path)   # Save the image to a temporary file
        
        image = Image.open(image_path).convert('RGB')

        # Prepare decoder inputs
        task_prompt = "<s_synthdog>"
        decoder_input_ids = tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt").input_ids

        # Preprocess the image
        pixel_values = processor(image, return_tensors="pt")
        
        # Perform encoding with debug statements
        with torch.no_grad():
            encoder_outputs = quantized_encoder(**pixel_values)
            # print("Encoder outputs shape:", encoder_outputs.last_hidden_state.shape)

        # Initialize the decoder input for the first time step
        generated_ids = decoder_input_ids

        # Perform decoding step-by-step
        max_length = 16  # Maximum length of the output sequence
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
