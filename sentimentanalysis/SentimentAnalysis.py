import os
import torch
from transformers import AutoTokenizer

# sentiment_analysis_quantized_model_path = './model_onnx/'
# #sentiment_analysis = onnx.load(sentiment_analysis_quantized_model_path)

# provider = "VitisAIExecutionProvider"
# # cache_dir = Path(__file__).parent.resolve()
# provider_options = {'config_file': 'vaip_config.json'}
# sess_options = ort.SessionOptions()
# # if args.num_inter_op_threads:
# #     sess_options.execution_mode  = ort.ExecutionMode.ORT_PARALLEL        
# sess_options.inter_op_num_threads = 1
# sess_options.add_session_config_entry("session.intra_op.allow_spinning", "0")
# # sess_options.add_session_config_entry("session.inter_op.allow_spinning", args.inter_op_spinning)

# session = AutoModelForSequenceClassification.from_pretrained(sentiment_analysis_quantized_model_path, provider=provider,use_cache=False, use_io_binding=False, session_options=sess_options, provider_options=provider_options)

# model = torch.load("./model_onnx_quantized/model.pt")

# sentence = "I am feeling sad today"
# tokenizer = AutoTokenizer.from_pretrained('./model')
# inputs = tokenizer(sentence, return_tensors="pt", max_length=128)
# #print(inputs['attention_mask'])
# print(model)
# model.eval()
# # outputs = model.generate(
# #     {'input_ids': inputs['input_ids'].numpy(), 
# #      'attention_mask' : inputs['attention_mask'].numpy(), 
# #      'token_type_ids' : inputs['token_type_ids'].numpy()}, max_new_tokens=30)
# node_args = ()
# quant_mode = 1
# node_kwargs = {'device': 'aie'}
# Utils.replace_node(    model, 
#                         torch.ao.nn.quantized.dynamic.modules.linear.Linear,
#                         qlinear.QLinear, 
#                         node_args, node_kwargs )
# print(model)
# with torch.no_grad():
#     outputs = model(**inputs)


# #print(outputs)

# logits = outputs.logits
# probabilities = torch.nn.functional.softmax(logits, dim=-1)
# predicted_class_index = probabilities.argmax().item()


# sentiment_labels = ['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise']


# print(f"Predicted sentiment: {sentiment_labels[predicted_class_index]}")
# print('fin')


class SentimentAnalysis:
    def __init__(self, settings):
        print("Intializing Sentiment Analysis based Text Analyzer")
        print("Loading models...")
        self.tokenizer = AutoTokenizer.from_pretrained("./models/sa")
        self.model = torch.load("./models/sa/quantized_sa_model.pt")
        self.settings = settings
        if (self.settings[0] == 'npu'):
            import qlinear 
            from utils import Utils
            node_args = ()
            node_kwargs = {'device': 'aie'}
            Utils.replace_node(self.model, torch.ao.nn.quantized.dynamic.modules.linear.Linear, qlinear.QLinear, node_args, node_kwargs )
        #Add gpu support
        self.sentiment_labels = ['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise']

    def run(self, dialogue):
        inputs = self.tokenizer(dialogue, return_tensors="pt", padding="max_length", truncation=True, max_length=128)
        with torch.no_grad():
            outputs = self.model(**inputs)
        logits = outputs.logits
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        predicted_class_index = probabilities.argmax().item()
        print(f"Predicted sentiment: {self.sentiment_labels[predicted_class_index]}")
        return self.sentiment_labels[predicted_class_index]