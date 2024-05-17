import numpy as np
import torch 
#from datasets import load_dataset, load_metric
from transformers import (AutoTokenizer, 
                          AutoModelForSequenceClassification, 
                          TrainingArguments, 
                          Trainer)


model_name = './model'
tokenizer = AutoTokenizer.from_pretrained(model_name)
device = "cpu"
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=6, local_files_only=True)
model = model.to(device)


def predict_sentiment(sentence):

    inputs = tokenizer(sentence, return_tensors="pt", padding="max_length", truncation=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    
    print(outputs)
    logits = outputs.logits
    print(logits)
    probabilities = torch.nn.functional.softmax(logits, dim=-1)
    print(probabilities)
    predicted_class_index = probabilities.argmax().item()
    sentiment_labels = ['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise']

    return sentiment_labels[predicted_class_index]

user_sentence = "I am feeling sad today"
sentiment = predict_sentiment(user_sentence)
print(f"Predicted sentiment: {sentiment}")