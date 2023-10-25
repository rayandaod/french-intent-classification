import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline

from src.helper import *

# Get the config
config = parse_config("config.yaml")

# Load translator (french to english)
translator = pipeline("translation", model=config['translator_fr_en_model_path'])

# Load pre-trained model (fine-tuned on original CLINC150, i.e in english)
model = AutoModelForSequenceClassification.from_pretrained(config['pretrained_english_model_path'])
tokenizer = AutoTokenizer.from_pretrained(config['pretrained_english_model_path'])


def predict(user_input: str):
    # Get the translated sentence
    translated_user_input = translator(user_input)[0]['translation_text']
    if config['verbose']: print(translated_user_input)

    # Tokenize the input
    inputs = tokenizer(translated_user_input, return_tensors="pt")

    # Get the predicted intent index
    outputs = model(**inputs)
    logits = outputs.logits
    intent_index = logits.argmax().item()

    # Get the intent label from the index
    intent_label = model.config.id2label[intent_index]

    # Replace 'oos' with 'out_of_scope'
    if intent_label == 'oos':
        intent_label = 'out_of_scope'

    # Since this method uses a model trained on the full CLINC150 dataset,
    # we return 'out_of_scope' if the predicted intent is not in config['classes']
    if intent_label not in config['classes']:
        intent_label = 'out_of_scope'

    return intent_label


if __name__ == '__main__':
    print('Prediction:', predict(sys.argv[1]))
