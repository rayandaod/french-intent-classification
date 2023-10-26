import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import argparse
import timeit
import torch

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline
from tqdm import tqdm

from src.helper import *

# Get the config
config = parse_config("config.yaml")


def get_en_model_tokenizer_trans(verbose: bool = False):
    if verbose: print('> Loading the model, tokenizer, and translator...')

    model = AutoModelForSequenceClassification.from_pretrained(config['pretrained_english_model_path'])
    tokenizer = AutoTokenizer.from_pretrained(config['pretrained_english_model_path'])
    translator = pipeline("translation", model=config['translator_fr_en_model_path'])

    return model, tokenizer, translator


def predict_en(model: object, tokenizer: object, translator: object, 
            df: pd.DataFrame, verbose: bool = False):
    
    # Copy the dataframe
    df = df.copy()

    # Translate 
    if verbose:
        print('> Translating to english...')
        tqdm.pandas()
        df['text_en'] = df['text'].progress_apply(lambda x: translator(x)[0]['translation_text'])
    else:
        df['text_en'] = df['text'].apply(lambda x: translator(x)[0]['translation_text'])

    # Tokenize
    if verbose:
        print('> Tokenizing the user input...')
        df['input_ids'] = df['text_en'].progress_apply(lambda x: tokenizer(x, return_tensors="pt")['input_ids'])
    else:
        df['input_ids'] = df['text_en'].apply(lambda x: tokenizer(x, return_tensors="pt")['input_ids'])

    # Predict
    if verbose:
        print('> Predicting the intent index...')
        df['intent_idx_pred'] = df['input_ids'].progress_apply(lambda x: torch.argmax(model(x)[0]).item())
    else:
        df['intent_idx_pred'] = df['input_ids'].apply(lambda x: torch.argmax(model(x)[0]).item())

    # Get the class names from the predicted indices
    if verbose:
        print('> Getting the class names...')
        df['intent_label_pred'] = df['intent_idx_pred'].progress_apply(lambda x: model.config.id2label[x])
    else:
        df['intent_label_pred'] = df['intent_idx_pred'].apply(lambda x: model.config.id2label[x])

    # Replace 'oos' with 'out_of_scope'
    if verbose:
        print('> Replacing \'oos\' with \'out_of_scope\'...')
        df['intent_label_pred'] = df['intent_label_pred'].progress_apply(lambda x: 'out_of_scope' if x == 'oos' else x)
    else:
        df['intent_label_pred'] = df['intent_label_pred'].apply(lambda x: 'out_of_scope' if x == 'oos' else x)

    # Since this method uses a model trained on the full CLINC150 dataset,
    # we return 'out_of_scope' if the predicted intent is not in config['classes']
    if verbose:
        print('> Replacing intents not in config[\'classes\'] with \'out_of_scope\'...')
        df['intent_label_pred'] = df['intent_label_pred'].progress_apply(lambda x: 'out_of_scope' if x not in config['classes'] else x)
    else:
        df['intent_label_pred'] = df['intent_label_pred'].apply(lambda x: 'out_of_scope' if x not in config['classes'] else x)

    # Match the idx of the intents not in config['classes'] to the idx of 'out_of_scope'
    if verbose:
        print('> Matching the idx of the intents not in config[\'classes\'] to the idx of \'out_of_scope\'...')
        df['intent_idx_pred'] = df['intent_label_pred'].progress_apply(lambda x: model.config.label2id['oos'] if x == 'out_of_scope' else model.config.label2id[x])
    else:
        df['intent_idx_pred'] = df['intent_label_pred'].apply(lambda x: model.config.label2id['oos'] if x == 'out_of_scope' else model.config.label2id[x])

    return df['intent_idx_pred'], df['intent_label_pred']


def prepare_and_predict(user_input: str, verbose: bool = False) -> (str, float):
    """
    Predicts the intent of the user input.
    """

    # Create a dataframe with the user input
    user_input_df = pd.DataFrame({'text': [user_input]})

    # Get the model and the tokenizer
    model, tokenizer, translator = get_en_model_tokenizer_trans(verbose=verbose)

    # Predict and time
    start = timeit.default_timer()
    _, prediction = predict_en(model=model,
                                tokenizer=tokenizer,
                                translator=translator,
                                df=user_input_df,
                                verbose=verbose)
    stop = timeit.default_timer()
    
    return prediction.iloc[0], stop - start


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--text', type=str, help='The text to predict the intent of.')
    parser.add_argument('--verbose', action='store_true', help='Whether to print logs.')
    args = parser.parse_args()

    prediction, speed = prepare_and_predict(args.text, verbose=args.verbose)
    print('Prediction:', prediction)
    print('Speed:', f'{speed:0.2f}', 'seconds')
