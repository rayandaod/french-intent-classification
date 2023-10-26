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


def get_en_model_tokenizer_trans(config: dict, verbose: bool = False):
    if verbose: print('> Loading the model, tokenizer, and translator...')

    model = AutoModelForSequenceClassification.from_pretrained(config['pretrained_english_model_path'])
    tokenizer = AutoTokenizer.from_pretrained(config['pretrained_english_model_path'])
    translator = pipeline("translation", model=config['translator_fr_en_model_path'])

    return model, tokenizer, translator


def predict_en(model: object, tokenizer: object, translator: object, 
            df: pd.DataFrame, config: dict, verbose: bool = False):
    
    # Copy the dataframe
    df = df.copy()

    # Translate
    translate = lambda x: translator(x)[0]['translation_text']
    if verbose:
        print('\n> Translating to english...')
        tqdm.pandas()
        df['text_en'] = df['text'].progress_apply(translate)
    else:
        df['text_en'] = df['text'].apply(translate)

    # Tokenize
    tokenize = lambda x: tokenizer(x, return_tensors="pt")['input_ids']
    if verbose:
        print('\n> Tokenizing the user input...')
        df['input_ids'] = df['text_en'].progress_apply(tokenize)
    else:
        df['input_ids'] = df['text_en'].apply(tokenize)

    # Predict
    predict = lambda x: torch.argmax(model(x)[0]).item()
    if verbose:
        print('\n> Predicting the intent index...')
        df['intent_idx_pred'] = df['input_ids'].progress_apply(predict)
    else:
        df['intent_idx_pred'] = df['input_ids'].apply(predict)

    # Get the class names from the predicted indices
    get_class_names_from_idx = lambda x: model.config.id2label[x]
    if verbose:
        print('\n> Getting the class names...')
        df['intent_label_pred'] = df['intent_idx_pred'].progress_apply(get_class_names_from_idx)
    else:
        df['intent_label_pred'] = df['intent_idx_pred'].apply(get_class_names_from_idx)

    # Replace 'oos' with 'out_of_scope'
    replace_oos = lambda x: 'out_of_scope' if x == 'oos' else x
    if verbose:
        print('\n> Replacing \'oos\' with \'out_of_scope\'...')
        df['intent_label_pred'] = df['intent_label_pred'].progress_apply(replace_oos)
    else:
        df['intent_label_pred'] = df['intent_label_pred'].apply(replace_oos)

    # Since this method uses a model trained on the full CLINC150 dataset,
    # we return 'out_of_scope' if the predicted intent is not in config['classes']
    map_labels_to_oos = lambda x: 'out_of_scope' if x not in config['classes'] else x
    if verbose:
        print('\n> Replacing intents not in config[\'classes\'] with \'out_of_scope\'...')
        df['intent_label_pred'] = df['intent_label_pred'].progress_apply(map_labels_to_oos)
    else:
        df['intent_label_pred'] = df['intent_label_pred'].apply(map_labels_to_oos)

    # Match the idx of the intents not in config['classes'] to the idx of 'out_of_scope'
    map_idx_to_oos_idx = lambda x: model.config.label2id['oos'] if x == 'out_of_scope' else model.config.label2id[x]
    if verbose:
        print('\n> Matching the idx of the intents not in config[\'classes\'] to the idx of \'out_of_scope\'...')
        df['intent_idx_pred'] = df['intent_label_pred'].progress_apply(map_idx_to_oos_idx)
    else:
        df['intent_idx_pred'] = df['intent_label_pred'].apply(map_idx_to_oos_idx)

    return df['intent_idx_pred'], df['intent_label_pred']


def prepare_and_predict(user_input: str, config:dict, verbose: bool = False) -> (str, float):
    """
    Predicts the intent of the user input.
    """

    # Create a dataframe with the user input
    user_input_df = pd.DataFrame({'text': [user_input]})

    # Get the model and the tokenizer
    model, tokenizer, translator = get_en_model_tokenizer_trans(config=config, verbose=verbose)

    # Predict and time
    start = timeit.default_timer()
    _, prediction = predict_en(model=model,
                                tokenizer=tokenizer,
                                translator=translator,
                                df=user_input_df,
                                config=config,
                                verbose=verbose)
    stop = timeit.default_timer()
    
    return prediction.iloc[0], stop - start


if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--text', '-t', type=str, help='The text to predict the intent of.')
    parser.add_argument('--verbose', '-v', action='store_true', help='Whether to print logs.')
    args = parser.parse_args()

    # Get the config
    config = parse_config("config.yaml")

    # Predict
    prediction, speed = prepare_and_predict(user_input=args.text, config=config, verbose=args.verbose)
    
    print('\nPrediction:', prediction)
    print('Speed:', f'{speed:0.2f}', 'seconds')
