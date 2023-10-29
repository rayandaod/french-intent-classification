import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import argparse
import timeit
import torch
import numpy as np
import pandas as pd

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline
from tqdm import tqdm

from src.helper import parse_config
from src import RANDOM_SEED

np.random.seed(RANDOM_SEED)


class IntentPredictorEnglish():
    """
    Predicts the intent of the user input.
    Loads the model, the label encoder, and any external models needed for inference.
    """

    def __init__(self, config_path: str, verbose: bool = False) -> None:
        """
        Constructor.
        """
        
        # Parse the config file
        self.config = parse_config(config_path)

        # Set the verbose attribute
        self.verbose = verbose

        # Load the model, the tokenizer, and the translator
        if self.verbose: print('> Loading the model, tokenizer, and translator...')
        self.translator = pipeline("translation", model=self.config['translator_fr_en_model_path'])
        self.model = AutoModelForSequenceClassification.from_pretrained(self.config['pretrained_english_model_path'])
        self.tokenizer = AutoTokenizer.from_pretrained(self.config['pretrained_english_model_path'])
    
        return


    def __call__(self, df: pd.DataFrame) -> (pd.Series, pd.Series):
        """
        Predicts the intent of the entries in the dataframe using the "english pipeline"
        (translator -> tokenizer -> pre-trained model)
        """

        # Copy the dataframe
        df = df.copy()

        # Translate
        translate = lambda x: self.translator(x)[0]['translation_text']
        if self.verbose:
            print('\n> Translating to english...')
            tqdm.pandas()
            df['text_en'] = df['text'].progress_apply(translate)
        else:
            df['text_en'] = df['text'].apply(translate)

        # Tokenize
        tokenize = lambda x: self.tokenizer(x, return_tensors="pt")['input_ids']
        if self.verbose:
            print('\n> Tokenizing the user input...')
            df['input_ids'] = df['text_en'].progress_apply(tokenize)
        else:
            df['input_ids'] = df['text_en'].apply(tokenize)

        # Predict
        predict = lambda x: torch.argmax(self.model(x)[0]).item()
        if self.verbose:
            print('\n> Predicting the intent index...')
            df['intent_idx_pred'] = df['input_ids'].progress_apply(predict)
        else:
            df['intent_idx_pred'] = df['input_ids'].apply(predict)

        # Get the class names from the predicted indices
        get_class_names_from_idx = lambda x: self.model.config.id2label[x]
        if self.verbose:
            print('\n> Getting the class names...')
            df['intent_label_pred'] = df['intent_idx_pred'].progress_apply(get_class_names_from_idx)
        else:
            df['intent_label_pred'] = df['intent_idx_pred'].apply(get_class_names_from_idx)

        # Replace 'oos' with 'out_of_scope'
        replace_oos = lambda x: 'out_of_scope' if x == 'oos' else x
        if self.verbose:
            print('\n> Replacing \'oos\' with \'out_of_scope\'...')
            df['intent_label_pred'] = df['intent_label_pred'].progress_apply(replace_oos)
        else:
            df['intent_label_pred'] = df['intent_label_pred'].apply(replace_oos)

        # Since this method uses a model trained on the full CLINC150 dataset,
        # we return 'out_of_scope' if the predicted intent is not in config['classes']
        map_labels_to_oos = lambda x: 'out_of_scope' if x not in self.config['classes'] else x
        if self.verbose:
            print('\n> Replacing intents not in config[\'classes\'] with \'out_of_scope\'...')
            df['intent_label_pred'] = df['intent_label_pred'].progress_apply(map_labels_to_oos)
        else:
            df['intent_label_pred'] = df['intent_label_pred'].apply(map_labels_to_oos)

        # Match the idx of the intents not in config['classes'] to the idx of 'out_of_scope'
        map_idx_to_oos_idx = lambda x: self.model.config.label2id['oos'] if x == 'out_of_scope' else self.model.config.label2id[x]
        if self.verbose:
            print('\n> Matching the idx of the intents not in config[\'classes\'] to the idx of \'out_of_scope\'...')
            df['intent_idx_pred'] = df['intent_label_pred'].progress_apply(map_idx_to_oos_idx)
        else:
            df['intent_idx_pred'] = df['intent_label_pred'].apply(map_idx_to_oos_idx)

        return df['intent_idx_pred'], df['intent_label_pred']


if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--text', '-t', type=str, help='The text to predict the intent of.')
    parser.add_argument('--config', '-c', type=str, default='config.yaml', help='The path to the config file.')
    parser.add_argument('--verbose', '-v', action='store_true', help='Whether to print logs.')
    args = parser.parse_args()

    intent_predictor_english = IntentPredictorEnglish(config_path=args.config,
                                                      verbose=args.verbose)
    
    # Create a dataframe with the user input
    user_input_df = pd.DataFrame({'text': [args.text]})

    # Predict and time
    start = timeit.default_timer()
    idx_pred, label_pred = intent_predictor_english(user_input_df)
    total_time = timeit.default_timer() - start
    
    print('\nPrediction:', label_pred.iloc[0])
    print('Speed:', f'{total_time:0.2f}', 'seconds')
