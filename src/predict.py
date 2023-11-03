import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pickle
import argparse
import timeit
import numpy as np
import pandas as pd
import torch

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline

from src.preprocess import DataPreprocessor
from src.helper import enhanced_apply, parse_config
from src import RANDOM_SEED

np.random.seed(RANDOM_SEED)


class IntentPredictor():
    """
    Predicts the intent of the user input.
    Loads the model, the label encoder, and any external models needed for inference.
    """

    def __init__(self,
                 model_name: str,
                 config_path: str,
                 verbose: bool = False) -> None:
        """
        Constructor.
        """
        
        # Set the verbose attribute
        self.verbose = verbose

        # Get the model, the inference data preprocessing function names, and the label encoder
        self.model = pickle.load(open(os.path.join('model_zoo', model_name, 'model.pkl'), 'rb'))
        self.prep_fn_shorts = open(os.path.join('model_zoo', model_name, 'inference_data_prep.txt'), 'r').read().splitlines()
        self.label_enc = pickle.load(open(os.path.join('model_zoo', 'label_encoder.pkl'), 'rb'))

        # Initialise the data preprocessor
        self.data_preprocessor = DataPreprocessor(prep_fn_shorts=self.prep_fn_shorts,
                                                  config_path=config_path,
                                                  verbose=self.verbose)
    
        return


    def __call__(self, df: pd.DataFrame) -> (pd.Series, pd.Series):
        """
        Predicts the intent of the user input.
        """

        # Preprocess the dataframe
        if 'embedding' not in df.columns:
            df = self.data_preprocessor(df)

        # Get the embeddings
        X = np.array(df['embedding'].tolist())

        # Predict
        if self.verbose: print('\n> Predicting...')
        intent_idx_pred = self.model.predict(X)
        
        # Get the intent name using the label encoder
        intent_label_pred = self.label_enc.inverse_transform(intent_idx_pred)
        
        return intent_idx_pred, intent_label_pred


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
        if self.verbose: print('\n> Translating to english...')
        df['text_en'] = enhanced_apply(translate, df['text'], verbose=self.verbose)

        # Tokenize
        tokenize = lambda x: self.tokenizer(x, return_tensors="pt")['input_ids']
        if self.verbose: print('\n> Tokenizing the user input...')
        df['input_ids'] = enhanced_apply(tokenize, df['text_en'], verbose=self.verbose)

        # Predict
        predict = lambda x: torch.argmax(self.model(x)[0]).item()
        if self.verbose: print('\n> Predicting the intent index...')
        df['intent_idx_pred'] = enhanced_apply(predict, df['input_ids'], verbose=self.verbose)

        # Get the class names from the predicted indices
        get_class_names_from_idx = lambda x: self.model.config.id2label[x]
        if self.verbose: print('\n> Getting the class names...')
        df['intent_label_pred'] = enhanced_apply(get_class_names_from_idx, df['intent_idx_pred'], verbose=self.verbose)

        # Replace 'oos' with 'out_of_scope'
        replace_oos = lambda x: 'out_of_scope' if x == 'oos' else x
        if self.verbose: print('\n> Replacing \'oos\' with \'out_of_scope\'...')
        df['intent_label_pred'] = enhanced_apply(replace_oos, df['intent_label_pred'], verbose=self.verbose)

        # Since this method uses a model trained on the full CLINC150 dataset,
        # we return 'out_of_scope' if the predicted intent is not in config['classes']
        map_labels_to_oos = lambda x: 'out_of_scope' if x not in self.config['classes'] else x
        if self.verbose: print('\n> Replacing intents not in config[\'classes\'] with \'out_of_scope\'...')
        df['intent_label_pred'] = enhanced_apply(map_labels_to_oos, df['intent_label_pred'], verbose=self.verbose)

        # Match the idx of the intents not in config['classes'] to the idx of 'out_of_scope'
        map_idx_to_oos_idx = lambda x: self.model.config.label2id['oos'] if x == 'out_of_scope' else self.model.config.label2id[x]
        if self.verbose: print('\n> Matching the idx of the intents not in config[\'classes\'] to the idx of \'out_of_scope\'...')
        df['intent_idx_pred'] = enhanced_apply(map_idx_to_oos_idx, df['intent_label_pred'], verbose=self.verbose)

        return df['intent_idx_pred'], df['intent_label_pred']


if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='logReg_camembert', help='The name of the model folder.')
    parser.add_argument('--text', '-t', type=str, help='The text to predict the intent of.')
    parser.add_argument('--config', '-c', type=str, default='config.yaml', help='The path to the configuration file.')
    parser.add_argument('--verbose', '-v', action='store_true', help='Whether to print logs.')
    args = parser.parse_args()

    if args.model != 'english':
        # Initialize the predictor
        intent_predictor = IntentPredictor(model_name=args.model,
                                        config_path=args.config, 
                                        verbose=args.verbose)
        
    if args.model == 'english':
        # Initialize the predictor
        intent_predictor = IntentPredictorEnglish(config_path=args.config,
                                                  verbose=args.verbose)
    
    # Create a dataframe with the user input
    df = pd.DataFrame({'text': [args.text]})

    # Predict and time the prediction
    start = timeit.default_timer()
    _, label_pred = intent_predictor(df)
    total_time = timeit.default_timer() - start

    print('\nPrediction:', label_pred[0])
    print('Speed:', f'{total_time:0.2f}', 'seconds')
