import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pickle
import argparse
import timeit
import numpy as np
import pandas as pd

from src.preprocess import preprocessing_fn_dict, get_ext_models
from src.helper import parse_config
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

        self.config = parse_config(config_path)
        self.verbose = verbose

        # Get the model, the inference data preprocessing function names, and the label encoder
        self.model = pickle.load(open(os.path.join('model_zoo', model_name, 'model.pkl'), 'rb'))
        self.prep_fn_shorts = open(os.path.join('model_zoo', model_name, 'inference_data_prep.txt'), 'r').read().splitlines()
        self.label_enc = pickle.load(open(os.path.join('model_zoo', 'label_encoder.pkl'), 'rb'))

        # Get the pretrained models
        self.ext_models = get_ext_models(prep_fn_shorts=self.prep_fn_shorts, 
                                        config=self.config,
                                        verbose=self.verbose)
    
        return


    def predict(self, df: pd.DataFrame) -> (str, float):
        """
        Predicts the intent of the user input.
        """

        # Preprocess the dataframe
        if 'embedding' not in df.columns:
            for preprocessing_fn_name in self.prep_fn_shorts:
                df = preprocessing_fn_dict[preprocessing_fn_name](df,
                                                                  self.ext_models,
                                                                  self.verbose)

        # Get the embeddings
        X = np.array(df['embedding'].tolist())

        # Predict
        if self.verbose: print('\n> Predicting...')
        intent_idx_pred = self.model.predict(X)
        
        # Get the intent name using the label encoder
        intent_label_pred = self.label_enc.inverse_transform(intent_idx_pred)[0]
        
        return intent_idx_pred, intent_label_pred


if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='logReg_camembert', help='The name of the model folder.')
    parser.add_argument('--text', '-t', type=str, help='The text to predict the intent of.')
    parser.add_argument('--config', '-c', type=str, default='config.yaml', help='The path to the configuration file.')
    parser.add_argument('--verbose', '-v', action='store_true', help='Whether to print logs.')
    args = parser.parse_args()

    # Initialize the predictor
    intent_predictor = IntentPredictor(model_name=args.model,
                                       config_path=args.config, 
                                       verbose=args.verbose)
    
    # Create a dataframe with the user input
    df = pd.DataFrame({'text': [args.text]})

    # Predict and time the prediction
    start = timeit.default_timer()
    prediction, speed = intent_predictor.predict(df)
    total_time = timeit.default_timer() - start

    print('\nPrediction:', prediction)
    print('Total time:', f'{total_time:0.2f}', 'seconds')
