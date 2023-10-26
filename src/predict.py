import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pickle
import argparse
import timeit
import numpy as np

from src.preprocess import preprocessing_fn_dict, get_ext_models
from src.helper import *


def get_model_and_prep_fn_shorts(model_name: str):
    """
    Get the model and the function names for inference data preprocessing.
    """

    # Load the model and the inference data preprocessing function names
    model = pickle.load(open(os.path.join('model_zoo', model_name, 'model.pkl'), 'rb'))
    preprocessing_fn_names = open(os.path.join('model_zoo', model_name, 'inference_data_prep.txt'), 'r').read().splitlines()

    return model, preprocessing_fn_names


def predict(model: object, label_enc: object, prep_fn_shorts: list[str],
            df: pd.DataFrame, prep_dict: dict, verbose: bool = False):
    """
    Predicts the intent of the entries in the dataframe using the loaded model and the label encoder.
    Preprocesses the data using the preprocessing functions if needed.
    """

    # Preprocess the user input
    if 'embedding' not in df.columns:
        for preprocessing_fn_name in prep_fn_shorts:
            df = preprocessing_fn_dict[preprocessing_fn_name](df=df,
                                                              ext_models=prep_dict,
                                                              verbose=verbose)

    # Get the embeddings
    X = np.array(df['embedding'].tolist())

    # Predict
    if verbose: print('\n> Predicting...')
    intent_idx_pred = model.predict(X)
    
    # Get the intent name using the label encoder
    intent_class_name = label_enc.inverse_transform(intent_idx_pred)[0]

    return intent_idx_pred, intent_class_name


def prepare_and_predict(user_input: str, model_name: str, config: dict, verbose: bool = False):
    """
    Predicts the intent of the user input.
    Loads the model, the label encoder, and any external models needed for inference.
    """

    # Create a dataframe with the user input
    user_input_df = pd.DataFrame({'text': [user_input]})

    # Get the model, the inference data preprocessing function names, and the label encoder
    model, prep_fn_shorts = get_model_and_prep_fn_shorts(model_name)
    label_enc = pickle.load(open(os.path.join('model_zoo', 'label_encoder.pkl'), 'rb'))

    # Get the pretrained models
    prep_dict = get_ext_models(prep_fn_shorts=prep_fn_shorts, 
                                            config=config,
                                            verbose=verbose)

    # Predict and time
    start = timeit.default_timer()
    _, prediction = predict(model=model,
                            label_enc=label_enc,
                            prep_fn_shorts=prep_fn_shorts,
                            df=user_input_df,
                            prep_dict=prep_dict,
                            verbose=verbose)
    stop = timeit.default_timer()
    
    return prediction, stop - start


if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default=None, help='The name of the model folder.')
    parser.add_argument('--text', '-t', type=str, help='The text to predict the intent of.')
    parser.add_argument('--verbose', '-v', action='store_true', help='Whether to print logs.')
    args = parser.parse_args()

    # Get the config and set the seed
    config = parse_config("config.yaml")
    np.random.seed(config['random_state'])

    # Predict
    prediction, speed = prepare_and_predict(user_input=args.text,
                                            model_name=args.model,
                                            config=config,
                                            verbose=args.verbose)
    print('\nPrediction:', prediction)
    print('Speed:', f'{speed:0.2f}', 'seconds')
