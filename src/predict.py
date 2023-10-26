import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pickle
import argparse
import timeit
import numpy as np

from src.preprocess import preprocessing_fn_dict, load_pretrained_models
from src.helper import *


def get_model_and_prep_fn_shorts(model_name: str, config: dict):
    # Load the model and the inference data preprocessing function names
    model = pickle.load(open(os.path.join('model_zoo', model_name, 'model.pkl'), 'rb'))
    preprocessing_fn_names = open(os.path.join('model_zoo', model_name, 'inference_data_prep.txt'), 'r').read().splitlines()

    return model, preprocessing_fn_names


def predict(model: object, label_enc: object, prep_fn_names: list[str],
            df: pd.DataFrame, pretrained_models: dict, config: dict, verbose: bool = False):
    # Preprocess the user input
    if 'embedding' not in df.columns:
        for preprocessing_fn_name in prep_fn_names:
            df = preprocessing_fn_dict[preprocessing_fn_name](df=df,
                                                              pretrained_models=pretrained_models,
                                                              verbose=verbose)

    # Get the embeddings
    X = np.array(df['embedding'].tolist())

    # Predict
    if verbose: print('\n> Predicting...')
    intent_idx_pred = model.predict(X)
    
    # Get the intent name using the label encoder
    intent_class_name = label_enc.inverse_transform(intent_idx_pred)[0]

    return intent_idx_pred, intent_class_name


def prepare_and_predict(model_name: str, user_input: str, config: dict, verbose: bool = False):
    """
    Predicts the intent of the user input.
    """

    # Create a dataframe with the user input
    user_input_df = pd.DataFrame({'text': [user_input]})

    # Get the model, the inference data preprocessing function names, and the label encoder
    model, prep_fn_shorts = get_model_and_prep_fn_shorts(model_name, config)
    label_enc = pickle.load(open(os.path.join('model_zoo', 'label_encoder.pkl'), 'rb'))

    # Get the pretrained models
    pretrained_models = load_pretrained_models(prep_fn_shorts=prep_fn_shorts, 
                                               config=config,
                                               verbose=verbose)

    # Predict and time
    start = timeit.default_timer()
    _, prediction = predict(model=model,
                            label_enc=label_enc,
                            prep_fn_names=prep_fn_shorts,
                            df=user_input_df,
                            pretrained_models=pretrained_models,
                            config=config,
                            verbose=verbose)
    stop = timeit.default_timer()
    
    return prediction, stop - start


if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--recipe', '-r', type=str, default=None, help='The name of the recipe to use for inference.')
    parser.add_argument('--text', '-t', type=str, help='The text to predict the intent of.')
    parser.add_argument('--verbose', '-v', action='store_true', help='Whether to print logs.')
    args = parser.parse_args()

    # Get the config
    config = parse_config("config.yaml")

    # Get the model name
    model_name = get_model_name_from_recipe(args.recipe, config)

    # Predict
    prediction, speed = prepare_and_predict(model_name=model_name,
                                            user_input=args.text,
                                            config=config,
                                            verbose=args.verbose)
    print('\nPrediction:', prediction)
    print('Speed:', f'{speed:0.2f}', 'seconds')
