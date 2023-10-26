import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pickle
import argparse
import timeit
import numpy as np

from src.preprocess import preprocessing_fn_dict
from src.helper import *

# Get the config
config = parse_config("config.yaml")


def get_model_preprocessingFns(model_name: str):
    # Load the model and the inference data preprocessing function names
    model = pickle.load(open(os.path.join('model_zoo', model_name, 'model.pkl'), 'rb'))
    preprocessing_fn_names = open(os.path.join('model_zoo', model_name, 'inference_data_prep.txt'), 'r').read().splitlines()

    return model, preprocessing_fn_names


def predict(model: object, label_enc: object, prep_fn_names: list[str],
            df: pd.DataFrame, verbose: bool = False):
    # Preprocess the user input
    if 'embedding' not in df.columns:
        for preprocessing_fn_name in prep_fn_names:
            df = preprocessing_fn_dict[preprocessing_fn_name](df, verbose=verbose)

    # Get the embeddings
    X = np.array(df['embedding'].tolist())

    # Predict
    intent_idx_pred = model.predict(X)
    
    # Get the intent name using the label encoder
    intent_class_name = label_enc.inverse_transform(intent_idx_pred)[0]

    return intent_idx_pred, intent_class_name


def prepare_and_predict(model_name: str, user_input: str, verbose: bool = False):
    """
    Predicts the intent of the user input.
    """

    # Create a dataframe with the user input
    user_input_df = pd.DataFrame({'text': [user_input]})

    # Get the model, and the inference data preprocessing function names
    model, preprocessing_fn_names = get_model_preprocessingFns(model_name)

    # Get the label encoder
    label_enc = pickle.load(open(os.path.join('model_zoo', 'label_encoder.pkl'), 'rb'))

    # Predict and time
    start = timeit.default_timer()
    _, prediction = predict(model=model,
                            label_enc=label_enc,
                            prep_fn_names=preprocessing_fn_names,
                            df=user_input_df,
                            verbose=verbose)
    stop = timeit.default_timer()
    
    return prediction, stop - start


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='logReg_on_PLUS_oos1_down_carry_trans_sentenceCamembertBase',
                        help='The name of the model to use for inference.')
    parser.add_argument('--text', type=str, help='The text to predict the intent of.')
    parser.add_argument('--verbose', action='store_true', help='Whether to print logs.')
    args = parser.parse_args()

    prediction, speed = prepare_and_predict(args.model, args.text, verbose=args.verbose)
    print('Prediction:', prediction)
    print('Speed:', f'{speed:0.2f}', 'seconds')
