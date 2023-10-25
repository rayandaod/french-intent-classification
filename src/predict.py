import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pickle

from src.preprocess import preprocessing_fn_dict
from src.helper import *

# Get the config
config = parse_config("config.yaml")

# Load the model and the label encoder
model = pickle.load(open(os.path.join('models', config['model_folder_name'], 'model.pkl'), 'rb'))
label_enc = pickle.load(open(os.path.join('models', config['model_folder_name'], 'label_encoder.pkl'), 'rb'))


def predict(user_input: str, verbose: bool = False):
    """
    Predicts the intent of the user input.
    """
    # Create a dataframe with the user input
    df = pd.DataFrame({'text': [user_input]})
    
    # Preprocess the user input
    for preprocessing_fn_name in config['training_inference_data_prep']:
        df = preprocessing_fn_dict[preprocessing_fn_name][0](df, verbose=verbose)

    # Get the embeddings
    enc = df['embedding'].iloc[0]
    
    # Predict
    pred = model.predict(enc.reshape(1, -1))[0]
    
    # Get the intent name using the label encoder
    intent_name = label_enc.inverse_transform([pred])[0]
    
    return intent_name


if __name__ == '__main__':
    print('\nPrediction:', predict(sys.argv[1]))