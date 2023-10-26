import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import argparse
import pickle
import numpy as np

from sklearn.linear_model import LogisticRegression

from src.helper import *


def train(recipe: dict, config: dict, verbose: bool = False) -> None:
    # Get the dataset folder name from the data preprocessing function short names
    prep_fn_shorts = recipe['training_data_prep'] + recipe['training_inference_data_prep']
    dataset_folder_name = '_'.join(prep_fn_shorts)

    # Get the training data
    train_data_path = os.path.join('data', dataset_folder_name, recipe['clinc150_version'], 'train', f'train_{dataset_folder_name}.pkl')
    train_df = pd.read_pickle(train_data_path)

    # If specified, add the validation data to the training data
    if recipe['add_val']:
        val_data_path = os.path.join('data', dataset_folder_name, recipe['clinc150_version'], 'validation', f'validation_{dataset_folder_name}.pkl')
        val_df = pd.read_pickle(val_data_path)
        train_df = pd.concat([train_df, val_df], ignore_index=True)

    # Get the embedding vectors and labels
    X = np.array(train_df['embedding'].tolist())
    y = train_df['label']

    # Load the label encoder and encode the labels
    label_encoder = pickle.load(open(os.path.join('model_zoo', 'label_encoder.pkl'), 'rb'))
    y = label_encoder.fit_transform(y)

    # Train the model
    if verbose: print('\n> Training the model...')
    model_type = recipe['model_type']
    model = LogisticRegression(random_state=config['random_state'], max_iter=1000)
    model.fit(X, y)

    # Save the classifier and the label encoder
    if verbose: print(f'\n> Saving the model in model_zoo/{model_type}_on_{recipe["clinc150_version"].upper()}_{dataset_folder_name}/model.pkl...')
    model_folder_name = model_type + '_on_' + recipe['clinc150_version'].upper() + '_' + dataset_folder_name
    model_path = f'model_zoo/{model_folder_name}'
    os.makedirs(model_path, exist_ok=True)
    pickle.dump(model, open(f'{model_path}/model.pkl', 'wb'))

    # Save the list of inference preprocessing function short names
    with open(f'{model_path}/inference_data_prep.txt', 'w') as f:
        for prep_fn_short in recipe['training_inference_data_prep']:
            f.write(f'{prep_fn_short}\n')

    return


if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--recipe', '-r', type=str, help='The recipe to use.')
    parser.add_argument('--verbose', '-v', action='store_true', help='Whether to print the logs or not.')
    args = parser.parse_args()

    # Get the config
    config = parse_config("config.yaml")

    # Get the recipe
    recipe = config['recipes'][args.recipe]

    # Train
    train(recipe=recipe, config=config, verbose=args.verbose)
