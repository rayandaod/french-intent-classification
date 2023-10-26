import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import pandas as pd
import nltk
import numpy as np
import argparse

from transformers import FlaubertModel, FlaubertTokenizer
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from datasets import DatasetDict
from datasets import load_dataset
from tqdm import tqdm

from src.helper import parse_config


def get_ext_models(prep_fn_shorts: list, config:dict, verbose: bool=False) -> dict:
    """
    Get the external models needed for preprocessing based on the passed preprocessing function short names.
    """

    # Create a translation pipeline
    if 'trans' in prep_fn_shorts:
        if verbose: print('\n> Creating a translation pipeline...')
        translator = pipeline("translation", model=config['translator_en_fr_model_path'])
    else:
        translator = None

    # Get the stopwords from nltk
    if 'stop' in prep_fn_shorts:
        if verbose: print('\n> Gathering french stopwords...')
        nltk_path = '/Users/rayandaod/Documents/Docs/Job Search/ILLUIN/intent_classification/data/_nltk_data'
        if not os.path.exists(nltk_path):
            nltk.download('stopwords', download_dir=nltk_path)
        nltk.data.path.append(nltk_path)
        french_stopwords = nltk.corpus.stopwords.words('french')
    else:
        french_stopwords = None

    # Load the FlauBERT model and tokenizer
    if 'flaubertSmallCased' in prep_fn_shorts \
        or 'flaubertBaseUncased' in prep_fn_shorts \
        or 'flaubertBaseCased' in prep_fn_shorts \
        or 'flaubertLargeCased' in prep_fn_shorts:
        if verbose: print('\n> Loading the FlauBERT model and tokenizer...')
        flaubert_model_short_name = [s for s in prep_fn_shorts if s.startswith('flaubert')][0]
        flaubert_model_path = config['flaubert_model_paths'][flaubert_model_short_name]
        flaubert_tokenizer = FlaubertTokenizer.from_pretrained(flaubert_model_path, do_lowercase=True)
        flaubert_model, log = FlaubertModel.from_pretrained(flaubert_model_path, output_loading_info=True)
    else:
        flaubert_tokenizer = None
        flaubert_model = None

    # Load the CamemBERT model and tokenizer
    if 'sentenceCamembertBase' in prep_fn_shorts or 'sentenceCamembertLarge' in prep_fn_shorts:
        if verbose: print('\n> Loading the CamemBERT model...')
        camembert_model_short_name = [s for s in prep_fn_shorts if s.startswith('sentenceCamembert')][0]
        sentence_camembert_model =  SentenceTransformer(config['sentence_camembert_model_paths'][camembert_model_short_name])
    else:
        sentence_camembert_model = None

    return {
        'translator_en_fr': translator,
        'french_stopwords': french_stopwords,
        'flaubert_tokenizer': flaubert_tokenizer,
        'flaubert_model': flaubert_model,
        'sentence_camembert_model': sentence_camembert_model
    }


def load_clinc150_dataset_split(clinc150_dataset:DatasetDict, split:str='train', verbose:bool=False) -> pd.DataFrame:
    """
    Load a split of the CLINC150 dataset as a dataframe.
    """

    if verbose: print(f'\n>Loading the {split} set of CLINC150...')

    # Get the data
    df = clinc150_dataset[split].to_pandas()

    # Get the class names and create a dictionary to map the class index to the class name
    labels = clinc150_dataset[split].features["intent"].names
    labels = {i: name for i, name in enumerate(labels)}

    # Add a new column to the dataframe with the class name
    df["label"] = df["intent"].map(labels)

    # Replace oos by out_of_scope
    df["label"] = df["label"].replace("oos", "out_of_scope")

    # Drop the intent index column
    df = df.drop("intent", axis=1)

    if verbose: print(df.head())
    return df


def oos_strat_1(df:pd.DataFrame, ext_models: dict, verbose:bool=False) -> pd.DataFrame:
    """
    Apply the first out_of_scope strategy, i.e replace the classes that are not in config['classes'] by out_of_scope.
    """
    
    if verbose: print('\n> Applying the OOS strategy 1...')

    # Copy the dataframe
    df = df.copy()

    # Consider the classes that are not in config['classes'] as out-of-scope classes
    df.loc[~df["label"].isin(config['classes']), "label"] = "out_of_scope"

    return df


def oos_strat_2(df:pd.DataFrame, ext_models: dict, verbose:bool=False) -> pd.DataFrame:
    """
    Apply the second out_of_scope strategy, i.e only keep the classes that are in config['classes'] (including out_of_scope).
    """

    if verbose: print('\n> Applying the OOS strategy 2...')

    # Copy the dataframe
    df = df.copy()

    # Only keep the classes we are interested in (including out_of_scope)
    df = df[df["label"].isin(config['classes'])]

    # Reset the index
    df = df.reset_index(drop=True)

    return df


def downsample_oos(df:pd.DataFrame, ext_models: dict, verbose:bool=False) -> pd.DataFrame:
    """
    Downsample the out-of-scope class to a certain proportion of the individual in-scope classes (e.g 2.5x)
    """

    if verbose: print('\n> Downsampling the out-of-scope class...')

    # Copy the dataframe
    df = df.copy()

    # Only keep a certain proportion of the out-of-scope examples compared to the number of in-scope examples
    df_oos = df[df['label'] == 'out_of_scope']
    n_oos_to_keep = int(config['proportion_oos'] * len(df[df['label'] == config['classes'][0]]))
    df_oos = df_oos.sample(n_oos_to_keep)
    if verbose: print(f'Kept {n_oos_to_keep} out_of_scope examples')

    # Merbe back with non-oos examples
    df = pd.concat([df[df['label'] != 'out_of_scope'], df_oos])
    if verbose: print(df['label'].value_counts())

    # Reset the index
    df = df.reset_index(drop=True)

    return df


def carry_on_enhancer_for_trans(df:pd.DataFrame, ext_models: dict, verbose:bool=False) -> pd.DataFrame:
    """
    Enhance the carry-on examples for english-to-french translation.
    It was noticed that the carry-on examples were not translated correctly,
    so this function replaces occurences of 'carry on' by 'carry on' and a random luggage candidate.
    """

    if verbose: print('\n> Enhancing carry-on examples for english-to-french translation...')

    luggage_candidates = ['luggage', 'baggage', 'suitcase', 'bag', 'case']
    luggages_candidates = [l + 's' for l in luggage_candidates]

    # Copy the dataframe
    df = df.copy()

    # Replace occurences of 'carry on' in the texts by 'carry on' and a random luggage candidate
    df['text'] = df[df['label' == 'carry_on']]['text'].apply(lambda x: x.replace('carry on', f'carry on {luggage_candidates[np.random.randint(len(luggage_candidates))]}'))

    # Replace occurences of 'carry ons' in the texts by 'carry on'  and a random luggages candidate
    df['text'] = df[df['label' == 'carry_on']]['text'].apply(lambda x: x.replace('carry ons', f'carry on {luggages_candidates[np.random.randint(len(luggages_candidates))]}'))

    # Replace occurences of 'carry-on' in the texts by 'carry on' and a random luggage candidate
    df['text'] = df[df['label' == 'carry_on']]['text'].apply(lambda x: x.replace('carry-on', f'carry on {luggage_candidates[np.random.randint(len(luggage_candidates))]}'))

    # Replace occurences of 'carry-ons' in the texts by 'carry on' and a random luggages candidate
    df['text'] = df[df['label' == 'carry_on']]['text'].apply(lambda x: x.replace('carry-ons', f'carry on {luggages_candidates[np.random.randint(len(luggages_candidates))]}'))

    return df


def translate_en_fr(df:pd.DataFrame, ext_models: dict, verbose:bool=False) -> pd.DataFrame:
    """
    Translate the text from English to French using the previously loaded translator.
    """

    if verbose: print('\n> Translating the text from English to French...')

    # Copy the dataframe
    df = df.copy()

    # Translate the text from English to French
    translator_fn = lambda x: ext_models['translator_en_fr'](x)[0]['translation_text']
    if verbose:
        tqdm.pandas()
        df['text_fr'] = df['text'].progress_apply(translator_fn)
    else:
        df['text_fr'] = df['text'].apply(translator_fn)

    # Rename the text column to text_en and the text_fr column to text
    df = df.rename(columns={'text': 'text_en'})
    df = df.rename(columns={'text_fr': 'text'})

    if verbose: print('First example translated:', df.iloc[0]['text'])

    return df


def remove_stopwords(df:pd.DataFrame, ext_models: dict, verbose:bool=False) -> pd.DataFrame:
    """
    Remove the stopwords from the text (french).
    """

    if verbose: print('\n> Removing the stopwords...')

    # Copy the dataframe
    df = df.copy()

    # Get the stopwords
    french_stopwords = ext_models['french_stopwords']

    # Remove the stopwords
    df['text'] = df['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in french_stopwords]))

    if verbose: print('First example without stopwords:', df.iloc[0]['text'])
    return df


def flaubert_encoder(df:pd.DataFrame, ext_models: dict, verbose:bool=False) -> pd.DataFrame:
    """
    Encode each word using the previously loaded FlauBERT model and tokenizer.
    """

    if verbose: print('\n> Encoding each word using FlauBERT...')

    # Copy the dataframe
    df = df.copy()

    # Get the FlauBERT tokenizer and model
    flaubert_tokenizer = ext_models['flaubert_tokenizer']
    flaubert_model = ext_models['flaubert_model']

    # Tokenize each row using the FlauBERT tokenizer
    # Use tqdm to display a progress bar, only if verbose is True
    flaubert_tokenizer_fn = lambda x: flaubert_tokenizer.tokenize(x)
    if verbose:
        tqdm.pandas()
        df['token'] = df['text'].progress_apply(flaubert_tokenizer_fn)
    else:
        df['token'] = df['text'].apply(flaubert_tokenizer_fn)

    # Convert the tokens to ids
    # Use tqdm to display a progress bar, only if verbose is True
    flaubert_token_to_id_fn = lambda x: flaubert_tokenizer.convert_tokens_to_ids(x)
    if verbose:
        df['token_id'] = df['token'].progress_apply(flaubert_token_to_id_fn)
    else:
        df['token_id'] = df['token'].apply(flaubert_token_to_id_fn)

    # Get the FlauBERT embeddings
    # Use tqdm to display a progress bar, only if verbose is True
    flaubert_fn = lambda x: flaubert_model(torch.tensor([x], dtype=torch.long))[0][0].detach().numpy()
    if verbose:
        df['word_embeddings'] = df['token_id'].progress_apply(flaubert_fn)
    else:
        df['word_embeddings'] = df['token_id'].apply(flaubert_fn)

    # Remove the token and token_id columns
    df = df.drop(['token', 'token_id'], axis=1)

    if verbose: print('Word embeddings shape:', df.iloc[0]['word_embeddings'].shape)
    return df


def average_word_emb(df:pd.DataFrame, ext_models: dict, verbose:bool=False) -> pd.DataFrame:
    """
    Merge the word embeddings by averaging them.
    """

    if verbose: print('\n> Averaging the word embeddings...')

    # Copy the dataframe
    df = df.copy()

    # Get the embeddings
    if verbose:
        tqdm.pandas()
        df['embedding'] = df['word_embeddings'].progress_apply(lambda x: np.mean(x, axis=0))
    else:
        df['embedding'] = df['word_embeddings'].apply(lambda x: np.mean(x, axis=0))

    # Remove the word_embeddings column
    df = df.drop('word_embeddings', axis=1)

    if verbose: print('Sentence embedding shape:', df.iloc[0]['embedding'].shape)
    return df


def sum_word_emb(df:pd.DataFrame, ext_models: dict, verbose:bool=False) -> pd.DataFrame:
    """
    Merge the word embeddings by summing them.
    """

    if verbose: print('\n> Summing the word embeddings...')

    # Copy the dataframe
    df = df.copy()

    # Get the embeddings
    if verbose:
        tqdm.pandas()
        df['embedding'] = df['word_embeddings'].progress_apply(lambda x: np.sum(x, axis=0))
    else:
        df['embedding'] = df['word_embeddings'].apply(lambda x: np.sum(x, axis=0))

    # Remove the word_embeddings column
    df = df.drop('word_embeddings', axis=1)

    if verbose: print('Sentence embedding shape:', df.iloc[0]['embedding'].shape)
    return df


def norm_emb(df:pd.DataFrame, ext_models: dict, verbose:bool=False) -> pd.DataFrame:
    """
    Normalize the final embedding.
    """

    if verbose: print('\n> Normalizing the embedding...')

    # Copy the dataframe
    df = df.copy()

    # Get the embeddings
    if verbose:
        tqdm.pandas()
        df['embedding'] = df['embedding'].progress_apply(lambda x: x / np.linalg.norm(x))
    else:
        df['embedding'] = df['embedding'].apply(lambda x: x / np.linalg.norm(x))

    if verbose: print('Sentence embedding shape:', df.iloc[0]['embedding'].shape)
    return df


def sentence_camembert(df:pd.DataFrame, ext_models: dict, verbose:bool=False) -> pd.DataFrame:
    """
    Encode the sentence using the previously loaded CamemBERT model.
    """

    if verbose: print('\n> Encoding the sentence using CamemBERT...')

    # Copy the dataframe
    df = df.copy()

    # Get the CamemBERT model
    sentence_camembert_model = ext_models['sentence_camembert_model']

    # Get the embeddings
    if verbose:
        tqdm.pandas()
        df['embedding'] = df['text'].progress_apply(lambda x: sentence_camembert_model.encode(x))
    else:
        df['embedding'] = df['text'].apply(lambda x: sentence_camembert_model.encode(x))

    if verbose: print('Sentence embedding shape:', df.iloc[0]['embedding'].shape)
    return df


# Store the preprocessing function references in a dictionary
# Each of these functions takes a dataframe as input and returns a dataframe as output
preprocessing_fn_dict = {
    'oos1': oos_strat_1,
    'oos2': oos_strat_2,
    'down': downsample_oos,
    'carry': carry_on_enhancer_for_trans,
    'trans': translate_en_fr,
    'stop': remove_stopwords,
    'flaubertSmallCased': flaubert_encoder,
    'flaubertBaseUncased': flaubert_encoder,
    'flaubertBaseCased': flaubert_encoder,
    'flaubertLargeCased': flaubert_encoder,
    'avg': average_word_emb,
    'sum': sum_word_emb,
    'norm': norm_emb,
    'sentenceCamembertBase': sentence_camembert,
    'sentenceCamembertLarge': sentence_camembert,
}


def preprocess_train_val(ext_models: dict, recipe:dict, verbose:bool=False) -> None:
    """
    Preprocess the training and validation sets.
    """

    # Load the dataset
    dataset = load_dataset(path='clinc_oos', name=recipe['clinc150_version'])

    # List the preprocessing functions to apply
    prep_fn_shorts = recipe['training_data_prep'] + recipe['training_inference_data_prep']

    for split in ['train', 'validation']:
        if verbose: print(f'\n> Preprocessing the {split} set...')

        # Create dataset folder if it does not exist
        dataset_folder = os.path.join('data', recipe['clinc150_version'], split)
        os.makedirs(dataset_folder, exist_ok=True)

        # Load and prepare the split
        df = load_clinc150_dataset_split(dataset, split=split, verbose=verbose)

        # Apply the preprocessing functions and save the intermediate dataframes 
        # by concatenating the short names of the preprocessing functions that happened until now
        concat_short_names = f'{split}'

        # Save the dataframe before any preprocessing
        if not os.path.exists(os.path.join(dataset_folder, f'{concat_short_names}.pkl')):
            df.to_pickle(os.path.join(dataset_folder, f'{concat_short_names}.pkl'))

        # Apply the preprocessing functions
        for preprocessing_fn_name in prep_fn_shorts:
            concat_short_names = '_'.join([concat_short_names, preprocessing_fn_name])
            
            # Apply the preprocessing function if the file does not already exist
            if not os.path.exists(os.path.join(dataset_folder, f'{concat_short_names}.pkl')):
                df = preprocessing_fn_dict[preprocessing_fn_name](df=df,
                                                                  ext_models=ext_models,
                                                                  verbose=verbose)
                df.to_pickle(os.path.join(dataset_folder, f'{concat_short_names}.pkl'))
            
            # Load the dataframe if the file already exists
            elif verbose:
                print(f'\nFile {concat_short_names}.pkl already exists, skipping...')
                df = pd.read_pickle(os.path.join(dataset_folder, f'{concat_short_names}.pkl'))


if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--recipe', '-r', type=str, default=None, help='The recipe to use.')
    parser.add_argument('--verbose', '-v', action='store_true', help='Whether to print the logs or not.')
    args = parser.parse_args()

    # Get the config and set the random seed
    config = parse_config('config.yaml')
    np.random.seed(config['random_state'])

    # Get the recipe
    if args.recipe is not None:
        recipe = config['recipes'][args.recipe]
    else:
        recipe = config['recipes'][config['best_recipe']]

    # Get the preprocessing function short names
    prep_fn_shorts = recipe['training_data_prep'] + recipe['training_inference_data_prep']

    # Load pretrained models
    ext_models = get_ext_models(prep_fn_shorts=prep_fn_shorts,
                                                    config=config)

    # Preprocess the data
    preprocess_train_val(ext_models=ext_models,
                              recipe=recipe,
                              verbose=args.verbose)
