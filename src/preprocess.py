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

from src.helper import parse_config, enhanced_apply
from src import RANDOM_SEED

np.random.seed(RANDOM_SEED)


class DataPreprocessor():
    """
    Preprocesses the data according to a recipe.
    """

    def __init__(self,
                 prep_fn_shorts: list[str],
                 config_path: str,
                 verbose: bool = False) -> None:
        """
        Constructor.
        """

        self.prep_fn_shorts = prep_fn_shorts

        # Get the config
        self.config = parse_config(config_path)

        # Create a translation pipeline
        translator = None
        if 'trans' in self.prep_fn_shorts:
            if verbose: print('\n> Creating a translation pipeline...')
            translator = pipeline("translation", model=self.config['translator_en_fr_model_path'])

        # Get the stopwords from nltk
        french_stopwords = None
        if 'stop' in self.prep_fn_shorts:
            if verbose: print('\n> Gathering french stopwords...')
            nltk_path = '../data/_nltk_data'
            if not os.path.exists(nltk_path):
                nltk.download('stopwords', download_dir=nltk_path)
            nltk.data.path.append(nltk_path)
            french_stopwords = nltk.corpus.stopwords.words('french')

        # Load the FlauBERT model and tokenizer
        flaubert_tokenizer = None
        flaubert_model = None
        if 'flaubertSmallCased' in self.prep_fn_shorts \
            or 'flaubertBaseUncased' in self.prep_fn_shorts \
            or 'flaubertBaseCased' in self.prep_fn_shorts \
            or 'flaubertLargeCased' in self.prep_fn_shorts:
            if verbose: print('\n> Loading the FlauBERT model and tokenizer...')
            flaubert_model_short_name = [s for s in self.prep_fn_shorts if s.startswith('flaubert')][0]
            flaubert_model_path = self.config['flaubert_model_paths'][flaubert_model_short_name]
            flaubert_tokenizer = FlaubertTokenizer.from_pretrained(flaubert_model_path, do_lowercase=True)
            flaubert_model, log = FlaubertModel.from_pretrained(flaubert_model_path, output_loading_info=True)

        # Load the CamemBERT model and tokenizer
        sentence_camembert_model = None
        if 'sentenceCamembertBase' in self.prep_fn_shorts \
            or 'sentenceCamembertLarge' in self.prep_fn_shorts:
            if verbose: print('\n> Loading the CamemBERT model...')
            camembert_model_short_name = [s for s in self.prep_fn_shorts if s.startswith('sentenceCamembert')][0]
            sentence_camembert_model =  SentenceTransformer(self.config['sentence_camembert_model_paths'][camembert_model_short_name])

        self.ext_models = {
            'translator_en_fr': translator,
            'french_stopwords': french_stopwords,
            'flaubert_tokenizer': flaubert_tokenizer,
            'flaubert_model': flaubert_model,
            'sentence_camembert_model': sentence_camembert_model
        }

        # Set the preprocessing function dictionary
        self.prep_fn_dict = {
            'oos1': self.oos_strat_1,
            'oos2': self.oos_strat_2,
            'down': self.downsample_oos,
            'carry': self.carry_on_enhancer_for_trans,
            'trans': self.translate_en_fr,
            'stop': self.remove_stopwords,
            'flaubertSmallCased': self.flaubert_enc,
            'flaubertBaseUncased': self.flaubert_enc,
            'flaubertBaseCased': self.flaubert_enc,
            'flaubertLargeCased': self.flaubert_enc,
            'avg': self.avg_word_emb,
            'sum': self.sum_word_emb,
            'norm': self.norm_emb,
            'sentenceCamembertBase': self.sentence_camembert,
            'sentenceCamembertLarge': self.sentence_camembert,
        }

        # Set the verbose attribute
        self.verbose = verbose

        return
    

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess the dataframe.
        """

        # Copy the dataframe
        df = df.copy()

        # Apply the preprocessing functions
        for preprocessing_fn_name in self.prep_fn_shorts:
            df = self.prep_fn_dict[preprocessing_fn_name](df)

        return df
    

    def oos_strat_1(self, df:pd.DataFrame) -> pd.DataFrame:
        """
        Apply the first out_of_scope strategy, i.e replace the classes that are not in config['classes'] by out_of_scope.
        """
        
        if self.verbose: print('\n> Applying the OOS strategy 1...')

        # Copy the dataframe
        df = df.copy()

        # Consider the classes that are not in config['classes'] as out-of-scope classes
        df.loc[~df["label"].isin(self.config['classes']), "label"] = "out_of_scope"

        return df


    def oos_strat_2(self, df:pd.DataFrame) -> pd.DataFrame:
        """
        Apply the second out_of_scope strategy, i.e only keep the classes that are in config['classes'] (including out_of_scope).
        """

        if self.verbose: print('\n> Applying the OOS strategy 2...')

        # Copy the dataframe
        df = df.copy()

        # Only keep the classes we are interested in (including out_of_scope)
        df = df[df["label"].isin(self.config['classes'])]

        # Reset the index
        df = df.reset_index(drop=True)

        return df


    def downsample_oos(self, df:pd.DataFrame) -> pd.DataFrame:
        """
        Downsample the out-of-scope class to a certain proportion of the individual in-scope classes (e.g 2.5x)
        """

        if self.verbose: print('\n> Downsampling the out-of-scope class...')

        # Copy the dataframe
        df = df.copy()

        # Only keep a certain proportion of the out-of-scope examples compared to the number of in-scope examples
        df_oos = df[df['label'] == 'out_of_scope']
        n_oos_to_keep = int(self.config['proportion_oos'] * len(df[df['label'] == self.config['classes'][0]]))
        df_oos = df_oos.sample(n_oos_to_keep)
        if self.verbose: print(f'Kept {n_oos_to_keep} out_of_scope examples')

        # Merbe back with non-oos examples
        df = pd.concat([df[df['label'] != 'out_of_scope'], df_oos])
        if self.verbose: print(df['label'].value_counts())

        # Reset the index
        df = df.reset_index(drop=True)

        return df


    def carry_on_enhancer_for_trans(self, df:pd.DataFrame) -> pd.DataFrame:
        """
        Enhance the carry-on examples for english-to-french translation.
        It was noticed that the carry-on examples were not translated correctly,
        so this function replaces occurences of 'carry on' by 'carry on' and a random luggage candidate.
        """

        if self.verbose: print('\n> Enhancing carry-on examples for english-to-french translation...')

        luggage_candidates = ['luggage', 'baggage', 'suitcase', 'bag', 'case']
        luggages_candidates = [l + 's' for l in luggage_candidates]

        # Copy the dataframe
        df = df.copy()

        # Replace occurences of 'carry on' in the carry_on label texts by 'carry on' and a random luggage candidate
        df['text'] = df.apply(lambda x: x['text'].replace('carry on', f"carry on {luggage_candidates[np.random.randint(len(luggage_candidates))]}") if x['label'] == 'carry_on' else x['text'], axis=1)
        df['text'] = df.apply(lambda x: x['text'].replace('carry ons', f"carry on {luggages_candidates[np.random.randint(len(luggages_candidates))]}") if x['label'] == 'carry_on' else x['text'], axis=1)
        df['text'] = df.apply(lambda x: x['text'].replace('carry-on', f"carry on {luggage_candidates[np.random.randint(len(luggages_candidates))]}") if x['label'] == 'carry_on' else x['text'], axis=1)
        df['text'] = df.apply(lambda x: x['text'].replace('carry-ons', f"carry on {luggages_candidates[np.random.randint(len(luggages_candidates))]}") if x['label'] == 'carry_on' else x['text'], axis=1)

        return df


    def translate_en_fr(self, df:pd.DataFrame) -> pd.DataFrame:
        """
        Translate the text from English to French using the previously loaded translator.
        """

        if self.verbose: print('\n> Translating the text from English to French...')

        # Copy the dataframe
        df = df.copy()

        # Translate the text from English to French
        translator_fn = lambda x: self.ext_models['translator_en_fr'](x)[0]['translation_text']
        df['text_fr'] = enhanced_apply(translator_fn, df['text'], verbose=self.verbose)

        # Rename the text column to text_en and the text_fr column to text
        df = df.rename(columns={'text': 'text_en'})
        df = df.rename(columns={'text_fr': 'text'})

        if self.verbose: print('First example translated:', df.iloc[0]['text'])

        return df


    def remove_stopwords(self, df:pd.DataFrame) -> pd.DataFrame:
        """
        Remove the stopwords from the text (french).
        """

        if self.verbose: print('\n> Removing the stopwords...')

        # Copy the dataframe
        df = df.copy()

        # Get the stopwords
        french_stopwords = self.ext_models['french_stopwords']

        # Remove the stopwords
        df['text'] = df['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in french_stopwords]))

        if self.verbose: print('First example without stopwords:', df.iloc[0]['text'])
        return df


    def flaubert_enc(self, df:pd.DataFrame) -> pd.DataFrame:
        """
        Encode each word using the previously loaded FlauBERT model and tokenizer.
        """

        if self.verbose: print('\n> Encoding each word using FlauBERT...')

        # Copy the dataframe
        df = df.copy()

        # Get the FlauBERT tokenizer and model
        flaubert_tokenizer = self.ext_models['flaubert_tokenizer']
        flaubert_model = self.ext_models['flaubert_model']

        # Tokenize each row using the FlauBERT tokenizer
        # Use tqdm to display a progress bar, only if verbose is True
        flaubert_tokenizer_fn = lambda x: flaubert_tokenizer.tokenize(x)
        df['token'] = enhanced_apply(flaubert_tokenizer_fn, df['text'], verbose=self.verbose)

        # Convert the tokens to ids
        # Use tqdm to display a progress bar, only if verbose is True
        flaubert_token_to_id_fn = lambda x: flaubert_tokenizer.convert_tokens_to_ids(x)
        df['token_id'] = enhanced_apply(flaubert_token_to_id_fn, df['token'], verbose=self.verbose)

        # Get the FlauBERT embeddings
        # Use tqdm to display a progress bar, only if verbose is True
        flaubert_fn = lambda x: flaubert_model(torch.tensor([x], dtype=torch.long))[0][0].detach().numpy()
        df['word_embeddings'] = enhanced_apply(flaubert_fn, df['token_id'], verbose=self.verbose)

        # Remove the token and token_id columns
        df = df.drop(['token', 'token_id'], axis=1)

        if self.verbose: print('Word embeddings shape:', df.iloc[0]['word_embeddings'].shape)
        return df


    def avg_word_emb(self, df:pd.DataFrame) -> pd.DataFrame:
        """
        Merge the word embeddings by averaging them.
        """

        if self.verbose: print('\n> Averaging the word embeddings...')

        # Copy the dataframe
        df = df.copy()

        # Get the embeddings
        df['embedding'] = enhanced_apply(lambda x: np.mean(x, axis=0), df['word_embeddings'], verbose=self.verbose)

        # Remove the word_embeddings column
        df = df.drop('word_embeddings', axis=1)

        if self.verbose: print('Sentence embedding shape:', df.iloc[0]['embedding'].shape)
        return df


    def sum_word_emb(self, df:pd.DataFrame) -> pd.DataFrame:
        """
        Merge the word embeddings by summing them.
        """

        if self.verbose: print('\n> Summing the word embeddings...')

        # Copy the dataframe
        df = df.copy()

        # Get the embeddings
        df['embedding'] = enhanced_apply(lambda x: np.sum(x, axis=0), df['word_embeddings'], verbose=self.verbose)

        # Remove the word_embeddings column
        df = df.drop('word_embeddings', axis=1)

        if self.verbose: print('Sentence embedding shape:', df.iloc[0]['embedding'].shape)
        return df


    def norm_emb(self, df:pd.DataFrame) -> pd.DataFrame:
        """
        Normalize the final embedding.
        """

        if self.verbose: print('\n> Normalizing the embedding...')

        # Copy the dataframe
        df = df.copy()

        # Get the embeddings
        df['embedding'] = enhanced_apply(lambda x: x / np.linalg.norm(x), df['embedding'], verbose=self.verbose)

        if self.verbose: print('Sentence embedding shape:', df.iloc[0]['embedding'].shape)
        return df


    def sentence_camembert(self, df:pd.DataFrame) -> pd.DataFrame:
        """
        Encode the sentence using the previously loaded CamemBERT model.
        """

        if self.verbose: print('\n> Encoding the sentence using CamemBERT...')

        # Copy the dataframe
        df = df.copy()

        # Get the CamemBERT model
        sentence_camembert_model = self.ext_models['sentence_camembert_model']

        # Get the embeddings
        df['embedding'] = enhanced_apply(lambda x: sentence_camembert_model.encode(x), df['text'], verbose=self.verbose)

        if self.verbose: print('Sentence embedding shape:', df.iloc[0]['embedding'].shape)
        return df



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


def preprocess_dataset(recipe_name:str, config_path: str, verbose: bool) -> None:
    """
    Preprocess the dataset according to the recipe.
    """

    # Get the config
    config = parse_config(config_path)

    # Get the recipe
    recipe = config['recipes'][recipe_name]

    # Get the preprocessing function short names
    prep_fn_shorts = recipe['training_data_prep'] + recipe['training_inference_data_prep']

    # Initialise the dataset preprocessor
    data_prep = DataPreprocessor(prep_fn_shorts=prep_fn_shorts,
                                 config_path=config_path,
                                 verbose=verbose)
    
    # Load the dataset
    dataset = load_dataset(path='clinc_oos', name=recipe['clinc150_version'])

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
                df = data_prep.prep_fn_dict[preprocessing_fn_name](df)
                df.to_pickle(os.path.join(dataset_folder, f'{concat_short_names}.pkl'))
            
            # Load the dataframe if the file already exists
            elif verbose:
                print(f'\nFile {concat_short_names}.pkl already exists, skipping...')
                df = pd.read_pickle(os.path.join(dataset_folder, f'{concat_short_names}.pkl'))



if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--recipe', '-r', type=str, default='camembert', help='The recipe to use.')
    parser.add_argument('--config', '-c', type=str, default='config.yaml', help='The path to the configuration file.')
    parser.add_argument('--verbose', '-v', action='store_true', help='Whether to print the logs or not.')
    args = parser.parse_args()

    preprocess_dataset(recipe_name=args.recipe,
                        config_path=args.config,
                        verbose=args.verbose)