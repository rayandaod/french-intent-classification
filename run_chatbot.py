import os
import timeit
import argparse
import readline
import pickle
import numpy as np

from src.preprocess import get_ext_models
from src.predict import predict, get_model_and_prep_fn_shorts
from src.predict_english import predict_en, get_en_model_tokenizer_trans
from src.helper import *

parser = argparse.ArgumentParser()


def run_chatbot(recipe_name: str, config: dict, verbose: bool = False) -> None:
    """
    Run the chatbot using the specified recipe.
    """

    # Get the model name
    model_name = get_model_name_from_recipe(recipe_name=recipe_name,
                                            config=config)

    if recipe_name != 'english':
        # Get the model, the inference data preprocessing function names, and the label encoder
        model, prep_fn_shorts = get_model_and_prep_fn_shorts(model_name, config)
        label_enc = pickle.load(open(os.path.join('model_zoo', 'label_encoder.pkl'), 'rb'))
        pretrained_models = get_ext_models(prep_fn_shorts=prep_fn_shorts, 
                                                    config=config,
                                                    verbose=args.verbose)
    else:
        # Get the model and the tokenizer
        model, tokenizer, translator = get_en_model_tokenizer_trans(config=config,
                                                                    verbose=args.verbose)
    # Introduce the bot
    print('\nILLUIN Bot:\tBonjour! Je suis ILLUIN Bot, votre assistant virtuel. Comment puis-je vous aider?')

    # Start the conversation
    while True:
        
        # Get user input and handle ctrl+c
        try:
            user_input = input('Vous: \t\t')
            start = timeit.default_timer()
        except KeyboardInterrupt:
            print('\nILLUIN Bot:\tAu revoir!')
            break

        # Convert the user input to a dataframe
        user_input_df = pd.DataFrame({'text': [user_input]})

        if recipe_name != 'english':
            _, prediction = predict(model=model,
                                label_enc=label_enc,
                                prep_fn_shorts=prep_fn_shorts,
                                df=user_input_df,
                                prep_dict=pretrained_models,
                                config=config,
                                verbose=args.verbose)
        else:
            _, prediction = predict_en(model=model,
                                        tokenizer=tokenizer,
                                        translator=translator,
                                        df=user_input_df,
                                        config=config,
                                        verbose=args.verbose)
            prediction = prediction.iloc[0]
        stop = timeit.default_timer()
        
        print(f'\n>> Prediction: {prediction}')
        print(f'>> Speed: {stop - start:0.7f} seconds\n')

        # Handle the lost_luggage prediction
        if prediction == 'lost_luggage':
            print("ILLUIN Bot:\tJe suis désolé de lire cela. Si j'ai bien compris, vous avez perdu vos bagages. Voici le numéro de téléphone de notre service client: 01 23 45 67 89. Veuillez noter que le coût de l'appel est de 0,15€/min.\n")

        print('ILLUIN Bot:\tQue puis-je faire d\'autre pour vous?')



if __name__ == '__main__':
    # Parse arguments
    parser.add_argument('--recipe', '-r', type=str, default=None, help='The recipe of the model to use for inference.')
    parser.add_argument('--verbose', '-v', action='store_true', help='Whether to print logs.')
    args = parser.parse_args()

    # Get the config and set seed
    config = parse_config("config.yaml")
    np.random.seed(config['random_state'])

    run_chatbot()
