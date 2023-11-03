import timeit
import argparse
import readline
import numpy as np
import pandas as pd

from src.predict import IntentPredictor, IntentPredictorEnglish
from src import RANDOM_SEED

np.random.seed(RANDOM_SEED)


def run_chatbot(model_name: str,
                config_path: str,
                verbose: bool = False) -> None:
    """
    Run the chatbot using the specified model.
    """

    if model_name != 'english':
        intent_predictor = IntentPredictor(model_name=model_name,
                                           config_path=config_path,
                                           verbose=verbose)
    else:
        intent_predictor = IntentPredictorEnglish(config_path=config_path,
                                                  verbose=verbose)
        

    # Introduce the bot
    print('\nBot:\tBonjour! Je suis votre assistant virtuel. Comment puis-je vous aider?')

    # Start the conversation
    while True:
        
        # Get user input and handle KeyboardInterrupt
        try:
            user_input = input('Vous:\t')
            start = timeit.default_timer()
        
        except KeyboardInterrupt:
            print('CTRL+C\nBot:\tAu revoir!')
            break

        # Convert the user input to a dataframe
        user_input_df = pd.DataFrame({'text': [user_input]})

        # Predict the intent
        _, prediction = intent_predictor(user_input_df)
        prediction = prediction[0]
        
        # Stop the timer
        total_time = timeit.default_timer() - start
        
        print(f'\n>> Prediction: {prediction}')
        print(f'>> Speed: {total_time:0.2f} seconds\n')

        # Handle the lost_luggage prediction
        if prediction == 'lost_luggage':
            print("Bot:\tJe suis désolé de lire cela. Si j'ai bien compris, vous avez perdu vos bagages. Voici le numéro de téléphone de notre service client: 01 23 45 67 89. Veuillez noter que le coût de l'appel est de 0,15€/min.\n")

        print('Bot:\tQue puis-je faire d\'autre pour vous?')



if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='logReg_camembert', help='The model to use for inference.')
    parser.add_argument('--config', '-c', type=str, default='config.yaml', help='The path to the config file.')
    parser.add_argument('--verbose', '-v', action='store_true', help='Whether to print logs.')
    args = parser.parse_args()

    run_chatbot(model_name=args.model,
                config_path=args.config,
                verbose=args.verbose)
