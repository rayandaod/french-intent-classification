import timeit
import argparse
import readline

from src.predict import predict
from src.predict_lazy import predict as translate_and_predict_with_pretrained

parser = argparse.ArgumentParser()


if __name__ == '__main__':
    # Parse arguments
    parser.add_argument('--model', type=str, default='best', help='The model to use for inference.')
    args = parser.parse_args()

    # Introduce the bot
    print('\nILLUIN Bot:\tBonjour! Je suis ILLUIN Bot, votre assistant virtuel. Comment puis-je vous aider?')

    # Start the conversation
    while True:
        # Get user input
        user_input = input('Vous: \t\t')

        # Predict and time
        start = timeit.default_timer()
        if args.model == 'best':
            prediction = predict(user_input, verbose=False)
        elif args.model == 'english':
            prediction = translate_and_predict_with_pretrained(user_input)
        stop = timeit.default_timer()
        
        print(f'\n>> Prediction: {prediction}')
        print(f'>> Speed: {stop - start:0.7f} seconds\n')

        if prediction == 'lost_luggage':
            print("ILLUIN Bot:\tJe suis désolé d\'entendre cela. Si j'ai bien compris, vous avez perdu vos bagages. Auquel cas, voici le numéro de téléphone de notre service client: 01 23 45 67 89. Veuillez noter que le coût de l'appel est de 0,15€/min.\n")

        print('ILLUIN Bot:\tQue puis-je faire d\'autre pour vous?')