import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import argparse
import pickle

from sklearn.metrics import confusion_matrix, classification_report

from src.helper import parse_config
from src.predict import get_model_preprocessingFns, predict
from src.predict_en import get_en_model_tokenizer_trans, predict_en

# Get the config
config = parse_config("config.yaml")


def evaluate(model_name: str, test_path: str, eval_name: str, verbose: bool=False) -> None:
    """Evaluate a model on a test set.

    Args:
        model_name (str): The name of the model to use for inference.
        test_path (str): The path to the test set csv file.
        eval_name (str): The name of the evaluation folder.
        verbose (bool, optional): Whether to print logs. Defaults to False.

    Returns:
        None
    """
    # Get the model path
    model_path = os.path.join('model_zoo', model_name)

    # Create the evaluation folder
    eval_path = os.path.join(model_path, eval_name)
    os.makedirs(eval_path, exist_ok=True)

    # Load the test set
    df_test = pd.read_csv(test_path)

    # Get the label encoder
    label_enc = pickle.load(open(os.path.join('model_zoo', 'label_encoder.pkl'), 'rb'))

    if model_name != 'en':
        model, preprocessing_fn_names = get_model_preprocessingFns(model_name)
        y_pred, _ = predict(model=model,
                            label_enc=label_enc,
                            prep_fn_names=preprocessing_fn_names,
                            df=df_test,
                            verbose=verbose)
        
        # Get the true labels and encode them
        y_true = label_enc.fit_transform(df_test['label'])
        
    else:
        model, tokenizer, translator = get_en_model_tokenizer_trans(verbose=verbose)
        _, y_labels = predict_en(model=model,
                                tokenizer=tokenizer,
                                translator=translator,
                                df=df_test,
                                verbose=verbose)

        # Map the y_labels to y_pred using the label encoder
        y_pred = label_enc.transform(y_labels)

        # Get the true labels using the label encoder
        y_true = label_enc.fit_transform(df_test['label'])

    # Get the classes
    classes = label_enc.classes_

    # Compute the confusion matrix and save it as an image in the evaluation folder
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 10))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=classes, yticklabels=classes)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(eval_path, 'confusion_matrix.png'))
    
    # Save the classification report
    class_report = classification_report(y_true, y_pred, target_names=classes)
    with open(os.path.join(eval_path, 'classification_report.txt'), 'w') as f:
        f.write(class_report)
    
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='The model to use for inference.')
    parser.add_argument('--test_path', type=str, help='The test set csv file to use for evaluation.')
    parser.add_argument('--eval_name', type=str, help='The name of the evaluation folder.')
    parser.add_argument('--verbose', action='store_true', help='Whether to print the translated sentence.')
    args = parser.parse_args()

    # Evaluate the model
    evaluate(model_name=args.model,
             test_path=args.test_path,
             eval_name=args.eval_name,
             verbose=args.verbose)
