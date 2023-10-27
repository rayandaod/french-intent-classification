import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import argparse
import pickle
import datetime
import timeit

from sklearn.metrics import confusion_matrix, classification_report

from src.helper import parse_config, get_model_name_from_recipe
from src.preprocess import get_ext_models
from src.predict import get_model_and_prep_fn_shorts, predict
from src.predict_english import get_en_model_tokenizer_trans, predict_en


def evaluate(model_name: str, test_path: str, eval_name: str,
             config: dict, verbose: bool=False) -> None:
    """
    Evaluate a model on a test set, and save the results in an evaluation folder.
    """
    
    # Get the model path
    model_path = os.path.join('model_zoo', model_name)

    # Create the evaluation folder
    datatime_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    eval_path = os.path.join(model_path, f'eval_{eval_name}_{datatime_str}')
    os.makedirs(eval_path, exist_ok=True)

    # Load the test set
    df_test = pd.read_csv(test_path)

    # Get the label encoder
    label_enc = pickle.load(open(os.path.join('model_zoo', 'label_encoder.pkl'), 'rb'))

    # If the model is not the English model
    if model_name != 'english':
        model, prep_fn_shorts = get_model_and_prep_fn_shorts(model_name)
        pretrained_models = get_ext_models(prep_fn_shorts=prep_fn_shorts,
                                                   config=config,
                                                   verbose=verbose)
        start = timeit.default_timer()
        y_pred, _ = predict(model=model,
                            label_enc=label_enc,
                            prep_fn_shorts=prep_fn_shorts,
                            df=df_test,
                            prep_dict=pretrained_models,
                            verbose=verbose)
        stop = timeit.default_timer()
        
        # Get the true labels and encode them
        y_true = label_enc.fit_transform(df_test['label'])
        
    # If the model is the English model
    else:
        model, tokenizer, translator = get_en_model_tokenizer_trans(config, verbose=verbose)

        start = timeit.default_timer()
        _, y_labels = predict_en(model=model,
                                tokenizer=tokenizer,
                                translator=translator,
                                df=df_test,
                                config=config,
                                verbose=verbose)
        stop = timeit.default_timer()

        # Map the y_labels to y_pred using the label encoder
        y_pred = label_enc.transform(y_labels)

        # Get the true labels using the label encoder
        y_true = label_enc.fit_transform(df_test['label'])

    # Get the classes
    classes = label_enc.classes_

    # Compute the confusion matrix and save it as an image in the evaluation folder
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(20, 20))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=classes, yticklabels=classes, annot_kws={"fontsize":20})
    plt.title('Confusion matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks(fontsize=20, rotation=45, ha='right')
    plt.yticks(fontsize=20, rotation=45)
    plt.savefig(os.path.join(eval_path, 'confusion_matrix.png'))

    # Print the classification report and save it 
    class_report = classification_report(y_true, y_pred, target_names=classes)
    print(class_report)
    with open(os.path.join(eval_path, 'classification_report.txt'), 'w') as f:
        f.write(class_report)

    # Print the accuracy of the in-scope classes (everything excep out_of_scope) and append it to the classification report
    in_scope_idx = label_enc.transform(classes[classes != 'out_of_scope'])
    in_scope_acc = np.sum(cm[in_scope_idx, in_scope_idx]) / np.sum(cm[in_scope_idx, :])
    print(f'\n>> Accuracy for in-scope classes: {in_scope_acc:0.2f}')
    with open(os.path.join(eval_path, 'classification_report.txt'), 'a') as f:
        f.write(f'\n>> Accuracy for in-scope classes: {in_scope_acc:0.2f}')

    # Print the out_of_scope class recall score and append it to the classification report
    oos_idx = label_enc.transform(['out_of_scope'])[0]
    oos_recall = cm[oos_idx, oos_idx] / np.sum(cm[oos_idx, :])
    print(f'\n>> Recall for out_of_scope: {oos_recall:0.2f}')
    with open(os.path.join(eval_path, 'classification_report.txt'), 'a') as f:
        f.write(f'\n>> Recall for out_of_scope: {oos_recall:0.2f}')

    # Print the average speed and append it to the classification report
    speed = (stop - start) / len(df_test)
    print(f'\n>> Average speed: {speed:0.2f} seconds')
    with open(os.path.join(eval_path, 'classification_report.txt'), 'a') as f:
        f.write(f'\n>> Average speed: {speed:0.2f} seconds')
    
    return


if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='logReg_camembert', help='The model to use for inference.')
    parser.add_argument('--test_path', '-t', type=str, default='data/examples.csv', help='The test set csv file to use for evaluation.')
    parser.add_argument('--eval_name', '-e', type=str, help='The name of the evaluation folder.')
    parser.add_argument('--verbose', '-v', action='store_true', help='Whether to print the translated sentence.')
    args = parser.parse_args()

    # Get the config and set seed
    config = parse_config("config.yaml")
    np.random.seed(config['random_state'])

    # Evaluate the model
    evaluate(model_name=args.model,
             test_path=args.test_path,
             eval_name=args.eval_name,
             config=config,
             verbose=args.verbose)
