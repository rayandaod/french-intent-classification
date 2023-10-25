import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
from src.helper import parse_config
from src.preprocess import preprocessing_fn_dict

# Get the config
config = parse_config("/Users/rayandaod/Documents/Docs/Job Search/ILLUIN/intent_classification/config.yaml")


def evaluate_model(classifier:object, df:pd.DataFrame, title:str) -> pd.DataFrame:
    """Evaluate a classifier on a dataframe.

    Args:
        classifier (object): The classifier to evaluate.
        df (pd.DataFrame): The dataframe to evaluate the classfiier on.
        title (str): The title of the plot.

    Returns:
        pd.DataFrame: The dataframe with the predicted classes.
    """
    # Copy the dataframe
    df = df.copy()

    # Encode the labels
    encoder = LabelEncoder()

    # If the column 'embedding' does not exist, run the inference preprocessing functions
    if 'embedding' not in df.columns:
        for preprocessing_fn_name in config['training_inference_data_prep']:
            df = preprocessing_fn_dict[preprocessing_fn_name][0](df, verbose=config['verbose'])

    # Get the embeddings
    X_eval = np.array(df['embedding'].tolist())

    # Get the labels and encode them
    y_eval = encoder.fit_transform(df['label'])

    # Predict
    y_pred = classifier.predict(X_eval)
    accuracy_score(y_eval, y_pred)

    # Compute the confusion matrix
    conf_mat = confusion_matrix(y_eval, y_pred)
    conf_mat_df = pd.DataFrame(conf_mat, index = encoder.classes_, columns = encoder.classes_)
    conf_mat_df.index.name = 'Actual'
    conf_mat_df.columns.name = 'Predicted'
    
    # Plot the confusion matrix
    plt.figure(figsize = (20, 14))
    sns.set(font_scale=1.4)
    sns.heatmap(conf_mat_df, annot=True, annot_kws={"size": 16}, fmt='d', cmap='Blues')
    plt.title(title)
    plt.show()
    
    # Print the classification report
    print(classification_report(y_eval, y_pred, target_names=encoder.classes_))

    # Add the predicted classes to the dataframe
    df['predictions'] = encoder.inverse_transform(y_pred)

    return df