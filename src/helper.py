import yaml
import pandas as pd
import matplotlib.pyplot as plt
import logging
import timeit

from tqdm import tqdm


def parse_config(config_path: str) -> dict:
    """
    Parse the config.yaml file
    """

    # Parse the config file
    with open(config_path, "r") as config_file:
        config = yaml.safe_load(config_file)
    
    return config


def enhanced_apply(function: callable, df: pd.DataFrame) -> pd.Series:
    """
    Apply a function to a dataframe, with a progress bar if logging is enabled.
    """

    if logging.getLogger().isEnabledFor(logging.INFO):
        tqdm.pandas()
        return df.progress_apply(function)
    else:
        return df.apply(function)
    

def timeit_decorator(method: callable) -> callable:
    """
    Decorator to time a method.
    """

    def timed(*args, **kwargs):
        start = timeit.default_timer()
        result = method(*args, **kwargs)
        total_time = timeit.default_timer() - start
        logging.info(f"Time taken: {total_time:0.2f} seconds")
        return result

    return timed


def plot_class_distribution(df: pd.DataFrame, title: str, highlighted_classes: list=[]) -> None:
    """
    Plot the class distribution of a dataframe.
    """

    plt.figure(figsize=(30, 8))
    df_value_counts = df["label"].value_counts()
    df_value_counts.plot(kind='bar',
                         color=['#6495ED' if c not in highlighted_classes else '#F08080' for c in df_value_counts.index])
    plt.xticks(rotation=90)
    plt.xlabel("Class name")
    plt.ylabel("Number of examples")
    plt.title(title)
    plt.show()


def plot_text_length_distribution(df:pd.DataFrame, title:str) -> None:
    """
    Plot the text length distribution of a dataframe.
    """

    plt.figure(figsize=(30, 8))
    df["text"].str.len().hist(bins=30)
    plt.xlabel("Text length")
    plt.ylabel("Number of examples")
    plt.title(title)
    plt.show()
