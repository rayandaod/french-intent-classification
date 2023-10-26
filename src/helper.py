import yaml
import pandas as pd
import matplotlib.pyplot as plt


def parse_config(config_path: str) -> dict:
    """Parse the config.yaml file

    Args:
        config_path (str): The path to the config.yaml file.

    Returns:
        dict: The parsed config.
    """
    # Parse the config file
    with open(config_path, "r") as config_file:
        config = yaml.safe_load(config_file)
    
    return config


def get_model_name_from_recipe(recipe_name: str, config: dict) -> str:
    """Get the model name from a recipe.

    Args:
        recipe (str): The name of the recipe.
        config (dict): The config.

    Returns:
        str: The model name.
    """
    # Get the recipe
    if recipe_name is not None:
        recipe = config['recipes'][recipe_name]
    else:
        recipe = config['recipes'][config['best_recipe']]

    # Get the model type (logReg, etc.)
    model_type = recipe['model_type']

    return f"{model_type}_{recipe_name}"


def plot_class_distribution(df: pd.DataFrame, title: str, highlighted_classes: list=[]) -> None:
    """Plot the class distribution of a dataframe.

    Args:
        df (pd.DataFrame): The dataframe to plot.
        title (str): The title of the plot.
        highlighted_classes (list): The list of classes to highlight.
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
    """Plot the text length distribution of a dataframe.

    Args:
        df (pd.DataFrame): The dataframe to plot.
        title (str): The title of the plot.
    """
    plt.figure(figsize=(30, 8))
    df["text"].str.len().hist(bins=30)
    plt.xlabel("Text length")
    plt.ylabel("Number of examples")
    plt.title(title)
    plt.show()
