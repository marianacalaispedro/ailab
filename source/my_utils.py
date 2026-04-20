"""

Utils Module
------------

Utility functions (file I/O, configuration loading, constants).

"""

import pandas as pd
import os


def load_dataset(dataset_path: str) -> pd.DataFrame:
    """
    Loads a dataset from a specified file path. Supports CSV and Excel file formats.
    Args:
        dataset_path (str): The file path to the dataset. Must be a .csv or .xlsx file.
    Returns:
        pandas.DataFrame: The loaded dataset as a pandas DataFrame.
    Raises:
        ValueError: If the file is not in CSV or Excel format.
    Notes:
        - If the first column of the dataset is named 'Unnamed: 0', it will be automatically dropped.
    """

    if dataset_path.endswith('.csv'):
        data = pd.read_csv(dataset_path)
    
    elif dataset_path.endswith('.xlsx'):
        data = pd.read_excel(dataset_path)
    
    else:
        raise ValueError("The dataset must be in CSV or EXCEL format.")
    
    if data.columns[0] == 'Unnamed: 0':
        data = data.drop(columns=data.columns[0])

    return data

def export_dataset(data: pd.DataFrame, path: str, name: str, format: str = 'csv') -> None:
    """
    Exports a pandas DataFrame to a specified file format and path.

    Args:
        data (pd.DataFrame): The DataFrame to be exported.
        path (str): The directory path where the file will be saved.
        name (str): The name of the file (without extension).
        format (str, optional): The file format ('csv' or 'xlsx'). Defaults to 'csv'.

    Raises:
        ValueError: If the specified format is not 'csv' or 'xlsx'.

    Returns:
        None
    """
    if format not in ['csv', 'xlsx']:
        raise ValueError("The format must be 'csv' or 'xlsx'.")

    full_path = f"{path}/{name}.{format}"

    # Check if the file exists and remove it if it does
    if os.path.exists(full_path):
        os.remove(full_path)

    if format == 'csv':
        data.to_csv(full_path, index=False)
    elif format == 'xlsx':
        data.to_excel(full_path, index=False)