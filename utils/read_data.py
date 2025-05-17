import glob
import pandas as pd


def get_all_csv_files(path: str = '../dataset/') -> list:
    """
    Get all CSV files from the specified path.
    
    Args:
        path (str): The path to search for CSV files.
        
    Returns:
        list: A list of paths to CSV files.
    """
    return glob.glob(path)


def read_data(path: str = '../dataset/') -> pd.DataFrame:
    
    files = get_all_csv_files(path + 'creditcard_part_*.csv')
    
    dfs = [
        pd.read_csv(file, index_col=0)
        for file in sorted(files,key=lambda x: int(x.split('_')[-1].split('.')[0]))
    ]
    
    return pd.concat(dfs, ignore_index=True)


def get_train_data(path: str = '../notebooks/df_full_train.csv') -> pd.DataFrame:
    """
    Read and return the training data from the specified path.
    
    Args:
        path (str): The path to the dataset.
    
    Returns:
        pd.DataFrame: The training data as a DataFrame.
    """
    return pd.read_csv(path)


def get_test_data(path: str = '../notebooks/df_test.csv') -> pd.DataFrame:
    """
    Read and return the test data from the specified path.
    
    Args:
        path (str): The path to the dataset.
    
    Returns:
        pd.DataFrame: The test data as a DataFrame.
    """
    return pd.read_csv(path)


if __name__ == "__main__":
    path = 'dataset/'
    
    print("CSV files found:", get_all_csv_files(path + 'creditcard_part_*.csv'))
    
    df = read_data(path)
    print("DataFrame shape:", df.shape)
    print("DataFrame columns:", df.columns.tolist())
    print("DataFrame head:\n", df.head())
    print("DataFrame tail:\n", df.tail())
    print("DataFrame info:\n", df.info())
    print("DataFrame describe:\n", df.describe())
    print("DataFrame memory usage:\n", df.memory_usage(deep=True))
