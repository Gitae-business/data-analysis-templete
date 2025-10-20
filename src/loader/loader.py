import os
import pandas as pd
from config import config

def load_data():
    train_path = config.TRAIN_DATA
    test_path = config.TEST_DATA

    try:
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        print("Train dataset loaded successfully")
        print(train_df.info())
        
        print("\nTest dataset loaded successfully")
        print(test_df.info())

        return train_df, test_df

    except FileNotFoundError:
        print(f"Error: Data files not found in '{config.DATA_DIR}' directory.")
        return None, None

if __name__ == '__main__':
    train_data, test_data = load_data()
    if train_data is not None and test_data is not None:
        print("\nTrain data head:")
        print(train_data.head())
        print("\nTest data head:")
        print(test_data.head())
