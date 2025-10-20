import pandas as pd
from sklearn.model_selection import train_test_split
import os
import urllib.request

def generate_adult_test_set():
    """
    Generates the test.csv file for the Adult dataset by downloading the raw
    data from the UCI repository and replicating the data splitting from its
    training notebook.
    """
    print("--- Generating test.csv for Adult dataset ---")
    output_dir = './data/adult/adult/'
    output_path = os.path.join(output_dir, 'test.csv')
    data_path = os.path.join(output_dir, 'adult.data')
    test_data_path = os.path.join(output_dir, 'adult.test')

    # Data URLs from the UCI Machine Learning Repository
    data_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'
    test_data_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test'

    if os.path.exists(output_path):
        print("✅ test.csv already exists for Adult dataset.\n")
        return

    print("File not found. Generating test.csv for Adult dataset...")
    
    # Ensure the target directory exists
    os.makedirs(output_dir, exist_ok=True)

    # --- Download the data if it's not already present ---
    if not os.path.exists(data_path):
        print(f"Downloading adult.data from {data_url}...")
        urllib.request.urlretrieve(data_url, data_path)
        print("Download complete.")
    
    if not os.path.exists(test_data_path):
        print(f"Downloading adult.test from {test_data_url}...")
        urllib.request.urlretrieve(test_data_url, test_data_path)
        print("Download complete.")


    print("Processing data files...")
    column_names = [
        'age', 'workclass', 'fnlwgt', 'education', 'educational-num', 'marital-status',
        'occupation', 'relationship', 'race', 'gender', 'capital-gain', 'capital-loss',
        'hours-per-week', 'native-country', 'income'
    ]

    try:
        df_train = pd.read_csv(data_path, header=None, names=column_names, sep=r'\s*,\s*', engine='python', na_values='?')
        df_test = pd.read_csv(test_data_path, header=None, names=column_names, sep=r'\s*,\s*', engine='python', na_values='?', skiprows=1)
    except FileNotFoundError:
        print(f"❌ Error: Failed to read data files even after download attempt.")
        return

    df = pd.concat([df_train, df_test], ignore_index=True)

    X = df.drop('income', axis=1)
    y = df['income']

    # Replicate the exact split from the notebook
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_test.to_csv(output_path, index=False)
    print(f"✅ Successfully created '{output_path}'.\n")


if __name__ == "__main__":
    generate_adult_test_set()

