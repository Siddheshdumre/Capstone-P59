import pandas as pd
from sklearn.model_selection import train_test_split
import os
import urllib.request

def generate_german_test_set():
    """
    Generates the test.csv file for the German Credit dataset by downloading the
    raw data from the UCI repository and replicating the data splitting from
    its training notebook.
    """
    print("--- Generating test.csv for German Credit dataset ---")
    output_dir = './data/german/german_credit_risk/'
    output_path = os.path.join(output_dir, 'test.csv')
    data_path = os.path.join(output_dir, 'german.data')

    # Data URL from the UCI Machine Learning Repository
    data_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data'

    if os.path.exists(output_path):
        print("✅ test.csv already exists for German Credit dataset.\n")
        return

    print("File not found. Generating test.csv for German Credit dataset...")
    
    # Ensure the target directory exists
    os.makedirs(output_dir, exist_ok=True)

    # --- Download the data if it's not already present ---
    if not os.path.exists(data_path):
        print(f"Downloading german.data from {data_url}...")
        urllib.request.urlretrieve(data_url, data_path)
        print("Download complete.")
    
    print("Processing data file...")
    
    # Column names are based on the dataset's documentation
    column_names = [
        'status', 'duration', 'credit_history', 'purpose', 'amount', 
        'savings', 'employment_duration', 'installment_rate', 
        'personal_status_sex', 'other_debtors', 'present_residence', 
        'property', 'age', 'other_installment_plans', 'housing', 
        'number_credits', 'job', 'people_liable', 'telephone', 
        'foreign_worker', 'credit_risk'
    ]

    try:
        # The raw data is space-separated
        df = pd.read_csv(data_path, header=None, names=column_names, sep=' ')
    except FileNotFoundError:
        print(f"❌ Error: Failed to read data file even after download attempt.")
        return

    # The target variable is 'credit_risk'
    X = df.drop('credit_risk', axis=1)
    y = df['credit_risk']

    # Replicate the exact split from the notebook (assuming standard parameters)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    X_test.to_csv(output_path, index=False)
    print(f"✅ Successfully created '{output_path}'.\n")


if __name__ == "__main__":
    generate_german_test_set()

