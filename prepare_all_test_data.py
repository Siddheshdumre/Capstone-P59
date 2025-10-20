import pandas as pd
from sklearn.model_selection import train_test_split
import os
import urllib.request

def prepare_german_test_set():
    """
    Downloads the raw German Credit data, replicates the exact preprocessing
    from the training notebook, and saves the final test set.
    """
    print("\n--- Preparing German Credit dataset ---")
    output_path = './data/german/german_credit_risk/test.csv'
    
    try:
        # Step 1: Download raw data from UCI repository
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data'
        data_path = './downloads/german.data'
        os.makedirs('./downloads', exist_ok=True)
        print("Downloading raw German credit data...")
        urllib.request.urlretrieve(url, data_path)

        # Step 2: Load data with correct column names from the training notebook
        column_names = [
            'Checking account', 'Duration', 'Credit history', 'Purpose', 'Credit amount', 
            'Saving accounts', 'Employment', 'Installment rate', 'Personal status', 
            'Other debtors', 'Present residence since', 'Property', 'Age', 
            'Other installment plans', 'Housing', 'Existing credits', 'Job', 
            'Number of dependents', 'Telephone', 'Foreign worker', 'Risk'
        ]
        df = pd.read_csv(data_path, sep=' ', header=None, names=column_names)

        print("Applying exact preprocessing steps from the training notebook...")
        
        # Step 3: Replicate feature engineering and selection
        df['Sex'] = df['Personal status'].apply(lambda x: 'male' if x in ['A91', 'A93', 'A94'] else 'female')
        
        # This is the crucial step: select ONLY the 9 features the model was trained on.
        features_to_keep = [
            'Age', 'Sex', 'Job', 'Housing', 'Saving accounts', 
            'Checking account', 'Credit amount', 'Duration', 'Purpose'
        ]
        X = df[features_to_keep].copy()
        y = df['Risk'].map({1: 0, 2: 1})
        
        # Step 4: Generate the identical test set split
        _, X_test, _, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        X_test.to_csv(output_path, index=False, sep=',') # Ensure comma separation
        print(f"✅ Successfully created '{output_path}' with the correct 9 features.")
        
    except Exception as e:
        print(f"❌ Error preparing German test set: {e}")

def prepare_compas_test_set():
    """
    Downloads and preprocesses the COMPAS data, ensuring the test set
    is saved with the correct format.
    """
    print("\n--- Preparing COMPAS dataset ---")
    output_path = './data/compas/test.csv'

    try:
        url = 'https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv'
        data_path = './downloads/compas-scores-two-years.csv'
        os.makedirs('./downloads', exist_ok=True)
        urllib.request.urlretrieve(url, data_path)
        
        print("Loading and preprocessing COMPAS data...")
        df = pd.read_csv(data_path)
        
        # Replicate the exact preprocessing steps
        features_to_keep = [
            'age', 'c_charge_degree', 'race', 'age_cat', 'score_text', 'sex', 
            'priors_count', 'days_b_screening_arrest', 'decile_score', 
            'is_recid', 'two_year_recid'
        ]
        df = df[features_to_keep]
        df = df.loc[(df['days_b_screening_arrest'] <= 30) & (df['days_b_screening_arrest'] >= -30)]
        df = df.loc[df['is_recid'] != -1]
        df = df.loc[df['c_charge_degree'] != "O"]
        df = df.loc[df['score_text'] != 'N/A']
        
        X = df.drop('two_year_recid', axis=1)
        y = df['two_year_recid']
        
        _, X_test, _, _ = train_test_split(X, y, test_size=0.2, random_state=42)
        
        X_test.to_csv(output_path, index=False, sep=',') # Ensure comma separation
        print(f"✅ Successfully created '{output_path}' with correct formatting.")
        
    except Exception as e:
        print(f"❌ Error preparing COMPAS test set: {e}")

if __name__ == '__main__':
    prepare_german_test_set()
    prepare_compas_test_set()
    print("\n--- All test data prepared. ---")

