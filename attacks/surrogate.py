import pandas as pd
import numpy as np
import joblib
import argparse
import os

def surrogate_attack(model, X, seed=42):
    try:
        from art.attacks.evasion import HopSkipJump
        from art.estimators.classification import SklearnClassifier
    except ImportError:
        print("Installing adversarial-robustness-toolbox...")
        import subprocess
        subprocess.check_call(["pip", "install", "adversarial-robustness-toolbox"])
        from art.attacks.evasion import HopSkipJump
        from art.estimators.classification import SklearnClassifier

    np.random.seed(seed)
    
    preprocessor = model.named_steps['preproc']
    classifier = model.named_steps['clf']
    
    X_processed = preprocessor.transform(X)
    art_classifier = SklearnClassifier(model=classifier)

    attack = HopSkipJump(classifier=art_classifier, targeted=False, max_iter=5, max_eval=100, init_eval=10)
    
    subset_size = min(20, len(X))
    print(f"Attacking a subset of {subset_size} samples...")
    X_attacked_processed = attack.generate(x=X_processed[:subset_size])
    
    # Inverse transform is complex; we will work with the processed version
    # and re-apply to the original dataframe for prediction
    perturbation = X_attacked_processed - X_processed[:subset_size]
    
    X_attacked = X.copy()
    X_processed_full = preprocessor.transform(X_attacked)
    X_processed_full[:subset_size] += perturbation
    
    # This step is an approximation, as a true inverse is not always possible
    try:
        X_attacked_final = preprocessor.inverse_transform(X_processed_full)
        X_attacked = pd.DataFrame(X_attacked_final, columns=X.columns)
    except Exception:
        print("Warning: Could not inverse transform. Using original features for non-numeric types.")
        # Fallback for complex transformers
        numeric_features = X.select_dtypes(include=np.number).columns
        num_preproc = preprocessor.named_transformers_['num']
        X_attacked[numeric_features] = num_preproc.inverse_transform(X_processed_full[:, :len(numeric_features)])


    print(f"Generated {len(X_attacked)} attacked samples.")
    
    predictions = model.predict(X_attacked)
    if hasattr(model, "predict_proba"):
        probas = model.predict_proba(X_attacked)
        logits = np.log(probas[:, 1] / (1 - probas[:, 0] + 1e-9))
    else:
        logits = model.decision_function(X_attacked) if hasattr(model, "decision_function") else np.zeros(len(X_attacked))

    return X_attacked, predictions, logits

def main():
    parser = argparse.ArgumentParser(description="Run Surrogate Model Attack.")
    parser.add_argument("--model_path", required=True, help="Path to model file.")
    parser.add_argument("--data_path", required=True, help="Path to test data CSV.")
    parser.add_argument("--output_dir", required=True, help="Directory to save results.")
    args = parser.parse_args()

    print(f"Loading model from {args.model_path}...")
    model = joblib.load(args.model_path)
    
    print(f"Loading data from {args.data_path}...")
    X_test = pd.read_csv(args.data_path)

    try:
        model_features = model.named_steps['preproc'].get_feature_names_out()
        X_test.columns = model_features
        print("Successfully aligned data columns with model features.")
    except Exception as e:
        print(f"Could not align columns, proceeding with caution. Error: {e}")

    print("Applying Surrogate attack...")
    X_attacked, predictions, logits = surrogate_attack(model, X_test)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    pd.DataFrame(X_attacked, columns=X_test.columns).to_csv(f"{args.output_dir}/attacked_features.csv", index=False)
    np.savez(f"{args.output_dir}/attacked_results.npz", predictions=predictions, logits=logits)

    print(f"✅ Surrogate attack complete. Results saved in {args.output_dir}")

if __name__ == "__main__":
    main()

