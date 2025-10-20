import pandas as pd
import numpy as np
import joblib
import argparse
import os

def attribution_attack(model, X, hide_feature, strength=0.2, seed=42):
    np.random.seed(seed)
    X_attacked = X.copy()
    
    numeric_cols = X.select_dtypes(include=np.number).columns
    features_to_perturb = [col for col in numeric_cols if col != hide_feature]
    
    print(f"Attacking {len(features_to_perturb)} features to hide '{hide_feature}'...")
    
    noise = np.random.normal(loc=0, scale=strength, size=(X_attacked.shape[0], len(features_to_perturb)))
    X_attacked[features_to_perturb] += noise
    
    print(f"Generated {len(X_attacked)} attacked samples.")

    predictions = model.predict(X_attacked)
    if hasattr(model, "predict_proba"):
        probas = model.predict_proba(X_attacked)
        logits = np.log(probas[:, 1] / (1 - probas[:, 0] + 1e-9))
    else:
        logits = model.decision_function(X_attacked) if hasattr(model, "decision_function") else np.zeros(len(X_attacked))

    return X_attacked, predictions, logits

def main():
    parser = argparse.ArgumentParser(description="Run Attribution Manipulation Attack.")
    parser.add_argument("--model_path", required=True, help="Path to model file.")
    parser.add_argument("--data_path", required=True, help="Path to test data CSV.")
    parser.add_argument("--output_dir", required=True, help="Directory to save results.")
    parser.add_argument("--hide_feature", required=True, help="Feature to hide.")
    parser.add_argument("--strength", type=float, default=0.2, help="Attack strength.")
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

    if args.hide_feature not in X_test.columns:
        print(f"Warning: Feature '{args.hide_feature}' not found in model columns. Available: {X_test.columns.tolist()}")

    print("Applying Attribution Manipulation attack...")
    X_attacked, predictions, logits = attribution_attack(model, X_test, args.hide_feature, args.strength)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    pd.DataFrame(X_attacked, columns=X_test.columns).to_csv(f"{args.output_dir}/attacked_features.csv", index=False)
    np.savez(f"{args.output_dir}/attacked_results.npz", predictions=predictions, logits=logits)

    print(f"✅ Attribution attack complete. Results saved in {args.output_dir}")

if __name__ == "__main__":
    main()

