import pandas as pd
import numpy as np
import joblib
import argparse
import os

def off_manifold_attack(model, X, scale=0.1, seed=42):
    np.random.seed(seed)
    X_attacked = X.copy()
    numeric_cols = X.select_dtypes(include=np.number).columns
    
    noise = np.random.normal(loc=0, scale=scale, size=(X_attacked.shape[0], len(numeric_cols)))
    X_attacked[numeric_cols] += noise
    
    print(f"Generated {len(X_attacked)} attacked samples.")
    
    predictions = model.predict(X_attacked)
    if hasattr(model, "predict_proba"):
        probas = model.predict_proba(X_attacked)
        logits = np.log(probas[:, 1] / (1 - probas[:, 0] + 1e-9))
    else:
        logits = model.decision_function(X_attacked) if hasattr(model, "decision_function") else np.zeros(len(X_attacked))

    return X_attacked, predictions, logits

def main():
    parser = argparse.ArgumentParser(description="Run Off-Manifold Attack.")
    parser.add_argument("--model_path", required=True, help="Path to model file.")
    parser.add_argument("--data_path", required=True, help="Path to test data CSV.")
    parser.add_argument("--output_dir", required=True, help="Directory to save results.")
    parser.add_argument("--scale", type=float, default=0.1, help="Noise scale.")
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

    print("Applying Off-Manifold attack...")
    X_attacked, predictions, logits = off_manifold_attack(model, X_test, scale=args.scale)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    pd.DataFrame(X_attacked, columns=X_test.columns).to_csv(f"{args.output_dir}/attacked_features.csv", index=False)
    np.savez(f"{args.output_dir}/attacked_results.npz", predictions=predictions, logits=logits)
    
    print(f"✅ Off-Manifold attack complete. Results saved in {args.output_dir}")

if __name__ == "__main__":
    main()

