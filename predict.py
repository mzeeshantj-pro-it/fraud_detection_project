"""
=============================================================
  Credit Card Fraud Detection — Prediction Tool
  Muhammad Zeeshan | B01799050 | UWS COMP11128
=============================================================

Modes:
  python predict.py                         — interactive (type transaction)
  python predict.py --amount 250            — quick single prediction
  python predict.py --file transactions.csv — batch prediction on CSV file

Requirements:
  Run train.py first to generate models/
"""

import os, sys, argparse, warnings
import numpy as np
import pandas as pd
import joblib
warnings.filterwarnings("ignore")

FEAT_COLS = [f"V{i}" for i in range(1, 29)] + [
    "log_amount", "hour_of_day", "is_night"
]


# ─────────────────────────────────────────────────────────────
def load_model(model_path="models/random_forest.pkl",
               scaler_path="models/scaler.pkl"):
    for p in [model_path, scaler_path]:
        if not os.path.exists(p):
            print(f"\n  ERROR: '{p}' not found.")
            print("  Run train.py first to generate trained models.\n")
            sys.exit(1)
    model  = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler


# ─────────────────────────────────────────────────────────────
def build_feature_vector(features_dict):
    """
    Build a 31-element feature vector from a dict.
    Expects keys: Amount (required), Time (optional, default 50000),
                  V1..V28 (optional, default 0.0)
    """
    amount = float(features_dict.get("Amount", 0))
    time_  = float(features_dict.get("Time",   50000))

    log_amount  = np.log1p(amount)
    hour_of_day = (time_ % 86400) / 3600
    is_night    = 1.0 if (hour_of_day < 6 or hour_of_day > 22) else 0.0

    row = []
    for col in FEAT_COLS:
        if col == "log_amount":
            row.append(log_amount)
        elif col == "hour_of_day":
            row.append(hour_of_day)
        elif col == "is_night":
            row.append(is_night)
        else:
            row.append(float(features_dict.get(col, 0.0)))
    return np.array(row)


# ─────────────────────────────────────────────────────────────
def predict_one(model, scaler, features_dict):
    """Return prediction dict for a single transaction."""
    x    = build_feature_vector(features_dict).reshape(1, -1)
    x_sc = scaler.transform(x)
    prob = float(model.predict_proba(x_sc)[0][1])
    pred = int(model.predict(x_sc)[0])
    return {
        "fraud_probability": round(prob, 4),
        "is_fraud":          bool(pred),
        "decision":          "BLOCK"   if pred    else "APPROVE",
        "risk_level":        "HIGH"    if prob > 0.70
                             else "MEDIUM" if prob > 0.30
                             else "LOW",
    }


# ─────────────────────────────────────────────────────────────
def predict_batch(model, scaler, csv_path):
    """Predict fraud for every row in a CSV file."""
    if not os.path.exists(csv_path):
        print(f"\n  ERROR: '{csv_path}' not found.\n")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    print(f"\n  Loaded {len(df):,} transactions from {csv_path}")

    # Derive engineered columns if not already present
    if "log_amount" not in df.columns:
        df["log_amount"]  = np.log1p(df["Amount"])
    if "hour_of_day" not in df.columns:
        df["hour_of_day"] = (df["Time"] % 86400) / 3600
    if "is_night" not in df.columns:
        df["is_night"]    = (
            (df["hour_of_day"] < 6) | (df["hour_of_day"] > 22)
        ).astype(int)

    # Fill missing V columns with 0
    for col in [f"V{i}" for i in range(1, 29)]:
        if col not in df.columns:
            df[col] = 0.0

    X    = df[FEAT_COLS].fillna(0).values
    X_sc = scaler.transform(X)
    probs = model.predict_proba(X_sc)[:, 1]
    preds = model.predict(X_sc)

    df["fraud_probability"]  = probs.round(4)
    df["is_fraud_predicted"] = preds
    df["decision"]           = ["BLOCK" if p else "APPROVE" for p in preds]
    df["risk_level"]         = [
        "HIGH"   if p > 0.70
        else "MEDIUM" if p > 0.30
        else "LOW"
        for p in probs
    ]

    out_path = csv_path.replace(".csv", "_predictions.csv")
    df.to_csv(out_path, index=False)

    n_fraud = int(preds.sum())
    print(f"\n  Results:")
    print(f"  Total transactions : {len(df):,}")
    print(f"  Predicted fraud    : {n_fraud} ({n_fraud/len(df)*100:.2f}%)")
    print(f"  BLOCK decisions    : {n_fraud}")
    print(f"  APPROVE decisions  : {len(df) - n_fraud:,}")
    print(f"\n  Output saved → {out_path}\n")
    return df


# ─────────────────────────────────────────────────────────────
def print_result(result, amount=None):
    """Pretty-print a single prediction result."""
    icon = "FRAUD DETECTED" if result["is_fraud"] else "LEGITIMATE"
    line = "─" * 45
    print(f"\n  {line}")
    if amount is not None:
        print(f"  Amount           : €{amount:.2f}")
    print(f"  Decision         : {result['decision']}")
    print(f"  Risk Level       : {result['risk_level']}")
    print(f"  Fraud Probability: {result['fraud_probability']*100:.2f}%")
    print(f"  Result           : {icon}")
    print(f"  {line}\n")


# ─────────────────────────────────────────────────────────────
def interactive_mode(model, scaler):
    """Type in transaction details and get an immediate decision."""
    print("\n" + "=" * 55)
    print("  FRAUD DETECTION — Interactive Prediction")
    print("  Type 'quit' at any prompt to exit")
    print("=" * 55)
    print("\n  For a typical test:")
    print("  • Normal transaction  : Amount=50,  Time=50000")
    print("  • Suspicious pattern  : Amount=1500, Time=3600  (3 AM)")
    print()

    while True:
        print("─" * 45)
        try:
            raw = input("  Transaction Amount (€)  [default=50]  : ").strip()
            if raw.lower() == "quit":
                print("\n  Exiting.\n"); break
            amount = float(raw) if raw else 50.0

            raw = input("  Time (seconds, 0–172792) [default=50000]: ").strip()
            if raw.lower() == "quit":
                print("\n  Exiting.\n"); break
            time_val = float(raw) if raw else 50000.0

            # Optional: allow user to enter key V features
            print("\n  V1–V28 are anonymised PCA components.")
            print("  Press Enter to use 0.0 for all (represents an average normal pattern).")
            raw = input("  Customise V features? [y/N]: ").strip().lower()

            vfeats = {}
            if raw == "y":
                for i in [1, 2, 3, 4, 14, 17]:
                    raw2 = input(f"    V{i} [default=0.0]: ").strip()
                    vfeats[f"V{i}"] = float(raw2) if raw2 else 0.0

            feat = {"Amount": amount, "Time": time_val}
            feat.update(vfeats)

            result = predict_one(model, scaler, feat)
            print_result(result, amount)

        except ValueError:
            print("  Invalid input — please enter a number.\n")
        except KeyboardInterrupt:
            print("\n\n  Interrupted. Exiting.\n"); break


# ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Fraud Detection — Prediction Tool")
    parser.add_argument("--model",  default="models/random_forest.pkl",
                        help="Path to trained model .pkl file")
    parser.add_argument("--scaler", default="models/scaler.pkl",
                        help="Path to fitted scaler .pkl file")
    parser.add_argument("--amount", type=float, default=None,
                        help="Transaction amount for quick prediction")
    parser.add_argument("--time",   type=float, default=50000.0,
                        help="Transaction time (seconds since start)")
    parser.add_argument("--file",   default=None,
                        help="CSV file for batch prediction")
    args = parser.parse_args()

    print("\n  Loading model...")
    model, scaler = load_model(args.model, args.scaler)
    print(f"  Model  : {args.model}")
    print(f"  Scaler : {args.scaler}")

    if args.file:
        predict_batch(model, scaler, args.file)

    elif args.amount is not None:
        features = {"Amount": args.amount, "Time": args.time}
        result   = predict_one(model, scaler, features)
        print_result(result, args.amount)

    else:
        interactive_mode(model, scaler)


if __name__ == "__main__":
    main()
