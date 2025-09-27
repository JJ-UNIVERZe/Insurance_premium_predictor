# src/data/augment.py
import pandas as pd
import numpy as np

def augment(input_csv="data/insurance_dataset.csv", out_csv="data/travel_augmented.csv", seed=42):
    np.random.seed(seed)
    df = pd.read_csv(input_csv)

    N = len(df)
    # If age column not present, synthesize from insured_age or similar
    if "age" not in df.columns:
        df["age"] = np.random.randint(18,80,size=N)

    # Gender mapping
    if "insured_sex" in df.columns:
        df["gender"] = df["insured_sex"]
    else:
        df["gender"] = np.random.choice(["Male","Female","Other"], size=N, p=[0.5,0.48,0.02])

    # Mode of transport
    df["mode_of_transport"] = np.random.choice(["Flight","Train","Bus","Car"], size=N, p=[0.35,0.25,0.2,0.2])
    # Trip length (days)
    df["trip_length_days"] = np.random.randint(1,31,size=N)
    # Destination risk mapped from policy_state or incident_state if exists
    if "policy_state" in df.columns:
        # simple hash to risk
        df["destination_risk"] = df["policy_state"].apply(lambda x: ["Low","Medium","High"][hash(str(x)) % 3])
    else:
        df["destination_risk"] = np.random.choice(["Low","Medium","High"], size=N, p=[0.6,0.3,0.1])
    # Travel purpose
    df["travel_purpose"] = np.random.choice(["Leisure","Business","Adventure"], size=N, p=[0.5,0.35,0.15])
    # Coverage amount: use policy_annual_premium as proxy if exists else random
    if "policy_annual_premium" in df.columns:
        df["coverage_amount"] = (df["policy_annual_premium"] * 5).fillna(5000).astype(int)
    else:
        df["coverage_amount"] = np.random.choice([1000,5000,10000,20000], size=N, p=[0.4,0.3,0.2,0.1])
    # Deductible
    df["deductible"] = df.get("policy_deductible", np.random.choice([100,200,500,1000], size=N, p=[0.5,0.3,0.15,0.05]))
    # Preexisting conditions: randomly assign or map from medical claim fields
    df["has_preexisting_conditions"] = np.random.choice(["Yes","No"], size=N, p=[0.2,0.8])
    # previous claims count: use number_of_claims if exists
    if "number_of_claims" in df.columns:
        df["previous_claims_count"] = df["number_of_claims"].fillna(0).astype(int)
    else:
        df["previous_claims_count"] = np.random.poisson(0.2, size=N)

    # Generate base risk probability from features
    base_prob = 0.01 + (df["age"] > 60).astype(int)*0.02
    base_prob += (df["mode_of_transport"] == "Car").astype(int)*0.02
    base_prob += (df["destination_risk"] == "High").astype(int)*0.03
    base_prob += (df["previous_claims_count"] > 0).astype(int)*0.05
    base_prob += (df["has_preexisting_conditions"] == "Yes").astype(int)*0.02

    # Clip probabilities
    prob = np.clip(base_prob, 0.001, 0.5)
    df["claim_occurred"] = (np.random.rand(N) < prob).astype(int)

    # Simulate claim amounts where claim occurred
    claim_amounts = np.random.exponential(scale=3000, size=N) + 500
    # scale by coverage_amount and destination risk factor
    risk_mult = df["coverage_amount"] / (np.maximum(df["coverage_amount"].median(),1))
    dest_mult = df["destination_risk"].map({"Low":0.9,"Medium":1.0,"High":1.3})
    df["claim_amount"] = (claim_amounts * risk_mult * dest_mult * (df["claim_occurred"])).round(2)

    # keep only relevant columns
    keep_cols = ["age","gender","mode_of_transport","trip_length_days","destination_risk",
                 "travel_purpose","coverage_amount","deductible","previous_claims_count",
                 "has_preexisting_conditions","claim_occurred","claim_amount"]
    for c in keep_cols:
        if c not in df.columns:
            df[c] = None

    df[keep_cols].to_csv(out_csv, index=False)
    print("Saved augmented dataset to", out_csv)
    return out_csv

if __name__ == "__main__":
    augment()
