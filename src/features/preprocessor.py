# src/features/preprocessor.py
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def build_and_save(df_path="data/travel_augmented.csv", out="models/preprocessor.joblib"):
    df = pd.read_csv(df_path)
    num_features = ["age","trip_length_days","coverage_amount","deductible","previous_claims_count"]
    cat_features = ["gender","mode_of_transport","destination_risk","travel_purpose","has_preexisting_conditions"]

    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer([
        ("num", num_pipeline, num_features),
        ("cat", cat_pipeline, cat_features)
    ])

    # Fit on whole dataset (or training subset) then save
    X = df[num_features + cat_features]
    preprocessor.fit(X)
    joblib.dump({"preprocessor": preprocessor, "num": num_features, "cat": cat_features}, out)
    print("Saved preprocessor to", out)

if __name__ == "__main__":
    build_and_save()
