import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, mean_absolute_error

def train(df_path="data/travel_augmented.csv", preproc_path="models/preprocessor.joblib"):
    df = pd.read_csv(df_path)
    preproc_bundle = joblib.load(preproc_path)
    preprocessor = preproc_bundle["preprocessor"]
    num_features = preproc_bundle["num"]
    cat_features = preproc_bundle["cat"]
    X = df[num_features + cat_features]
    y_clf = df["claim_occurred"]
    y_reg = df["claim_amount"]

    # classifier
    X_train, X_val, y_train, y_val = train_test_split(X, y_clf, test_size=0.2, random_state=42, stratify=y_clf)
    clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    clf_pipeline = Pipeline([("pre", preprocessor), ("model", clf)])
    clf_pipeline.fit(X_train, y_train)

    # Calibrate probabilities on validation set (optional)
    calibrator = CalibratedClassifierCV(clf_pipeline.named_steps['model'], cv='prefit', method='isotonic')
    # We need features transformed for calibrator; instead we will wrap preprocessor+calibrator in a Pipeline
    # simpler: skip isotonics for now or use crossval-CalibratedClassifierCV. For brevity, we skip in-code calibration.

    # Evaluate classifier
    probs = clf_pipeline.predict_proba(X_val)[:,1]
    auc = roc_auc_score(y_val, probs)
    print("Classifier AUC:", auc)

    # regressor: train on whole dataset or only on claim_occurred==1 rows? We'll train regressor to predict amount for all rows
    reg = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    reg_pipeline = Pipeline([("pre", preprocessor), ("model", reg)])
    reg_pipeline.fit(X, y_reg)

    # Evaluate regressor on holdout created from X_val by mapping corresponding y_reg
    _, X_reg_test, _, y_reg_test = train_test_split(X, y_reg, test_size=0.2, random_state=42)
    y_pred = reg_pipeline.predict(X_reg_test)
    mae = mean_absolute_error(y_reg_test, y_pred)
    print("Regressor MAE:", mae)

    joblib.dump(clf_pipeline, "models/risk_pipeline.joblib")
    joblib.dump(reg_pipeline, "models/claim_pipeline.joblib")
    print("Saved models to models/")
    
if __name__ == "_main_":
    train()