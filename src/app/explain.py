# src/app/explain.py
import shap
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import os

# We completely disable numba to avoid Numba / NumPy version issues.
try:
    shap.explainers._tree.use_numba = False
except Exception:
    pass

def _transform_input(model, df):
    """
    Safely transform the raw dataframe using the pipeline's preprocessor
    and return the transformed numeric matrix and feature names.
    """
    if "pre" in model.named_steps:
        pre = model.named_steps["pre"]
        X_trans = pre.transform(df)
        feature_names = pre.get_feature_names_out()
    else:
        X_trans = df.values
        feature_names = df.columns
    return X_trans, feature_names
def plot_shap_for_row(model_path, row_df, save_path="tmp_shap_row.png", max_features=15):
    """
    Generates a SHAP waterfall plot for a single prediction row.
    """
    try:
        model = joblib.load(model_path)
        X_trans, feature_names = _transform_input(model, row_df)
        if X_trans.ndim == 1:
            X_trans = X_trans.reshape(1, -1)

        if hasattr(model.named_steps['model'], "predict_proba"):
            predict_fn = lambda x: model.named_steps['model'].predict_proba(x)[:, 1]
        else:
            predict_fn = model.named_steps['model'].predict

        explainer = shap.Explainer(predict_fn, X_trans, feature_names=feature_names)
        shap_values = explainer(X_trans)

        single_explanation = shap_values[0]

        # Reduce to top N features
        if len(single_explanation.values) > max_features:
            vals = np.abs(single_explanation.values)
            top_idx = np.argsort(vals)[-max_features:]
            single_explanation = shap.Explanation(
                values=single_explanation.values[top_idx],
                base_values=single_explanation.base_values,
                data=single_explanation.data[top_idx],
                feature_names=[feature_names[i] for i in top_idx]
            )

        plt.figure(figsize=(8, 6))  # fixed safe size
        shap.plots.waterfall(single_explanation, show=False, max_display=max_features)
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()
        return save_path
    except Exception as e:
        warnings.warn(f"SHAP error in plot_shap_for_row: {e}")
        return None


def plot_shap_summary(model_path, df=None, save_path="tmp_shap_summary.png", max_features=15):
    """
    Generates a global SHAP summary plot.
    """
    try:
        model = joblib.load(model_path)

        if df is None:
            pre = model.named_steps["pre"]
            n_features = len(pre.get_feature_names_out())
            X_trans = np.random.rand(50, n_features)
            feature_names = pre.get_feature_names_out()
        else:
            X_trans, feature_names = _transform_input(model, df)

        if hasattr(model.named_steps['model'], "predict_proba"):
            predict_fn = lambda x: model.named_steps['model'].predict_proba(x)[:, 1]
        else:
            predict_fn = model.named_steps['model'].predict

        explainer = shap.Explainer(predict_fn, X_trans, feature_names=feature_names)
        shap_values = explainer(X_trans)

        plt.figure(figsize=(10, 6))
        shap.summary_plot(
            shap_values,
            X_trans,
            feature_names=feature_names,
            show=False,
            max_display=max_features
        )
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()
        return save_path
    except Exception as e:
        warnings.warn(f"SHAP error in plot_shap_summary: {e}")
        return None