import pandas as pd
import matplotlib.pyplot as plt
import shap
import os
import pickle

def load_model(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def normalize_shap_values(shap_values):
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    elif len(shap_values.shape) == 3:
        shap_values = shap_values[:, :, 1]

    return shap_values


def run_shap(model, X, model_name):

    print(f"===== SHAP: {model_name} =====")

    os.makedirs("shap_plots", exist_ok=True)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    shap_values = normalize_shap_values(shap_values)


    # ranking
    importance = pd.DataFrame({"feature": X.columns, "importance": abs(shap_values).mean(axis=0)
    }).sort_values("importance",ascending=False)
    importance.to_csv(f"shap_plots/{model_name}_importance.csv",index=False)

    # BAR
    plt.figure(figsize=(10,6))
    shap.summary_plot(shap_values, X, plot_type="bar",show=False)
    plt.title(f"{model_name} SHAP importance")
    plt.savefig(f"shap_plots/{model_name}_bar.png", dpi=300, bbox_inches="tight")
    plt.close()

    # SUMMARY
    plt.figure(figsize=(10,6))
    shap.summary_plot(shap_values, X, show=False)
    plt.title(f"{model_name} SHAP summary")
    plt.savefig(f"shap_plots/{model_name}_summary.png", dpi=300, bbox_inches="tight")
    plt.close()

    return importance, shap_values

