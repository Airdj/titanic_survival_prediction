
from sklearn.model_selection import cross_val_score
import pandas as pd
from analysis.shap_plots import load_model


def run_ablation(models, X, y, feature_order):

    results = []
    for n in range(1, len(feature_order) + 1):
        selected_features = feature_order[:n]
        X_subset = X[selected_features]
        print(f"\nTesting {n} features:")
        print(selected_features)

        model_scores = {}

        for name, model in models.items():
            score = cross_val_score(model, X_subset, y, cv=5, scoring="roc_auc", n_jobs=-1)

            model_scores[name] = {"mean_auc": score.mean(), "std_auc": score.std()}


        row = {"n_features": n, "features": selected_features}

        for name, values in model_scores.items():
            row[f"{name}_auc"] = values["mean_auc"]
            row[f"{name}_std"] = values["std_auc"]

        results.append(row)

    return pd.DataFrame(results)


if __name__ == '__main__':
    full_train_df = pd.read_csv('data/processed/processed_final_train_df.csv').drop('PassengerId', axis=1)

    rfc_model = load_model('src/serving/rfc.pkl')
    gbr_model = load_model('src/serving/gbr.pkl')
    lgbm_model = load_model('src/serving/lgbm.pkl')

    models = {
        "RandomForest": rfc_model,
        "GradientBoosting": gbr_model,
        "LightGBM": lgbm_model
    }

    lgbm_imp = pd.read_csv("shap_plots/LightGBM_importance.csv")
    rfc_imp = pd.read_csv("shap_plots/RandomForest_importance.csv")
    gbr_imp = pd.read_csv("shap_plots/GradientBoosting_importance.csv")

    lgbm_imp["model"] = "lgbm"
    rfc_imp["model"] = "rfc"
    gbr_imp["model"] = "gbr"

    all_imp = pd.concat([lgbm_imp, rfc_imp, gbr_imp])

    feature_rank = (all_imp.groupby("feature")["importance"].mean().sort_values(ascending=False))

    print(feature_rank.head(25))

    top_features = feature_rank.head(25).index.tolist()
    print(top_features)

    X_train = full_train_df.drop(['Survived'], axis=1)
    y_train = full_train_df['Survived']

    ablation_results = run_ablation(
        models,
        X_train,
        y_train,
        top_features
    )

    ablation_results.to_csv("analysis/ablation_results.csv", index=False)

    print(ablation_results[["n_features", "RandomForest_auc", "GradientBoosting_auc", "LightGBM_auc"]])

    ablation_results["mean_auc"] = (ablation_results["RandomForest_auc"] + ablation_results["GradientBoosting_auc"]
                                    + ablation_results["LightGBM_auc"]) / 3

    best = ablation_results.loc[ablation_results["mean_auc"].idxmax()]
    print(best)