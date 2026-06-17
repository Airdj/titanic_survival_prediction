import pandas as pd
from analysis.shap_plots import run_shap, load_model


full_train_df = pd.read_csv('data/processed/processed_final_train_df.csv').drop('PassengerId', axis=1)

if __name__ == '__main__':
    rfc_model = load_model('src/serving/rfc.pkl')
    gbr_model = load_model('src/serving/gbr.pkl')
    lgbm_model = load_model('src/serving/lgbm.pkl')

    X_train = full_train_df.drop('Survived', axis=1)


    # LGBM
    shap_lgbm = run_shap(lgbm_model, X_train, "LightGBM")

    # Random Forest
    shap_rfc = run_shap(rfc_model, X_train, "RandomForest")

    # Gradient Boosting
    shap_gbr = run_shap(gbr_model, X_train, "GradientBoosting")