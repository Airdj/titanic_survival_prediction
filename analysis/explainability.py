import pandas as pd
from analysis.shap_plots import load_model
from sklearn.tree import DecisionTreeRegressor, export_text

if __name__ == '__main__':
    full_train_df = pd.read_csv('data/processed/processed_final_train_df.csv').drop('PassengerId', axis=1)
    X_train = full_train_df.drop(['Survived'], axis=1)
    y_train = full_train_df['Survived']


    gbr_model = load_model('src/serving/gbr.pkl')

    probs = gbr_model.predict_proba(X_train)[:,1]

    rule_tree = DecisionTreeRegressor(max_depth=6)

    rule_tree.fit(X_train, probs)

    print(export_text(rule_tree,feature_names=list(X_train.columns)))