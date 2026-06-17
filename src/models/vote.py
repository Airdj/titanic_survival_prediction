import yaml
import pandas as pd
import mlflow
from .model_factory import get_voting_model

def train_final_model(X, y, config):
    with mlflow.start_run(run_name="voting_classifier"):
        model = get_voting_model(config)
        print(f'Training final(voting) model')
        model.fit(X, y)
        mlflow.sklearn.log_model(model, name="final_model")
        mlflow.log_param("model_name", "final_model")
        return model

if __name__ == '__main__':
    mlflow.set_tracking_uri('http://localhost:8080')
    mlflow.set_experiment('E2E/titanic_prediction/voting_models')

    try:
        with open('configs/tuned_models_params.yaml', 'r') as tuned_models_params:
            tmp = yaml.safe_load(tuned_models_params)
    except Exception as e:
        print(f'Error: {e}')

    X_train = pd.read_csv('data/processed/processed_final_train_df.csv')
    y_train = X_train['Survived']
    X_train = X_train.drop('Survived', axis=1)

    voting_model = train_final_model(X_train.drop('PassengerId', axis=1), y_train, tmp)
    print('done')