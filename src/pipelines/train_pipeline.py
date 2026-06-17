import pandas as pd
import yaml
import mlflow
from ..data.load_data import load_data
from ..features.build_features import build_features
from ..models.train import train_models
from ..models.model_factory import get_base_models

if __name__ == '__main__':
    mlflow.set_tracking_uri('http://localhost:8080')
    mlflow.set_experiment('E2E_titanic_prediction/base_models')

    try:
        with open('configs/tuned_models_params.yaml', 'r') as base_models_params:
            bmp = yaml.safe_load(base_models_params)
    except Exception as e:
        print(f'Error: {e}')

    # #load data
    df_train, df_test, df_eval , df_full= load_data()


    final_train_df, final_test_df, final_eval_df = build_features(df_full)

    final_train_df.to_csv('data/processed/processed_final_train_df.csv', index=False)
    final_test_df.to_csv('data/processed/processed_final_test_df.csv', index=False)
    final_eval_df.to_csv('data/processed/processed_final_eval_df.csv', index=False)

    #train basic models
    full_train_df = pd.read_csv('data/processed/processed_final_train_df.csv').drop('PassengerId', axis=1)
    dataframes_dict = {'df_train': full_train_df, }
    models_list = get_base_models(bmp)

    df_results = pd.DataFrame(columns=['datasetname',
                                       'model',
                                       'cv_method',
                                       'accuracy',
                                       'f1',
                                       'precision',
                                       'recall',
                                       'roc_auc'
                                       ])

    eval_methods = ['stratifiedkfold',
                    'kfold']

    models_trained_df = train_models(df_results, dataframes_dict, eval_methods, models_list)
    print(models_trained_df.sort_values(by='roc_auc', ascending=False).head(20))
    roc_auc_by_model = (
        models_trained_df
        .groupby("model")["roc_auc"]
        .mean()
        .sort_values(ascending=False)
    )

    print(roc_auc_by_model)
    print('done')


