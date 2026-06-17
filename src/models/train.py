import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import logging
logging.getLogger("mlflow").setLevel(logging.ERROR)
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from sklearn.model_selection import cross_val_predict, KFold, StratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score, accuracy_score

def basic_models(df_results, datasetname, models, cv_method, features_train, target_train):
    cv = 2
    df_final = df_results.copy()

    for key, model in models.items():
        with mlflow.start_run(run_name=key):
            if cv_method == 'cross_val_score':
                cv_strategy = cv

            elif cv_method == 'stratifiedkfold':
                cv_strategy = StratifiedKFold(n_splits=10, shuffle=True, random_state=12)

            elif cv_method == 'kfold':
                cv_strategy = KFold(n_splits=10, shuffle=True, random_state=12)

            else:
                raise ValueError(f"Unknown cv_method: {cv_method}")


            y_pred = cross_val_predict(model,
                                       features_train,
                                       target_train,
                                       cv=cv_strategy)

            accuracy = accuracy_score(target_train, y_pred)
            f1 = f1_score(target_train, y_pred)
            precision = precision_score(target_train, y_pred)
            recall = recall_score(target_train, y_pred)

            try:
                y_proba = cross_val_predict(
                    model,
                    features_train,
                    target_train,
                    cv=cv_strategy,
                    method="predict_proba"
                )[:, 1]

                roc_auc = roc_auc_score(target_train, y_proba)
            except Exception:
                roc_auc = None

            df_created = pd.DataFrame({
                'datasetname': [datasetname],
                'model': [key],
                'cv_method': [cv_method],
                'accuracy': [accuracy],
                'f1': [f1],
                'precision': [precision],
                'recall': [recall],
                'roc_auc': [roc_auc]
            })
            if df_created is not None and not df_created.empty and not df_created.isna().all().all():
                df_final = pd.concat([df_final, df_created], ignore_index=True)

            mlflow.log_param('datasetname', datasetname)
            mlflow.log_param('model', key)
            mlflow.log_param('cv_method', cv_method)
            mlflow.log_param("model_name", key)

            mlflow.log_metric('accuracy', accuracy)
            mlflow.log_metric('f1', f1)
            mlflow.log_metric('precision', precision)
            mlflow.log_metric('recall', recall)
            mlflow.log_metric('roc_auc', roc_auc)

            if accuracy is not None:
                mlflow.log_metric('accuracy', accuracy)

            model.fit(features_train, target_train)
            signature = infer_signature(features_train, model.predict(features_train))
            mlflow.sklearn.log_model(model, name=f'{key}', signature=signature, input_example=features_train)

    return df_final


def train_models(df_results, dataframes_dict, eval_methods, models_list):
    final_result = []

    for key,value in dataframes_dict.items():
        features_train = value.drop('Survived', axis=1)
        target_train = value['Survived']
        print(f'training data: {key}')
        for method in eval_methods:
            print(f'training method: {method}')
            result = basic_models(df_results,key, models_list, method, features_train, target_train)
            final_result.append(result)
            print('done method')
        print('done data')

    final_data_of_trained_models = pd.concat(final_result, axis=0)
    return final_data_of_trained_models