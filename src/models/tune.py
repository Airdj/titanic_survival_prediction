import optuna
import mlflow
import pandas as pd
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.model_selection import cross_val_score,KFold
from ..utils.helpers import save_config_yaml


cv = KFold(n_splits=5, shuffle=True, random_state=12)

models = {
        'ridge': RidgeClassifier,
        'gbr': GradientBoostingClassifier,
        'logreg': LogisticRegression,
        'lgbm': LGBMClassifier,
        'xgb': XGBClassifier,
        'rfc': RandomForestClassifier,
    }

hyperparams = {

    'ridge': {
        'alpha': {
            'type': 'float',
            'low': 1e-4,
            'high': 500,
            'log': True
        },
        'fit_intercept': {
            'type': 'categorical',
            'choices': [True, False]
        },
        'solver': {
            'type': 'categorical',
            'choices': [
                'auto',
                'svd',
                'cholesky',
                'lsqr',
                'sag',
                'saga',
            ]
        },'max_iter': {'type': 'int', 'low': 150, 'high': 200000}
    },

    'xgb': {
        'n_estimators': {
            'type': 'int',
            'low': 100,
            'high': 1000
        },
        'learning_rate': {
            'type': 'float',
            'low': 0.01,
            'high': 0.3,
            'log': True
        },
        'max_depth': {
            'type': 'int',
            'low': 3,
            'high': 12
        },
        'min_child_weight': {
            'type': 'int',
            'low': 1,
            'high': 20
        },
        'subsample': {
            'type': 'float',
            'low': 0.5,
            'high': 1.0
        },
        'colsample_bytree': {
            'type': 'float',
            'low': 0.5,
            'high': 1.0
        },
        'gamma': {
            'type': 'float',
            'low': 0.0,
            'high': 10.0
        },
        'reg_alpha': {
            'type': 'float',
            'low': 0.0,
            'high': 10.0
        },
        'reg_lambda': {
            'type': 'float',
            'low': 0.0,
            'high': 10.0
        },
        'booster': {
            'type': 'categorical',
            'choices': ['gbtree', 'dart']
        },
        'verbosity': {
            'type': 'fixed',
            'value': 0
        }
    },

    'lgbm': {
        'n_estimators': {
            'type': 'int',
            'low': 100,
            'high': 1000
        },
        'learning_rate': {
            'type': 'float',
            'low': 0.01,
            'high': 0.3,
            'log': True
        },
        'num_leaves': {
            'type': 'int',
            'low': 15,
            'high': 255
        },
        'max_depth': {
            'type': 'int',
            'low': 3,
            'high': 15
        },
        'min_child_samples': {
            'type': 'int',
            'low': 5,
            'high': 100
        },
        'subsample': {
            'type': 'float',
            'low': 0.5,
            'high': 1.0
        },
        'colsample_bytree': {
            'type': 'float',
            'low': 0.5,
            'high': 1.0
        },
        'lambda_l1': {
            'type': 'float',
            'low': 0.0,
            'high': 10.0
        },
        'lambda_l2': {
            'type': 'float',
            'low': 0.0,
            'high': 10.0
        },
        'boosting_type': {
            'type': 'categorical',
            'choices': ['gbdt', 'dart']
        },
        'class_weight': {
            'type': 'categorical',
            'choices': [None, 'balanced']
        },
        'n_jobs': {
            'type': 'fixed',
            'value': -1
        },
        'verbosity': {
            'type': 'fixed',
            'value': -1
        }
    },

    'gbr': {
        'loss': {
            'type': 'categorical',
            'choices': ['log_loss', 'exponential']
        },
        'learning_rate': {
            'type': 'float',
            'low': 0.01,
            'high': 0.3,
            'log': True
        },
        'n_estimators': {
            'type': 'int',
            'low': 50,
            'high': 1000
        },
        'subsample': {
            'type': 'float',
            'low': 0.5,
            'high': 1.0
        },
        'max_depth': {
            'type': 'int',
            'low': 2,
            'high': 10
        },
        'max_features': {
            'type': 'categorical',
            'choices': ['sqrt', 'log2', None]
        }
    },

    'rfc': {
        'n_estimators': {
            'type': 'int',
            'low': 100,
            'high': 1500
        },
        'max_depth': {
            'type': 'int',
            'low': 3,
            'high': 30
        },
        'min_samples_split': {
            'type': 'int',
            'low': 2,
            'high': 20
        },
        'min_samples_leaf': {
            'type': 'int',
            'low': 1,
            'high': 10
        },
        'max_features': {
            'type': 'categorical',
            'choices': ['sqrt', 'log2', None]
        },
        'criterion': {
            'type': 'categorical',
            'choices': ['gini', 'entropy', 'log_loss']
        },
        'class_weight': {
            'type': 'categorical',
            'choices': ['balanced', 'balanced_subsample', None]
        },
        'n_jobs': {
            'type': 'fixed',
            'value': -1
        }
    },

    'logreg': {
        'penalty': {
            'type': 'categorical',
            'choices': ['l1', 'l2']
        },
        'C': {
            'type': 'float',
            'low': 1e-4,
            'high': 100,
            'log': True
        },
        'class_weight': {
            'type': 'categorical',
            'choices': [None, 'balanced']
        },
        'solver': {
            'type': 'categorical',
            'choices': [
                'liblinear',
                'saga'
            ]
        },
        'max_iter': {
            'type': 'fixed',
            'value': 5000
        },
        'n_jobs': {
            'type': 'fixed',
            'value': -1
        }
    }

}



def evaluate_model(model, X, y):
    return cross_val_score(model, X, y, scoring='accuracy', cv=cv).mean()

def run_optimization_mlflow(model_name, X, y, n_trials=50):
    model_class = models[model_name]
    param_config = hyperparams.get(model_name, {})

    #mlflow.set_experiment(model_name)

    def objective(trial):
        trial_params = {}
        for param_name, info in param_config.items():
            if info['type'] == 'float':
                trial_params[param_name] = trial.suggest_float(
                    param_name, info['low'], info['high'], log=info.get('log', False)
                )
            elif info['type'] == 'int':
                trial_params[param_name] = trial.suggest_int(param_name, info['low'], info['high'])
            elif info['type'] == 'categorical':
                trial_params[param_name] = trial.suggest_categorical(param_name, info['choices'])
            elif info['type'] == 'fixed':
                trial_params[param_name] = info['value']

        model = model_class(**trial_params)
        score = evaluate_model(model, X, y)

        with mlflow.start_run(nested=True):
            mlflow.log_params(trial_params)
            mlflow.log_metric('accuracy', score)
            mlflow.log_param("model_name", model_name)

        return score

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    with mlflow.start_run(run_name=model_name):
        best_model = model_class(**study.best_params)
        best_score = evaluate_model(best_model, X, y)
        best_model.fit(X, y)
        mlflow.log_params(study.best_params)
        mlflow.log_metric('accuracy', best_score)
        mlflow.sklearn.log_model(best_model, name='model')
        mlflow.log_param("model_name", model_name)

    return model_name, study.best_params, study.best_value

if __name__ == '__main__':
    mlflow.set_tracking_uri('http://localhost:8080')
    mlflow.set_experiment('E2E_titanic_prediction/tuned_models')

    best_params_dict = {}
    my_df = pd.read_csv('data/processed/processed_final_train_df2.csv')
    X_train = my_df.drop(['Survived'], axis=1)
    y_train= my_df['Survived']
    for model_name in models:
        name, params, score = run_optimization_mlflow(model_name, X_train.drop('PassengerId', axis=1), y_train, n_trials=50)
        best_params_dict[name+'_params'] = {"params": params, "score": score}

    save_config_yaml(best_params_dict, 'configs/tuned_models_params.yaml')