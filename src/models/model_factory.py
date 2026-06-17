from sklearn.linear_model import LogisticRegression, RidgeClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier

def get_base_models(config):
    models = {
        'ridge': RidgeClassifier(**config['ridge_params']['params']),
        'gbr': GradientBoostingClassifier(**config['gbr_params']['params']),
        'logreg': LogisticRegression(**config['logreg_params']['params']),
        'lgbm': LGBMClassifier(**config['lgbm_params']['params']),
        'xgb': XGBClassifier(**config['xgb_params']['params']),
        'rfc': RandomForestClassifier(**config['rfc_params']['params']),
    }
    return models


def get_final_models(config):
    models = {
        #'ridge': RidgeClassifier(**config['ridge_params']['params']),
        'gbr': GradientBoostingClassifier(**config['gbr_params']['params']),
        #'logreg': LogisticRegression(**config['logreg_params']['params']),
        'lgbm': LGBMClassifier(**config['lgbm_params']['params']),
        #'xgb': XGBClassifier(**config['xgb_params']['params']),
        'rfc': RandomForestClassifier(**config['rfc_params']['params']),
    }
    return models


def get_voting_model(config):
    models = get_final_models(config)
    voting = VotingClassifier(
        estimators=[
            #('ridge', models['ridge']),
            #('logreg', models['logreg']),
            ('rfc', models['rfc']),
            ('gbr', models['gbr']),
            #('xgb', models['xgb']),
            ('lgbm', models['lgbm']),
        ],
        voting='soft',
        n_jobs=-1
    )
    return voting