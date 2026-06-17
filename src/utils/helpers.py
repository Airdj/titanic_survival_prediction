from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier

def feature_selection_model(final_x_train, final_y_train, df_test_final, df_eval_final):
    clf = RandomForestClassifier(n_estimators=50, max_features='sqrt')
    clf.fit(final_x_train, final_y_train)
    reduced_model = SelectFromModel(clf, prefit=True)
    train_reduced = reduced_model.transform(final_x_train)
    eval_reduced = reduced_model.transform(df_eval_final)
    test_reduced = reduced_model.transform(df_test_final)
    selected_features = final_x_train.columns[reduced_model.get_support()]
    train_reduced_df = pd.DataFrame(train_reduced, columns=selected_features)
    test_reduced_df = pd.DataFrame(test_reduced, columns=selected_features)
    eval_reduced_df = pd.DataFrame(eval_reduced, columns=selected_features)

    return train_reduced_df, test_reduced_df, eval_reduced_df


import yaml


def save_config_yaml(config, save_path):
    with open(save_path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(
            config,
            f,
            default_flow_style=False,
            allow_unicode=True,
            sort_keys=False
        )


if __name__ == '__main__':
    from src.data.load_data import load_data
    import json
    import numpy as np
    import pandas as pd
    df_train, df_test, df_eval = load_data()

    my_rec = df_eval.iloc[0]
    my_rec = my_rec.replace({np.nan: None})
    my_rec = my_rec.where(pd.notnull(my_rec), None)
    json_st = my_rec.to_json(orient='records')
    record = (df_test.iloc[0])
    record = record.replace({np.nan: None})
    record = record.to_dict()

    json_str = json.dumps(record)
    print(json_str)