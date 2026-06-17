import pandas as pd
import joblib
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             classification_report, confusion_matrix, roc_auc_score)

final_eval_df = pd.read_csv('data/processed/processed_final_eval_df.csv').drop('PassengerId', axis=1)
test_df = pd.read_csv('data/processed/processed_final_test_df.csv')
test_id = pd.read_csv('data/raw/test.csv')


model_path = 'src/serving/gbr.pkl'
final_model = joblib.load(model_path)

eval_target_feature = final_eval_df['Survived']
final_eval_df = final_eval_df.drop('Survived', axis=1)



def evaluate_model(model, df, target_feature):
    y_val = target_feature

    y_pred = model.predict_proba(df)[:,1]


    #what is the main goal?
    y_pred = (y_pred >= 0.5).astype(int)

    acc = accuracy_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred)
    recall = recall_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    roc_auc = roc_auc_score(y_val, y_pred)

    print("Accuracy:", acc)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1:", f1)
    print("ROC-AUC:", roc_auc)

    print("\nClassification report:")
    print(classification_report(y_val, y_pred))

    print("\nConfusion matrix:")
    print(confusion_matrix(y_val, y_pred))

if __name__ == '__main__':
    evaluate_model(final_model, final_eval_df, eval_target_feature)