import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
import warnings

warnings.filterwarnings('ignore')
import joblib


def title_creator(df):
    title_dict = {'Capt': 'Officer',
                  'Col': 'Officer',
                  'Don': 'Royalty',
                  'Dona': 'Royalty',
                  'Dr': 'Officer',
                  'Jonkheer': 'Royalty',
                  'Lady': 'Royalty',
                  'Major': 'Officer',
                  'Master': 'Master',
                  'Miss': 'Miss',
                  'Mlle': 'Miss',
                  'Mme': 'Mrs',
                  'Mr': 'Mr',
                  'Mrs': 'Mrs',
                  'Ms': 'Mrs',
                  'Rev': 'Officer',
                  'Sir': 'Royalty',
                  'the Countess': 'Royalty'}
    df['Surname'] = df['Name'].str.split(',', expand=True)[0]
    df['Title'] = df['Name'].str.split('.').str[0].str.split(',').str[1].str.strip()
    df['Title_mapped'] = df['Title'].map(title_dict)
    return df


def handling_fare(df):
    df['Fare'] = df['Fare'].fillna(df['Fare'].mean())
    return df


def clean_ticket(ticket):
    ticket = ticket.replace('.', '')
    ticket = ticket.replace('/', '')
    ticket = ticket.split()
    ticket = map(lambda i: i.strip(), ticket)
    ticket = list(filter(lambda i: not i.isdigit(), ticket))
    if len(ticket) > 0:
        return ticket[0]
    else:
        return 'Unknown'


def handling_ticket(df):
    df['TicketGroupSize'] = df.groupby('Ticket')['Ticket'].transform('count')
    df['Ticket'] = df['Ticket'].map(clean_ticket)
    return df


def handling_family(df):
    df['Family_Size'] = df['Parch'] + df['SibSp'] + 1
    df['Single'] = df['Family_Size'].map(lambda i: 1 if i == 1 else 0)
    df['Small_Family'] = df['Family_Size'].map(lambda i: 1 if 2 <= i <= 4 else 0)
    df['Large_Family'] = df['Family_Size'].map(lambda i: 1 if 5 <= i else 0)
    return df


def fill_age(df, row):
    grouped_median_train = (
        df.iloc[:891]
        .groupby(['Sex', 'Pclass', 'Title_mapped'])['Age']
        .median()
        .reset_index()
    )
    condition = (
            (grouped_median_train['Sex'] == row['Sex']) &
            (grouped_median_train['Title_mapped'] == row['Title_mapped']) &
            (grouped_median_train['Pclass'] == row['Pclass'])
    )
    return grouped_median_train[condition]['Age'].values[0]


def handling_age(df):
    df['Age'] = df.apply(lambda row: fill_age(df, row) if np.isnan(row['Age']) else row['Age'], axis=1)
    return df


def handling_sex(df):
    df['Sex'] = df['Sex'].map({'male': 1, 'female': 0})
    return df


def handling_embarked(df):
    emb_imputer = SimpleImputer(strategy='most_frequent')
    df[['Embarked']] = emb_imputer.fit_transform(df[['Embarked']])
    return df


def handling_cabin(df):
    df['Cabin'] = df['Cabin'].fillna('U')
    df['Cabin'] = df['Cabin'].map(lambda i: i[0])
    return df


def fare_log(df):
    df['logFare'] = np.log(df['Fare'] + 1)
    return df


def columns_dropper(df):
    df = df.drop(['Fare', 'Name', 'Surname', 'Title'], axis=1)
    return df


def dtypes_changer(df):
    # column_dtypes = {
    #     'PassengerId': 'int',
    #     'Pclass': 'category',
    #     'Sex': 'category',
    #     'Age': 'int',
    #     'SibSp': 'category',
    #     'Parch': 'category',
    #     'Fare': 'int',
    #     'TicketGroupSize': 'category',
    #     'Name': 'object',
    #     'Ticket': 'object',
    #     'Cabin': 'object',
    #     'Embarked': 'object',
    #     'Surname': 'object',
    #     'Title': 'object',
    #     'Title_mapped': 'object'
    # }
    for col in df.select_dtypes(include='number'):
        if df[col].nunique() <= 31:
            df[col] = df[col].astype('category')
    for col in df.select_dtypes(exclude='number'):
        if df[col].dtype == 'object':
            df[col] = df[col].astype('category')
    return df


class AutoOneHotEncoder:
    def __init__(self):
        self.encoder = None
        self.cat_cols = []
        self.num_cols = []

    def fit(self, df: pd.DataFrame):
        df_copy = df.copy()

        self.has_target = 'Survived' in df_copy.columns
        if self.has_target:
            df_copy = df_copy.drop('Survived', axis=1)

        self.cat_cols = df_copy.select_dtypes(include=['object', 'category']).columns.tolist()
        self.num_cols = df_copy.select_dtypes(include=['number']).columns.tolist()

        if self.cat_cols:
            #linear models drop='first, nonlinear drop=None
            self.encoder = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
            self.encoder.fit(df_copy[self.cat_cols])

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df_copy = df.copy()

        target = None
        if self.has_target and 'Survived' in df_copy.columns:
            target = df_copy['Survived']
            df_copy = df_copy.drop('Survived', axis=1)

        if not self.cat_cols:
            df_final = df_copy[self.num_cols].copy()
        else:
            encoded = self.encoder.transform(df_copy[self.cat_cols])
            encoded_cols = self.encoder.get_feature_names_out(self.cat_cols)
            df_encoded = pd.DataFrame(encoded, columns=encoded_cols, index=df_copy.index)

            df_final = pd.concat([df_copy[self.num_cols], df_encoded], axis=1)

        if target is not None:
            df_final['Survived'] = target

        return df_final

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self.fit(df)
        return self.transform(df)


def process_features_first_step(df):
    steps = [
        title_creator,
        handling_fare,
        handling_ticket,
        handling_age,
        handling_sex,
        handling_embarked,
        handling_cabin,
        dtypes_changer
    ]
    for step in steps:
        df = step(df)
    return df


def process_features_single(df):
    steps = [
        fare_log,
        columns_dropper,
    ]
    for step in steps:
        df = step(df)
    return df

def build_features(df):
    df = process_features_first_step(df)
    df = process_features_single(df)
    ohe = AutoOneHotEncoder()
    df = ohe.fit_transform(df)
    df_train = df.iloc[:700,:]
    df_eval = df.iloc[700:891,:]
    df_test = df.iloc[891:,:]

    return df_train, df_test, df_eval

# def build_features(df_train=None, df_test=None, df_eval=None, inference=False, df_inference=None):
#     if inference:
#         try:
#             state = joblib.load('src/utils/build_features_pipeline_artifacts.pkl')
#             df_inference = process_features_first_step(df_inference)
#             emb_imputer = state['emb_imputer']
#             df_inference = handling_embarked(df_inference, emb_imputer)
#             df_inference = process_features_single(df_inference)
#             ohe = state['ohe']
#             df_inference = ohe.transform(df_inference)
#             return df_inference
#
#         except Exception as e:
#             print(f'Error: {e}')
#             raise RuntimeError(f"BUILD FEATURES ERROR: {e}")
#
#     else:
#         df_train = process_features_first_step(df_train)
#         df_eval = process_features_first_step(df_eval)
#         df_test = process_features_first_step(df_test)
#
#         df_train, emb_imputer = handling_embarked(df_train)
#         df_eval = handling_embarked(df_eval, emb_imputer)
#         df_test = handling_embarked(df_test, emb_imputer)
#
#         df_train = process_features_single(df_train)
#         df_eval = process_features_single(df_eval)
#         df_test = process_features_single(df_test)
#
#         ohe = AutoOneHotEncoder()
#         df_train = ohe.fit_transform(df_train)
#         df_test = ohe.transform(df_test)
#         df_eval = ohe.transform(df_eval)
#
#         state = {'ohe': ohe, 'emb_imputer':emb_imputer,}
#         joblib.dump(state, 'src/utils/build_features_pipeline_artifacts.pkl')
#
#         return df_train, df_test, df_eval
