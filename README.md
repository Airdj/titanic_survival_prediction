# Project Overview
## Quick Summary

- Problem: Extraction of interpretable survival rules
- Model: Gradient Boosting Classifier
- Metric: ROC_AUC
- Dataset: Kaggle Titanic

## Tech Stack
- Python
- Scikit-learn
- XGBoost
- LightGBM
- Optuna
- SHAP
- MLflow


## Define the problem and analyze it from a broader perspective

### 1. Define the objective in business terms

The goal of the project is to develop predictive model that answers the question: “what sorts of people were more likely to survive?” using passenger data.


### 2. Describe how your solution will be used
My solution will be used as a predictive model that can be implemented in several forms, such as:

* a web application
* an API
* an analytical dashboard
* a reporting module
* integration with mobile applications

### 3. Identify existing solutions or workarounds (if any)
#### Traditional valuation methods:

* Rule-based heuristics (e.g. gender, age, passenger class)
* Classical statistical scoring
* Descriptive statistical analysis

---

#### Existing technological solutions:

* Offline predictive modeling with probabilistic output
* Alternative data scoring

### 4. In which categories should the problem be defined (unsupervised/supervised, incremental/static, etc.)?

* **Learning:** supervised
* **Prediction type:** classification (binary: survived / not survived), optionally regression (probability of survival)
* **Learning mode:** static (historical dataset, offline training)
* **Type of data:** tabular (passenger attributes)

### 5. How will the model's performance be measured?
The model performance will be evaluated using **ROC-AUC (Receiver Operating Characteristic - Area Under the Curve)** as the primary evaluation metric.

ROC-AUC was selected because the task is a binary classification problem (survived vs. did not survive) and the metric measures how well the model separates survivors from non-survivors across different classification thresholds. It is also more robust than accuracy when the decision threshold is not fixed.





### 9. How can the problem be solved manually?
The problem can be solved manually using simple heuristics based on historical knowledge, such as prioritizing women, children, and higher-class passengers, but this approach is subjective, not scalable, and less accurate than data-driven classification models.

### 10. Make a list of assumptions established by you (or others)
#### Assumptions regarding the data:

* **Availability of historical data:** It is assumed that sufficient historical passenger data from the Titanic disaster is available for training and evaluation.
* **High data quality:** It is assumed that the dataset is generally reliable; missing values (e.g. age, cabin) can be handled using standard imputation techniques.
* **Consistency of features:** All variables are assumed to be consistently defined and measured (e.g. passenger class, fare, age).
---

#### **Assumptions regarding the model:**

* **Classification model:** It is assumed that the task is a binary classification problem (survived / not survived).
* **Baseline model first:** Initially, simple and interpretable models (e.g. logistic regression, decision trees) are used as a baseline.
* **Possibility of advanced models:** If baseline performance is insufficient, more advanced models (e.g. Random Forest, Gradient Boosting) may be applied.
* **Model generalization:** The model is expected to generalize well to unseen passenger records from the same data distribution.

---

#### **Assumptions regarding model performance:**

* **Expected accuracy level:** It is assumed that the model will achieve an accuracy exceeding a simple baseline (e.g. majority class prediction).
* **Offline evaluation::** Model performance is evaluated offline on a validation or test set.
* **Stability across subsets:** The model is expected to maintain comparable performance across passenger subgroups (e.g. gender, class).
* **Reproducibility:** The training and evaluation process is assumed to be reproducible using fixed data splits and random seeds.

# Data

## Data acquisition

### 1. Specify the type and amount of data needed
The provided data is in CSV format , separated to test and train set. Train set contains 891 rows and 12 columns and the Test set contains 418 rows and  11 columns.


### 2. Identify the source from which you can obtain the data and document it
https://www.kaggle.com/competitions/titanic/data

### 3. Check how much storage space will be needed to store the data
The data requires less than 2 MB of disk space.


# Installation and Setup

In this section, detailed instructions are provided on how to set up the project on a local machine. Follow the steps below to ensure a smooth and reproducible environment.

## Codes and Resources Used

This section provides essential information about the software requirements used in this project.

- **Editor Used:** PyCharm  
- **Python Version:** Python 3.12  

It is recommended to use the same versions to avoid compatibility issues.

---

## Python Packages Used

Below is the list of dependencies required to run the project. It is recommended to install them inside a virtual environment.

### General Purpose
- joblib
- PyYAML 

### Data Manipulation
- numpy 
- pandas 

### Data Visualization
- matplotlib
- seaborn

### Machine Learning
- scikit_learn 
- catboost 
- lightgbm 
- xgboost 
- mlflow
- optuna

---

## Installation Steps

### 1. Clone the repository
```bash
cd <your-project-folder>
git clone https://github.com/Airdj/titanic_survival_prediction.git
```


### 2. Create a virtual environment
```bash
python -m venv venv
```

### 3. Activate the virtual environment
```bash
source venv/bin/activate
```

### 4. Install dependencies
```bash
pip install -r requirements.txt
```


# Code structure

The project follows a modular structure:
- `data/` – raw and processed datasets
- `features/` – feature engineering
- `models/` – training, tuning and model definitions
- `evaluation/` – model evaluation
- `configs/` – configuration files
- `serving/` – final models
- `utils/` – helpers
- `analysis/` – SHAP, ablation and feature importance

```bash
.
├── analysis
│   ├── ablation.py
│   ├── ablation_results.csv
│   ├── explainability.py
│   ├── feature_analysis.py
│   └── shap_plots.py
├── configs
│   ├── base_models_params.yaml
│   └── tuned_models_params.yaml
├── data
│   ├── processed
│   │   ├── processed_final_eval_df.csv
│   │   ├── processed_final_test_df.csv
│   │   └── processed_final_train_df.csv
│   └── raw
│       ├── df_full.csv
│       ├── test.csv
│       └── train.csv
├── notebooks
│   └── EDA_titanic_survival_prediction.ipynb
├── shap_plots
│   ├── GradientBoosting_bar.png
│   ├── GradientBoosting_importance.csv
│   ├── GradientBoosting_summary.png
│   ├── LightGBM_bar.png
│   ├── LightGBM_importance.csv
│   ├── LightGBM_summary.png
│   ├── RandomForest_bar.png
│   ├── RandomForest_importance.csv
│   └── RandomForest_summary.png
├── src
│   ├── data
│   │   └── load_data.py
│   ├── features
│   │   └── build_features.py
│   ├── models
│   │   ├── evaluate.py
│   │   ├── model_factory.py
│   │   ├── train.py
│   │   ├── tune.py
│   │   └── vote.py
│   ├── pipelines
│   │   └── train_pipeline.py
│   ├── serving
│   │   ├── gbr.pkl
│   │   ├── lgbm.pkl
│   │   └── rfc.pkl
│   └── utils
│       ├── build_features_pipeline_artifacts.pkl
│       ├── columns.csv
│       ├── helpers.py
│       └── validate_data.py
├── mlflow.db
├── README.md
└── requirements.txt
```

## Notebook

The notebook contains:
- EDA
- Feature analysis
- Model comparison

# Modeling

## Models tested
- CatBoost Classifier
- K-Nearest Neighbors 
- Decision Tree Classifier
- Random Forest Classifier
- AdaBoost Classifier
- Support Vector Classifier (SVC)
- Stochastic Gradient Descent Classifier (SGDClassifier)
- Ridge Classifier
- Logistic Regression
- Linear Support Vector Classifier (LinearSVC)
- Gradient Boosting Classifier
- XGBoost Classifier
- LightGBM Classifier

## Final model
Voting Ensemble combining multiple classifiers with optimized weights (Ridge, Logistic Regression,Random Forest, Gradient Boosting, XGBoost, LightGBM)

## Feature engineering
- Drop columns (Low correlation with the target variable, High category dominance,
Lack of statistical significance, High proportion of missing values)
- Handling missing values
- Combine new features
- Encoding categorical variables
- Handling outliers

# Results and evaluation
The evaluation process consists of several stages:

1. **Baseline model comparison**

Multiple machine learning algorithms were trained and compared:

- Gradient Boosting Classifier (GBR)
- XGBoost
- LightGBM
- Random Forest
- Logistic Regression

Each model was evaluated using cross-validation ROC-AUC score.

2. **Hyperparameter tuning evaluation**

Models were optimized using hyperparameter tuning and compared against their baseline performance.

The best performing model was:

- Gradient Boosting Classifier
- ROC-AUC: **0.8729**

3. **Feature selection validation**

SHAP (SHapley Additive exPlanations) was used to identify the most influential features.

An ablation study was performed by gradually adding SHAP-ranked features and measuring the change in ROC-AUC.

The optimal feature subset achieved:

- Gradient Boosting ROC-AUC: **0.8742**
- using the top 25 most important features

This confirmed that a smaller set of meaningful features can maintain or improve model performance.

4. **Business interpretation evaluation**

Beyond predictive performance, the model was analyzed using interpretable methods.

A surrogate decision tree was trained on model predictions to extract human-readable survival rules.
## Model Performance Comparison

### Base Models

| Rank | Model | ROC-AUC |
|----|--------|----------:|
| 1 | Gradient Boosting (GBR) | **0.8652** |
| 2 | XGBoost (XGB) | **0.8603** |
| 3 | LightGBM (LGBM) | **0.8600** |
| 4 | Random Forest (RFC) | **0.8586** |
| 5 | Logistic Regression | **0.8576** |

### Tuned Models

| Rank | Model | ROC-AUC |
|-----|--------|----------:|
| 1 | Gradient Boosting (GBR) | **0.8729** |
| 2 | Random Forest (RFC) | **0.8671** |
| 3 | LightGBM (LGBM) | **0.8570** |
| 4 | Logistic Regression | **0.8445** |
| 5 | XGBoost (XGB) | **0.8435** |

### Performance Improvement After Hyperparameter Tuning

| Model | Base ROC-AUC | Tuned ROC-AUC | Improvement |
|--------|----------:|----------:|----------:|
| Gradient Boosting | 0.8652 | **0.8729** | **+0.0077** |
| Random Forest | 0.8586 | **0.8671** | **+0.0085** |
| LightGBM | 0.8600 | 0.8570 | -0.0031 |
| Logistic Regression | 0.8576 | 0.8445 | -0.0131 |
| XGBoost | 0.8603 | 0.8435 | -0.0167 |

### Best Performing Model

**Gradient Boosting Classifier (GBR)** achieved the highest cross-validated 
ROC-AUC score of **0.8729** after hyperparameter tuning.

----------------------------------------------------------------------------------------------------


## Feature Selection Using SHAP and Ablation Study

Features were ranked according to their SHAP importance and progressively added to evaluate their contribution to model performance.

### Ablation Results

| Number of Features | Random Forest | Gradient Boosting | LightGBM |
|-------------------:|--------------:|------------------:|----------:|
| 1 | 0.7759 | 0.7759 | 0.7759 |
| 2 | 0.8300 | 0.8300 | 0.8300 |
| 3 | 0.8353 | 0.8353 | 0.8353 |
| 4 | 0.8529 | **0.8665** | 0.8560 |
| 5 | 0.8459 | 0.8513 | 0.8520 |
| 6 | 0.8492 | 0.8540 | 0.8528 |
| 7 | 0.8516 | 0.8594 | 0.8587 |
| 8 | 0.8534 | 0.8601 | 0.8593 |
| 9 | 0.8531 | 0.8599 | 0.8599 |
| 10 | 0.8575 | 0.8627 | 0.8602 |
| 15 | **0.8579** | 0.8633 | 0.8600 |
| 20 | 0.8558 | 0.8625 | 0.8599 |
| 25 | 0.8555 | **0.8742** | **0.8613** |

### Key Findings

- The first few SHAP-ranked features captured most of the predictive signal.
- Performance increased rapidly up to approximately 10 features.
- Gradient Boosting consistently outperformed the other models.
- The best result was achieved by **Gradient Boosting** with the top **25 SHAP-ranked features**, reaching a ROC-AUC of **0.8742**.
- The ablation study confirmed that a relatively small subset of engineered features explains most of the survival prediction performance.

### Best Feature Subset

The optimal feature subset contained the top **25 SHAP-ranked features** and produced the highest average model performance.

| Model | Best ROC-AUC |
|--------|----------:|
| Random Forest | 0.8555 |
| Gradient Boosting | **0.8742** |
| LightGBM | 0.8613 |

-------------------------------------------------------------------------------------------
## Extracted Survival Rules
A surrogate decision tree was trained on Gradient Boosting predictions
to extract interpretable survival rules.

The highest predicted survival probability was associated with:

| Profile | Predicted Survival |
|---|---:|
| Female passenger, 1st/2nd class, non-officer | ~92% |
| Female passenger, 3rd class, lower fare | ~60% |
| Male passenger with known cabin and age < 36 | ~53% |
| Male passenger, unknown cabin | ~12% |

Main survival drivers:
- Passenger gender/title
- Ticket class
- Cabin information
- Passenger group size







