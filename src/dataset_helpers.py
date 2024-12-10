from copy import deepcopy
import numpy as np
import pandas as pd
import sklearn

def get_data_education(maxn=5_000, with_original=False):
    df = pd.read_csv('../datasets/studentinfo.csv').drop(columns=['id_student', 'code_module', 'code_presentation'])
    df['final_result'] = df['final_result'].isin(['Withdrawn', 'Failed']).astype(int)
    df = df.dropna()
    if with_original: df_original = df.copy()
    numeric_cols = ['studied_credits']
    for col in numeric_cols:
        df[col] = pd.qcut(df[col], 10, duplicates='drop')
        df[col] = df[col].apply(lambda x: str(round(x.left, 2)) + ' - ' + str(round(x.right,2)))

    X = df.drop(columns=['final_result'])[:maxn]
    y = df['final_result'][:maxn]

    context = """This dataset comprises various attributes of student profiles and their engagement in an online learning platform, including demographics, previous academic background, and interaction with the system."""
    context_target = """student's final result as a binary classification of success (0) or withdrawal/failure (1)"""

    if with_original:
        return X, y, context, context_target, df_original.drop(columns=['final_result'])[:maxn], df_original['final_result'][:maxn]
    else:
        return X, y, context, context_target

def get_data_compas(maxn=5000, with_original=False):
    compas = pd.read_csv('../datasets/compas-scores-two-years-violent.csv')
    cols = ['sex', 'age_cat', 'race', 'juv_fel_count', 'juv_misd_count', 'juv_other_count', 'priors_count', 'c_charge_degree', 'v_decile_score']
    compas = compas[cols]
    compas = compas.dropna()
    if with_original: df_original = compas.copy()

    X = compas.drop('v_decile_score', axis=1)[:maxn]
    y = compas['v_decile_score'][:maxn]

    numeric_cols = [col for col in X.columns if 'count' in col]
    for col in numeric_cols:
        X[col] = pd.qcut(X[col], 10, duplicates='drop')
        X[col] = X[col].apply(lambda x: str(round(x.left, 2)) + ' - ' + str(round(x.right,2)))

    y_binary = (y > 5).astype(int)

    context = """This dataset includes demographic and criminal history data used in the COMPAS risk assessment tool, which is intended to assist courts in determining the likelihood of a defendant reoffending."""
    context_target = """violent crime recidivism risk score (v_decile_score)"""
    if with_original:
        X_orig = df_original.drop(columns=['v_decile_score'])
        y_orig = (df_original['v_decile_score'] > 5).astype(int)
        return X, y_binary, context, context_target, X_orig[:maxn], y_orig[:maxn]
    else:
        return X, y_binary, context, context_target
    
def get_data_breast_cancer(maxn=5000, with_original=False):
    df = pd.read_csv('../datasets/coimbra_breast_cancer.csv')
    df = df.dropna()
    if with_original: df_original = df.copy()
    numeric_cols = ['Age', 'BMI', 'Glucose', 'Insulin', 'HOMA', 'Leptin', 'Adiponectin', 'Resistin', 'MCP.1']
    if with_original: df_original = df.copy()
    for col in numeric_cols:
        df[col] = pd.qcut(df[col], 10, duplicates='drop')
        df[col] = df[col].apply(lambda x: str(round(x.left, 2)) + ' - ' + str(round(x.right,2)))
    
    # Dropping MCP.1 to have 8 variables in each dataset.
    X = df.drop(columns=['Outcome', 'MCP.1'])[:maxn]
    y = df['Outcome'][:maxn]
    context = """The dataset contains biomedical measurements from female patients, which are used to determine the likelihood of having breast cancer."""
    context_target = """breast cancer outcome, where 1 indicates presence of cancer and 0 indicates absence"""

    if with_original:
        return X, y, context, context_target, df_original.drop(columns=['Outcome', 'MCP.1'])[:maxn], df_original['Outcome'][:maxn]
    else:
        return X, y, context, context_target
    
def get_data_loan_default(maxn=5_000, with_original=False):
    df = pd.read_csv('../datasets/Loan_default.csv')
    df = df.dropna()
    if with_original: df_original = df.copy()

    numeric_cols = ['Age', 'Income', 'LoanAmount', 'CreditScore', 'MonthsEmployed', 'InterestRate', 'LoanTerm', 'DTIRatio']
    for col in numeric_cols:
        df[col] = pd.qcut(df[col], 10, duplicates='drop')
        df[col] = df[col].apply(lambda x: str(round(x.left, 2)) + ' - ' + str(round(x.right,2)))

    X = df.drop(columns=['Default', 'LoanID'])[:maxn]
    y = df['Default'][:maxn]
    
    cols_loan = ['LoanAmount', 'CreditScore', 'MonthsEmployed', 'NumCreditLines', 'InterestRate', 'LoanTerm', 'DTIRatio', 'HasCoSigner']
    X = X.drop(columns=cols_loan)


    context = """This dataset represents financial and personal information about borrowers, which is utilized to predict the likelihood of defaulting on a loan."""
    context_target = """loan default probability, with 1 indicating default and 0 indicating non-default"""

    if with_original:
        return X, y, context, context_target, df_original.drop(columns=['Default', 'LoanID'])[:maxn], df_original['Default'][:maxn]
    else:
        return X, y, context, context_target
    
def get_data_diabetes(with_original=False):
    df = pd.read_csv('../datasets/diabetes.csv')
    df = df.dropna()
    if with_original: df_original = df.copy()

    numeric_cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    for col in numeric_cols:
        df[col] = pd.qcut(df[col], 10, duplicates='drop')
        df[col] = df[col].apply(lambda x: str(round(x.left, 2)) + ' - ' + str(round(x.right,2)))

    X = df.drop(columns=['Outcome'])
    y = df['Outcome']

    context = """This dataset includes medical diagnostic measurements, potentially helpful in predicting diabetes among patients."""
    context_target = """diabetes diagnosis, where 1 indicates diabetes and 0 indicates no diabetes"""

    if with_original:
        return X, y, context, context_target, df_original.drop(columns=['Outcome']), df_original['Outcome']
    else:
        return X, y, context, context_target

def load_seer_cutract_dataset(name, seed, dataset_path, prop=1.0):
    """
    Loads the SEER/CUTRACT dataset, and returns the features, labels, and the entire dataset

    Note this returns a balanced data!

    Args:
      name: the name of the dataset to load.
      seed: the random seed used to generate the data
      prop: proportion of the dataset to return 0-1

    Returns:
      The features, labels, and the entire dataset.
    """

    def aggregate_grade(row):
        if row["grade_1.0"] == 1:
            return 1
        if row["grade_2.0"] == 1:
            return 2
        if row["grade_3.0"] == 1:
            return 3
        if row["grade_4.0"] == 1:
            return 4
        if row["grade_5.0"] == 1:
            return 5

    def aggregate_stage(row):
        if row["stage_1"] == 1:
            return 1
        if row["stage_2"] == 1:
            return 2
        if row["stage_3"] == 1:
            return 3
        if row["stage_4"] == 1:
            return 4
        if row["stage_5"] == 1:
            return 5

    random_seed = seed


    features = [
        "age",
        "psa",
        "comorbidities",
        "treatment_CM",
        "treatment_Primary hormone therapy",
        "treatment_Radical Therapy-RDx",
        "treatment_Radical therapy-Sx",
        "grade",
        "stage_1",
        "stage_2",
        "stage_3",
        "stage_4",
    ]

    label = "mortCancer"
    df = pd.read_csv(f"{dataset_path}/{name}.csv")

    df["grade"] = df.apply(aggregate_grade, axis=1)
    df["stage"] = df.apply(aggregate_stage, axis=1)
    df["mortCancer"] = df["mortCancer"].astype(int)
    df["mort"] = df["mort"].astype(int)

    mask = df[label] == True
    df_dead = df[mask]
    df_survive = df[~mask]

    if name == "seer":
        total_balanced = 10000
        n_samples_per_group = int(prop * total_balanced)

        n_samples = n_samples_per_group  
        ns = n_samples_per_group  
    else:
        total_balanced = 1000
        n_samples_per_group = int(prop * total_balanced)

        n_samples = n_samples_per_group
        ns = n_samples_per_group
    df = pd.concat(
        [
            df_dead.sample(ns, random_state=random_seed),
            df_survive.sample(n_samples, random_state=random_seed),
        ]
    )
    df = sklearn.utils.shuffle(df, random_state=random_seed)
    df = df.reset_index(drop=True)

    # Renaming dictionary
    rename_dict = {
        "psa": "prostate_specific_antigen",
        "treatment_CM": "treatment_conservative_management",
        "treatment_Primary hormone therapy": "treatment_primary_hormone_therapy",
        "treatment_Radical Therapy-RDx": "treatment_radical_radiotherapy",
        "treatment_Radical therapy-Sx": "treatment_radical_prostatectomy",
        "grade": "gleason_score",
    }

    # Rename the columns
    df = df.rename(columns=rename_dict)

    # New feature list
    features = [rename_dict.get(f, f) for f in features]
    return df[features], df[label], df

