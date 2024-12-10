# subgroup_model_evaluation.py

import pandas as pd
import numpy as np
import warnings
import imp
import time
import sys
import sklearn

from copy import deepcopy
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer
import xgboost as xgb

from comp_metrics import get_metrics
from subgroup_helpers import *
from dataset_helpers import *
from metrics_helpers import calculate_group_statistics
from subgroup_discovery import LLMSubgroupFinder
from openai import AzureOpenAI
from openai_config import get_openai_config

warnings.filterwarnings('ignore')

# Azure OpenAI configuration
config = get_openai_config()
deployment_names = {
    'gpt-4': 'SWNorth-gpt-4-0613-20231016-3',
    'gpt-35': 'SWNorth-gpt-35-turbo-0613-20231016-4'
}

llm = AzureOpenAI(
    api_version=config["api_version"],
    azure_endpoint=config["api_base"],
    api_key=config["api_key"]
)

config['engine'] = deployment_names['gpt-4']
subgroup_finder = LLMSubgroupFinder(llm=llm, config=config, verbose=False)

# Load dataset function
def load_seer_cutract_dataset(name, seed, dataset_path, prop=1.0):
    def aggregate_grade(row):
        for i in range(1, 6):
            if row[f"grade_{i}.0"] == 1:
                return i

    def aggregate_stage(row):
        for i in range(1, 6):
            if row[f"stage_{i}"] == 1:
                return i

    df = pd.read_csv(f"{dataset_path}/{name}.csv")

    df["grade"] = df.apply(aggregate_grade, axis=1)
    df["stage"] = df.apply(aggregate_stage, axis=1)
    df["mortCancer"] = df["mortCancer"].astype(int)

    mask = df["mortCancer"] == True
    df_dead = df[mask]
    df_survive = df[~mask]

    total_balanced = 10000 if name == "seer" else 1000
    n_samples = int(prop * total_balanced)

    df = pd.concat([
        df_dead.sample(n_samples, random_state=seed),
        df_survive.sample(n_samples, random_state=seed)
    ])

    df = sklearn.utils.shuffle(df, random_state=seed).reset_index(drop=True)

    rename_dict = {
        "psa": "prostate_specific_antigen",
        "treatment_CM": "treatment_conservative_management",
        "treatment_Primary hormone therapy": "treatment_primary_hormone_therapy",
        "treatment_Radical Therapy-RDx": "treatment_radical_radiotherapy",
        "treatment_Radical therapy-Sx": "treatment_radical_prostatectomy",
        "grade": "gleason_score"
    }

    df = df.rename(columns=rename_dict)
    features = [col for col in df.columns if col != "mortCancer"]

    return df[features], df["mortCancer"], df

# Load datasets
features_uk, labels_uk, df_uk = load_seer_cutract_dataset('cutract', 0, '../datasets/', prop=1.0)

# Clean features
def clean_features(features):
    return [f.lower().replace(" ", "_") for f in features]

features_uk.columns = clean_features(features_uk.columns)

# Instantiate models
model_lr = LogisticRegression()
model_rf = RandomForestClassifier()
model_svc = SVC(probability=True)
model_xgb = xgb.XGBClassifier()
model_mlp = MLPClassifier(hidden_layer_sizes=(2,))
models = [model_lr, model_rf, model_svc, model_xgb, model_mlp]

# Function to get queries for different subgroup discovery methods
def get_queries_many_exp(X_train, y_train, model, is_numeric_string=False, n_subgroups=5):
    kb = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='uniform')
    X_train = X_train.copy()
    y_train = y_train.copy()

    # MDSS
    subset, _ = mdss_group(X_train, y_train)
    q_mdss = generate_mdss_query_condition(subset)

    # Pysubgroup (Beam search)
    res_beam = pysubgroup_group(X_train, y_train, n_subgroups=n_subgroups, beamsearch=True)
    query_beam = transform_for_query(transform_conditions(res_beam))

    # Slicefinder
    X_train_sf = X_train.copy()
    X_train_sf[['age', 'prostate_specific_antigen', 'comorbidities']] = kb.fit_transform(
        X_train_sf[['age', 'prostate_specific_antigen', 'comorbidities']]
    )
    sf_dict = slicefinder_query(X_train_sf, y_train, model, n_subgroups=n_subgroups)
    sf_query = slicefinder_query_transform(sf_dict, is_numeric_string=is_numeric_string)

    # Divexplorer
    FP_sorted = divergence_group(X_train_sf, y_train, model, n_subgroups=n_subgroups)
    query_divexplorer = create_query_dictionary(FP_sorted['itemsets'], is_numeric_string=is_numeric_string)

    return {
        'mdss': q_mdss,
        'pysubgroup_beam': query_beam,
        'slicefinder': sf_query,
        'divexplorer': query_divexplorer
    }, kb

# Train and evaluate models with queries
results_list = []
for i in range(5):
    X_train_uk, X_test_uk, y_train_uk, y_test_uk = train_test_split(features_uk, labels_uk, test_size=0.5, random_state=i)
    models_results = {}

    for model in models:
        model_name = model.__class__.__name__
        print(f"Evaluating Model: {model_name}")
        model.fit(X_train_uk, y_train_uk)

        queries, kb = get_queries_many_exp(X_train_uk, y_train_uk, model, is_numeric_string=True, n_subgroups=5)
        models_results[model_name] = {}

        for query_name, query in queries.items():
            query_main = query[0]
            results = calculate_group_statistics(X_test_uk, y_test_uk, model, query_main)
            models_results[model_name][query_name] = results

    results_list.append(models_results)

# Aggregate and display results
data = {}
for entry in results_list:
    for model in entry:
        if model not in data:
            data[model] = {}
        for method in entry[model]:
            accuracy_diff = entry[model][method].get('accuracy_diff', None)
            p_value_bootstrap = entry[model][method].get('p_value_bootstrap', None)
            if method not in data[model]:
                data[model][method] = {'accuracy_diff': [], 'p_value_bootstrap': []}
            data[model][method]['accuracy_diff'].append(accuracy_diff)
            data[model][method]['p_value_bootstrap'].append(p_value_bootstrap)

# Function to calculate mean and standard deviation
def mean_std_format(values):
    mean_val = pd.Series(values).mean()
    std_val = pd.Series(values).std()
    return f"{mean_val:.2f} ± {std_val:.2f}"

# Create DataFrame with results
table = pd.DataFrame({
    (model, stat): [mean_std_format(data[model][method][stat]) for method in data[model]]
    for model in data for stat in ['accuracy_diff', 'p_value_bootstrap']
})

table.index = data[list(data.keys())[0]].keys()

# Display the table
print(table)

# Export the table to LaTeX
print(table.to_latex())
