# imports
import pandas as pd
import numpy as np
import os
import warnings
import time
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from subgroup_helpers import mdss_group, generate_mdss_query_condition, pysubgroup_group, transform_conditions, transform_for_query, divergence_group, create_query_dictionary, slicefinder_query, slicefinder_query_transform
from subgroup_discovery import LLMSubgroupFinder
from openai_config import get_openai_config
from dataset_helpers import get_data_education, get_data_compas, get_data_breast_cancer, get_data_loan_default, get_data_diabetes

# Ignore warnings
warnings.filterwarnings('ignore')

# Instantiate the AzureOpenAI client
config = get_openai_config()
deployment_names = {'gpt-4': 'SWNorth-gpt-4-0613-20231016-3'}

llm = AzureOpenAI(
    api_version=config["api_version"],
    azure_endpoint=config["api_base"],
    api_key=config["api_key"]
)

config['engine'] = deployment_names['gpt-4']

# Context and target definitions
context = "This dataset contains information about whether a person will default on their loan based on covariates such as how much a person runs, annual rainfall, favorite color, and others."
context_target = "loan default"

# Initialize subgroup finder
subgroup_finder = LLMSubgroupFinder(llm=llm, config=config, verbose=False)

# Function to get queries and measure time
def get_queries_with_time(X_train, y_train, model, llm_test=False, divexpl=True, verbose=False):
    time_dict = {}
    n_subgroups = 1

    start_time = time.time()
    subset, _ = mdss_group(X_train, y_train)
    q_mdss = generate_mdss_query_condition(subset)
    time_dict['mdss'] = time.time() - start_time

    start_time = time.time()
    res = pysubgroup_group(X_train, y_train, n_subgroups=n_subgroups, beamsearch=True)
    res = transform_conditions(res)
    query_beam = transform_for_query(res)[0]
    time_dict['pysubgroup_beam'] = time.time() - start_time

    start_time = time.time()
    res2 = pysubgroup_group(X_train, y_train, n_subgroups, beamsearch=False)
    res2 = transform_conditions(res2)
    query_apriori = transform_for_query(res2)[0]
    time_dict['pysubgroup_apriori'] = time.time() - start_time

    if divexpl:
        start_time = time.time()
        FP_sorted = divergence_group(X_train, y_train, model, n_subgroups=n_subgroups)
        query_divexplorer = create_query_dictionary(FP_sorted['itemsets'])[0]
        time_dict['divexplorer'] = time.time() - start_time
    else:
        query_divexplorer = None

    start_time = time.time()
    sf_dict = slicefinder_query(X_train, y_train, model, n_subgroups=n_subgroups)
    sf_query = slicefinder_query_transform(sf_dict)
    time_dict['slicefinder'] = time.time() - start_time

    if llm_test:
        start_time = time.time()
        subgroup_finder.fit(X_train, context, context_target, evaluate_feasibility=False)
        sciencetest_query = subgroup_finder.subgroups
        if len(sciencetest_query) > 0:
            sciencetest_query = sciencetest_query[0]
        time_dict['science_test'] = time.time() - start_time

    queries = {
        'mdss': q_mdss,
        'pysubgroup_beam': query_beam,
        'pysubgroup_apriori': query_apriori,
        'divexplorer': query_divexplorer,
        'slicefinder': sf_query,
    }

    if llm_test:
        queries['science_test'] = sciencetest_query

    return queries, time_dict

# Function to add synthetic categorical variables
def add_categorical_variables(X, num_new_variables, num_categories, probabilities):
    X = X.copy()
    if len(probabilities) != num_categories:
        raise ValueError("Length of probabilities must match the number of categories.")
    if not np.isclose(sum(probabilities), 1):
        raise ValueError("Probabilities must sum up to 1.")
    
    for i in range(1, num_new_variables + 1):
        new_column_name = f'x_{i}'
        if np.random.rand() < 0.5:
            new_values = np.random.choice([0, 1], size=len(X), p=[0.90, 0.1])
        else:
            new_values = np.random.choice(4, size=len(X), p=probabilities)
        X[new_column_name] = new_values

    return X

# Experiment function
def run_experiment(data_getter, i, irrelevant_feature_counts=[1, 2, 4, 8, 16]):
    num_categories = 4
    probabilities = [0.1, 0.3, 0.4, 0.2]

    X, y, context, context_target = data_getter()
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=123 + i)

    np.random.seed(123 + i)
    try:
        ind = np.random.choice(len(X_train), size=150, replace=False)
    except ValueError:
        ind = np.arange(len(X_train))
    X_train = X_train.iloc[ind]
    y_train = y_train.iloc[ind]
    results_ls = []

    for num_irrelevant in irrelevant_feature_counts:
        print(f"Running experiment for {num_irrelevant} irrelevant features")
        X_train_augmented = add_categorical_variables(X_train.copy(), num_irrelevant, num_categories, probabilities)
        model = LogisticRegression()
        queries, time_dict = get_queries_with_time(X_train_augmented, y_train, model, llm_test=True, verbose=True)
        results = {'num_irrelevant': num_irrelevant, 'queries': queries, 'time': time_dict}
        results_ls.append(results)

    return results_ls

# Run experiments and save results
data_getters = [get_data_education, get_data_compas, get_data_breast_cancer, get_data_loan_default, get_data_diabetes]
names = ['education', 'compas', 'breast_cancer', 'loan_default', 'diabetes']
results_dir = 'experiment_results/exp1/'

for n_run in range(5):
    for i, (name, getter) in enumerate(zip(names, data_getters)):
        print(f"Running experiment for dataset {i+1}: {name}, run {n_run}")
        experiment_results = run_experiment(getter, n_run)
        results_df = pd.DataFrame(experiment_results)
        timestamp = pd.Timestamp.now().strftime("%Y%m%d-%H%M%S")
        results_df.to_csv(os.path.join(results_dir, f'results_{name}_{timestamp}_run_{n_run}.csv'), index=False)
        print(f"---------- Dataset {name} and run {n_run} finished ----------")
