# tns_experiment.py

import numpy as np
import pandas as pd
from itertools import islice
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder

from subgroup_helpers import *
from metrics_helpers import calculate_group_statistics
from subgroup_discovery import LLMSubgroupFinder
from openai import AzureOpenAI
from openai_config import get_openai_config

# Set up OpenAI configuration
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
subgroup_finder2 = LLMSubgroupFinder(llm=llm, config=config, verbose=False)

# Generate synthetic dataset function
def generate_dataset(seed):
    np.random.seed(seed)
    n_samples = 5000

    gender = np.random.choice(['Male', 'Female'], n_samples)
    ethnicity = np.random.choice(['White', 'Black'], n_samples)
    age = np.random.randint(18, 70, n_samples)
    income_level = np.random.choice(['Low', 'Medium', 'High'], n_samples)
    education_level = np.random.choice(['No_Degree', 'Bachelor', 'Master', 'PhD'], n_samples)

    def logistic(x):
        return 1 / (1 + np.exp(-x))

    noise = np.zeros(n_samples)
    recidivism = logistic(-0.05 * age + 0.2 * (income_level == 'High') + 0.2 * (education_level == 'PhD') + noise)
    median = pd.Series(recidivism).median()
    recidivism_dummy = (recidivism > median).astype(int)

    df = pd.DataFrame({
        'gender': gender,
        'ethnicity': ethnicity,
        'age': age,
        'income_level': income_level,
        'education_level': education_level,
        'recidivism': recidivism_dummy
    })

    encoded_df = pd.get_dummies(df, columns=['gender', 'ethnicity', 'income_level', 'education_level'], drop_first=False)
    X = encoded_df.drop('recidivism', axis=1)
    y = encoded_df['recidivism']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

# Custom model class with two corruptions
class CustomModel_two_corruptions:
    def __init__(self, model_base, model_corrupted, corruption_condition):
        self.model_base = model_base
        self.model_corrupted = model_corrupted
        self.corruption_condition = corruption_condition

    def fit(self, X_train, y_train):
        self.model_base.fit(X_train, y_train)
        self.model_corrupted.fit(X_train, y_train)

    def predict(self, X):
        base_preds = self.model_base.predict(X)
        corrupted_preds = self.model_corrupted.predict(X)
        mask = eval(self.corruption_condition, {'X': X})
        combined_preds = np.where(mask, corrupted_preds, base_preds)
        return combined_preds

# Function to get subgroup discovery queries
def get_queries_many_exp(X_train, y_train, model, context, context_target, n_subgroups=5):
    subgroup_finder2.fit(X_train, context, context_target, evaluate_feasibility=False, n=n_subgroups)
    optimal_queries = subgroup_finder2.get_optimal_queries(X_train, y_train, model, min_group_size=10, n_groups=n_subgroups)
    return optimal_queries, None

# List of corruption conditions
conditions = [
    "X['ethnicity_White'] == 1",
    "X['ethnicity_Black'] == 1"
]
condition_names = [
    ['White subgroup corruption'],
    ['Black subgroup corruption']
]

# Initialize base and corrupted models
model_base = LogisticRegression()
model_corrupted = LogisticRegression()

# Run the TNs experiment
scores_ls = []

for condition, names in zip(conditions, condition_names):
    print(f"Condition: {condition}, Names: {names}")

    for seed in range(5):
        print(f"Seed: {seed}")
        
        # Generate dataset
        X_train, X_test, y_train, y_test = generate_dataset(seed)
        
        # Create custom model
        model = CustomModel_two_corruptions(model_base, model_corrupted, condition)

        # Resetting the index
        X_train = X_train.reset_index(drop=True)
        y_train.index = X_train.index

        # Fit the model
        model.fit(X_train, y_train)
        y_preds = model.predict(X_train)

        # Generate context
        cat_cols = X_train.select_dtypes(include=['object', 'boolean']).columns.tolist()
        X_train_desc = X_train.copy()
        X_train_desc['preds'] = (y_preds == y_train)

        text = []
        for col in cat_cols:
            pred_proportions = X_train_desc.groupby(col)['preds'].mean()
            for category, proportion in pred_proportions.items():
                text_str = f"In the {col} category of {category}, the proportion of correct predictions is {proportion:.2f}"
                text.append(text_str)

        text_input = "; ".join(text)
        context = f"""This dataset contains information about whether a person is prone to recidivism. The covariates are age, gender, race, income, education. This is the description: {X_train.describe()}. This is the distribution of outcomes based on each covariate: {text_input}"""
        context_target = """recidivism prediction"""

        # Get subgroup discovery queries
        queries, _ = get_queries_many_exp(X_train, y_train, model, context, context_target)
        print("Queries:", queries)

        # Scoring the queries
        scores = {}
        for name in names:
            for key, inner_dict in queries.items():
                top_5_values = list(islice(inner_dict.values(), 5))
                score_for_name = 1/3 if any(name in value for value in top_5_values) else 0
                scores[key] = scores.get(key, 0) + score_for_name

        print("Scores:", scores)
        scores_ls.append(scores)

# Aggregate scores and export to CSV
df_scores = pd.DataFrame(scores_ls)
df_scores.to_csv("tns_experiment_results.csv", index=False)
print("TNs experiment results saved to tns_experiment_results.csv")
