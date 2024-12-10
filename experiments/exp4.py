# bias_experiment.py

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
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

# Biased Logistic Regression class
class BiasedLogisticRegression:
    def __init__(self, model, bias_subgroup, percentage_bias=0.5):
        self.model = model
        self.bias_subgroup = bias_subgroup
        self.percentage_bias = percentage_bias

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        y_pred = self.model.predict(X_test)
        is_bias_subgroup = (X_test[self.bias_subgroup] == 1)
        n_subgroup_people = np.sum(is_bias_subgroup)
        change_mask = np.random.choice([True, False], n_subgroup_people, p=[self.percentage_bias, 1 - self.percentage_bias])
        y_pred[is_bias_subgroup] = y_pred[is_bias_subgroup] ^ change_mask
        return y_pred

# Evaluate white and black subgroup scores
def calculcate_white_black_scores(top_5_queries):
    conditions_white = ['ethnicity_Black <= 0.5', 'ethnicity_Black < 0.5', 'ethnicity_White > 0.5', 'ethnicity_White >= 0.5']
    conditions_black = ['ethnicity_Black >= 0.5', 'ethnicity_Black > 0.5', 'ethnicity_White <= 0.5', 'ethnicity_White < 0.5']
    
    values = list(top_5_queries.values())
    
    max_score_white = max(1 if any(condition in value for value in values) else 0 for condition in conditions_white)
    max_score_black = max(1 if any(condition in value for value in values) else 0 for condition in conditions_black)
    
    return max_score_white, max_score_black

# Main experiment loop
def run_bias_experiment(bias_subgroup, bias_ls, iterations=10):
    scores_global = {}
    for percentage_bias in bias_ls:
        print(f"Calculating percentage bias: {percentage_bias}")
        scores_global[percentage_bias] = {'white_score': 0, 'black_score': 0}

        for seed in range(iterations):
            X_train, X_test, y_train, y_test = generate_dataset(seed)
            model = BiasedLogisticRegression(LogisticRegression(), bias_subgroup, percentage_bias)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            context = f"""This dataset contains information about whether a person is prone to recidivism. The covariates are age, gender, ethnicity, income, education."""
            context_target = "recidivism prediction"

            subgroup_finder2.find_subgroup_variables(X_test, context, context_target, n=30)
            queries = subgroup_finder2.get_optimal_queries(X_test, y_test, model, min_group_size=15)
            top_5_queries = dict(list(queries.items())[:5])

            max_score_white, max_score_black = calculcate_white_black_scores(top_5_queries)
            scores_global[percentage_bias]['white_score'] += max_score_white
            scores_global[percentage_bias]['black_score'] += max_score_black

    return scores_global

# Run white and black experiments
bias_ls = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5]

# White subgroup bias experiment
scores_white_experiment = run_bias_experiment('ethnicity_White', bias_ls)
df_scores_white = pd.DataFrame({k: v for k, v in scores_white_experiment.items()}).T
df_scores_white.to_csv("white_bias_experiment_results.csv")
print("White bias experiment results saved to white_bias_experiment_results.csv")

# Black subgroup bias experiment
scores_black_experiment = run_bias_experiment('ethnicity_Black', bias_ls)
df_scores_black = pd.DataFrame({k: v for k, v in scores_black_experiment.items()}).T
df_scores_black.to_csv("black_bias_experiment_results.csv")
print("Black bias experiment results saved to black_bias_experiment_results.csv")

# Display results
print("White Bias Experiment Results:")
print(df_scores_white)

print("Black Bias Experiment Results:")
print(df_scores_black)
