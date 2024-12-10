# Imports
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from statsmodels.stats.contingency_tables import mcnemar
from copy import deepcopy
from scipy import stats
from scipy.stats import ttest_ind, chi2_contingency
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

def welchs_t_test_for_accuracy(df, model, query):
    # Subgroup defined by the query
    subgroup = df.query(query)
    subgroup_predictions = model.predict(subgroup.drop('y', axis=1))
    subgroup_accuracies = (subgroup['y'] == subgroup_predictions).astype(int)

    # Complementary group (not in the subgroup)
    complementary_group = df.query(f"not ({query})")
    complementary_predictions = model.predict(complementary_group.drop('y', axis=1))
    complementary_accuracies = (complementary_group['y'] == complementary_predictions).astype(int)

    # Apply Welch's t-test
    result = ttest_ind(subgroup_accuracies, complementary_accuracies, equal_var=False)

    return result.pvalue
    
def chi_square_test_for_accuracy(df, model, query):
    # Subgroup defined by the query
    subgroup = df.query(query)
    subgroup_predictions = model.predict(subgroup.drop('y', axis=1))
    # Counting correct and incorrect predictions in the subgroup
    correct_subgroup = np.sum(subgroup['y'] == subgroup_predictions)
    incorrect_subgroup = np.sum(subgroup['y'] != subgroup_predictions)

    # Complementary group (not in the subgroup)
    complementary_group = df.query(f"not ({query})")
    complementary_predictions = model.predict(complementary_group.drop('y', axis=1))
    # Counting correct and incorrect predictions in the complementary group
    correct_complementary = np.sum(complementary_group['y'] == complementary_predictions)
    incorrect_complementary = np.sum(complementary_group['y'] != complementary_predictions)

    # Constructing the contingency table
    table = [[correct_subgroup, correct_complementary],
              [incorrect_subgroup, incorrect_complementary]]

    # Perform Chi-Square test
    chi2, p, dof, expected = chi2_contingency(table)

    return p


def bootstrapping_test_for_accuracy_string(df, model, subgroup, num_bootstrap_samples=200):
    # Get the remainder based on indices which are not in subgroup

    remainder = df.loc[~df.index.isin(subgroup.index)].copy()

    # Combine accuracies from both subgroup and remainder
    pooled_accuracies = np.concatenate([
        (subgroup['y'] == model.predict(subgroup.drop('y', axis=1))).astype(int),
        (remainder['y'] == model.predict(remainder.drop('y', axis=1))).astype(int)
    ])

    # Observed accuracy difference
    observed_diff = np.mean(subgroup['y'] == model.predict(subgroup.drop('y', axis=1))) - \
                    np.mean(remainder['y'] == model.predict(remainder.drop('y', axis=1)))

    bootstrap_diffs = []

    # Bootstrapping under the null hypothesis
    for _ in range(num_bootstrap_samples):
        # Resampling with replacement from the pooled accuracies
        resampled_indices = np.random.choice(len(pooled_accuracies), size=len(pooled_accuracies), replace=True)
        resampled_accuracies = pooled_accuracies[resampled_indices]

        # Splitting the resampled accuracies into "subgroup" and "remainder"
        resampled_subgroup_acc = resampled_accuracies[:len(subgroup)]
        resampled_remainder_acc = resampled_accuracies[len(remainder):]

        # Difference in accuracies for the resampled data
        resampled_diff = np.mean(resampled_subgroup_acc) - np.mean(resampled_remainder_acc)
        bootstrap_diffs.append(resampled_diff)

    # Calculating p-value
    p_value = np.sum(np.abs(bootstrap_diffs) >= np.abs(observed_diff)) / num_bootstrap_samples

    return p_value



def bootstrapping_test_for_accuracy(df, model, query, num_bootstrap_samples=200):
    # Subgroup and remainder of the dataset
    subgroup = df.query(query)
    remainder = df.query(f"not ({query})")

    # Combine accuracies from both subgroup and remainder
    pooled_accuracies = np.concatenate([
        (subgroup['y'] == model.predict(subgroup.drop('y', axis=1))).astype(int),
        (remainder['y'] == model.predict(remainder.drop('y', axis=1))).astype(int)
    ])

    # Observed accuracy difference
    observed_diff = np.mean(subgroup['y'] == model.predict(subgroup.drop('y', axis=1))) - \
                    np.mean(remainder['y'] == model.predict(remainder.drop('y', axis=1)))

    bootstrap_diffs = []

    # Bootstrapping under the null hypothesis
    for _ in range(num_bootstrap_samples):
        # Resampling with replacement from the pooled accuracies
        resampled_indices = np.random.choice(len(pooled_accuracies), size=len(pooled_accuracies), replace=True)
        resampled_accuracies = pooled_accuracies[resampled_indices]

        # Splitting the resampled accuracies into "subgroup" and "remainder"
        resampled_subgroup_acc = resampled_accuracies[:len(subgroup)]
        resampled_remainder_acc = resampled_accuracies[len(remainder):]

        # Difference in accuracies for the resampled data
        resampled_diff = np.mean(resampled_subgroup_acc) - np.mean(resampled_remainder_acc)
        bootstrap_diffs.append(resampled_diff)

    # Calculating p-value
    p_value = np.sum(np.abs(bootstrap_diffs) >= np.abs(observed_diff)) / num_bootstrap_samples

    return p_value


def welchs_t_test_for_accuracy(df, model, query):
    # Subgroup defined by the query
    subgroup = df.query(query)
    subgroup_predictions = model.predict(subgroup.drop('y', axis=1))
    subgroup_accuracies = (subgroup['y'] == subgroup_predictions).astype(int)
    #subgroup_accuracy = accuracy_score(subgroup['y'], subgroup_predictions)

    # Complementary group (not in the subgroup)
    complementary_group = df.query(f"not ({query})")
    complementary_predictions = model.predict(complementary_group.drop('y', axis=1))
    complementary_accuracies = (complementary_group['y'] == complementary_predictions).astype(int)
    #complementary_accuracy = accuracy_score(complementary_group['y'], complementary_predictions)

    # Apply Welch's t-test
    result = ttest_ind(subgroup_accuracies, complementary_accuracies, equal_var=False)

    return result.pvalue

def mcnemars_test(df, model, query):
    # Step 1: Calculate overall model accuracy
    overall_accuracy = accuracy_score(df['y'], model.predict(df.drop('y', axis=1)))

    # Subgroup
    subgroup = df.query(query)
    subgroup_size = len(subgroup)
    subgroup_predictions = model.predict(subgroup.drop('y', axis=1))
    subgroup_actual = subgroup['y'].to_numpy()

    # Step 2: Calculate expected proportions in subgroup
    expected_correct = round(overall_accuracy * subgroup_size)
    expected_incorrect = subgroup_size - expected_correct

    # Step 3: Observe actual predictions in subgroup
    actual_correct = np.sum(subgroup_predictions == subgroup_actual)
    actual_incorrect = subgroup_size - actual_correct

    # Step 4: Apply McNemar's Test
    # Constructing the contingency table
    table = [[actual_correct, actual_incorrect],
             [expected_correct, expected_incorrect]]

    # Perform McNemar's test
    result = mcnemar(table, exact=False, correction=True)

    return result.pvalue



def calculate_weighted_relative_outcomes(df, query):
    df_subgroup = df.query(query)

    support = df_subgroup.shape[0] / df.shape[0]
    diff_outcomes = df_subgroup['y'].mean() - df['y'].mean()

    return support * diff_outcomes

def calculate_weighted_relative_accuracy(df, query, model):

    df_subgroup = df.query(query)
    support = df_subgroup.shape[0] / df.shape[0]
    diff_accuracy = accuracy_score(df_subgroup['y'], model.predict(df_subgroup.drop('y', axis=1))) - accuracy_score(df['y'], model.predict(df.drop('y', axis=1)))

    return support * diff_accuracy

def calculate_odds_ratio(df, query):
    """Odds ratio: (p1 * (1-p1) / (p0 * (1-p0))), 
    where p1 is the probability of the outcome in the subgroup, 
    and p0 is the probability of the outcome in the rest of the dataset."""
    
    df_subgroup = df.query(query)
    df_rest = df.query(f"not ({query})")

    p1 = df_subgroup['y'].mean()
    p0 = df_rest['y'].mean()

    return p1 * (1-p1) / (p0 * (1-p0))

def calculate_odds_ratio_acc(df, query, model):
    """Odds ratio: (p1 * (1-p1) / (p0 * (1-p0))), 
    where p1 is the % accuracy in the subgroup, and p0 is the % accuracy in the rest of the dataset."""

    df_subgroup = df.query(query)
    df_rest = df.query(f"not ({query})")

    p1 = accuracy_score(df_subgroup['y'], model.predict(df_subgroup.drop('y', axis=1)))
    p0 = accuracy_score(df_rest['y'], model.predict(df_rest.drop('y', axis=1)))

    return p1 * (1-p1) / (p0 * (1-p0))

def calculate_lift(df, query):
    """Lift: p1 / p, where p is the probability of the outcome in the entire dataset"""
    
    df_subgroup = df.query(query)
    p1 = df_subgroup['y'].mean()
    p = df['y'].mean()

    return p1 / p

def calculate_lift_outcome(df, query, model):
    """Lift: p1 / p, where p is the accuracy of the entire dataset, and p1 is the accuracy of the subgroup"""
    
    df_subgroup = df.query(query)
    p1 = accuracy_score(df_subgroup['y'], model.predict(df_subgroup.drop('y', axis=1)))
    p = accuracy_score(df['y'], model.predict(df.drop('y', axis=1)))

    return p1 / p

def calculate_group_statistics(X, y, model, query, X_tr=None, num_iterations=250):
    # Calculate the dataframe
    df = deepcopy(X)
    df['y'] = y
    # Filter the subgroup
    if len(query) == 0 or len(df.query(query)) == 0 or len(df.query(query)) == len(df):
        return {
            'group_size': 0,
            'support': 0,
            'p_value_mc': 1,
            'p_value_t': 1,
            'p_value_chi': 1,
             'p_value_bootstrap': 1, 
            'num_criteria': 0,
            'outcome_diff': 0,
            'accuracy_diff': 0,
            'odds_ratio_outcome': np.nan,
             'odds_ratio_acc': np.nan, # odds ratio of the accuracy
            'query': query,
            'lift_outcome': np.nan,
            'lift_acc': np.nan, # lift of the accuracy
            'weighted_relative_outcome': np.nan,
            'weighted_relative_accuracy': np.nan
        }
    
    subgroup = df.query(query)
    # Calculate statistics
    group_size = len(subgroup)
    relative_size = group_size / len(df)
    num_criteria = query.count("and") + 1 # Counting 'and' and adding 1 for the first condition

    # Outcome difference
    avg_outcome_dataset = y.mean()
    avg_outcome_subgroup = subgroup['y'].mean()
    outcome_diff = abs(avg_outcome_dataset - avg_outcome_subgroup)

    # Model accuracy difference
    if X_tr is not None:
        subgroup_tr = X_tr.loc[subgroup.index]
        subgroup_y = y.loc[subgroup.index]
        accuracy_dataset = accuracy_score(y, model.predict(X_tr))
        accuracy_subgroup = accuracy_score(subgroup_y, model.predict(subgroup_tr))
        accuracy_diff = abs(accuracy_dataset - accuracy_subgroup)
   
    else:
        accuracy_dataset = accuracy_score(y, model.predict(X))
        accuracy_subgroup = accuracy_score(subgroup['y'], model.predict(subgroup.drop('y', axis=1)))
        accuracy_diff = abs(accuracy_dataset - accuracy_subgroup)
   
   # P-value calculation (randomization-based testing)
    p_value = mcnemars_test(df, model, query)
    p_value_t = welchs_t_test_for_accuracy(df, model, query)
    p_value_chi = chi_square_test_for_accuracy(df, model, query)

    # Get odds ratio
    odds_ratio = calculate_odds_ratio(df, query)

    # Calculate lift for outcome
    lift = calculate_lift(df, query)

    # Calculate lift for accuracy
    lift_acc = calculate_lift_outcome(df, query, model)

    # Calculate weighted relative accuracy and outcomes
    wro = calculate_weighted_relative_outcomes(df, query)
    wre = calculate_weighted_relative_accuracy(df, query, model)

    # Odds ratio of accuracy
    odds_ratio_acc = calculate_odds_ratio_acc(df, query, model)

    # Bootstrap p-value acc
    pval_bootstrap = bootstrapping_test_for_accuracy(df, model, query)
    
    return {
        'group_size': group_size, # size of the subgroup
        'support': relative_size, # support of the subgroup
        'p_value_mc': p_value, # p-value for evaluating whether the accuracy is different in the subgroup from average accuracy
        'p_value_t': p_value_t, # p-value for evaluating whether the accuracy is different in the subgroup from average accuracy
        'p_value_chi': p_value_chi, # p-value for evaluating whether the accuracy is different in the subgroup from average accuracy
        'p_value_bootstrap': pval_bootstrap, # p-value for evaluating whether the accuracy is different in the subgroup from average accuracy
        'num_criteria': num_criteria, # number of criteria in the subgroup
        'outcome_diff': outcome_diff, # difference in the outcome between the subgroup and the entire dataset
        'accuracy_diff': accuracy_diff, # difference in the accuracy between the subgroup and the entire dataset
        'odds_ratio_outcome': odds_ratio, # odds ratio of the outcome
        'odds_ratio_acc': odds_ratio_acc, # odds ratio of the accuracy
        'query': query,
        'lift_outcome': lift,
        'lift_acc': lift_acc, # lift of the accuracy
        'weighted_relative_outcome': wro, # Weighted relative outcomes
        'weighted_relative_accuracy': wre # Weighted relative accuracy
    }

def calculate_group_statistics_string(X, y, model, query, ohe, num_iterations=250):
    # Calculate the dataframe
    df = deepcopy(X)
    df['y'] = y
    # Filter the subgroup
    if len(query) == 0 or len(df.query(query)) == 0 or len(df.query(query)) == len(df):
        return {
            'group_size': 0,
            'support': 0,
            'p_value_mc': 1,
            'p_value_t': 1,
            'p_value_chi': 1,
             'p_value_bootstrap': 1, 
            'num_criteria': 0,
            'outcome_diff': 0,
            'accuracy_diff': 0,
            'odds_ratio_outcome': np.nan,
             'odds_ratio_acc': np.nan, # odds ratio of the accuracy
            'query': query,
            'lift_outcome': np.nan,
            'lift_acc': np.nan, # lift of the accuracy
            'weighted_relative_outcome': np.nan,
            'weighted_relative_accuracy': np.nan
        }
    
    subgroup = df.query(query)
    # Calculate statistics
    group_size = len(subgroup)
    relative_size = group_size / len(df)
    num_criteria = query.count("and") + 1 # Counting 'and' and adding 1 for the first condition

    # Outcome difference
    avg_outcome_dataset = y.mean()
    avg_outcome_subgroup = subgroup['y'].mean()
    outcome_diff = abs(avg_outcome_dataset - avg_outcome_subgroup)

    # Transform the dataframe (except for y) to one-hot encoding
    df_ohe = pd.DataFrame(ohe.transform(df.drop('y', axis=1)), columns=ohe.get_feature_names_out())
    df_ohe['y'] = df['y']

    subgroup_ohe = pd.DataFrame(ohe.transform(subgroup.drop('y', axis=1)), columns=ohe.get_feature_names_out())
    subgroup_ohe['y'] = subgroup['y']

    X_ohe = df_ohe.drop('y', axis=1)

    accuracy_dataset = accuracy_score(y, model.predict(X_ohe))
    accuracy_subgroup = accuracy_score(subgroup['y'], model.predict(subgroup_ohe.drop('y', axis=1)))
    accuracy_diff = abs(accuracy_dataset - accuracy_subgroup)

    
    # Bootstrap p-value acc
    pval_bootstrap = bootstrapping_test_for_accuracy_string(df_ohe, model, subgroup_ohe)
    
    return {
        'group_size': group_size, # size of the subgroup
        'support': relative_size, # support of the subgroup
      'p_value_bootstrap': pval_bootstrap, # p-value for evaluating whether the accuracy is different in the subgroup from average accuracy
        'num_criteria': num_criteria, # number of criteria in the subgroup
        'outcome_diff': outcome_diff, # difference in the outcome between the subgroup and the entire dataset
        'accuracy_diff': accuracy_diff, # difference in the accuracy between the subgroup and the entire dataset
        'query': query,

    }

def compute_differences_metrics_two_datasets(metrics_1, metrics_2):
    """Computes the differences between many metrics between the two datasets X1 and X2 that 
    might come from different populations, or be simple train-test splits.
    
    IMPORTANT: Differences are calculated as X2 - X1, so a positive difference means that X2 is higher than X1."""

    # Calculate the differences
    metrics_diff = {}
    for metric in metrics_1:
        # CHeck if numeric
        if isinstance(metrics_1[metric], (int, float)):
            metrics_diff[metric] = metrics_2[metric] - metrics_1[metric]
        else:
            metrics_diff[metric] = None

    return metrics_diff