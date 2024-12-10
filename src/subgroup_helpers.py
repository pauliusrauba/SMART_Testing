import pandas as pd

import warnings
from time import perf_counter

from pandas.api.types import is_numeric_dtype

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
model = LogisticRegression(random_state=0, max_iter=1000)
ohe = OneHotEncoder()
from benchmarks.aif360_old.detectors.mdss.ScoringFunctions.BerkJones import BerkJones
from benchmarks.aif360_old.detectors.mdss.MDSS import MDSS

import pysubgroup as ps
import ast
import re
import numpy as np
import sys
from benchmarks.divexplorer.divexplorer.FP_DivergenceExplorer import FP_DivergenceExplorer
from benchmarks.divexplorer.divexplorer.FP_Divergence import FP_Divergence
from benchmarks.slicefinder.SliceLogisticRegression import MyFakeLR
from benchmarks.slicefinder.slice_finder import SliceFinder
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import accuracy_score
from statsmodels.stats.contingency_tables import mcnemar
from benchmarks.sliceline.slicefinder import Slicefinder

def mdss_group(X_train, y_train):
    """Get the MDSS group as a dictionary. Returns a single subgroup"""

    X_train = X_train.copy()
    y_train = y_train.copy()
    
    # Get the expected probability
    expected_prob = y_train.mean()
    direction = 'positive'
    try:
        penalty = 1
        num_iters = 10
        search_space = list(X_train.columns)
        X_train['expected_prob'] = expected_prob

        scoring_function = BerkJones(direction=direction, alpha = expected_prob)
        scanner = MDSS(scoring_function)

        start = perf_counter()
        subset, score = scanner.parallel_scan(coordinates = X_train[search_space], outcomes = y_train, expectations = X_train['expected_prob'], penalty = penalty, num_iters = num_iters)
        end = perf_counter()

    except:
        penalty = 0
        num_iters = 10
        search_space = list(X_train.columns)
        X_train['expected_prob'] = expected_prob

        scoring_function = BerkJones(direction=direction, alpha = expected_prob)
        scanner = MDSS(scoring_function)

        start = perf_counter()
        subset, score = scanner.parallel_scan(coordinates = X_train[search_space], outcomes = y_train, expectations = X_train['expected_prob'], penalty = penalty, num_iters = num_iters)
        end = perf_counter()

    return subset, score

def generate_mdss_query_condition(subset):
    """
    Generates a query condition string from a given subset dictionary.
    The condition generated will be usable with DataFrame.query() method.

    Args:
    subset (dict): A dictionary where keys are column names and values are lists of acceptable values.

    Returns:
    str: A string that represents the query condition.
    """

    conditions = []
    for key, values in subset.items():
        # For each key, create a condition string
        condition = f"{key} in {values}"
        conditions.append(condition)

    # Join all conditions with ' and '
    query_condition = ' and '.join(conditions)
    return query_condition


def pysubgroup_group(X_train, y_train, n_subgroups=5, beamsearch=True):
    # Merge the X_train and y_train
    df = pd.concat([X_train, y_train], axis=1)
    target_name = y_train.name
    target = ps.BinaryTarget(target_name, True)
    search_space = ps.create_selectors(df, ignore=[target_name])

    task = ps.SubgroupDiscoveryTask (
        data = df, 
        target = target, 
        search_space = search_space, 
        result_set_size=n_subgroups, 
        depth=5, 
        qf=ps.WRAccQF())

    start = perf_counter()   
    if beamsearch: result = ps.BeamSearch().execute(task)
    else: result = ps.Apriori().execute(task)
    end = perf_counter()

    desc = result.to_dataframe()
    return list(desc['subgroup'])

def transform_conditions(conditions):
    final_desc_format = {}

    for i, condition in enumerate(conditions):
        # Convert to string if it's not already
        condition_str = str(condition) if not isinstance(condition, str) else condition

        # Replace 'AND' with 'and'
        condition_str = condition_str.replace("AND", "and")

        # Replace the custom range notation
        condition_str = re.sub(r'\[(\d+):(\d+)\[', lambda m: f">= {m.group(1)} and < {m.group(2)}", condition_str)

        # Replace equality '==' with '=' for pandas query compatibility
        #condition_str = condition_str.replace("==", "=")

        final_desc_format[i] = condition_str

    return final_desc_format

def transform_for_query(conditions):
    transformed_conditions = {}

    for key, condition in conditions.items():
        # Identify and transform range conditions
        condition = re.sub(r'(\w+):\s*>=\s*(\d+)\s+and\s+<\s*(\d+)', r'\1 >= \2 and \1 < \3', condition)

        # Ensure single quotes are used for string literals
        condition = condition.replace('"', "'")

        transformed_conditions[key] = condition

    return transformed_conditions


def divergence_group(X_train, y_train, model, n_subgroups=5):

    # Get the predicted class
    le = LabelEncoder()
    X_model = X_train.copy()
    for column in X_model.columns:
        if X_model.dtypes[column] == "object" or X_model[column].dtypes.name == "category":
            X_model[column] = le.fit_transform(X_model[column])
    model.fit(X_model.values, y_train)
    y_train_pred = model.predict(X_model)

    # Merge the X_train and y_train
    y_train_pred = pd.Series(y_train_pred)
    y_train_pred.name = "predicted_class"
    df = pd.concat([X_train, y_train, y_train_pred], axis=1)
    target_name = y_train.name
    ypred_name = y_train_pred.name
    min_sup=0.02

    start = perf_counter()
    fp_diver=FP_DivergenceExplorer(df, true_class_name = target_name, predicted_class_name = ypred_name, class_map={"P":1, "N":0})
    FP_fm=fp_diver.getFrequentPatternDivergence(min_support=min_sup, metrics=["d_accuracy"],FPM_type='fpgrowth')
    fp_divergence=FP_Divergence(FP_fm, "d_accuracy")
    FP_sorted=fp_divergence.getDivergence(th_redundancy=0)
    FP_sorted = FP_sorted.sort_values(by = "d_accuracy").head(n_subgroups)
    end = perf_counter()
    return FP_sorted

def create_query_dictionary(ls_conditions, is_numeric_string=True):
    """
    Corrects the query creation by adding quotes around the string values in each condition.

    Args:
    ls_conditions (list): A list of frozensets, each containing conditions as strings.

    Returns:
    dict: A dictionary where keys are indices and values are corrected query strings.
    """
    query_dict = {}

    for index, condition_set in enumerate(ls_conditions):
        corrected_conditions = []
        for condition in condition_set:
            key, value = condition.split('=')
  
            if is_numeric_string: corrected_condition = f"{key} == '{value}'"
            else: corrected_condition = f"{key} == {value}"
            corrected_conditions.append(corrected_condition)

        # Join corrected conditions with ' and '
        query_dict[index] = ' and '.join(corrected_conditions)

    return query_dict


def slicefinder_query(X_train, y_train, model, n_subgroups=5):
    """Get a subset for slicefinder"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        search_space = list(X_train.columns)

        for column in search_space:
            if X_train[column].dtype != "int64":
                X_train[column] = X_train[column].astype('category')

        encoders = {}
        for column in search_space:
            if X_train.dtypes[column] == "object" or X_train[column].dtypes.name == "category":
                le = LabelEncoder()
                X_train[column] = le.fit_transform(X_train[column])
                encoders[column] = le

        model.fit(X_train.values, y_train)
        sf = SliceFinder(model, (X_train[search_space], y_train))

        start = perf_counter()
        #recommendations = sf.find_slice(k=50, epsilon=0.1, degree=5,max_workers=4, max_time=300)
        recommendations = sf.find_slice(k=n_subgroups, epsilon=0.1, degree=5,max_workers=4, max_time=300)
        end = perf_counter()

        subset = {}
        for i, s in enumerate(recommendations):
            subset[i] = {}
            for k, v in list(s.filters.items()):
                values = ''
                if k in encoders:
                    le = encoders[k]
                    for v_ in v:
                        values += '%s '%(le.inverse_transform(v_)[0])
                else:
                    for v_ in sorted(v, key=lambda x: x[0]):
                        if len(v_) > 1:
                            values += '%s ~ %s'%(v_[0], v_[1])
                        else:
                            values += '%s '%(v_[0])
                #print ('%s:%s'%(k, values))
                temp_v = values.strip()
                subset[i][k] = [temp_v]

        # Return only the top n_subgroups entries from this dictionary
        subset = {k: v for k, v in subset.items() if k < n_subgroups}

    return subset

def slicefinder_query_transform(input_dict, is_numeric_string=True):
    """
    Generate a dictionary of pandas query strings from a given dictionary, where each key in the original 
    dictionary corresponds to a separate query.

    Args:
    input_dict (dict): A dictionary where keys are integers and values are dictionaries,
                       with keys as column names and values as lists of conditions.

    Returns:
    dict: A dictionary where each key corresponds to an individual query string.
    """

    all_copies = {}

    # Iterate over each item in the dictionary
    for i, keycondition in enumerate(input_dict.items()):

        key, conditions = keycondition

        sub_query_parts = []

        # Iterate over each condition for a given key
        for column, values in conditions.items():
            for value in values:
                if '~' in value:
                    # Handle range values
                    range_values = value.split('~')
                    if len(range_values) == 2:
                        start, end = range_values
                        column_conditions = f"{column}.between({start}, {end})"
                else:
                    # Handle exact match values
                    if is_numeric_string: column_conditions = f"{column} == '{value}'"
                    else: column_conditions = f"{column} == {value}"
                
                sub_query_parts.append(f"({column_conditions})")

        # Combine all sub-query parts for the current key using 'and'
        combined_sub_query = " and ".join(sub_query_parts)
        all_copies[i] = combined_sub_query

    return all_copies


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


def calculate_group_statistics(df, y, model, query, num_iterations=1000):
    df['y'] = y
    
    # Filter the subgroup
    if len(query) == 0 or len(df.query(query)) == 0:
        return {
            'group_size': 0,
            'relative_size': 0,
            'p_value': 1,
            'num_criteria': 0,
            'outcome_diff': 0,
            'accuracy_diff': 0
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
    accuracy_dataset = accuracy_score(y, model.predict(df.drop('y', axis=1)))
    accuracy_subgroup = accuracy_score(subgroup['y'], model.predict(subgroup.drop('y', axis=1)))
    accuracy_diff = abs(accuracy_dataset - accuracy_subgroup)

    # P-value calculation (randomization-based testing)
    p_value = mcnemars_test(df, model, query)
    
    return {
        'group_size': group_size,
        'relative_size': relative_size,
        'p_value': p_value,
        'num_criteria': num_criteria,
        'outcome_diff': outcome_diff,
        'accuracy_diff': accuracy_diff
    }


def sliceline_dataframe(X, y, model, ohe=None):
    sf = Slicefinder(k=1, alpha=0.95, max_l=20, verbose=False)

    y_pred = model.predict(X)
    training_errors = (y - y_pred)**2
    sf.fit(X, training_errors)

    # Get the slice dataframe
    df_sliceline = pd.DataFrame(sf.top_slices_, columns=sf.feature_names_in_, index=sf.get_feature_names_out())

    return df_sliceline

def sliceline_dataframe_to_query_dict(df):
    query_dict = {}

    # Iterate over each row in the DataFrame
    for i, (index, row) in enumerate(df.iterrows()):
        conditions = []

        # Iterate over each column in the row
        for col in df.columns:
            value = row[col]

            # If the value is not None, add a condition to the list
            if pd.notna(value):
                if isinstance(value, str):
                    # Add quotes around string values
                    conditions.append(f"'{col}' == '{value}'")
                else:
                    conditions.append(f"`{col}` == {value}")

        # Join conditions with 'and' and add to the dictionary
        if conditions:
            query_dict[i] = ' and '.join(conditions)

    return query_dict