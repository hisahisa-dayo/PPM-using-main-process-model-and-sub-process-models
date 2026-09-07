import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from attribute_filter import is_excluded_attribute
from dt_router import _categorical_columns, replace_rare_categories
from pm_evaluator import get_next_activity_probabilities


ROUTER_DROP_COLUMNS = [
    'case_id', 'step_index_x', 'ground_truth', 'predicted_event',
    'is_correct', 'log_step_index', 'case', 'event', 'time',
    'step_index_y', 'step_index', 'previous_activity', 'current_activity',
    'prefix_length', 'current_activity_count', 'consecutive_loop_count',
    'previous_same_activity_count',
    'Variant', 'Variant index', 'Variant.1'
]

PREDICTOR_DROP_COLUMNS = [
    'case_id', 'step_index_x', 'ground_truth', 'predicted_event',
    'is_correct', 'log_step_index', 'case', 'event', 'time',
    'step_index_y', 'step_index',
    'Variant', 'Variant index', 'Variant.1',
    'hybrid_predicted_event', 'pre_attribute_predicted_event',
    'used_model', 'selected_model_key',
    'attribute_adjusted_prediction', 'attribute_prediction_confidence',
    'process_model_probability_margin'
]


def _encode_frame(raw_frame, frequent_category_values=None, columns=None,
                  min_category_frequency=10, categorical_columns=None):
    if categorical_columns is None:
        categorical_columns = _categorical_columns(raw_frame)
    if frequent_category_values is None:
        frequent_category_values = {}
        for column in categorical_columns:
            values = raw_frame[column].fillna('__MISSING__').astype(str)
            counts = values.value_counts(dropna=False)
            frequent_category_values[column] = set(
                counts[counts > min_category_frequency].index.astype(str)
            )

    encoded_raw = replace_rare_categories(raw_frame, frequent_category_values)
    encoded = pd.get_dummies(encoded_raw, drop_first=True).fillna(0)
    for column in categorical_columns:
        other_feature = f'{column}_OTHER'
        if columns is None and other_feature not in encoded.columns:
            encoded[other_feature] = 0

    if columns is not None:
        encoded = encoded.reindex(columns=columns, fill_value=0)
    return encoded, frequent_category_values


def _router_leaf_indices(df_activity, router_model):
    clf, encoded_feature_names, frequent_category_values = router_model
    raw = df_activity.drop(
        columns=[c for c in ROUTER_DROP_COLUMNS if c in df_activity.columns]
    )
    raw = raw[
        [column for column in raw.columns
         if not is_excluded_attribute(column)]
    ]
    encoded, _ = _encode_frame(
        raw,
        frequent_category_values=frequent_category_values,
        columns=encoded_feature_names
    )
    return clf.apply(encoded)


def assign_model_keys(df, router_models, selected_rules):
    """Reproduce the existing router and label each prediction point."""
    model_keys = pd.Series('main', index=df.index, dtype='object')
    rule_lookup = {
        (rule['Activity'], rule['Leaf_ID']): f"sub_{rank}"
        for rank, rule in enumerate(selected_rules, 1)
    }

    for activity, router_model in router_models.items():
        activity_mask = df['current_activity'] == activity
        if not activity_mask.any():
            continue
        df_activity = df.loc[activity_mask]
        leaf_indices = _router_leaf_indices(df_activity, router_model)
        for index, leaf_id in zip(df_activity.index, leaf_indices):
            model_keys.at[index] = rule_lookup.get(
                (activity, leaf_id), 'main'
            )
    return model_keys


def _predictor_raw_features(frame, excluded_columns):
    drop_columns = set(PREDICTOR_DROP_COLUMNS) | set(excluded_columns)
    raw = frame.drop(columns=[c for c in drop_columns if c in frame.columns])
    return raw[
        [column for column in raw.columns
         if not is_excluded_attribute(column, excluded_columns)]
    ]


def train_attribute_transition_predictors(
    df_train, router_models, selected_rules, process_model_sources,
    activity_to_id, args
):
    """Train trees only on points where the selected PM is ambiguous."""
    model_keys = assign_model_keys(df_train, router_models, selected_rules)
    eligible_mask = pd.Series(False, index=df_train.index, dtype='bool')

    for index, row in df_train.iterrows():
        model_key = model_keys.at[index]
        model_source = process_model_sources.get(model_key)
        if model_source is None:
            continue
        trans_matrix, second_order_model = model_source
        probabilities = get_next_activity_probabilities(
            trans_matrix,
            second_order_model,
            activity_to_id,
            row['current_activity'],
            row['previous_activity']
        )
        if probabilities is None:
            continue
        if (
            process_model_probability_margin(probabilities)
            <= args.attribute_probability_margin
        ):
            eligible_mask.at[index] = True

    predictors = {}

    eligible_model_keys = model_keys.loc[eligible_mask]
    for model_key in eligible_model_keys.unique():
        subset = df_train.loc[eligible_mask & (model_keys == model_key)]
        labels = subset['ground_truth'].astype(str)
        if (
            len(subset) < args.attribute_tree_min_samples
            or labels.nunique() < 2
        ):
            continue

        raw = _predictor_raw_features(
            subset, args.attribute_predictor_excluded_columns
        )
        encoded, frequent_values = _encode_frame(
            raw,
            min_category_frequency=args.categorical_min_frequency
        )
        if encoded.shape[1] == 0:
            continue

        clf = DecisionTreeClassifier(
            max_depth=args.attribute_tree_max_depth,
            min_samples_leaf=args.attribute_tree_min_samples_leaf,
            random_state=args.random_state
        )
        clf.fit(encoded, labels)
        predictors[model_key] = {
            'classifier': clf,
            'encoded_feature_names': encoded.columns.tolist(),
            'frequent_category_values': frequent_values,
            'categorical_columns': _categorical_columns(raw),
            'num_samples': len(subset),
            'num_classes': labels.nunique()
        }

    return predictors, eligible_model_keys


def process_model_probability_margin(probabilities):
    positive = probabilities[probabilities > 0.0]
    if len(positive) < 2:
        return 1.0
    top_two = np.partition(positive, -2)[-2:]
    return float(top_two.max() - top_two.min())


def predict_attribute_aware_next_activity(
    predictor, row, outgoing_activities, excluded_columns,
    min_confidence
):
    """Predict after masking classes not present in the selected process model."""
    raw = _predictor_raw_features(pd.DataFrame([row.to_dict()]), excluded_columns)
    encoded, _ = _encode_frame(
        raw,
        frequent_category_values=predictor['frequent_category_values'],
        columns=predictor['encoded_feature_names'],
        categorical_columns=predictor['categorical_columns']
    )
    classifier = predictor['classifier']
    probabilities = classifier.predict_proba(encoded)[0]
    masked = {
        str(activity): float(probability)
        for activity, probability in zip(classifier.classes_, probabilities)
        if str(activity) in outgoing_activities and probability > 0.0
    }
    probability_sum = sum(masked.values())
    if probability_sum <= 0.0:
        return '', 0.0

    prediction, probability = max(masked.items(), key=lambda item: item[1])
    confidence = probability / probability_sum
    if confidence < min_confidence:
        return '', confidence
    return prediction, confidence
