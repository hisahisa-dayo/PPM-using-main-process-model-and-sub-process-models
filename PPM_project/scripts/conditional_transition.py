"""Attribute-conditioned transition correction for the proposed method."""

from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from attribute_transition import assign_model_keys
from attribute_filter import is_excluded_attribute
from pm_evaluator import get_next_activity_probabilities


EVALUATION_COLUMNS = {
    'case_id', 'step_index_x', 'step_index_y', 'log_step_index',
    'previous_activity', 'current_activity', 'prefix_length',
    'current_activity_count', 'consecutive_loop_count',
    'previous_same_activity_count', 'ground_truth', 'predicted_event',
    'is_correct', 'hybrid_predicted_event', 'used_model',
    '_model_key', '_pm_prediction', '_pm_confidence',
    '_allowed_activities'
}


def candidate_attributes(frame, args):
    """Return attributes eligible for conditional-probability selection."""
    excluded = set(args.conditional_excluded_columns) | EVALUATION_COLUMNS
    return [
        column for column in frame.columns
        if not is_excluded_attribute(column, excluded)
        and frame[column].isna().mean() <= args.conditional_max_missing_ratio
        and frame[column].nunique(dropna=True) >= 2
    ]


def _previous_key(value):
    return '__START__' if pd.isna(value) else str(value)


def _numeric_attribute(column, series, args):
    name = column.lower()
    return pd.api.types.is_numeric_dtype(series) and any(
        token in name for token in args.conditional_numeric_name_tokens
    )


def _fit_encoder(series, column, args):
    if _numeric_attribute(column, series, args):
        values = pd.to_numeric(series, errors='coerce').dropna()
        if values.nunique() < 2:
            return None
        quantiles = np.linspace(0, 1, args.conditional_numeric_bins + 1)[1:-1]
        internal = np.unique(values.quantile(quantiles).to_numpy(dtype=float))
        edges = np.concatenate(([-np.inf], internal, [np.inf]))
        if len(edges) < 3:
            return None
        return {'kind': 'numeric', 'edges': edges.tolist()}

    values = series.fillna('__MISSING__').astype(str)
    counts = values.value_counts()
    frequent = set(
        counts[counts > args.categorical_min_frequency].index.astype(str)
    )
    if not frequent:
        return None
    return {'kind': 'categorical', 'frequent': frequent}


def _transform(series, encoder):
    if encoder['kind'] == 'numeric':
        values = pd.to_numeric(series, errors='coerce')
        bins = pd.cut(
            values, encoder['edges'], labels=False, include_lowest=True
        )
        return bins.map(
            lambda value: (
                '__MISSING__' if pd.isna(value) else f'BIN_{int(value)}'
            )
        ).astype(str)

    values = series.fillna('__MISSING__').astype(str)
    return values.where(values.isin(encoder['frequent']), 'OTHER')


def _build_table(frame, attribute_values):
    table = defaultdict(Counter)
    for (_, row), value in zip(frame.iterrows(), attribute_values):
        key = (_previous_key(row['previous_activity']), str(value))
        table[key][str(row['ground_truth'])] += 1
    return dict(table)


def _conditional_predict(row, value, table, min_support):
    counts = table.get((_previous_key(row['previous_activity']), str(value)))
    if not counts:
        return '', 0, 0.0

    allowed = row['_allowed_activities']
    usable = {
        activity: count for activity, count in counts.items()
        if activity in allowed
    }
    support = int(sum(usable.values()))
    if support < min_support or not usable:
        return '', support, 0.0

    max_count = max(usable.values())
    tied = {activity for activity, count in usable.items() if count == max_count}
    pm_prediction = str(row['_pm_prediction'])
    if pm_prediction in tied:
        prediction = pm_prediction
    else:
        prediction = sorted(tied)[0]
    return prediction, support, float(max_count / support)


def _split_by_case(frame, args):
    case_ids = frame['case_id'].astype(str).unique()
    if len(case_ids) < 2:
        return frame.iloc[0:0], frame.iloc[0:0]
    build_ids, validation_ids = train_test_split(
        case_ids,
        test_size=args.conditional_validation_size,
        random_state=args.random_state,
        shuffle=True
    )
    values = frame['case_id'].astype(str)
    return (
        frame.loc[values.isin(set(build_ids))],
        frame.loc[values.isin(set(validation_ids))]
    )


def add_process_model_columns(
    frame, model_keys, model_sources, activity_to_id, id_to_activity
):
    """Attach the routed PM prediction, confidence, and outgoing activities."""
    result = frame.copy()
    predictions, confidences, allowed_values = [], [], []

    for index, row in result.iterrows():
        model_key = str(model_keys.at[index])
        source = model_sources.get(model_key, model_sources['main'])
        probabilities = get_next_activity_probabilities(
            source[0], source[1], activity_to_id,
            row['current_activity'], row['previous_activity']
        )
        if probabilities is None:
            predictions.append('')
            confidences.append(np.nan)
            allowed_values.append(set())
            continue

        prediction_id = int(np.argmax(probabilities))
        predictions.append(id_to_activity[prediction_id])
        confidences.append(float(probabilities[prediction_id]))
        allowed_values.append({
            id_to_activity[i]
            for i, probability in enumerate(probabilities)
            if probability > 0.0
        })

    result['_model_key'] = model_keys.reindex(result.index).astype(str)
    result['_pm_prediction'] = predictions
    result['_pm_confidence'] = confidences
    result['_allowed_activities'] = allowed_values
    return result


def prepare_training_predictions(
    frame, router_models, selected_rules, model_sources,
    activity_to_id, id_to_activity
):
    model_keys = assign_model_keys(frame, router_models, selected_rules)
    return add_process_model_columns(
        frame, model_keys, model_sources, activity_to_id, id_to_activity
    )


def select_conditional_attributes(df_train, attributes, args):
    """Select one validated attribute per (model, current activity)."""
    selectors, diagnostics = {}, []
    uncertain = df_train[
        df_train['_pm_confidence'] < args.pm_confidence_threshold
    ]

    for (model_key, current_activity), subset in uncertain.groupby(
        ['_model_key', 'current_activity'], dropna=False
    ):
        if len(subset) < args.conditional_min_training_samples:
            continue

        build, validation = _split_by_case(subset, args)
        if (
            len(build) < args.conditional_min_training_samples
            or len(validation) < args.conditional_min_validation_samples
        ):
            continue

        truth = validation['ground_truth'].astype(str).to_numpy()
        baseline = validation['_pm_prediction'].astype(str).to_numpy()
        baseline_correct = int(np.sum(truth == baseline))
        best = None

        for attribute in attributes:
            if attribute not in build.columns:
                continue
            encoder = _fit_encoder(build[attribute], attribute, args)
            if encoder is None:
                continue
            table = _build_table(
                build, _transform(build[attribute], encoder)
            )
            validation_values = _transform(validation[attribute], encoder)
            final_predictions, applied = [], 0
            for (_, row), value in zip(
                validation.iterrows(), validation_values
            ):
                prediction, _, _ = _conditional_predict(
                    row, value, table, args.conditional_min_value_support
                )
                final_predictions.append(
                    prediction or str(row['_pm_prediction'])
                )
                applied += bool(prediction)

            corrected = int(np.sum(
                truth == np.asarray(final_predictions, dtype=object)
            ))
            candidate = {
                'attribute': attribute,
                'corrected': corrected,
                'gain': corrected - baseline_correct,
                'applied': applied
            }
            if best is None or (
                candidate['corrected'], candidate['applied']
            ) > (best['corrected'], best['applied']):
                best = candidate

        gain_points = (
            best['gain'] / len(validation) * 100.0 if best else 0.0
        )
        accepted = bool(
            best
            and best['gain'] >= args.conditional_min_validation_net_gain
            and gain_points >= args.conditional_min_validation_gain_points
        )
        diagnostics.append({
            'model_key': model_key,
            'current_activity': current_activity,
            'eligible_samples': len(subset),
            'construction_samples': len(build),
            'validation_samples': len(validation),
            'baseline_correct': baseline_correct,
            'selected_attribute': best['attribute'] if best else '',
            'conditional_correct': (
                best['corrected'] if best else baseline_correct
            ),
            'validation_net_gain': best['gain'] if best else 0,
            'validation_gain_points': gain_points,
            'validation_applied': best['applied'] if best else 0,
            'accepted': accepted
        })

        if accepted:
            attribute = best['attribute']
            encoder = _fit_encoder(subset[attribute], attribute, args)
            selectors[(str(model_key), str(current_activity))] = {
                'attribute': attribute,
                'encoder': encoder,
                'table': _build_table(
                    subset, _transform(subset[attribute], encoder)
                )
            }

    return selectors, pd.DataFrame(diagnostics)


def apply_conditional_correction(df_test, selectors, args):
    """Apply selected corrections and return predictions plus diagnostics."""
    predictions, attributes, supports, probabilities, applied_values = (
        [], [], [], [], []
    )

    for _, row in df_test.iterrows():
        prediction = str(row['_pm_prediction'])
        attribute, support = '', 0
        conditional_probability, applied = np.nan, False

        if row['_pm_confidence'] < args.pm_confidence_threshold:
            selector = selectors.get((
                str(row['_model_key']), str(row['current_activity'])
            ))
            if selector:
                attribute = selector['attribute']
                value = _transform(
                    pd.Series([row[attribute]]), selector['encoder']
                ).iloc[0]
                replacement, support, conditional_probability = (
                    _conditional_predict(
                        row, value, selector['table'],
                        args.conditional_min_value_support
                    )
                )
                if replacement:
                    prediction = replacement
                    applied = True

        predictions.append(prediction)
        attributes.append(attribute)
        supports.append(support)
        probabilities.append(conditional_probability)
        applied_values.append(applied)

    return {
        'predictions': predictions,
        'selected_attributes': attributes,
        'supports': supports,
        'conditional_probabilities': probabilities,
        'applied': applied_values
    }
