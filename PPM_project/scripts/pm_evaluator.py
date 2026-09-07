import numpy as np
import pandas as pd


def get_next_activity_probabilities(
    trans_matrix, second_order_model, matrix_act_to_id,
    current_act, previous_act=None
):
    """Return the probability vector used by the process-model predictor."""
    if (
        previous_act is not None
        and previous_act in matrix_act_to_id
        and current_act in matrix_act_to_id
    ):
        second_order_key = (
            matrix_act_to_id[previous_act], matrix_act_to_id[current_act]
        )
        probs = second_order_model.get(second_order_key)
        if probs is not None and np.max(probs) > 0.0:
            return probs

    if current_act in matrix_act_to_id:
        probs = trans_matrix[matrix_act_to_id[current_act]]
        if np.max(probs) > 0.0:
            return probs

    return None


def predict_next_activity(trans_matrix, second_order_model, matrix_act_to_id, id_to_act, current_act, previous_act=None):
    probs = get_next_activity_probabilities(
        trans_matrix, second_order_model, matrix_act_to_id,
        current_act, previous_act
    )
    return id_to_act[np.argmax(probs)] if probs is not None else ""


def evaluate_pm_on_train(trans_matrix, matrix_act_to_id, train_data_with_ids, second_order_model=None):
    id_to_act = {v: k for k, v in matrix_act_to_id.items()}
    if second_order_model is None:
        second_order_model = {}

    results = []
    for case_id, trace in train_data_with_ids:
        trace_len = len(trace)
        if trace_len < 2:
            continue

        for step_idx in range(1, trace_len):
            previous_act = trace[step_idx - 2] if step_idx >= 2 else None
            current_act = trace[step_idx - 1]
            ground_truth = trace[step_idx]
            observed_prefix = trace[:step_idx]
            current_activity_count = observed_prefix.count(current_act)
            consecutive_loop_count = 0
            for activity in reversed(observed_prefix):
                if activity != current_act:
                    break
                consecutive_loop_count += 1
            predicted_event = predict_next_activity(
                trans_matrix,
                second_order_model,
                matrix_act_to_id,
                id_to_act,
                current_act,
                previous_act
            )

            results.append({
                'case_id': case_id,
                'step_index': step_idx,
                'previous_activity': previous_act,
                'current_activity': current_act,
                'prefix_length': step_idx,
                'current_activity_count': current_activity_count,
                'consecutive_loop_count': consecutive_loop_count,
                'previous_same_activity_count': current_activity_count - 1,
                'ground_truth': ground_truth,
                'predicted_event': predicted_event,
                'is_correct': 1 if predicted_event == ground_truth else 0
            })

    return pd.DataFrame(results)

