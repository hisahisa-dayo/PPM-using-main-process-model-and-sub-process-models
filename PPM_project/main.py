import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import argparse
import sys
import io
import contextlib
import time
import numpy as np
import pandas as pd
from pathlib import Path
import pm4py
from sklearn.model_selection import KFold

sys.path.append('./scripts')
from preprocessor import Preprocessor
from attribute_filter import COMMON_EXCLUDED_ATTRIBUTE_COLUMNS
from pm4py_dfg import ProcessModelGenerator
from pm_evaluator import evaluate_pm_on_train, predict_next_activity
from dt_router import train_decision_tree_router, replace_rare_categories
from conditional_transition import (
    add_process_model_columns,
    apply_conditional_correction,
    candidate_attributes,
    prepare_training_predictions,
    select_conditional_attributes,
)
from bi_lstm import run_bi_lstm_fold
from process_transformer import run_process_transformer_fold

IMPLEMENTED_METHODS = {
    'proposed', 'proposed_attr_transition',
    'single_pm', 'bi_lstm', 'transformer'
}


class Args:
    def __init__(self):
        #self.data_set = 'Helpdesk.csv'
        #self.data_set = 'BPIC2012.csv'
        #self.data_set = 'BPIC2017.csv'
        #self.data_set = 'Hospital.csv'
        self.data_set = 'RequestForPayment.csv'
        #self.data_set = 'PrepaidTravelCost.csv'

        self.data_dir = './data/'
        self.result_dir = './results/'

        # Select one method:
        # proposed, proposed_attr_transition, single_pm, bi_lstm, transformer
        self.experiment_method = 'proposed'

        self.num_folds = 10
        self.random_state = 42
        self.use_transition_matrix = True
        self.cross_validation = False
        self.task = 'act_pred_'

        self.min_support_ratio = 0.005
        self.error_threshold = 0.50

        # Attribute-conditioned transition correction settings.
        self.pm_confidence_threshold = 0.75
        self.conditional_validation_size = 0.20
        self.conditional_min_training_samples = 100
        self.conditional_min_validation_samples = 100
        self.conditional_min_value_support = 20
        self.conditional_min_validation_net_gain = 3
        self.conditional_min_validation_gain_points = 1.0
        self.conditional_numeric_bins = 4
        self.conditional_max_missing_ratio = 0.90
        self.conditional_excluded_columns = list(
            COMMON_EXCLUDED_ATTRIBUTE_COLUMNS
        )
        self.conditional_numeric_name_tokens = [
            'amount', 'cost', 'price', 'quantity', 'qty', 'duration',
            'value', 'number', 'count'
        ]
        self.save_conditional_diagnostics = True

        # Saving DFG images for every fold can be slow and creates many files.
        self.save_model_images = False
        self.num_saved_sub_models = 5
        self.print_submodel_diagnostics = False

        # Bi-LSTM baseline settings.
        self.bi_lstm_units = 100
        self.bi_lstm_dropout = 0.2
        self.bi_lstm_learning_rate = 0.002
        self.bi_lstm_epochs = 100
        self.bi_lstm_batch_size = 128
        self.bi_lstm_validation_split = 0.1
        self.bi_lstm_patience = 10
        self.bi_lstm_min_delta = 0.0
        self.bi_lstm_verbose = 1
        # Elkhawaga et al. (2022): keep categorical values occurring more
        # than 10 times in training data; group the rest into OTHER.
        self.categorical_min_frequency = 10
        self.bi_lstm_excluded_attribute_columns = list(
            COMMON_EXCLUDED_ATTRIBUTE_COLUMNS
        )

        # One-Tr attribute-aware ProcessTransformer settings. Model dimensions
        # follow the official ProcessTransformer implementation.
        self.transformer_embedding_dim = 36
        self.transformer_num_heads = 4
        self.transformer_key_dim = 36
        self.transformer_ff_dim = 64
        self.transformer_dropout = 0.1
        self.transformer_learning_rate = 0.001
        self.transformer_batch_size = 12
        self.transformer_epochs = 100
        self.transformer_validation_split = 0.1
        self.transformer_patience = 10
        self.transformer_min_delta = 0.0
        self.transformer_verbose = 1


def build_clean_cases(preprocessor):
    ids_list = preprocessor.data_structure['data']['ids_process_instances']
    instances_list = preprocessor.data_structure['data']['process_instances']
    all_cases_with_ids = list(zip(ids_list, instances_list))

    cleaned_cases_with_ids = []
    noise_activities = ['!', '_END_']
    for case_id, trace in all_cases_with_ids:
        cleaned_trace = [act for act in trace if act not in noise_activities]
        if len(cleaned_trace) > 1:
            cleaned_cases_with_ids.append((case_id, cleaned_trace))

    return cleaned_cases_with_ids


def build_global_activity_dict(cleaned_cases_with_ids):
    global_activities = set()
    for _, trace in cleaned_cases_with_ids:
        global_activities.update(trace)
    return {act: idx for idx, act in enumerate(sorted(global_activities))}


def encode_test_features(
    df_act, encoded_feature_names, drop_columns, frequent_category_values
):
    X_test_raw = df_act.drop(columns=[col for col in drop_columns if col in df_act.columns])
    X_test_raw = replace_rare_categories(
        X_test_raw, frequent_category_values
    )
    X_test_encoded = pd.get_dummies(X_test_raw, drop_first=True)

    X_test_encoded = X_test_encoded.reindex(
        columns=encoded_feature_names, fill_value=0
    )
    return X_test_encoded.replace([np.inf, -np.inf], np.nan).fillna(0)


def apply_command_line_overrides(args):
    """Override experiment settings without editing this source file."""
    parser = argparse.ArgumentParser(
        description='Run one next-activity prediction experiment.'
    )
    parser.add_argument('--data-set')
    parser.add_argument('--experiment-method', choices=sorted(IMPLEMENTED_METHODS))
    parser.add_argument('--error-threshold', type=float)
    parser.add_argument('--pm-confidence-threshold', type=float)
    options = parser.parse_args()

    if options.data_set is not None:
        args.data_set = options.data_set
    if options.experiment_method is not None:
        args.experiment_method = options.experiment_method
    if options.error_threshold is not None:
        args.error_threshold = options.error_threshold
    if options.pm_confidence_threshold is not None:
        args.pm_confidence_threshold = options.pm_confidence_threshold
    return args


def run_bi_lstm_cv_fold(
    args, fold_no, train_data_with_ids, test_data_with_ids,
    global_act_to_id, df_log_base
):
    print(f"\n=== Fold {fold_no}/{args.num_folds}: Bi-LSTM ===")
    print(f"Train Cases: {len(train_data_with_ids)}")
    print(f"Test Cases:  {len(test_data_with_ids)}")

    result = run_bi_lstm_fold(
        args,
        train_data_with_ids,
        test_data_with_ids,
        global_act_to_id,
        df_log_base
    )
    print(
        f"  > [Fold {fold_no}] Bi-LSTM: {result['correct']}/{result['test_events']} "
        f"({result['accuracy']:.2f}%)"
    )
    print(f"  > [Fold {fold_no}] Epochs: {result['epochs_trained']}")
    print(f"  > [Fold {fold_no}] Attribute features: {result['num_attribute_features']}")
    print(f"  > [Fold {fold_no}] Offline time: {result['offline_seconds']:.6f} sec")
    print(f"  > [Fold {fold_no}] Online time:  {result['online_seconds']:.6f} sec")

    return {
        'fold': fold_no,
        'method': args.experiment_method,
        'train_cases': len(train_data_with_ids),
        'test_cases': len(test_data_with_ids),
        'test_events': result['test_events'],
        'num_rules': 0,
        'num_sub_models': 0,
        'routed_events': 0,
        'main_correct': result['correct'],
        'hybrid_correct': result['correct'],
        'main_acc': result['accuracy'],
        'hybrid_acc': result['accuracy'],
        'improvement': 0.0,
        'method_correct': result['correct'],
        'method_acc': result['accuracy'],
        'epochs_trained': result['epochs_trained'],
        'num_attribute_features': result['num_attribute_features'],
        'offline_seconds': result['offline_seconds'],
        'online_seconds': result['online_seconds'],
        'online_ms_per_event': result['online_ms_per_event']
    }


def run_process_transformer_cv_fold(
    args, fold_no, train_data_with_ids, test_data_with_ids,
    global_act_to_id, df_log_base
):
    print(f"\n=== Fold {fold_no}/{args.num_folds}: ProcessTransformer ===")
    print(f"Train Cases: {len(train_data_with_ids)}")
    print(f"Test Cases:  {len(test_data_with_ids)}")

    result = run_process_transformer_fold(
        args,
        train_data_with_ids,
        test_data_with_ids,
        global_act_to_id,
        df_log_base
    )
    print(
        f"  > [Fold {fold_no}] ProcessTransformer: "
        f"{result['correct']}/{result['test_events']} "
        f"({result['accuracy']:.2f}%)"
    )
    print(f"  > [Fold {fold_no}] Epochs: {result['epochs_trained']}")
    print(f"  > [Fold {fold_no}] Attribute features: {result['num_attribute_features']}")
    print(f"  > [Fold {fold_no}] Offline time: {result['offline_seconds']:.6f} sec")
    print(f"  > [Fold {fold_no}] Online time:  {result['online_seconds']:.6f} sec")

    return {
        'fold': fold_no,
        'method': args.experiment_method,
        'train_cases': len(train_data_with_ids),
        'test_cases': len(test_data_with_ids),
        'test_events': result['test_events'],
        'num_rules': 0,
        'num_sub_models': 0,
        'routed_events': 0,
        'main_correct': result['correct'],
        'hybrid_correct': result['correct'],
        'main_acc': result['accuracy'],
        'hybrid_acc': result['accuracy'],
        'improvement': 0.0,
        'method_correct': result['correct'],
        'method_acc': result['accuracy'],
        'epochs_trained': result['epochs_trained'],
        'num_attribute_features': result['num_attribute_features'],
        'offline_seconds': result['offline_seconds'],
        'online_seconds': result['online_seconds'],
        'online_ms_per_event': result['online_ms_per_event']
    }


def run_fold(args, fold_no, train_data_with_ids, test_data_with_ids, global_act_to_id, log_file_path, df_log_base):
    if args.experiment_method == 'bi_lstm':
        return run_bi_lstm_cv_fold(
            args, fold_no, train_data_with_ids, test_data_with_ids,
            global_act_to_id, df_log_base
        )
    if args.experiment_method == 'transformer':
        return run_process_transformer_cv_fold(
            args, fold_no, train_data_with_ids, test_data_with_ids,
            global_act_to_id, df_log_base
        )

    fold_result_dir = os.path.join(args.result_dir, 'process_model', f'cv_fold_{fold_no:02d}')
    Path(fold_result_dir).mkdir(parents=True, exist_ok=True)

    print(f"\n=== Fold {fold_no}/{args.num_folds}: Data Split ===")
    print(f"Train Cases: {len(train_data_with_ids)}")
    print(f"Test Cases:  {len(test_data_with_ids)}")

    train_instances = [trace for _, trace in train_data_with_ids]
    offline_time_sec = 0.0

    print("\n=== Generating Process Model ===")
    started_at = time.perf_counter()
    pm_gen = ProcessModelGenerator(train_instances, global_act_to_id)
    trans_matrix, matrix_act_to_id, dfg_main, start_main, end_main, second_order_model = pm_gen.calculate_transition_matrix()
    offline_time_sec += time.perf_counter() - started_at

    if args.save_model_images:
        main_vis_path = os.path.join(fold_result_dir, 'dfg_main_model.png')
        pm4py.save_vis_dfg(dfg_main, start_main, end_main, main_vis_path)

    all_models = {}
    selected_rules = []
    df_router_train = None
    proposed_methods = {'proposed', 'proposed_attr_transition'}
    if args.experiment_method in proposed_methods:
        print("\n=== Evaluating Process Model on Train Data ===")
        started_at = time.perf_counter()
        df_eval_results = evaluate_pm_on_train(trans_matrix, matrix_act_to_id, train_data_with_ids, second_order_model)

        with contextlib.redirect_stdout(io.StringIO()):
            all_models, selected_rules, df_router_train = train_decision_tree_router(
                df_eval=df_eval_results,
                log_csv_path=log_file_path,
                min_support_ratio=args.min_support_ratio,
                total_train_cases=len(train_data_with_ids),
                error_threshold=args.error_threshold,
                csv_sep=',',
                verbose=False,
                min_category_frequency=args.categorical_min_frequency
            )
        offline_time_sec += time.perf_counter() - started_at
        print(f"Selected Rules: {len(selected_rules)}")

    print("\n=== Generating Sub-Process Models for Exceptions ===")
    sub_models = []
    sub_model_vis_dir = os.path.join(fold_result_dir, 'sub_models')
    if args.save_model_images:
        Path(sub_model_vis_dir).mkdir(parents=True, exist_ok=True)

    for i, rule in enumerate(selected_rules, 1):
        target_act = rule['Activity']
        rule_case_ids = set(rule['Case_IDs'])
        sub_train_instances = [trace for case_id, trace in train_data_with_ids if case_id in rule_case_ids]

        started_at = time.perf_counter()
        pm_gen_sub = ProcessModelGenerator(sub_train_instances, global_act_to_id)
        sub_trans_matrix, _, dfg_sub, start_sub, end_sub, sub_second_order_model = pm_gen_sub.calculate_transition_matrix()
        offline_time_sec += time.perf_counter() - started_at

        if args.save_model_images and i <= args.num_saved_sub_models:
            sub_vis_path = os.path.join(sub_model_vis_dir, f'dfg_sub_model_{i:02d}.png')
            pm4py.save_vis_dfg(dfg_sub, start_sub, end_sub, sub_vis_path)

        sub_models.append({
            'rule_rank': i,
            'target_activity': target_act,
            'rule_text': rule['Rule'],
            'trans_matrix': sub_trans_matrix,
            'second_order_model': sub_second_order_model,
            'leaf_id': rule['Leaf_ID']
        })

    selectors = {}
    conditional_diagnostics = pd.DataFrame()
    model_sources = {
        'main': (trans_matrix, second_order_model),
        **{
            f"sub_{sub['rule_rank']}": (
                sub['trans_matrix'], sub['second_order_model']
            )
            for sub in sub_models
        }
    }
    id_to_act_global = {v: k for k, v in global_act_to_id.items()}

    if args.experiment_method == 'proposed_attr_transition':
        print("\n=== Selecting Attribute-Conditioned Transitions ===")
        started_at = time.perf_counter()
        df_conditional_train = prepare_training_predictions(
            df_router_train,
            all_models,
            selected_rules,
            model_sources,
            global_act_to_id,
            id_to_act_global
        )
        attributes = candidate_attributes(df_conditional_train, args)
        selectors, conditional_diagnostics = select_conditional_attributes(
            df_conditional_train, attributes, args
        )
        offline_time_sec += time.perf_counter() - started_at
        print(f"Candidate Attributes: {len(attributes)}")
        print(f"Candidate States: {len(conditional_diagnostics)}")
        print(f"Accepted Attributes: {len(selectors)}")

        if args.save_conditional_diagnostics:
            diagnostics_dir = os.path.join(
                args.result_dir,
                Path(args.data_set).stem,
                'proposed_attr_diagnostics'
            )
            Path(diagnostics_dir).mkdir(parents=True, exist_ok=True)
            diagnostics_path = os.path.join(
                diagnostics_dir,
                f'fold_{fold_no:02d}_attribute_selection.csv'
            )
            conditional_diagnostics.to_csv(
                diagnostics_path, index=False, encoding='utf-8-sig'
            )

    print("\n=== Evaluating Hybrid Predictor on Test Data ===")
    online_started_at = time.perf_counter()
    df_test_eval = evaluate_pm_on_train(trans_matrix, matrix_act_to_id, test_data_with_ids, second_order_model)
    df_test_eval['log_step_index'] = df_test_eval['step_index'] - 1
    df_test_eval['case_id'] = df_test_eval['case_id'].astype(str)

    df_test_merged = pd.merge(
        df_test_eval,
        df_log_base,
        left_on=['case_id', 'log_step_index'],
        right_on=['case', 'step_index'],
        how='inner'
    )

    df_test_merged['hybrid_predicted_event'] = df_test_merged['predicted_event']
    df_test_merged['used_model'] = 'Main Model'

    drop_columns = [
        'case_id', 'step_index_x', 'previous_activity', 'ground_truth',
        'predicted_event', 'is_correct', 'log_step_index', 'case',
        'event', 'time', 'step_index_y', 'step_index', 'current_activity',
        'prefix_length', 'current_activity_count', 'consecutive_loop_count',
        'previous_same_activity_count',
        'Variant', 'Variant index', 'Variant.1',
        'hybrid_predicted_event', 'used_model'
    ]
    for act in all_models.keys():
        mask_act = df_test_merged['current_activity'] == act
        if not mask_act.any():
            continue

        df_act = df_test_merged[mask_act]
        clf, encoded_feature_names, frequent_category_values = all_models[act]
        X_test_encoded = encode_test_features(
            df_act, encoded_feature_names, drop_columns,
            frequent_category_values
        )
        leaf_indices = clf.apply(X_test_encoded)

        for sub in sub_models:
            if sub['target_activity'] != act:
                continue

            match_mask = (leaf_indices == sub['leaf_id'])
            indices = df_act[match_mask].index
            for idx in indices:
                previous_act = df_test_merged.loc[idx, 'previous_activity']
                current_act = df_test_merged.loc[idx, 'current_activity']
                pred_ev = predict_next_activity(
                    sub['trans_matrix'],
                    sub['second_order_model'],
                    global_act_to_id,
                    id_to_act_global,
                    current_act,
                    previous_act
                )
                if pred_ev:
                    df_test_merged.at[idx, 'hybrid_predicted_event'] = pred_ev
                    df_test_merged.at[idx, 'used_model'] = f"Sub-Model {sub['rule_rank']}"

    df_test_merged['improved_predicted_event'] = (
        df_test_merged['hybrid_predicted_event']
    )
    conditional_applied = pd.Series(
        False, index=df_test_merged.index, dtype='bool'
    )
    if args.experiment_method == 'proposed_attr_transition':
        test_model_keys = df_test_merged['used_model'].map(
            lambda value: (
                'main' if value == 'Main Model'
                else f"sub_{str(value).split()[-1]}"
            )
        )
        df_test_conditional = add_process_model_columns(
            df_test_merged,
            test_model_keys,
            model_sources,
            global_act_to_id,
            id_to_act_global
        )
        # Preserve the exact existing Proposed prediction as the baseline.
        df_test_conditional['_pm_prediction'] = (
            df_test_merged['hybrid_predicted_event'].astype(str)
        )
        correction = apply_conditional_correction(
            df_test_conditional, selectors, args
        )
        df_test_merged['improved_predicted_event'] = correction['predictions']
        df_test_merged['selected_attribute'] = correction['selected_attributes']
        df_test_merged['conditional_support'] = correction['supports']
        df_test_merged['conditional_probability'] = (
            correction['conditional_probabilities']
        )
        conditional_applied = pd.Series(
            correction['applied'], index=df_test_merged.index, dtype='bool'
        )
        df_test_merged['conditional_correction_applied'] = conditional_applied

    online_time_sec = time.perf_counter() - online_started_at

    if args.print_submodel_diagnostics:
        print(f"\n=== Sub-Model vs Main Model: Deep Diagnostic (Fold {fold_no}) ===")
        print(f"{'Sub-Model':<12} | {'Routed':<8} | {'Main Acc':<10} | {'Sub Acc':<10} | {'Gain'}")
        print("-" * 60)
        for sub in sub_models:
            rank = sub['rule_rank']
            subset = df_test_merged[df_test_merged['used_model'] == f"Sub-Model {rank}"]
            if len(subset) > 0:
                main_acc_sub = (subset['predicted_event'] == subset['ground_truth']).mean() * 100
                sub_acc = (subset['hybrid_predicted_event'] == subset['ground_truth']).mean() * 100
                print(f"Sub-Model {rank:<3}  | {len(subset):<8} | {main_acc_sub:<9.1f}% | {sub_acc:<9.1f}% | {sub_acc-main_acc_sub:+.1f}%")
        print("-" * 60)

    total = len(df_test_merged)
    main_correct = int((df_test_merged['predicted_event'] == df_test_merged['ground_truth']).sum())
    hybrid_correct = int((df_test_merged['hybrid_predicted_event'] == df_test_merged['ground_truth']).sum())
    improved_correct = int((df_test_merged['improved_predicted_event'] == df_test_merged['ground_truth']).sum())
    routed_total = int((df_test_merged['used_model'] != 'Main Model').sum())

    main_acc = (main_correct / total) * 100 if total else 0.0
    hybrid_acc = (hybrid_correct / total) * 100 if total else 0.0
    improved_acc = (improved_correct / total) * 100 if total else 0.0
    improvement = hybrid_acc - main_acc
    conditional_improvement = improved_acc - hybrid_acc

    original_correct_mask = (
        df_test_merged['hybrid_predicted_event'] == df_test_merged['ground_truth']
    )
    improved_correct_mask = (
        df_test_merged['improved_predicted_event'] == df_test_merged['ground_truth']
    )
    changed_mask = (
        df_test_merged['hybrid_predicted_event']
        != df_test_merged['improved_predicted_event']
    )
    helped = int((changed_mask & ~original_correct_mask & improved_correct_mask).sum())
    hurt = int((changed_mask & original_correct_mask & ~improved_correct_mask).sum())
    wrong_to_wrong = int((changed_mask & ~original_correct_mask & ~improved_correct_mask).sum())

    if args.experiment_method == 'proposed_attr_transition':
        method_correct = improved_correct
        method_acc = improved_acc
    elif args.experiment_method == 'proposed':
        method_correct = hybrid_correct
        method_acc = hybrid_acc
    else:
        method_correct = main_correct
        method_acc = main_acc

    print(f"  > [Fold {fold_no}] Main Model: {main_correct}/{total} ({main_acc:.2f}%)")
    print(f"  > [Fold {fold_no}] Hybrid:     {hybrid_correct}/{total} ({hybrid_acc:.2f}%)")
    print(f"  > [Fold {fold_no}] Improvement: {improvement:+.2f}%")
    if args.experiment_method == 'proposed_attr_transition':
        print(f"  > [Fold {fold_no}] Attr Transition: {improved_correct}/{total} ({improved_acc:.2f}%)")
        print(f"  > [Fold {fold_no}] Attr Improvement: {conditional_improvement:+.2f}%")
        print(
            f"  > [Fold {fold_no}] Corrections: applied={int(conditional_applied.sum())}, "
            f"changed={int(changed_mask.sum())}, helped={helped}, hurt={hurt}"
        )
    print(f"  > [Fold {fold_no}] Offline time: {offline_time_sec:.6f} sec")
    print(f"  > [Fold {fold_no}] Online time:  {online_time_sec:.6f} sec")

    return {
        'fold': fold_no,
        'method': args.experiment_method,
        'train_cases': len(train_data_with_ids),
        'test_cases': len(test_data_with_ids),
        'test_events': total,
        'num_rules': len(selected_rules),
        'num_sub_models': len(sub_models),
        'routed_events': routed_total,
        'main_correct': main_correct,
        'hybrid_correct': hybrid_correct,
        'improved_correct': improved_correct,
        'main_acc': main_acc,
        'hybrid_acc': hybrid_acc,
        'improved_acc': improved_acc,
        'improvement': improvement,
        'conditional_improvement': conditional_improvement,
        'accepted_attributes': len(selectors),
        'correction_used': int(conditional_applied.sum()),
        'prediction_changed': int(changed_mask.sum()),
        'helped': helped,
        'hurt': hurt,
        'wrong_to_wrong': wrong_to_wrong,
        'method_correct': method_correct,
        'method_acc': method_acc,
        'epochs_trained': np.nan,
        'num_attribute_features': np.nan,
        'offline_seconds': offline_time_sec,
        'online_seconds': online_time_sec,
        'online_ms_per_event': (online_time_sec * 1000.0) / total if total else 0.0
    }


if __name__ == '__main__':
    args = apply_command_line_overrides(Args())
    if args.experiment_method not in IMPLEMENTED_METHODS:
        raise ValueError(
            f"Unknown method: {args.experiment_method}. "
            f"Choose one of: {sorted(IMPLEMENTED_METHODS)}"
        )
    log_params = {'case_id_key': 'case', 'activity_key': 'event', 'timestamp_key': 'time'}

    Path(args.result_dir).mkdir(parents=True, exist_ok=True)

    print("=== Initialization & Data Loading ===")
    preprocessor = Preprocessor(args, log_params)
    cleaned_cases_with_ids = build_clean_cases(preprocessor)
    global_act_to_id = build_global_activity_dict(cleaned_cases_with_ids)

    log_file_path = os.path.join(args.data_dir, args.data_set)
    df_log_base = pd.read_csv(log_file_path, sep=',')
    df_log_base = df_log_base.sort_values(by=['case', 'time'])
    df_log_base['step_index'] = df_log_base.groupby('case').cumcount()
    df_log_base['case'] = df_log_base['case'].astype(str)

    print("\n=== 10-Fold Cross Validation ===")
    print(f"Dataset: {args.data_set}")
    print(f"Method: {args.experiment_method}")
    print(f"Total Cases: {len(cleaned_cases_with_ids)}")
    print(f"Folds: {args.num_folds}")
    print(f"Random State: {args.random_state}")

    kf = KFold(n_splits=args.num_folds, shuffle=True, random_state=args.random_state)
    fold_results = []

    for fold_no, (train_idx, test_idx) in enumerate(kf.split(cleaned_cases_with_ids), 1):
        train_data_with_ids = [cleaned_cases_with_ids[i] for i in train_idx]
        test_data_with_ids = [cleaned_cases_with_ids[i] for i in test_idx]
        fold_results.append(
            run_fold(args, fold_no, train_data_with_ids, test_data_with_ids, global_act_to_id, log_file_path, df_log_base)
        )

    df_cv = pd.DataFrame(fold_results)
    dataset_name = Path(args.data_set).stem
    dataset_result_dir = os.path.join(args.result_dir, dataset_name)
    Path(dataset_result_dir).mkdir(parents=True, exist_ok=True)

    if args.experiment_method == 'proposed':
        summary_filename = (
            f'summary_{dataset_name}_{args.experiment_method}_{args.error_threshold:.2f}.csv'
        )
    elif args.experiment_method == 'proposed_attr_transition':
        summary_filename = (
            f'summary_{dataset_name}_proposed_attr_'
            f'{args.error_threshold:.2f}_'
            f'{args.pm_confidence_threshold:.2f}.csv'
        )
    else:
        summary_filename = f'summary_{dataset_name}_{args.experiment_method}.csv'

    summary_path = os.path.join(dataset_result_dir, summary_filename)

    total_events = df_cv['test_events'].sum()
    summary_row = {column: np.nan for column in df_cv.columns}
    summary_row.update({
        'fold': 'mean_total',
        'method': args.experiment_method,
        'test_events': total_events,
        'main_correct': df_cv['main_correct'].sum(),
        'hybrid_correct': df_cv['hybrid_correct'].sum(),
        'method_correct': df_cv['method_correct'].sum(),
        'main_acc': df_cv['main_acc'].mean(),
        'hybrid_acc': df_cv['hybrid_acc'].mean(),
        'improvement': df_cv['improvement'].mean(),
        'method_acc': df_cv['method_acc'].mean(),
        'epochs_trained': df_cv['epochs_trained'].mean(),
        'num_attribute_features': df_cv['num_attribute_features'].mean(),
        'offline_seconds': df_cv['offline_seconds'].sum(),
        'online_seconds': df_cv['online_seconds'].sum(),
        'online_ms_per_event': (
            df_cv['online_seconds'].sum() * 1000.0 / total_events if total_events else 0.0
        )
    })
    if args.experiment_method == 'proposed_attr_transition':
        summary_row.update({
            'improved_correct': df_cv['improved_correct'].sum(),
            'improved_acc': df_cv['improved_acc'].mean(),
            'conditional_improvement': df_cv['conditional_improvement'].mean(),
            'accepted_attributes': df_cv['accepted_attributes'].mean(),
            'correction_used': df_cv['correction_used'].sum(),
            'prediction_changed': df_cv['prediction_changed'].sum(),
            'helped': df_cv['helped'].sum(),
            'hurt': df_cv['hurt'].sum(),
            'wrong_to_wrong': df_cv['wrong_to_wrong'].sum()
        })
    df_summary = pd.concat([df_cv, pd.DataFrame([summary_row])], ignore_index=True)
    df_summary.to_csv(summary_path, index=False, encoding='utf-8-sig')

    print("\n=== Cross-Validation Results by Fold ===")
    if args.experiment_method in {'bi_lstm', 'transformer'}:
        result_columns = [
            'fold', 'test_events', 'method_acc', 'epochs_trained',
            'num_attribute_features',
            'offline_seconds', 'online_seconds'
        ]
    elif args.experiment_method == 'proposed_attr_transition':
        result_columns = [
            'fold', 'test_events', 'num_rules', 'accepted_attributes',
            'correction_used', 'prediction_changed',
            'hybrid_acc', 'improved_acc', 'conditional_improvement',
            'helped', 'hurt', 'offline_seconds', 'online_seconds'
        ]
    else:
        result_columns = [
            'fold', 'test_events', 'num_rules', 'routed_events',
            'main_acc', 'hybrid_acc', 'improvement',
            'offline_seconds', 'online_seconds'
        ]
    print(df_cv[result_columns].to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    print("\n=== Cross-Validation Summary ===")
    if args.experiment_method in {'bi_lstm', 'transformer'}:
        method_label = (
            'Bi-LSTM' if args.experiment_method == 'bi_lstm'
            else 'ProcessTransformer'
        )
        accuracy_columns = [('method_acc', method_label)]
    elif args.experiment_method == 'proposed_attr_transition':
        accuracy_columns = [
            ('main_acc', 'Main Model'),
            ('hybrid_acc', 'Proposed'),
            ('improved_acc', 'Proposed+Attr'),
            ('conditional_improvement', 'Attr Gain')
        ]
    else:
        accuracy_columns = [
            ('main_acc', 'Main Model'),
            ('hybrid_acc', 'Hybrid'),
            ('improvement', 'Improvement')
        ]
    for col, label in accuracy_columns:
        mean_val = df_cv[col].mean()
        std_val = df_cv[col].std(ddof=1)
        print(f"{label:<12}: {mean_val:.2f}% ± {std_val:.2f}%")

    pooled_main_acc = (df_cv['main_correct'].sum() / total_events) * 100 if total_events else 0.0
    pooled_hybrid_acc = (df_cv['hybrid_correct'].sum() / total_events) * 100 if total_events else 0.0
    pooled_method_acc = (df_cv['method_correct'].sum() / total_events) * 100 if total_events else 0.0
    print("\n=== Pooled Accuracy Across All Folds ===")
    if args.experiment_method == 'bi_lstm':
        print(f"Bi-LSTM     : {pooled_method_acc:.2f}%")
    elif args.experiment_method == 'transformer':
        print(f"Transformer : {pooled_method_acc:.2f}%")
    elif args.experiment_method == 'proposed_attr_transition':
        print(f"Main Model  : {pooled_main_acc:.2f}%")
        print(f"Proposed    : {pooled_hybrid_acc:.2f}%")
        print(f"Proposed+Attr: {pooled_method_acc:.2f}%")
        print(f"Attr Gain   : {pooled_method_acc - pooled_hybrid_acc:+.2f}%")
        print(
            f"Corrections : used={int(df_cv['correction_used'].sum())}, "
            f"changed={int(df_cv['prediction_changed'].sum())}, "
            f"helped={int(df_cv['helped'].sum())}, "
            f"hurt={int(df_cv['hurt'].sum())}"
        )
    else:
        print(f"Main Model  : {pooled_main_acc:.2f}%")
        print(f"Hybrid      : {pooled_hybrid_acc:.2f}%")
        print(f"Improvement : {pooled_hybrid_acc - pooled_main_acc:+.2f}%")
    print(f"Offline time: {df_cv['offline_seconds'].sum():.6f} sec (total)")
    print(f"Online time : {df_cv['online_seconds'].sum():.6f} sec (total)")
    print(
        f"Online/event: {df_cv['online_seconds'].sum() * 1000.0 / total_events:.6f} ms"
        if total_events else "Online/event: 0.000000 ms"
    )
    print(f"Results saved to: {summary_path}")
