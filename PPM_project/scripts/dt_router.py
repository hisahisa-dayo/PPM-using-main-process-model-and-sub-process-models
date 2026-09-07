import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, _tree

from attribute_filter import is_excluded_attribute


def _categorical_columns(frame):
    return [
        column for column in frame.columns
        if (
            pd.api.types.is_object_dtype(frame[column])
            or pd.api.types.is_string_dtype(frame[column])
            or isinstance(frame[column].dtype, pd.CategoricalDtype)
            or pd.api.types.is_bool_dtype(frame[column])
        )
    ]


def replace_rare_categories(frame, frequent_category_values):
    result = frame.copy()
    for column, frequent_values in frequent_category_values.items():
        values = result[column].fillna('__MISSING__').astype(str)
        result[column] = values.where(values.isin(frequent_values), 'OTHER')
    return result


def extract_error_rules_from_tree(clf, X, y, feature_names, min_samples, case_ids):
    """
    決定木からルールを抽出し、該当するCase IDのリストとLeaf IDと共に返す。
    """
    tree_ = clf.tree_
    leaf_indices = clf.apply(X)
    
    paths = {}
    def recurse(node, path):
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_names[tree_.feature[node]]
            threshold = tree_.threshold[node]
            
            if threshold == 0.5:
                left_cond = f"【{name}】 ではない"
                right_cond = f"【{name}】 である"
            elif threshold % 1 == 0.5:
                left_cond = f"【{name}】 が {int(threshold)} 以下"
                right_cond = f"【{name}】 が {int(threshold) + 1} 以上"
            else:
                left_cond = f"【{name}】 <= {threshold:.2f}"
                right_cond = f"【{name}】 > {threshold:.2f}"
                
            recurse(tree_.children_left[node], path + [left_cond])
            recurse(tree_.children_right[node], path + [right_cond])
        else:
            paths[node] = " AND\n         ".join(path) if path else "ALL DATA (Root)"
            
    recurse(0, [])
    
    rules = []
    for leaf_id in np.unique(leaf_indices):
        mask = (leaf_indices == leaf_id)
        leaf_y = y[mask]
        
        total_cases = len(leaf_y)
        # 動的に計算された最小ケース数でフィルタリング
        if total_cases < min_samples:
            continue
            
        error_count = np.sum(leaf_y == 0) # 0がエラー
        error_rate = error_count / total_cases
        
        leaf_case_ids = np.unique(case_ids[mask]).tolist()
        
        rules.append({
            'Rule': paths[leaf_id],
            'Error_Rate': error_rate,
            'Total_Cases': total_cases,
            'Error_Count': error_count,
            'Case_IDs': leaf_case_ids,
            'Leaf_ID': leaf_id
        })
            
    return rules

def train_decision_tree_router(
    df_eval, log_csv_path, min_support_ratio=0.001, total_train_cases=0,
    error_threshold=0.50, csv_sep=',', verbose=True,
    min_category_frequency=10
):
    
    # 割合から実際の最小ケース数（絶対数）を計算
    min_total_cases = max(1, int(total_train_cases * min_support_ratio))
    
    print(f"\n=== Training Decision Tree Router ===")
    print(f"  > Parameters: Error Threshold >= {error_threshold*100}%, Min Support Ratio >= {min_support_ratio*100}% (Calculated min_cases: {min_total_cases})")
    
    df_log = pd.read_csv(log_csv_path, sep=csv_sep)
    df_log = df_log.sort_values(by=['case', 'time'])
    df_log['step_index'] = df_log.groupby('case').cumcount()
    df_eval['log_step_index'] = df_eval['step_index'] - 1

    df_eval['case_id'] = df_eval['case_id'].astype(str)
    df_log['case'] = df_log['case'].astype(str)

    df_merged = pd.merge(
        df_eval, df_log, 
        left_on=['case_id', 'log_step_index'], 
        right_on=['case', 'step_index'], 
        how='inner'
    )

    drop_columns = [
        'case_id', 'step_index_x', 'ground_truth', 'predicted_event', 
        'is_correct', 'log_step_index', 'case', 'event', 'time', 
        'step_index_y', 'step_index', 'previous_activity', 'current_activity',
        'prefix_length', 'current_activity_count', 'consecutive_loop_count',
        'previous_same_activity_count',
        'Variant', 'Variant index', 'Variant.1'
    ]

    all_models = {}
    global_rules = []
    
    activities = df_merged['current_activity'].unique()
    
    for act in activities:
        df_act = df_merged[df_merged['current_activity'] == act]
        y = df_act['is_correct'].values
        total_act_cases = len(y)

        # 動的計算された min_total_cases を使用
        if total_act_cases < min_total_cases or np.sum(y == 0) == 0:
            continue

        X_raw = df_act.drop(columns=[col for col in drop_columns if col in df_act.columns])
        X_raw = X_raw[
            [column for column in X_raw.columns
             if not is_excluded_attribute(column)]
        ]
        
        categorical_columns = _categorical_columns(X_raw)
        frequent_category_values = {}
        for col in categorical_columns:
            values = X_raw[col].fillna('__MISSING__').astype(str)
            counts = values.value_counts(dropna=False)
            frequent_category_values[col] = set(
                counts[counts > min_category_frequency].index.astype(str)
            )
        X_raw = replace_rare_categories(X_raw, frequent_category_values)

        # エンコーディングと欠損値補完
        X_encoded = pd.get_dummies(X_raw, drop_first=True)
        X_encoded = X_encoded.fillna(0)
        for col in categorical_columns:
            other_feature = f'{col}_OTHER'
            if other_feature not in X_encoded.columns:
                X_encoded[other_feature] = 0
        encoded_feature_names = X_encoded.columns.tolist()

        # 決定木の学習
        clf = DecisionTreeClassifier(max_depth=5, min_samples_leaf=20, random_state=42, class_weight='balanced')
        clf.fit(X_encoded, y)

        rules = extract_error_rules_from_tree(
            clf, X_encoded, y, encoded_feature_names, 
            min_samples=min_total_cases,
            case_ids=df_act['case_id'].values
        )
        
        for r in rules:
            r['Activity'] = act
            global_rules.append(r)

        all_models[act] = (
            clf, encoded_feature_names, frequent_category_values
        )
    
    # 1. 閾値（例: 60%）以上のルールのみを抽出
    filtered_rules = [r for r in global_rules if r['Error_Rate'] >= error_threshold]
    
    # 2. エラー率の降順にソートして採用
    selected_rules = sorted(filtered_rules, key=lambda x: x['Error_Rate'], reverse=True)

    if not selected_rules:
        print(f"\n  > 条件（最小割合: {min_support_ratio*100}%以上, エラー率: {error_threshold*100}%以上）を満たす例外ルールは見つかりませんでした。")
    else:
        print(f"\n📍 閾値条件を満たしたクリティカルな例外ルール (Total: {len(selected_rules)} 個採用)")
        for i, r in enumerate(selected_rules, 1):
            print(f"\n [Rule {i}] 対象ノード：【{r['Activity']}】")
            print(f"  🚨 エラー率 {r['Error_Rate']*100:.1f}% (Wrong: {int(r['Error_Count'])} / Total: {int(r['Total_Cases'])})")
            print(f"     IF  {r['Rule']}")
            print("-" * 60)

    return all_models, selected_rules, df_merged
