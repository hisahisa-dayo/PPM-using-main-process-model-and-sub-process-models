from __future__ import division
import csv
import numpy
import copy
import xnap.utils as utils
from sklearn.model_selection import KFold, ShuffleSplit

from collections import defaultdict

class Preprocessor(object):
    data_structure = {
        'support': {
            'num_folds': 1,
            'data_dir': "",
            'ascii_offset': 161,
            'data_format': "%d.%m.%Y-%H:%M:%S",
            'train_index_per_fold': [],
            'test_index_per_fold': [],
            'iteration_cross_validation': 0,
            'elements_per_fold': 0,
            'event_labels': [],
            'event_types': [],
            'map_event_label_to_event_id': [],
            'map_event_id_to_event_label': [],
            'map_event_type_to_event_id': [],
            'map_event_id_to_event_type': [],
            'end_process_instance': '!',
            'transition_matrix': None,
            'matrix_act_to_id': None
        },

        'meta': {
            'num_features': 0,
            'num_event_ids': 0,
            'max_length_process_instance': 0,
            'num_attributes_control_flow': 3,  # process instance id, event id and timestamp
            'num_process_instances': 0
        },

        'data': {
            'process_instances': [],
            'ids_process_instances': [],

            'train': {
                'features_data': numpy.array([]),
                'labels': numpy.ndarray([])
            },

            'test': {
                'process_instances': [],
                'event_ids': []
            }
        }
    }

    def __init__(self, args, log_params: dict):

        self.args = args

        if isinstance(args, dict):
            self.data_structure = args
        else:
            utils.llprint("Initialization ... \n")
            self.data_structure['support']['num_folds'] = args.num_folds
            self.data_structure['support']['data_dir'] = args.data_dir + args.data_set
            self.process_model_size = getattr(args, 'process_model_size', float('inf'))
            self.get_sequences_from_eventlog(log_params)
            self.data_structure['support']['elements_per_fold'] = \
                int(round(
                    self.data_structure['meta']['num_process_instances'] / self.data_structure['support']['num_folds']))

            # add end marker of process instance
            self.data_structure['data']['process_instances'] = list(
                map(lambda x: x + ['!'], self.data_structure['data']['process_instances']))
            self.data_structure['meta']['max_length_process_instance'] = max(
                map(lambda x: len(x), self.data_structure['data']['process_instances']))

            # structures for predicting next activities
            self.data_structure['support']['event_labels'] = list(
                map(lambda x: set(x), self.data_structure['data']['process_instances']))
            self.data_structure['support']['event_labels'] = list(
                set().union(*self.data_structure['support']['event_labels']))
            self.data_structure['support']['event_labels'].sort()
            self.data_structure['support']['event_types'] = copy.copy(self.data_structure['support']['event_labels'])

            args.dim = len(self.data_structure['support']['event_labels'])

            if args.cross_validation:
                self.set_indices_k_fold_validation()
            else:
                self.set_indices_split_validation(args)

        # Computed attributes
        self.data_structure['support']['map_event_label_to_event_id'] = dict(
            (c, i) for i, c in enumerate(self.data_structure['support']['event_labels']))
        self.data_structure['support']['map_event_id_to_event_label'] = dict(
            (i, c) for i, c in enumerate(self.data_structure['support']['event_labels']))
        self.data_structure['support']['map_event_type_to_event_id'] = dict(
            (c, i) for i, c in enumerate(self.data_structure['support']['event_types']))
        self.data_structure['support']['map_event_id_to_event_type'] = dict(
            (i, c) for i, c in enumerate(self.data_structure['support']['event_types']))
        #self.data_structure['meta']['num_event_ids'] = len(self.data_structure['support']['event_labels'])

        #self.data_structure['meta']['num_features'] = len(self.data_structure['support']['event_labels'])

        self.data_structure['meta']['num_event_ids'] = len(self.data_structure['support']['event_labels'])

        # ★★★ 変更点: 特徴量の数を「アクティビティ数 × 2」にする ★★★
        # (前半: One-hot, 後半: 遷移確率)
        self.data_structure['meta']['num_features'] = len(self.data_structure['support']['event_labels']) * 2


        # --- ここから追加 ---
        utils.llprint("\n--- First 3 Processed Instances ---\n")
        num_to_print = min(3, len(self.data_structure['data']['process_instances']))
        for i in range(num_to_print):
            case_id = self.data_structure['data']['ids_process_instances'][i]
            activities = self.data_structure['data']['process_instances'][i]
            utils.llprint(f"Case ID: {case_id}, Activities: {activities}\n")
        if num_to_print < len(self.data_structure['data']['process_instances']):
            utils.llprint("...\n")
        utils.llprint("--------------------------------------\n")
        # --- ここまで追加 ---

    def get_pure_dict(self) -> dict:
        """
        Get a pure dict representation of self.data_structure which can be json serialized.
        """
        from copy import deepcopy
        pure_dict = deepcopy(self.data_structure)
        pure_dict['support']['train_index_per_fold'] = []#[[int(i) for i in x] for x in pure_dict["support"]["train_index_per_fold"]]
        pure_dict['support']['test_index_per_fold'] = []#[[int(i) for i in x] for x in pure_dict["support"]["test_index_per_fold"]]
        pure_dict['data']['train']['features_data'] = []
        pure_dict['data']['train']['labels'] = []
        pure_dict['data']['test']['process_instances'] = []
        pure_dict['data']['test']['event_ids'] = []

        return pure_dict

    def get_sequences_from_eventlog(self, log_params: dict):
        """
        Get sequences form event log.


        id_latest_process_instance = ''
        process_instance = []
        first_event_of_process_instance = True
        output = True

        file = open(self.data_structure['support']['data_dir'], 'r')
        try:
            dialect = csv.Sniffer().sniff(file.read(4096))
            file.seek(0)
            reader = csv.reader(file, dialect)
        except csv.Error:
            file.seek(0)
            reader = csv.reader(file)
        header = next(reader, None)
        if len(header) == 1 or len(header) == 2:  # Very stupid behavior of csv.Sniffer
            file.seek(0)
            reader = csv.reader(file)
            header = next(reader, None)

        # get index of case id and activity columns
        case_id_col_index = header.index(log_params['case_id_key'])
        activity_col_index = header.index(log_params['activity_key'])

        for event in reader:

            id_current_process_instance = event[case_id_col_index]

            if output:
                output = False

            if id_current_process_instance != id_latest_process_instance:
                self.add_data_to_data_structure(id_current_process_instance, 'ids_process_instances')
                id_latest_process_instance = id_current_process_instance

                if not first_event_of_process_instance:
                    self.add_data_to_data_structure(process_instance, 'process_instances')

                process_instance = []

                self.data_structure['meta']['num_process_instances'] += 1

            process_instance.append(event[activity_col_index])
            first_event_of_process_instance = False

        file.close()

        self.add_data_to_data_structure(process_instance, 'process_instances')

        self.data_structure['meta']['num_process_instances'] += 1
        """

        # ★★★ 変更点1: defaultdict を使用して、各Case IDに紐づくアクティビティを格納 ★★★
        # 各値はアクティビティ名のリスト
        case_activities = defaultdict(list)

        file = open(self.data_structure['support']['data_dir'], 'r')
        dialect = csv.Sniffer().sniff(file.read(4096))
        file.seek(0)
        reader = csv.reader(file, dialect)
        header = next(reader, None)
        if len(header) == 1 or len(header) == 2:  # Very stupid behavior of csv.Sniffer
            file.seek(0)
            reader = csv.reader(file)
            header = next(reader, None)

        # get index of case id and activity columns
        case_id_col_index = header.index(log_params['case_id_key'])
        activity_col_index = header.index(log_params['activity_key'])

        # ★★★ 変更点2: 全てのイベントを読み込み、Case IDごとにアクティビティをグループ化 ★★★
        for event_row in reader:
            case_id = event_row[case_id_col_index]
            activity = event_row[activity_col_index]

            # 各Case IDのアクティビティリストにアクティビティを追加
            # CSVが既にタイムスタンプ順であることを前提とするため、ここではソートしない
            case_activities[case_id].append(activity)

        file.close()

        # ★★★ 変更点3: グループ化されたアクティビティからプロセスインスタンスを構築 ★★★

        for case_id in case_activities.keys():
            process_instance_activities = case_activities[case_id]

            # 空のプロセスインスタンスは追加しない
            if process_instance_activities:
                self.add_data_to_data_structure(case_id, 'ids_process_instances')
                self.add_data_to_data_structure(process_instance_activities, 'process_instances')
                self.data_structure['meta']['num_process_instances'] += 1

    def set_training_set(self):
        """
        Set training set
        """

        utils.llprint("Get training instances ... \n")
        process_instances_train, _ = \
            self.get_instances_of_fold('train')

        utils.llprint("Create cropped training instances ... \n")
        cropped_process_instances, next_events = \
            self.get_cropped_instances(process_instances_train)

        utils.llprint("Create training set data as 3d-tensor ... \n")
        features_data = self.get_data_tensor(cropped_process_instances,
                                             'train')

        utils.llprint("Create training set label as tensor ... \n")
        labels = self.get_label_matrix(cropped_process_instances,
                                       next_events)

        self.data_structure['data']['train']['features_data'] = features_data
        self.data_structure['data']['train']['labels'] = labels

    def get_event_type_max_prob(self, predictions):
        """
        Get most likely activity from a probability distribution.
        :param predictions:
        :return: activity.
        """

        max_prediction = 0
        event_type = ''
        index = 0

        for prediction in predictions:
            if prediction >= max_prediction:
                max_prediction = prediction
                event_type = self.data_structure['support']['map_event_id_to_event_type'][index]
            index += 1

        return event_type

    def get_event_type(self, index):
        """
        Get activity label for activity id.
        :param index:
        :return: activity.
        """

        return self.data_structure['support']['map_event_id_to_event_type'][index]

    def add_data_to_data_structure(self, values, structure):
        """
        Add data to general data structure.
        :param values:
        :param structure:
        """

        self.data_structure['data'][structure].append(values)

    def set_indices_k_fold_validation(self):
        """
        Performs k fold cross-validation.
        """

        kFold = KFold(n_splits=self.data_structure['support']['num_folds'])

        for train_indices, test_indices in kFold.split(self.data_structure['data']['process_instances']):
            self.data_structure['support']['train_index_per_fold'].append(train_indices)
            self.data_structure['support']['test_index_per_fold'].append(test_indices)

    def set_indices_split_validation(self, args):
        """
        total_instances = len(self.data_structure['data']['process_instances'])
        test_size = int(total_instances * args.split_rate_test)

        # 時系列順に先頭から分割
        train_indices = list(range(0, total_instances - test_size))
        test_indices = list(range(total_instances - test_size, total_instances))

        self.data_structure['support']['train_index_per_fold'].append(train_indices)
        self.data_structure['support']['test_index_per_fold'].append(test_indices)
        """
        # データ総数
        num_process_instances = self.data_structure['meta']['num_process_instances']
        
        # ★修正: shuffle=True, random_state=42 (固定) を追加
        # これにより時系列が破壊され、純粋なランダム10分割になります
        kf = KFold(n_splits=self.data_structure['support']['num_folds'], 
                   shuffle=True, 
                   random_state=42)
        
        train_indices = []
        test_indices = []
        
        # ダミー配列を作ってsplitにかける
        X_dummy = numpy.zeros(num_process_instances)
        
        for train_index, test_index in kf.split(X_dummy):
            train_indices.append(list(train_index))
            test_indices.append(list(test_index))
            
        self.data_structure['support']['train_index_per_fold'] = train_indices
        self.data_structure['support']['test_index_per_fold'] = test_indices

    def get_instances_of_fold(self, mode):
        """
        Retrieves process instances of a fold.
        :param mode:
        :return: process instances.
        """

        process_instances_of_fold = []
        event_ids_of_fold = []

        for index, value in enumerate(self.data_structure['support'][mode + '_index_per_fold'][
                                          self.data_structure['support']['iteration_cross_validation']]):
            process_instances_of_fold.append(self.data_structure['data']['process_instances'][value])
            event_ids_of_fold.append(self.data_structure['data']['ids_process_instances'][value])

        if mode == 'test':
            self.data_structure['data']['test']['process_instances'] = process_instances_of_fold
            self.data_structure['data']['test']['event_ids'] = event_ids_of_fold
            return

        return process_instances_of_fold, event_ids_of_fold

    def get_cropped_instances(self, process_instances):
        """
        Crops prefixes out of process instances.
        :param process_instances:
        :return: Cropped process instances, next events
        """

        cropped_process_instances = []
        next_events = []

        for process_instance in process_instances:
            for i in range(0, len(process_instance)):

                if i == 0:
                    continue
                cropped_process_instances.append(process_instance[0:i])
                # label
                next_events.append(process_instance[i])

        return cropped_process_instances, next_events

    def get_cropped_instance_label(self, prefix_size, process_instance):
        """
        Crops the next activity label out of a single process instance.
        :param prefix_size:
        :param process_instance:
        :return: Next activity label
        """

        if prefix_size == len(process_instance) - 1:
            # end marker
            return str(self.data_structure["support"]["end_process_instance"])
        else:
            # label of next act
            return str(process_instance[prefix_size])

    def get_cropped_instance(self, prefix_size, process_instance):
        """
        Crops prefixes out of a single process instance.
        :param prefix_size:
        :param process_instance:
        :return: prefixes.
        """

        return process_instance[:prefix_size]

    def get_data_tensor(self, cropped_process_instances, mode):
        """
        Get data tensor from process instances.
        If args.use_transition_matrix is True:
            Creates [One-hot Activity] + [Transition Probability] (Size: 2 * num_event_ids)
        Else:
            Creates [One-hot Activity] only (Size: num_event_ids)
        """
        
        # --- 1. フラグの確認 ---
        # argsに属性がない場合はデフォルトTrue（提案手法）とする
        use_tm = getattr(self.args, 'use_transition_matrix', True)

        max_len = self.data_structure['meta']['max_length_process_instance']
        num_event_ids = self.data_structure['meta']['num_event_ids']
        
        # --- 2. テンソルサイズ（特徴量次元数）の決定 ---
        if use_tm:
            # 提案手法: One-hot(N) + 確率(N) = 2N
            # __init__で計算済みの num_features (2N) を使う
            tensor_dim = self.data_structure['meta']['num_features']
        else:
            # 従来手法: One-hot(N) のみ
            tensor_dim = num_event_ids

        # --- 3. データの入れ物作成 ---
        if mode == 'train':
            data_set = numpy.zeros((
                len(cropped_process_instances),
                max_len,
                tensor_dim), dtype=numpy.float64) # tensor_dimを使用
        else:
            data_set = numpy.zeros((
                1,
                max_len,
                tensor_dim), dtype=numpy.float32) # tensor_dimを使用

        # 必要な辞書などを取得
        trans_matrix = self.data_structure['support']['transition_matrix']
        matrix_act_to_id = self.data_structure['support']['matrix_act_to_id']
        global_map = self.data_structure['support']['map_event_label_to_event_id']
        
        matrix_id_to_act = {}
        if matrix_act_to_id is not None:
            matrix_id_to_act = {v: k for k, v in matrix_act_to_id.items()}

        # --- データを埋めるループ ---
        for index, cropped_process_instance in enumerate(cropped_process_instances):
            for index_, activity in enumerate(cropped_process_instance):
                
                act_str = str(activity)
                
                # 1. Global ID を取得して One-hot ベクトルをセット (共通)
                if act_str in global_map:
                    global_id = global_map[act_str]
                    data_set[index, index_, global_id] = 1
                    
                    # 2. 遷移確率ベクトルを取得して結合 (Trueの場合のみ)
                    if use_tm:
                        if trans_matrix is not None and matrix_act_to_id is not None and act_str in matrix_act_to_id:
                            
                            local_id = matrix_act_to_id[act_str]
                            raw_probs = trans_matrix[local_id]
                            
                            full_prob_vector = numpy.zeros(num_event_ids)
                            
                            for local_col_idx, prob in enumerate(raw_probs):
                                if prob > 0:
                                    target_act_name = matrix_id_to_act[local_col_idx]
                                    if target_act_name in global_map:
                                        target_global_id = global_map[target_act_name]
                                        full_prob_vector[target_global_id] = prob
                            
                            # 後半部分 (N ~ 2N-1) に確率を代入
                            data_set[index, index_, num_event_ids:] = full_prob_vector

        return data_set

    def get_data_tensor_for_single_prediction(self, cropped_process_instance):
        """
        Get three-order data tensor from a single prefix of a process instances.
        The prefix represents a running process instance.
        :param cropped_process_instance:
        :return: data tensor
        """

        data_set = self.get_data_tensor(
            [cropped_process_instance],
            'test')

        return data_set

    def get_label_matrix(self, cropped_process_instances, next_events):
        """
        Get matrix from process instances.
        :param next_events:
        :param cropped_process_instances:
        :return: label matrix
        """

        label = numpy.zeros((len(cropped_process_instances), len(self.data_structure['support']['event_types'])),
                            dtype=numpy.float64)

        for index, cropped_process_instance in enumerate(cropped_process_instances):

            for event_type in self.data_structure['support']['event_types']:

                if event_type == next_events[index]:
                    label[index, self.data_structure['support']['map_event_type_to_event_id'][event_type]] = 1
                else:
                    label[index, self.data_structure['support']['map_event_type_to_event_id'][event_type]] = 0

        return label

    def get_random_process_instance(self, lower_bound, upper_bound):
        """
        Selects a random process instance from the complete event log.
        :param lower_bound:
        :param upper_bound:
        :return: process instance.
        """

        process_instances = self.data_structure['data']['process_instances']

        while True:
            rand = numpy.random.randint(len(process_instances))
            size = len(process_instances[rand])

            if lower_bound <= size <= upper_bound:
                break

        return process_instances[rand]

    def stream_test_instances(self):
            """
            Yields test instances one by one.
            :return: A generator for process instances and their IDs.
            """
            test_indices = self.data_structure['support']['test_index_per_fold'][
                self.data_structure['support']['iteration_cross_validation']]

            for index in test_indices:
                # '!'マーカーを除いた純粋なアクティビティシーケンスを返す
                process_instance = self.data_structure['data']['process_instances'][index]
                yield process_instance[:-1], self.data_structure['data']['ids_process_instances'][index]

 
    def split_frequent_and_exception(self, ratio):
        from collections import Counter
        
        train_instances, _ = self.get_instances_of_fold('train')
        
        variants = [tuple(trace) for trace in train_instances]
        counts = Counter(variants)
        total_cases = len(variants)
        
        sorted_variants = counts.most_common()
        freq_variants = set()
        accumulated = 0
        
        # ★比率が0より大きい場合のみ、頻出パターンの抽出を行う
        if ratio > 0.0:
            for var, count in sorted_variants:
                freq_variants.add(var)
                accumulated += count
                if (accumulated / total_cases) >= ratio:
                    break
                
        self.frequent_variants_set = freq_variants
        
        # 実際のリストに振り分け
        train_freq = [list(v) for v in variants if v in freq_variants]
        train_exc = [list(v) for v in variants if v not in freq_variants]

        num_exc_variants = len(sorted_variants) - len(freq_variants)
        
        print(f"  > Variant Split: Total Cases={total_cases}")
        print(f"  > Frequent Patterns: {len(freq_variants)} variants (Cases: {len(train_freq)})")
        print(f"  > Exception Patterns: Rest {num_exc_variants} variants (Cases: {len(train_exc)})")
        
        return train_freq, train_exc
