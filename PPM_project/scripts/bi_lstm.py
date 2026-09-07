"""Bi-LSTM baseline for next-activity prediction.

The split is supplied by ``main.py`` so that this baseline uses exactly the
same case-level cross-validation folds as the process-model methods.
"""

import time

import numpy as np
import pandas as pd

from attribute_filter import is_excluded_attribute


def _make_prefix_sequence_class(tf):
    """Create a Keras sequence that encodes only the current mini-batch.

    Materialising every padded prefix can require tens of GiB for large logs.
    This sequence stores only two integer indices per sample and constructs the
    dense one-hot tensor one mini-batch at a time.
    """
    class PrefixSequence(tf.keras.utils.Sequence):
        def __init__(
            self, traces, activity_to_id, max_length, attribute_sequences,
            batch_size, shuffle=False, random_state=42
        ):
            super().__init__()
            self.traces = traces
            self.activity_to_id = activity_to_id
            self.max_length = max_length
            self.attribute_sequences = attribute_sequences
            self.batch_size = batch_size
            self.shuffle = shuffle
            self.rng = np.random.default_rng(random_state)
            self.num_activities = len(activity_to_id)
            self.num_attributes = (
                attribute_sequences[0].shape[1] if attribute_sequences else 0
            )
            self.samples = np.asarray(
                [
                    (trace_index, next_index)
                    for trace_index, trace in enumerate(traces)
                    for next_index in range(1, len(trace))
                ],
                dtype=np.int32
            ).reshape(-1, 2)
            self.order = np.arange(len(self.samples), dtype=np.int32)
            self.on_epoch_end()

        def __len__(self):
            return (len(self.samples) + self.batch_size - 1) // self.batch_size

        def __getitem__(self, batch_index):
            start = batch_index * self.batch_size
            sample_ids = self.order[start:start + self.batch_size]
            features = np.zeros(
                (
                    len(sample_ids), self.max_length,
                    self.num_activities + self.num_attributes
                ),
                dtype=np.float32
            )
            labels = np.empty(len(sample_ids), dtype=np.int32)

            for row, sample_id in enumerate(sample_ids):
                trace_index, next_index = self.samples[sample_id]
                trace = self.traces[trace_index]
                prefix_start = max(0, int(next_index) - self.max_length)
                prefix = trace[prefix_start:next_index]
                prefix_length = len(prefix)
                attributes = (
                    self.attribute_sequences[trace_index]
                    if self.attribute_sequences else None
                )
                activity_ids = np.fromiter(
                    (self.activity_to_id[activity] for activity in prefix),
                    dtype=np.int32,
                    count=prefix_length
                )
                features[row, np.arange(prefix_length), activity_ids] = 1.0
                if attributes is not None:
                    features[row, :prefix_length, self.num_activities:] = attributes[
                        prefix_start:next_index
                    ]
                labels[row] = self.activity_to_id[trace[next_index]]
            return features, labels

        def on_epoch_end(self):
            if self.shuffle:
                self.rng.shuffle(self.order)

        def labels_in_prediction_order(self):
            return np.asarray(
                [
                    self.activity_to_id[self.traces[trace_index][next_index]]
                    for trace_index, next_index in self.samples
                ],
                dtype=np.int32
            )

    return PrefixSequence


def _select_attribute_columns(df_log, excluded_columns):
    selected = []
    for column in df_log.columns:
        if is_excluded_attribute(column, excluded_columns):
            continue
        selected.append(column)
    return selected


def _fit_attribute_encoder(train_rows, attribute_columns, min_category_frequency):
    categorical_columns = [
        column for column in attribute_columns
        if (
            pd.api.types.is_object_dtype(train_rows[column])
            or pd.api.types.is_string_dtype(train_rows[column])
            or isinstance(train_rows[column].dtype, pd.CategoricalDtype)
            or pd.api.types.is_bool_dtype(train_rows[column])
        )
    ]
    numeric_columns = [
        column for column in attribute_columns if column not in categorical_columns
    ]

    numeric_medians = train_rows[numeric_columns].median(numeric_only=True)
    numeric_filled = train_rows[numeric_columns].fillna(numeric_medians).astype(float)
    numeric_means = numeric_filled.mean()
    numeric_stds = numeric_filled.std(ddof=0).replace(0.0, 1.0)

    categorical_frame = train_rows[categorical_columns].fillna('__MISSING__').astype(str)
    frequent_category_values = {}
    for column in categorical_columns:
        counts = categorical_frame[column].value_counts(dropna=False)
        frequent_category_values[column] = set(
            counts[counts > min_category_frequency].index.astype(str)
        )
        categorical_frame[column] = categorical_frame[column].where(
            categorical_frame[column].isin(frequent_category_values[column]),
            'OTHER'
        )
    categorical_encoded = pd.get_dummies(categorical_frame, dtype=np.float32)
    for column in categorical_columns:
        other_feature = f'{column}_OTHER'
        if other_feature not in categorical_encoded.columns:
            categorical_encoded[other_feature] = 0.0

    return {
        'categorical_columns': categorical_columns,
        'numeric_columns': numeric_columns,
        'numeric_medians': numeric_medians,
        'numeric_means': numeric_means,
        'numeric_stds': numeric_stds,
        'frequent_category_values': frequent_category_values,
        'categorical_feature_names': categorical_encoded.columns.tolist()
    }


def _transform_attributes(rows, encoder):
    numeric_columns = encoder['numeric_columns']
    categorical_columns = encoder['categorical_columns']

    numeric = rows[numeric_columns].fillna(encoder['numeric_medians']).astype(float)
    numeric = (numeric - encoder['numeric_means']) / encoder['numeric_stds']

    categorical = rows[categorical_columns].fillna('__MISSING__').astype(str)
    for column in categorical_columns:
        categorical[column] = categorical[column].where(
            categorical[column].isin(encoder['frequent_category_values'][column]),
            'OTHER'
        )
    categorical = pd.get_dummies(categorical, dtype=np.float32).reindex(
        columns=encoder['categorical_feature_names'], fill_value=0.0
    )
    return np.concatenate(
        [numeric.to_numpy(dtype=np.float32), categorical.to_numpy(dtype=np.float32)],
        axis=1
    )


def _prepare_attribute_sequences(args, df_log, train_records, test_records):
    """Fit attribute encoding on train events and transform both folds."""
    log = df_log.copy()
    log['case'] = log['case'].astype(str)
    log = log[~log['event'].isin(['!', '_END_'])]

    train_case_ids = {str(case_id) for case_id, _ in train_records}
    train_rows = log[log['case'].isin(train_case_ids)]
    attribute_columns = _select_attribute_columns(
        train_rows,
        args.bi_lstm_excluded_attribute_columns
    )
    encoder = _fit_attribute_encoder(
        train_rows, attribute_columns, args.categorical_min_frequency
    )
    encoded_all = _transform_attributes(log, encoder)

    case_to_attributes = {}
    for case_id, positions in log.groupby('case', sort=False).indices.items():
        case_to_attributes[case_id] = encoded_all[np.asarray(positions)]

    def sequences_for(records):
        sequences = []
        for case_id, trace in records:
            sequence = case_to_attributes.get(str(case_id))
            if sequence is None or len(sequence) != len(trace):
                raise ValueError(
                    f"Attribute rows do not match trace length for case {case_id}: "
                    f"rows={0 if sequence is None else len(sequence)}, trace={len(trace)}"
                )
            sequences.append(sequence)
        return sequences

    return sequences_for(train_records), sequences_for(test_records), encoded_all.shape[1]


def _split_training_cases(traces, validation_split, random_state):
    """Create a case-level validation split to avoid prefix leakage."""
    if validation_split <= 0.0 or len(traces) < 10:
        return traces, []

    rng = np.random.default_rng(random_state)
    indices = rng.permutation(len(traces))
    validation_size = max(1, int(round(len(traces) * validation_split)))
    validation_indices = set(indices[:validation_size].tolist())

    training_traces = [trace for i, trace in enumerate(traces) if i not in validation_indices]
    validation_traces = [trace for i, trace in enumerate(traces) if i in validation_indices]
    return training_traces, validation_traces


def run_bi_lstm_fold(
    args, train_data_with_ids, test_data_with_ids, activity_to_id, df_log
):
    """Train and evaluate one Bi-LSTM fold.

    Offline time includes case-level validation splitting, prefix encoding,
    model construction, and training. Online time includes test-prefix
    encoding and one batched prediction call.
    """
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise ImportError(
            "Bi-LSTM requires TensorFlow. Install a TensorFlow version "
            "compatible with the active Python environment."
        ) from exc

    train_traces = [trace for _, trace in train_data_with_ids]
    test_traces = [trace for _, trace in test_data_with_ids]
    if not train_traces:
        raise ValueError("Bi-LSTM received an empty training fold.")

    offline_started_at = time.perf_counter()
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(args.random_state)

    fitting_records, validation_records = _split_training_cases(
        train_data_with_ids,
        args.bi_lstm_validation_split,
        args.random_state
    )
    all_train_attributes, test_attributes, num_attribute_features = _prepare_attribute_sequences(
        args, df_log, train_data_with_ids, test_data_with_ids
    )
    attributes_by_case = {
        str(case_id): attributes
        for (case_id, _), attributes in zip(train_data_with_ids, all_train_attributes)
    }
    fitting_traces = [trace for _, trace in fitting_records]
    validation_traces = [trace for _, trace in validation_records]
    fitting_attributes = [attributes_by_case[str(case_id)] for case_id, _ in fitting_records]
    validation_attributes = [
        attributes_by_case[str(case_id)] for case_id, _ in validation_records
    ]
    max_length = max(len(trace) - 1 for trace in fitting_traces)
    PrefixSequence = _make_prefix_sequence_class(tf)
    train_sequence = PrefixSequence(
        fitting_traces, activity_to_id, max_length, fitting_attributes,
        args.bi_lstm_batch_size, shuffle=True, random_state=args.random_state
    )

    validation_data = None
    monitor = 'loss'
    if validation_traces:
        validation_data = PrefixSequence(
            validation_traces, activity_to_id, max_length, validation_attributes,
            args.bi_lstm_batch_size, shuffle=False,
            random_state=args.random_state
        )
        monitor = 'val_loss'

    model = tf.keras.Sequential([
        tf.keras.layers.Input(
            shape=(max_length, len(activity_to_id) + num_attribute_features),
            name='activity_prefix'
        ),
        tf.keras.layers.Masking(mask_value=0.0),
        tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(
                args.bi_lstm_units,
                dropout=args.bi_lstm_dropout,
                return_sequences=False
            )
        ),
        tf.keras.layers.Dense(len(activity_to_id), activation='softmax')
    ], name='bi_lstm_next_activity')
    model.compile(
        optimizer=tf.keras.optimizers.Nadam(
            learning_rate=args.bi_lstm_learning_rate
        ),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor=monitor,
        patience=args.bi_lstm_patience,
        min_delta=args.bi_lstm_min_delta,
        mode='min',
        restore_best_weights=True,
        verbose=0
    )
    history = model.fit(
        train_sequence,
        validation_data=validation_data,
        epochs=args.bi_lstm_epochs,
        callbacks=[early_stopping],
        verbose=args.bi_lstm_verbose,
    )
    offline_seconds = time.perf_counter() - offline_started_at

    online_started_at = time.perf_counter()
    test_sequence = PrefixSequence(
        test_traces, activity_to_id, max_length, test_attributes,
        args.bi_lstm_batch_size, shuffle=False, random_state=args.random_state
    )
    test_labels = test_sequence.labels_in_prediction_order()
    probabilities = model.predict(
        test_sequence,
        verbose=0
    )
    predictions = np.argmax(probabilities, axis=1)
    online_seconds = time.perf_counter() - online_started_at

    total = int(test_labels.size)
    correct = int(np.sum(predictions == test_labels))
    accuracy = (correct / total) * 100.0 if total else 0.0

    return {
        'test_events': total,
        'correct': correct,
        'accuracy': accuracy,
        'offline_seconds': offline_seconds,
        'online_seconds': online_seconds,
        'online_ms_per_event': (online_seconds * 1000.0) / total if total else 0.0,
        'epochs_trained': len(history.history['loss']),
        'num_attribute_features': num_attribute_features
    }
