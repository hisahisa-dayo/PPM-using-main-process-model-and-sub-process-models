"""One-Tr attribute-aware ProcessTransformer for next-activity prediction."""

import gc
import time

import numpy as np

from bi_lstm import _prepare_attribute_sequences, _split_training_cases


def _make_prefix_sequence_class(tf):
    """Materialise all-prefix training samples one mini-batch at a time."""

    class TransformerPrefixSequence(tf.keras.utils.Sequence):
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
            activity_tokens = np.zeros(
                (len(sample_ids), self.max_length), dtype=np.int32
            )
            attributes = np.zeros(
                (len(sample_ids), self.max_length, self.num_attributes),
                dtype=np.float32
            )
            labels = np.empty(len(sample_ids), dtype=np.int32)

            for row, sample_id in enumerate(sample_ids):
                trace_index, next_index = self.samples[sample_id]
                trace = self.traces[trace_index]
                prefix_start = max(0, int(next_index) - self.max_length)
                prefix = trace[prefix_start:next_index]
                prefix_length = len(prefix)
                # Zero is padding, so real activity IDs are shifted by one.
                activity_tokens[row, :prefix_length] = np.fromiter(
                    (
                        self.activity_to_id[activity] + 1
                        for activity in prefix
                    ),
                    dtype=np.int32,
                    count=prefix_length
                )
                if self.num_attributes:
                    attributes[row, :prefix_length] = self.attribute_sequences[
                        trace_index
                    ][prefix_start:next_index]
                labels[row] = self.activity_to_id[trace[next_index]]

            return {
                'activity_tokens': activity_tokens,
                'event_attributes': attributes
            }, labels

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

    return TransformerPrefixSequence


def _build_model(tf, args, max_length, num_activities, num_attributes):
    """Build the official one-block ProcessTransformer with One-Tr fusion."""

    embed_dim = args.transformer_embedding_dim
    activity_tokens = tf.keras.layers.Input(
        shape=(max_length,), dtype='int32', name='activity_tokens'
    )
    event_attributes = tf.keras.layers.Input(
        shape=(max_length, num_attributes),
        dtype='float32',
        name='event_attributes'
    )

    activity_embedding = tf.keras.layers.Embedding(
        input_dim=num_activities + 1,
        output_dim=embed_dim,
        mask_zero=True,
        name='activity_embedding'
    )(activity_tokens)
    attribute_projection = tf.keras.layers.Dense(
        embed_dim, use_bias=False, name='attribute_projection'
    )(event_attributes)
    x = tf.keras.layers.Concatenate(name='one_tr_feature_fusion')(
        [activity_embedding, attribute_projection]
    )
    x = tf.keras.layers.Dense(embed_dim, name='fused_event_projection')(x)

    position_ids = tf.keras.layers.Lambda(
        lambda tokens: tf.tile(
            tf.range(tf.shape(tokens)[1])[None, :],
            [tf.shape(tokens)[0], 1]
        ),
        output_shape=(max_length,),
        name='position_ids'
    )(activity_tokens)
    position_embedding = tf.keras.layers.Embedding(
        input_dim=max_length,
        output_dim=embed_dim,
        name='position_embedding'
    )(position_ids)
    x = tf.keras.layers.Add(name='add_position_embedding')(
        [x, position_embedding]
    )

    valid_positions = tf.keras.layers.Lambda(
        lambda tokens: tf.not_equal(tokens, 0),
        output_shape=(max_length,),
        dtype='bool',
        name='valid_position_mask'
    )(activity_tokens)
    attention_mask = tf.keras.layers.Lambda(
        lambda valid: tf.logical_and(
            valid[:, :, None], valid[:, None, :]
        ),
        output_shape=(max_length, max_length),
        dtype='bool',
        name='self_attention_mask'
    )(valid_positions)

    attention_output = tf.keras.layers.MultiHeadAttention(
        num_heads=args.transformer_num_heads,
        key_dim=args.transformer_key_dim,
        dropout=0.0,
        name='multi_head_attention'
    )(x, x, attention_mask=attention_mask)
    attention_output = tf.keras.layers.Dropout(
        args.transformer_dropout, name='attention_dropout'
    )(attention_output)
    x = tf.keras.layers.Add(name='attention_residual')([x, attention_output])
    x = tf.keras.layers.LayerNormalization(
        epsilon=1e-6, name='attention_layer_norm'
    )(x)

    feed_forward = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(
                args.transformer_ff_dim, activation='relu'
            ),
            tf.keras.layers.Dense(embed_dim)
        ],
        name='feed_forward_network'
    )(x)
    feed_forward = tf.keras.layers.Dropout(
        args.transformer_dropout, name='feed_forward_dropout'
    )(feed_forward)
    x = tf.keras.layers.Add(name='feed_forward_residual')([x, feed_forward])
    x = tf.keras.layers.LayerNormalization(
        epsilon=1e-6, name='feed_forward_layer_norm'
    )(x)

    pooled = tf.keras.layers.Lambda(
        lambda values: (
            tf.reduce_sum(
                values[0] * tf.cast(
                    values[1][:, :, None], values[0].dtype
                ),
                axis=1
            )
            / tf.maximum(
                tf.reduce_sum(
                    tf.cast(values[1][:, :, None], values[0].dtype),
                    axis=1
                ),
                tf.constant(1.0, dtype=values[0].dtype)
            )
        ),
        output_shape=(embed_dim,),
        name='masked_global_average_pooling'
    )([x, valid_positions])
    pooled = tf.keras.layers.Dropout(
        args.transformer_dropout, name='pooling_dropout'
    )(pooled)
    pooled = tf.keras.layers.Dense(
        64, activation='relu', name='classification_hidden'
    )(pooled)
    pooled = tf.keras.layers.Dropout(
        args.transformer_dropout, name='classification_dropout'
    )(pooled)
    outputs = tf.keras.layers.Dense(
        num_activities, activation='linear', name='next_activity_logits'
    )(pooled)

    return tf.keras.Model(
        inputs=[activity_tokens, event_attributes],
        outputs=outputs,
        name='one_tr_process_transformer'
    )


def run_process_transformer_fold(
    args, train_data_with_ids, test_data_with_ids, activity_to_id, df_log
):
    """Train and evaluate one case-level cross-validation fold."""

    try:
        import tensorflow as tf
    except ImportError as exc:
        raise ImportError(
            'ProcessTransformer requires TensorFlow compatible with the '
            'active Python environment.'
        ) from exc

    train_traces = [trace for _, trace in train_data_with_ids]
    test_traces = [trace for _, trace in test_data_with_ids]
    if not train_traces:
        raise ValueError('ProcessTransformer received an empty training fold.')

    offline_started_at = time.perf_counter()
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(args.random_state)

    fitting_records, validation_records = _split_training_cases(
        train_data_with_ids,
        args.transformer_validation_split,
        args.random_state
    )
    all_train_attributes, test_attributes, num_attribute_features = (
        _prepare_attribute_sequences(
            args, df_log, train_data_with_ids, test_data_with_ids
        )
    )
    attributes_by_case = {
        str(case_id): attributes
        for (case_id, _), attributes in zip(
            train_data_with_ids, all_train_attributes
        )
    }
    fitting_traces = [trace for _, trace in fitting_records]
    validation_traces = [trace for _, trace in validation_records]
    fitting_attributes = [
        attributes_by_case[str(case_id)] for case_id, _ in fitting_records
    ]
    validation_attributes = [
        attributes_by_case[str(case_id)] for case_id, _ in validation_records
    ]
    max_length = max(len(trace) - 1 for trace in fitting_traces)

    PrefixSequence = _make_prefix_sequence_class(tf)
    train_sequence = PrefixSequence(
        fitting_traces, activity_to_id, max_length, fitting_attributes,
        args.transformer_batch_size, shuffle=True,
        random_state=args.random_state
    )
    validation_data = None
    monitor = 'loss'
    if validation_traces:
        validation_data = PrefixSequence(
            validation_traces, activity_to_id, max_length,
            validation_attributes, args.transformer_batch_size,
            shuffle=False, random_state=args.random_state
        )
        monitor = 'val_loss'

    model = _build_model(
        tf, args, max_length, len(activity_to_id), num_attribute_features
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=args.transformer_learning_rate
        ),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor=monitor,
        patience=args.transformer_patience,
        min_delta=args.transformer_min_delta,
        mode='min',
        restore_best_weights=True,
        verbose=0
    )
    history = model.fit(
        train_sequence,
        validation_data=validation_data,
        epochs=args.transformer_epochs,
        callbacks=[early_stopping],
        verbose=args.transformer_verbose
    )
    offline_seconds = time.perf_counter() - offline_started_at

    online_started_at = time.perf_counter()
    test_sequence = PrefixSequence(
        test_traces, activity_to_id, max_length, test_attributes,
        args.transformer_batch_size, shuffle=False,
        random_state=args.random_state
    )
    test_labels = test_sequence.labels_in_prediction_order()
    logits = model.predict(test_sequence, verbose=0)
    predictions = np.argmax(logits, axis=1)
    online_seconds = time.perf_counter() - online_started_at

    total = int(test_labels.size)
    correct = int(np.sum(predictions == test_labels))
    accuracy = (correct / total) * 100.0 if total else 0.0
    result = {
        'test_events': total,
        'correct': correct,
        'accuracy': accuracy,
        'offline_seconds': offline_seconds,
        'online_seconds': online_seconds,
        'online_ms_per_event': (
            (online_seconds * 1000.0) / total if total else 0.0
        ),
        'epochs_trained': len(history.history['loss']),
        'num_attribute_features': num_attribute_features
    }

    del model, train_sequence, validation_data, test_sequence, logits
    tf.keras.backend.clear_session()
    gc.collect()
    return result
