"""CNN-LSTM captioning model with additive attention."""

from __future__ import annotations

import tensorflow as tf
from tensorflow.keras.layers import (
    LSTM,
    Concatenate,
    Dense,
    Dropout,
    Embedding,
    Input,
)
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


class BahdanauAttention(tf.keras.layers.Layer):
    """Additive attention over LSTM time steps using the image embedding as query."""

    def __init__(self, units: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self.units = int(units)
        self.W1 = Dense(self.units)
        self.W2 = Dense(self.units)
        self.V = Dense(1)

    def call(self, features, hidden):
        query = tf.expand_dims(hidden, axis=1)
        score = tf.nn.tanh(self.W1(features) + self.W2(query))
        attention_weights = tf.nn.softmax(self.V(score), axis=1)
        context_vector = tf.reduce_sum(attention_weights * features, axis=1)
        return context_vector, attention_weights

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({"units": self.units})
        return config


def build_model(
    vocab_size: int,
    max_length: int,
    feat_dim: int,
    embedding_dim: int = 256,
    lstm_units: int = 512,
    attention_units: int = 512,
    dropout_rate: float = 0.5,
    learning_rate: float = 1e-4,
) -> Model:
    """Build and compile the image captioning model."""
    image_input = Input(shape=(49, feat_dim), name="image_features")
    decoder_input = Input(shape=(max_length,), name="decoder_input")

    image_embedding = Dense(lstm_units, activation="relu", name="image_embedding")(image_input)
    token_embedding = Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        mask_zero=True,
        name="word_embedding",
    )(decoder_input)

    lstm_out = LSTM(
        lstm_units,
        return_sequences=False,
        dropout=dropout_rate,
        name="caption_lstm",
    )(token_embedding)

    context_vector, attention_weights = BahdanauAttention(attention_units, name="bahdanau_attention")(
        image_embedding, lstm_out
    )
    merged = Concatenate(name="fusion")([context_vector, lstm_out])
    x = Dropout(dropout_rate, name="fusion_dropout")(merged)
    x = Dense(lstm_units, activation="relu", name="decoder_dense")(x)
    outputs = Dense(vocab_size, activation="softmax", name="word_softmax")(x)

    model = Model(inputs=[image_input, decoder_input], outputs=outputs, name="cnn_lstm_captioner")
    model.compile(
        optimizer=Adam(learning_rate=learning_rate, clipnorm=1.0),
        loss=CategoricalCrossentropy(label_smoothing=0.1),
        metrics=["accuracy"],
    )
    model.attention_weights = attention_weights
    return model

