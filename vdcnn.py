"""Very Deep CNN model. https://arxiv.org/abs/1606.01781"""
import tensorflow as tf

from k_maxpooling import KMaxPooling


def identity_block(inputs, filters, kernel_size=3, use_bias=False, shortcut=False):
    x = tf.keras.layers.Conv1D(
        filters=filters, kernel_size=kernel_size, strides=1, padding="same"
    )(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.activations.relu(x)
    x = tf.keras.layers.Conv1D(
        filters=filters, kernel_size=kernel_size, strides=1, padding="same"
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    if shortcut:
        x = tf.keras.layers.Add()([x, inputs])
    return tf.keras.activations.relu(x)


def conv_block(
    inputs,
    filters,
    kernel_size=3,
    use_bias=False,
    shortcut=False,
    pool_type="max",
    sort=True,
    stage=1,
):
    x = tf.keras.layers.Conv1D(
        filters=filters, kernel_size=kernel_size, strides=1, padding="same"
    )(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.activations.relu(x)

    x = tf.keras.layers.Conv1D(
        filters=filters, kernel_size=kernel_size, strides=1, padding="same"
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)

    if shortcut:
        residual = tf.keras.layers.Conv1D(
            filters=filters, kernel_size=1, strides=2, name="shortcut_conv1d_%d" % stage
        )(inputs)
        residual = tf.keras.layers.BatchNormalization(
            name="shortcut_batch_normalization_%d" % stage
        )(residual)
        x = downsample(x, pool_type=pool_type, sort=sort, stage=stage)
        x = tf.keras.layers.Add()([x, residual])
        x = tf.keras.activations.relu(x)
    else:
        x = tf.keras.activations.relu(x)
        x = downsample(x, pool_type=pool_type, sort=sort, stage=stage)
    if pool_type is not None:
        x = tf.keras.layers.Conv1D(
            filters=2 * filters,
            kernel_size=1,
            strides=1,
            padding="same",
            name="1_1_conv_%d" % stage,
        )(x)
        x = tf.keras.layers.BatchNormalization(
            name="1_1_batch_normalization_%d" % stage
        )(x)
    return x


def downsample(inputs, pool_type="max", sort=True, stage=1):
    if pool_type == "max":
        x = tf.keras.layers.MaxPooling1D(
            pool_size=3, strides=2, padding="same", name="pool_%d" % stage
        )(inputs)
    elif pool_type == "k_max":
        k = int(inputs._keras_shape[1] / 2)
        x = KMaxPooling(k=k, sort=sort, name="pool_%d" % stage)(inputs)
    elif pool_type == "conv":
        x = tf.keras.layers.Conv1D(
            filters=inputs._keras_shape[-1],
            kernel_size=3,
            strides=2,
            padding="same",
            name="pool_%d" % stage,
        )(inputs)
        x = tf.keras.layers.BatchNormalization()(x)
    elif pool_type is None:
        x = inputs
    else:
        raise ValueError("unsupported pooling type!")
    return x


def VDCNN(
    num_classes,
    depth=9,
    sequence_length=1024,
    embedding_dim=16,
    shortcut=False,
    pool_type="max",
    sort=True,
    use_bias=False,
    embedding_input=False,
    input_tensor=None,
):
    if depth == 9:
        num_conv_blocks = (1, 1, 1, 1)
    elif depth == 17:
        num_conv_blocks = (2, 2, 2, 2)
    elif depth == 29:
        num_conv_blocks = (5, 5, 2, 2)
    elif depth == 49:
        num_conv_blocks = (8, 8, 5, 3)
    else:
        raise ValueError("unsupported depth for VDCNN.")

    if embedding_input:
        # Input is a n x m matrix of n ordered m-dimenstional vector embeddings
        inputs = tf.keras.Input(shape=(sequence_length, embedding_dim), name="inputs")
        embedded_chars = inputs
    else:
        # Input is raw text
        inputs = tf.keras.Input(shape=(sequence_length,), name="inputs")
        embedded_chars = tf.keras.layers.Embedding(
            input_dim=sequence_length, output_dim=embedding_dim
        )(inputs)
    x = tf.keras.layers.Conv1D(
        filters=64, kernel_size=3, strides=1, padding="same", name="temp_conv"
    )(embedded_chars)

    # Convolutional Block 64
    for _ in range(num_conv_blocks[0] - 1):
        x = identity_block(
            x, filters=64, kernel_size=3, use_bias=use_bias, shortcut=shortcut
        )
    x = conv_block(
        x,
        filters=64,
        kernel_size=3,
        use_bias=use_bias,
        shortcut=shortcut,
        pool_type=pool_type,
        sort=sort,
        stage=1,
    )

    # Convolutional Block 128
    for _ in range(num_conv_blocks[1] - 1):
        x = identity_block(
            x, filters=128, kernel_size=3, use_bias=use_bias, shortcut=shortcut
        )
    x = conv_block(
        x,
        filters=128,
        kernel_size=3,
        use_bias=use_bias,
        shortcut=shortcut,
        pool_type=pool_type,
        sort=sort,
        stage=2,
    )

    # Convolutional Block 256
    for _ in range(num_conv_blocks[2] - 1):
        x = identity_block(
            x, filters=256, kernel_size=3, use_bias=use_bias, shortcut=shortcut
        )
    x = conv_block(
        x,
        filters=256,
        kernel_size=3,
        use_bias=use_bias,
        shortcut=shortcut,
        pool_type=pool_type,
        sort=sort,
        stage=3,
    )

    # Convolutional Block 512
    for _ in range(num_conv_blocks[3] - 1):
        x = identity_block(
            x, filters=512, kernel_size=3, use_bias=use_bias, shortcut=shortcut
        )
    x = conv_block(
        x,
        filters=512,
        kernel_size=3,
        use_bias=use_bias,
        shortcut=False,
        pool_type=None,
        stage=4,
    )

    # k-max pooling with k = 8
    k = min(x.shape[1], 8)
    x = KMaxPooling(k=k, sort=True)(x)
    x = tf.keras.layers.Flatten()(x)

    # Dense Layers
    x = tf.keras.layers.Dense(2048, activation="relu")(x)
    x = tf.keras.layers.Dense(2048, activation="relu")(x)
    x = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    if input_tensor is not None:
        inputs = tf.keras.get_source_inputs(input_tensor)
    else:
        inputs = inputs

    # Create model.
    model = tf.keras.Model(inputs=inputs, outputs=x, name="VDCNN")
    return model


def main():
    model = VDCNN(10, depth=9, shortcut=False, pool_type="max")
    model.summary()


if __name__ == "__main__":
    main()
