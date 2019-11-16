"""Train a VDCNN on text data."""
import datetime
import sys
import pathlib

import tensorflow as tf
from absl import flags

import custom_callbacks
from vdcnn import VDCNN
from data_helper import DataHelper

# Dataset
flags.DEFINE_string(
    "dataset_path", "data/ag_news_csv/", "Path for the dataset to be used."
)
flags.DEFINE_string("train_log_dir", "logs/", "Location to write training logs.")

# Model hyperparameters
flags.DEFINE_integer("sequence_length", 1024, "Sequence Max Length (default: 1024)")
flags.DEFINE_string(
    "pool_type",
    "max",
    "Types of downsampling methods, use either three of max (maxpool), "
    "k_max (k-maxpool) or conv (linear) (default: 'max')",
)
flags.DEFINE_integer(
    "depth", 9, "Depth for VDCNN, use either 9, 17, 29 or 47 (default: 9)"
)
flags.DEFINE_boolean("shortcut", False, "Use optional shortcut (default: False)")
flags.DEFINE_boolean("sort", False, "Sort during k-max pooling (default: False)")
flags.DEFINE_boolean(
    "use_bias", False, "Use bias for all conv1d layers (default: False)"
)

# Training parameters
flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
flags.DEFINE_integer("num_epochs", 100, "Number of training epochs")
flags.DEFINE_integer(
    "evaluate_every", 500, "Evaluate model on validation dataset after this many steps"
)

FLAGS = flags.FLAGS
FLAGS(sys.argv)
print("\nParameters:")
print("-" * 20)
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr, value.value))
print("")

data_helper = DataHelper(sequence_max_length=FLAGS.sequence_length)


def preprocess():
    # Load data
    print("Loading data...")
    train_data, train_label, test_data, test_label = data_helper.load_dataset(
        FLAGS.dataset_path
    )
    print("Loading data succees...")

    # Preprocessing steps can go here

    return train_data, train_label, test_data, test_label


def train(x_train, y_train, x_test, y_test):

    session_ts = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    log_dir = str(pathlib.Path(FLAGS.train_log_dir / session_ts))

    # Init Keras Model here
    model = VDCNN(
        num_classes=y_train.shape[1],
        depth=FLAGS.depth,
        sequence_length=FLAGS.sequence_length,
        shortcut=FLAGS.shortcut,
        pool_type=FLAGS.pool_type,
        sort=FLAGS.sort,
        use_bias=FLAGS.use_bias,
    )

    model.compile(
        optimizer=tf.keras.optimizers.SGD(lr=0.01, momentum=0.9),
        loss="categorical_crossentropy",
        metrics=["acc"],
    )

    # Save model architecture
    model_json = model.to_json()
    with open("vdcnn_model.json", "w") as json_file:
        json_file.write(model_json)
    time_str = datetime.datetime.now().isoformat()
    print("{}: Model saved as json.".format(time_str))
    print("")

    # Trainer
    # Tensorboard and extra callback to support steps history
    tensorboard = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=50,
        write_graph=True,
        write_images=True,
    )
    checkpointer = tf.keras.callbacks.ModelCheckpoint(
        filepath="./checkpoints/vdcnn_weights_val_acc_{val_acc:.4f}.h5",
        save_freq="epoch",
        verbose=1,
        save_best_only=True,
        mode="max",
        monitor="val_acc",
    )
    loss_history = custom_callbacks.LossHistory(
        model, tensorboard, logdir=log_dir
    )
    evaluate_step = custom_callbacks.EvaluateStep(
        model,
        checkpointer,
        tensorboard,
        FLAGS.evaluate_every,
        FLAGS.batch_size,
        x_test,
        y_test,
        FLAGS.train_log_dir,
    )

    # Fit model
    model.fit(
        x_train,
        y_train,
        batch_size=FLAGS.batch_size,
        epochs=FLAGS.num_epochs,
        validation_data=(x_test, y_test),
        verbose=1,
        callbacks=[
            checkpointer,
            tensorboard,
            # loss_history,
            evaluate_step
        ],
    )
    print("-" * 30)
    time_str = datetime.datetime.now().isoformat()
    print("{}: Done training.".format(time_str))

    tf.keras.backend.clear_session()
    print("-" * 30)
    print()


def main():
    x_train, y_train, x_test, y_test = preprocess()
    train(x_train, y_train, x_test, y_test)


if __name__ == "__main__":
    main()
