import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from helper import plot_loss, plot_accuracy
import model_dispatcher
from pathlib import Path
import argparse
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# directory paths
PROJECT_PATH = Path().resolve().parent
DATA_PATH = PROJECT_PATH.joinpath("data")
SAVED_MODEL_PATH = PROJECT_PATH.joinpath("models")

if not DATA_PATH.exists():
    DATA_PATH.mkdir()

if not SAVED_MODEL_PATH.exists():
    SAVED_MODEL_PATH.mkdir()


def run(model_name, batch_size, epochs):

    batch_size = batch_size
    epochs = epochs
    img_height = 224
    img_width = 224

    # Create Train, Validation, and Test set
    df = pd.read_csv(DATA_PATH.joinpath('train.csv'))
    df['label'] = df['label'].astype('str')

    trainval_df, test_df = train_test_split(
        df, test_size=0.1, stratify=df['label'], random_state=42)

    train_df, valid_df = train_test_split(
        trainval_df, test_size=0.2, stratify=trainval_df['label'], random_state=42
    )

    # Training set with Image Augmentation
    train_datagen = ImageDataGenerator(
        preprocessing_function=tf.keras.applications.efficientnet.preprocess_input,
        horizontal_flip=True,
        vertical_flip=True,
        shear_range=0.2,
        zoom_range=0.2,
        height_shift_range=0.2,
        width_shift_range=0.2
    )

    train = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        directory=DATA_PATH.joinpath("train_images"),
        x_col="image_id",
        y_col="label",
        target_size=(img_height, img_width),
        class_mode="sparse",
        batch_size=batch_size
    )
    # validation set
    valid_datagen = ImageDataGenerator(
        preprocessing_function=tf.keras.applications.efficientnet.preprocess_input
    )
    valid = valid_datagen.flow_from_dataframe(
        dataframe=valid_df,
        directory=DATA_PATH.joinpath("train_images"),
        x_col="image_id",
        y_col="label",
        target_size=(img_height, img_width),
        class_mode="sparse",
        batch_size=batch_size
    )

    # test set
    test_datagen = ImageDataGenerator(
        preprocessing_function=tf.keras.applications.efficientnet.preprocess_input
    )
    test = test_datagen.flow_from_dataframe(
        dataframe=test_df,
        directory=DATA_PATH.joinpath("train_images"),
        x_col="image_id",
        y_col="label",
        target_size=(img_height, img_width),
        class_mode="sparse",
        batch_size=batch_size
    )

    model = model_dispatcher.models[model_name]
    print(model.summary())

    # compile the model
    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    # train the model
    early_stopping_cb = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True)

    reduce_lr_cb = tf.keras.callbacks.ReduceLROnPlateau(
        monitot="val_loss", patience=5, factor=0.2)

    model_checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        SAVED_MODEL_PATH.joinpath("{0}".format(model_name) + ".h5"),
        save_best_only=True
    )

    history = model.fit(train, validation_data=valid, epochs=epochs,
                        callbacks=[early_stopping_cb, reduce_lr_cb, model_checkpoint_cb])

    results = pd.DataFrame(history.history)

    # plot accuracy and loss
    plot_loss(results)
    plot_accuracy(results)

    # evaluate the model
    res = model.evaluate(test)
    print("Accuracy on Test set:", res[1])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str
    )
    parser.add_argument(
        "--batch_size",
        type=int
    )
    parser.add_argument(
        "--epochs",
        type=int
    )

    args = parser.parse_args()

    run(
        model_name=args.model_name,
        batch_size=args.batch_size,
        epochs=args.epochs
    )
