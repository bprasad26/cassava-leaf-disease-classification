import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, BatchNormalization, Dropout
from tensorflow.keras.models import Model
import argparse
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# create models


def efb0a():
    # Image specifications
    img_height = 224
    img_width = 224
    # train every layers of the EfficientNetB0
    model = Sequential()
    model.add(EfficientNetB0(include_top=False, weights='imagenet',
                             input_shape=(img_height, img_width, 3)))
    model.add(GlobalAveragePooling2D())
    model.add(Dense(5, activation="softmax"))
    return model


def efb0t():
    # image specifications
    img_height = 224
    img_width = 224
    # transfer learning + fine tunning of EfficientNetB0
    base_model = tf.keras.applications.EfficientNetB0(input_shape=(
        img_height, img_width, 3), include_top=False, weights="imagenet")

    # unfreeze the top layers of the model
    print("Number of layers in the base model:", len(base_model.layers))

    # Fine-tune from this layer onwards
    fine_tune_at = 150
    # Freeze all layers before the `fine_tune_at` layer
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    # build the model
    last_layer = base_model.get_layer('top_activation')
    last_output = last_layer.output
    x = GlobalAveragePooling2D()(last_output)
    x = Dense(32, activation="relu", name="FC_1")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(5, activation="softmax", name="softmax")(x)
    new_model = Model(inputs=base_model.input, outputs=x)
    return new_model


models = {
    "efb0a": efb0a(),
    "efb0t": efb0t()
}


def print_summary(model):
    print(models[model].summary())


if __name__ == "__main__":
    # initiate parser
    parser = argparse.ArgumentParser()
    # define arguments
    parser.add_argument(
        "--model",
        type=str
    )
    args = parser.parse_args()

    print_summary(args.model)
