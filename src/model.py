import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0

#Model config
INPUT_SHAPE = (224, 224, 3)
LEARNING_RATE = 0.0001


#CNN
def build_model_CNN():

    model = tf.keras.Sequential([

        #Input layer
        tf.keras.layers.Input(shape=INPUT_SHAPE),

        #Block 1
        tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), padding="same", activation="relu", kernel_initializer="he_normal"), 
        tf.keras.layers.ReLU(),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        #Block 2
        tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu", kernel_initializer="he_normal"), 
        tf.keras.layers.ReLU(),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        #Block 3
        tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu", kernel_initializer="he_normal"), 
        tf.keras.layers.ReLU(),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        #Classification head
        tf.keras.layers.GlobalAveragePooling2D(),

        tf.keras.layers.Dense(units=128, activation="relu", kernel_initializer="he_normal"),
        tf.keras.layers.Dropout(rate=0.2),

        tf.keras.layers.Dense(units=1, activation="sigmoid")
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

    model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])

    return model










from tensorflow.keras.applications import EfficientNetB0

#EFFICIENTNETB0
def build_model_EfficientNetB0():

    inputs = tf.keras.Input(shape=(224, 224, 3))

    #Data augmentation
    x = tf.keras.layers.RandomFlip("horizontal")(inputs)
    x = tf.keras.layers.RandomRotation(0.10)(x)
    x = tf.keras.layers.RandomZoom(0.10)(x)

    base_model = EfficientNetB0(
        weights="imagenet",
        include_top=False,
        input_shape=(224, 224, 3),
        name="efficientnetb0"
    )

    base_model.trainable = False

    x = base_model(x, training=False)

    #Classification head
    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    x = tf.keras.layers.Dropout(0.3)(x)

    outputs = tf.keras.layers.Dense(
        1,
        activation="sigmoid"
    )(x)

    model = tf.keras.Model(
        inputs,
        outputs
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=1e-4
        ),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model











#RESNET18
def residual_block(x, filters, stride=1):

    shortcut = x

    #First conv
    x = tf.keras.layers.Conv2D(
        filters,
        kernel_size=3,
        strides=stride,
        padding="same",
        use_bias=False,
        kernel_initializer="he_normal",
        kernel_regularizer=tf.keras.regularizers.l2(1e-4)
    )(x)

    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    #Second conv
    x = tf.keras.layers.Conv2D(
        filters,
        kernel_size=3,
        strides=1,
        padding="same",
        use_bias=False,
        kernel_initializer="he_normal",
        kernel_regularizer=tf.keras.regularizers.l2(1e-4)
    )(x)

    x = tf.keras.layers.BatchNormalization()(x)

    #Projection shortcut
    if stride != 1 or shortcut.shape[-1] != filters:

        shortcut = tf.keras.layers.Conv2D(
            filters,
            kernel_size=1,
            strides=stride,
            padding="same",
            use_bias=False,
            kernel_initializer="he_normal",
            kernel_regularizer=tf.keras.regularizers.l2(1e-4)
        )(shortcut)

        shortcut = tf.keras.layers.BatchNormalization()(shortcut)

    #Residual connection
    x = tf.keras.layers.Add()([x, shortcut])
    x = tf.keras.layers.ReLU()(x)

    return x


def build_model_ResNet18():

    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(0.2),
        tf.keras.layers.RandomZoom(0.1)
    ], name="data_augmentation")

    inputs = tf.keras.Input(shape=INPUT_SHAPE)

    x = data_augmentation(inputs)

    x = tf.keras.layers.Conv2D(
        64,
        kernel_size=7,
        strides=2,
        padding="same",
        use_bias=False,
        kernel_initializer="he_normal",
        kernel_regularizer=tf.keras.regularizers.l2(1e-4)
    )(x)

    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.MaxPooling2D(
        pool_size=3,
        strides=2,
        padding="same"
    )(x)

    #Stage 1---------------------------
    x = residual_block(x, 64, stride=1)
    x = residual_block(x, 64, stride=1)

    #Stage 2---------------------------
    x = residual_block(x, 128, stride=2)
    x = residual_block(x, 128, stride=1)

    #Stage 3---------------------------
    x = residual_block(x, 256, stride=2)
    x = residual_block(x, 256, stride=1)

    #Stage 4---------------------------
    x = residual_block(x, 512, stride=2)
    x = residual_block(x, 512, stride=1)

    #Classificator
    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    x = tf.keras.layers.Dropout(0.5)(x)

    outputs = tf.keras.layers.Dense(
        1,
        activation="sigmoid",
        kernel_initializer="he_normal",
        kernel_regularizer=tf.keras.regularizers.l2(1e-4)
    )(x)

    model = tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="resnet18"
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model






def get_model(name="cnn"):

    name = name.lower()

    if name == "cnn":
        return build_model_CNN()

    if name == "resnet18":
        return build_model_ResNet18()

    if name == "efficientnetb0":
        return build_model_EfficientNetB0()

    raise ValueError(f"Unknown model: {name}")