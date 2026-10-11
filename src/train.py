import tensorflow as tf

from src.model import get_model
from src.utils import create_tf_dataset

import matplotlib.pyplot as plt

#TRAIN FUNCTION
def train_model(
    model_name,
    train_df,
    val_df,
    image_size=(224, 224),
    batch_size=32,
    epochs=20
):

    print(f"\nSelected model: {model_name}\n")

    #TF.DATA DATASETS
    train_ds = create_tf_dataset(
        train_df,
        image_size,
        batch_size,
        model_name,
        shuffle=True
    )

    for images, labels in train_ds.take(1):
        print("Images shape:", images.shape)
        print("Labels:", labels.numpy()[:20])
        print("Min pixel:", images.numpy().min())
        print("Max pixel:", images.numpy().max())

    val_ds = create_tf_dataset(
        val_df,
        image_size,
        batch_size,
        model_name,
        shuffle=False
    )

    for images, labels in train_ds.take(1):

        plt.figure(figsize=(12, 6))

        for i in range(10):
            ax = plt.subplot(2, 5, i + 1)

            plt.imshow(images[i].numpy())

            plt.title(f"Label: {labels[i].numpy()}")

            plt.axis("off")

        plt.show()

        break

    #MODEL
    model = get_model(model_name)

    preds = model.predict(train_ds.take(1))
    print(preds[:20])

    print(model.metrics_names)

    model.summary()

    print(f"\nTraining {model_name}...\n")

    loss, acc = model.evaluate(train_ds, verbose=0)
    print(loss, acc)

    #Training
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=[

            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_accuracy",
                factor=0.5,
                patience=2,
                verbose=1
            ),

            tf.keras.callbacks.EarlyStopping(
                monitor="val_accuracy",
                patience=5,
                restore_best_weights=True,
                mode="max",
                verbose=1
            ),

            tf.keras.callbacks.ModelCheckpoint(
                filepath=f"models/{model_name}.keras",
                monitor="val_accuracy",
                save_best_only=True,
                verbose=1
            )

        ]
    )

    #Fine tuning (EfficientNetB0)
    if model_name == "efficientnetb0":

        print("\nStarting Fine-Tuning...\n")

        base_model = model.get_layer("efficientnetb0")

        base_model.trainable = True

        #Freezing the network
        for layer in base_model.layers[:-20]:
            layer.trainable = False

        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=1e-5
            ),
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )

        history_finetune = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=10,
            callbacks=[

                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor="val_accuracy",
                    factor=0.5,
                    patience=2,
                    verbose=1
                ),

                tf.keras.callbacks.EarlyStopping(
                    monitor="val_accuracy",
                    patience=5,
                    restore_best_weights=True,
                    mode="max",
                    verbose=1
                ),

                tf.keras.callbacks.ModelCheckpoint(
                    filepath=f"models/{model_name}.keras",
                    monitor="val_accuracy",
                    save_best_only=True,
                    verbose=1
                )

            ]
        )

    print(history.history)

    preds = model.predict(train_ds.take(1))
    print(preds[:20])

    for images, labels in train_ds.take(1):
        preds = model.predict(images, verbose=0)
        print("Labels:")
        print(labels.numpy())
        print("Predictions:")
        print(preds[:20].flatten())

    print(f"\nSaved: models/{model_name}.keras\n")


    return history