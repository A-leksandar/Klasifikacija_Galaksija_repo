import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split

from tensorflow.keras.applications.efficientnet import preprocess_input


#SAMPLE DATASET
def sample_dataset(
    df,
    samples_per_class=1000,
    random_state=42
):
    
    df_e = df[
        df["label"] == "E"
    ].sample(
        n=samples_per_class,
        random_state=random_state
    )

    df_s = df[
        df["label"] == "S"
    ].sample(
        n=samples_per_class,
        random_state=random_state
    )

    df = pd.concat(
        [df_e, df_s]
    ).sample(
        frac=1,
        random_state=random_state
    ).reset_index(
        drop=True
    )

    return df


#Converting the labels
def encode_labels(df):
    #Converting E -> 0 , i S -> 1

    df = df.copy()

    df["label"] = (
        df["label"] == "S"
    ).astype(int)

    return df


#SPLIT
def split_dataset(
    df,
    test_size=0.20,
    random_state=42
):

    train_df, temp_df = train_test_split(

        df,

        test_size=test_size,

        stratify=df["label"],

        random_state=random_state

    )

    val_df, test_df = train_test_split(

        temp_df,

        test_size=0.50,

        stratify=temp_df["label"],

        random_state=random_state

    )

    print(train_df.head())
    print(val_df.head())
    print(test_df.head())
    print(train_df["label"].value_counts())
    print(val_df["label"].value_counts())
    print(test_df["label"].value_counts())

    return train_df, val_df, test_df




def load_image(
    path,
    label,
    image_size,
    model_name
):

    image = tf.io.read_file(path)

    image = tf.image.decode_jpeg(
        image,
        channels=3
    )

    image = tf.image.resize(
        image,
        image_size
    )

    image = tf.cast(image, tf.float32)
    if model_name == "efficientnetb0":
        image = preprocess_input(image)
    else:
        image = image / 255.0

    tf.debugging.assert_all_finite(image, "Image contains NaN!")


    return image, label




def create_tf_dataset(
    dataframe,
    image_size,
    batch_size,
    model_name,
    shuffle=False
):

    dataset = tf.data.Dataset.from_tensor_slices(
        (
            dataframe["image_path"].values,
            dataframe["label"].values
        )
    )

    dataset = dataset.map(
        lambda path, label: load_image(
            path,
            label,
            image_size,
            model_name
        ),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(dataframe), seed=42, reshuffle_each_iteration=False)

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset