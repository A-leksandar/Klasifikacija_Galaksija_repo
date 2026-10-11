import pandas as pd
import os
import random
import numpy as np

SEED = 42

import tensorflow as tf

from src.train import train_model
from src.evaluate import evaluate_model
from src.utils import (
    sample_dataset,
    encode_labels,
    split_dataset
)

MODELS = [
    "cnn",
    "resnet18",
    "efficientnetb0"
]

DATASET_PATH = "datasets/dataset_ES.csv"
SAMPLES_PER_CLASS = 2000


if __name__ == "__main__":

    os.environ["PYTHONHASHSEED"] = str(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    print("\n")
    print("PREPARING DATASET")
    print("\n")

    #Load dataset
    df = pd.read_csv(DATASET_PATH)

    print(df.head(20))

    #Equal number of samples of galaxies for each class
    df = sample_dataset(
        df,
        samples_per_class=SAMPLES_PER_CLASS
    )

    df["image_path"] = df["image_path"].str.replace("\\", "/", regex=False)

    #Converting the labels:  e-> 0 i s->1
    df = encode_labels(df)

    #Equal split for all models
    train_df, val_df, test_df = split_dataset(df)

    print(train_df['label'].value_counts())
    print(val_df['label'].value_counts())
    print(test_df['label'].value_counts())
    print(train_df.sample(10))

    #Training and evaluation
    for model_name in MODELS:

        print("\n")
        print(f"MODEL: {model_name.upper()}")
        print("\n")

        history = train_model(
            model_name=model_name,
            train_df=train_df,
            val_df=val_df
        )

        evaluate_model(
            model_name=model_name,
            test_df=test_df
        )

    print("\nALL MODELS FINISHED SUCCESSFULLY.\n")