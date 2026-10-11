import os
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from src.utils import create_tf_dataset

def evaluate_model(
    model_name,
    test_df,
    image_size=(224, 224),
    batch_size=32
):
    
    #Config
    MODEL_NAME = model_name
    MODEL_PATH = f"models/{MODEL_NAME}.keras"

    #Output paths
    METRICS_PATH = f"outputs/metrics/{MODEL_NAME}_metrics.txt"
    PREDICTIONS_PATH = f"outputs/predictions/{MODEL_NAME}_predictions.csv"
    CONF_MATRIX_PATH = f"outputs/confusion_matrices/{MODEL_NAME}_cm.png"

    #Making of output directories
    os.makedirs("outputs/metrics", exist_ok=True)
    os.makedirs("outputs/predictions", exist_ok=True)
    os.makedirs("outputs/confusion_matrices", exist_ok=True)

    print("Preparing test dataset...")

    test_ds = create_tf_dataset(
        dataframe=test_df,
        image_size=image_size,
        batch_size=batch_size,
        model_name=MODEL_NAME,
        shuffle=False
    )

    print("Loading model...")

    if MODEL_NAME == "resnet18":

        model = tf.keras.models.load_model(MODEL_PATH)

    else:

        model = tf.keras.models.load_model(MODEL_PATH)

    print("Model loaded.")

    #Predictions
    print("Running predictions...")
    probs = model.predict(test_ds, verbose=1)
    preds = (probs > 0.5).astype(int).flatten()
    true = test_df["label"].values

    #Metrics
    accuracy = accuracy_score(true, preds)
    precision = precision_score(true, preds, zero_division=0)
    recall = recall_score(true,preds,zero_division=0)
    f1 = f1_score(true,preds,zero_division=0)
    macro_f1 = f1_score(true,preds,average="macro",zero_division=0)

    report = classification_report(
        true,
        preds,
        target_names=[
            "Smooth (E)",
            "Spiral (S)"
        ],
        zero_division=0
    )

    print("\n")
    print(report)

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}")
    print(f"Macro F1 : {macro_f1:.4f}")

    #Saving of metrics
    with open(METRICS_PATH, "w") as f:

        f.write(report)
        f.write("\n")

        f.write(f"Accuracy : {accuracy:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall   : {recall:.4f}\n")
        f.write(f"F1-score : {f1:.4f}\n")
        f.write(f"Macro F1 : {macro_f1:.4f}\n")

    print("Metrics saved:", METRICS_PATH)

    #Saving predictions
    pred_df = pd.DataFrame({

        "image_path": test_df["image_path"].values,
        "true_label": true,
        "predicted_label": preds,
        "probability_spiral": probs.flatten()

    })

    print("\nProbability statistics")
    print("Min :", probs.min())
    print("Max :", probs.max())
    print("Mean:", probs.mean())

    pred_df.to_csv(
        PREDICTIONS_PATH,
        index=False
    )

    print("Predictions saved:", PREDICTIONS_PATH)

    #Confusion matrix
    cm = confusion_matrix(true, preds)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=[
            "Smooth",
            "Spiral"
        ]
    )

    fig, ax = plt.subplots(figsize=(6, 6))

    disp.plot(
        ax=ax,
        colorbar=False
    )

    plt.title(f"{MODEL_NAME.upper()} Confusion Matrix")

    plt.savefig(
        CONF_MATRIX_PATH,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("Confusion matrix saved:", CONF_MATRIX_PATH)

    print("\nEvaluation completed.\n")