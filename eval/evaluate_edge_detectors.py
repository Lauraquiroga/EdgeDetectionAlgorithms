import os
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_score, recall_score, f1_score, jaccard_score, accuracy_score,
    confusion_matrix, ConfusionMatrixDisplay
)

def binarize(img):
    """
    Convert grayscale edge image to binary (0 or 1).
    Any non-zero pixel is treated as edge.
    
    input:
    - img: np.array, grayscale image.
    Returns:
    - np.array of same shape, values in {0,1}
    """
    return (img > 0).astype(np.uint8)

def evaluate_image(gt_path, pred_path, plot_cm=False):
    """
    Evaluate predicted edge masks against ground truth masks.
    input:
    - gt_dir: str, path to ground truth mask directory.
    - pred_dir: str, path to predicted mask directory.
    - plot_cm: bool, to plot the confusion matrix
    Returns a dictionary of evaluation metrics.
    """
    gt = binarize(cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE))
    pred = binarize(cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE))

    # Ensure shape match
    if gt.shape != pred.shape:
        raise ValueError(f"Shape mismatch: {gt.shape} vs {pred.shape}")

    gt_flat = gt.flatten()
    pred_flat = pred.flatten()

    # Basic metrics
    precision = precision_score(gt_flat, pred_flat, zero_division=0)
    recall = recall_score(gt_flat, pred_flat, zero_division=0)
    f1 = f1_score(gt_flat, pred_flat, zero_division=0)
    # Intersection over Union (IoU): TP/(TP+FP+FN)
    #how well your predicted edge overlap with the actual edge
    iou = jaccard_score(gt_flat, pred_flat, zero_division=0)
    #accuracy is not a good metric here but I report it anyway
    accuracy = accuracy_score(gt_flat, pred_flat)

    # Confusion matrix (format: [positive_class, negative_class])
    cm = confusion_matrix(gt_flat, pred_flat, labels=[1, 0])
    TP, FN = cm[0, 0], cm[0, 1]
    FP, TN = cm[1, 0], cm[1, 1]

    if plot_cm:
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Edge", "Non-Edge"])
        disp.plot(cmap="Blues", values_format='d')
        plt.title(f"Confusion Matrix - {os.path.basename(gt_path)}")
        plt.show()

    return {
        "filename": os.path.basename(gt_path),
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "iou": iou,
        "accuracy": accuracy,
        "TP": TP,
        "FP": FP,
        "FN": FN,
        "TN": TN
    }

def evaluate_all_images(gt_dir, pred_dir, output_csv="edge_metrics.csv", plot_cm=False):
    """
    Evaluate all images in a directory and save results to CSV.
    returns: none ( saved the metric file for all the images to edge_metrics.csv)
    """
    results = []
    filenames = sorted(os.listdir(gt_dir))

    for file in filenames:
        gt_path = os.path.join(gt_dir, file)
        pred_path = os.path.join(pred_dir, file)

        if not os.path.exists(pred_path):
            print(f"[!] Missing prediction for {file}")
            continue

        try:
            metrics = evaluate_image(gt_path, pred_path, plot_cm=plot_cm)
            results.append(metrics)
            print(f"{file}: F1={metrics['f1_score']:.4f}, IoU={metrics['iou']:.4f}")
        except Exception as e:
            print(f"Error with {file}: {e}")

    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"\n Metrics saved to '{output_csv}'")

    if not df.empty:
        print("\n--- Summary ---")
        print(df.describe().loc[["mean", "std"]])
    else:
        print("\nNo valid results found.")



