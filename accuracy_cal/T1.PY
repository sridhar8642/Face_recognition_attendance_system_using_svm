import numpy as np
from sklearn.metrics import confusion_matrix

# Confusion Matrix from the previous result
cm = np.array([
    [10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 10, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 10, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 10, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 10, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0, 9, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 9, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10]
])

# Initialize lists to store TP, TN, FP, FN for each class
TP = []
TN = []
FP = []
FN = []

# Number of classes
n_classes = cm.shape[0]

# Total sum of the confusion matrix
total_sum = cm.sum()

# Loop through each class (row)
for i in range(n_classes):
    TP_i = cm[i, i]  # True Positive: Diagonal element
    FN_i = cm[i, :].sum() - TP_i  # False Negative: Row sum minus diagonal
    FP_i = cm[:, i].sum() - TP_i  # False Positive: Column sum minus diagonal
    TN_i = total_sum - (TP_i + FP_i + FN_i)  # True Negative: Total sum minus TP, FP, FN

    TP.append(TP_i)
    TN.append(TN_i)
    FP.append(FP_i)
    FN.append(FN_i)

# Calculate and print the metrics for each class
for i in range(n_classes):
    # Accuracy (per class)
    accuracy = (TP[i] + TN[i]) / (TP[i] + TN[i] + FP[i] + FN[i])

    # Precision (per class)
    precision = TP[i] / (TP[i] + FP[i]) if (TP[i] + FP[i]) != 0 else 0

    # Recall (per class)
    recall = TP[i] / (TP[i] + FN[i]) if (TP[i] + FN[i]) != 0 else 0

    # F1-Score (per class)
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

    # Print results
    print(f"Class {i}:")
    print(f"  TP: {TP[i]}")
    print(f"  TN: {TN[i]}")
    print(f"  FP: {FP[i]}")
    print(f"  FN: {FN[i]}")
    print(f"  Accuracy: {accuracy:.2f}")
    print(f"  Precision: {precision:.2f}")
    print(f"  Recall: {recall:.2f}")
    print(f"  F1-Score: {f1_score:.2f}")
    print("")
