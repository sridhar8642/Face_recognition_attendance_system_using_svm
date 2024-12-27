import pickle
from collections import Counter
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
# Load face embeddings
embeddingFile = "output/embeddings.pickle"
print("Loading face embeddings...")
data = pickle.loads(open(embeddingFile, "rb").read())

# Count the number of samples per class
class_counts = Counter(data["names"])
max_samples = max(class_counts.values())

# Balance the dataset by oversampling
balanced_embeddings = []
balanced_names = []

for name in class_counts:
    idxs = np.where(np.array(data["names"]) == name)[0]
    oversampled_idxs = np.random.choice(idxs, max_samples, replace=True)
    for idx in oversampled_idxs:
        balanced_embeddings.append(data["embeddings"][idx])
        balanced_names.append(data["names"][idx])

# Update the data dictionary with balanced data
data["embeddings"] = np.array(balanced_embeddings)
data["names"] = np.array(balanced_names)

print("Dataset successfully balanced!")

# Verify the class-wise sample counts
balanced_class_counts = Counter(balanced_names)
print("Balanced Class Counts:")
for name, count in balanced_class_counts.items():
    print(f"{name}: {count}")

# Ensure all classes have the same number of samples
assert len(set(balanced_class_counts.values())) == 1, "Class support is not equal!"

# Prepare features and labels
X = np.array(data["embeddings"])  # Features
y = np.array(data["names"])       # Labels

# Split data into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Re-check support in training and testing sets
train_class_counts = Counter(y_train)
test_class_counts = Counter(y_test)

print("\nTraining Set Class Counts:")
for name, count in train_class_counts.items():
    print(f"{name}: {count}")

print("\nTesting Set Class Counts:")
for name, count in test_class_counts.items():
    print(f"{name}: {count}")

# Train the SVM model
print("\nTraining the SVM model...")
svm_model = SVC(C=1.0,kernel='linear', probability=True)  # Linear kernel
svm_model.fit(X_train, y_train)

# Make predictions
y_pred = svm_model.predict(X_test)

# Calculate Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy:.2f}")

# Generate and print Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Generate and print Confusion Matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# Generate and print Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

# Plot the Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
plt.title("Confusion Matrix")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.savefig("confusion_matrix.png")
plt.show()
