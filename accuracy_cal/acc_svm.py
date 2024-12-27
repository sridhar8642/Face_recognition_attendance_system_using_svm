import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle

# Paths to embedding and model files
embeddingFile = "output/embeddings.pickle"
recognizerFile = "output/recognizer.pickle"
labelEncFile = "output/le.pickle"

# Load face embeddings
print("Loading face embeddings...")
data = pickle.loads(open(embeddingFile, "rb").read())

# Encode labels
print("Encoding labels...")
labelEnc = LabelEncoder()
labels = labelEnc.fit_transform(data["names"])

# Perform stratified splitting to preserve class balance
X_train, X_test, y_train, y_test = train_test_split(
    data["embeddings"], labels, test_size=0.20, random_state=42, stratify=labels
)


# Train the model
print("Training model...")
recognizer = SVC(C=1.0, kernel="linear", probability=True)
recognizer.fit(X_train, y_train)
print("Training completed.")

# Evaluate the model on the test set
y_pred = recognizer.predict(X_test)

# Compute accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Model accuracy: {:.2f}%".format(accuracy * 100))

# Compute classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=labelEnc.classes_))

# Compute confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Visualize confusion matrix
plt.figure(figsize=(12, 10))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=labelEnc.classes_, yticklabels=labelEnc.classes_)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.show()

# Save the trained model and label encoder
with open(recognizerFile, "wb") as f:
    pickle.dump(recognizer, f)

with open(labelEncFile, "wb") as f:
    pickle.dump(labelEnc, f)
