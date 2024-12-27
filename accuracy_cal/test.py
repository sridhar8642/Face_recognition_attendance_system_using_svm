import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.manifold import TSNE
from matplotlib.colors import ListedColormap
from matplotlib import cm

# Paths to embedding and model files
embeddingFile = "output/embeddings.pickle"
recognizerFile = "output/recognizer.pickle"
labelEncFile = "output/le.pickle"

# Load the embeddings and labels
print("Loading face embeddings and labels...")
data = pickle.loads(open(embeddingFile, "rb").read())

# Load the trained SVM model and the label encoder
recognizer = pickle.loads(open(recognizerFile, "rb").read())
labelEnc = pickle.loads(open(labelEncFile, "rb").read())

# Convert the embeddings to a NumPy array
embeddings_array = np.array(data["embeddings"])

# Reduce dimensionality to 2D using t-SNE for visualization
print("Reducing dimensionality using t-SNE...")
tsne = TSNE(n_components=2, random_state=42)
reduced_embeddings = tsne.fit_transform(embeddings_array)

# Updated colormap for 30 classes
cmap = cm.get_cmap('tab20', 30)

# Scatter plot with t-SNE reduced embeddings
plt.figure(figsize=(15, 10))
scatter = plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1],
                      c=labelEnc.transform(data["names"]), cmap=cmap, s=50, alpha=0.7)

# Generate handles for the legend dynamically
handles = []
for idx, label in enumerate(labelEnc.classes_):
    handles.append(plt.Line2D([0], [0], marker='o', color='w', label=label,
                                markerfacecolor=cmap(idx), markersize=10))

plt.legend(handles=handles, title="Classes", bbox_to_anchor=(1.05, 1), loc='upper left', ncol=2)
plt.title("t-SNE Visualization of Face Embeddings (30 Classes)")
plt.xlabel("t-SNE component 1")
plt.ylabel("t-SNE component 2")
plt.show()

# Train SVM on reduced embeddings
print("Training SVM with reduced 2D embeddings")
reduced_svm = SVC(kernel='linear', C=1.0).fit(reduced_embeddings, labelEnc.transform(data["names"]))

# Create a mesh grid for decision boundary visualization
h = 0.3  # Larger step size for better performance with 30 classes
x_min, x_max = reduced_embeddings[:, 0].min() - 1, reduced_embeddings[:, 0].max() + 1
y_min, y_max = reduced_embeddings[:, 1].min() - 1, reduced_embeddings[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Predict decision boundaries
Z = reduced_svm.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot decision boundaries
plt.figure(figsize=(15, 10))
plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.6)

# Overlay the scatter plot
scatter = plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1],
                      c=labelEnc.transform(data["names"]), cmap=cmap, s=50, edgecolors='k')

plt.legend(handles=handles, title="Classes", bbox_to_anchor=(1.05, 1), loc='upper left', ncol=2)
plt.title("SVM Decision Boundary on t-SNE Reduced Embeddings (30 Classes)")
plt.xlabel("t-SNE component 1")
plt.ylabel("t-SNE component 2")
plt.show()
