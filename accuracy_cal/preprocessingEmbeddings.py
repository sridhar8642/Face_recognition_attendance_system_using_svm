import numpy as np
import imutils
import pickle
import cv2
import os
import dlib
from imutils import paths
from imgaug import augmenters as iaa
from sklearn.preprocessing import normalize  # Import normalize function

# Paths to your dataset and output files
dataset = "dataset"
embeddingFile = "output/embeddings.pickle"

# Load the face detector and the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("model/shape_predictor_68_face_landmarks.dat")
embedder = dlib.face_recognition_model_v1("model/dlib_face_recognition_resnet_model_v1.dat")

# Define augmentation sequence
augmenters = iaa.Sequential([
    iaa.Affine(rotate=(-30, 30)),                    
    iaa.Affine(scale=(0.8, 1.2)),                    
    iaa.Fliplr(0.5),                                 
    iaa.Multiply((0.8, 1.2)),                        
    iaa.LinearContrast((0.75, 1.5)),                 
    iaa.AdditiveGaussianNoise(scale=(0, 0.05*255)),  
    iaa.CropAndPad(percent=(-0.1, 0.1)),             
    iaa.GammaContrast(gamma=(0.5, 1.5)),             
])

# Get image paths
imagePaths = list(paths.list_images(dataset))

# Initialization
knownEmbeddings = []
knownNames = []
total = 0

# Process images one by one to apply face detection and embedding extraction
for (i, imagePath) in enumerate(imagePaths):
    print("Processing image {}/{}".format(i + 1, len(imagePaths)))
    name = imagePath.split(os.path.sep)[-2]
    image = cv2.imread(imagePath)
    image = imutils.resize(image, width=600)

    # Convert the image to RGB (dlib uses RGB format)
    rgbImage = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Apply augmentations
    for aug_image in augmenters(images=[rgbImage]):
        # Detect faces in the augmented image
        boxes = detector(aug_image, 1)
        
        # Loop over the face detections
        for box in boxes:
            # Extract landmarks using the shape predictor
            shape = predictor(aug_image, box)
            face_descriptor = embedder.compute_face_descriptor(aug_image, shape)
            
            # Normalize the embedding
            normalized_embedding = normalize([face_descriptor])[0]

            # Store the name and normalized embeddings
            knownNames.append(name)
            knownEmbeddings.append(np.array(normalized_embedding))
            total += 1

print("Total embeddings processed: {}".format(total))
data = {"embeddings": knownEmbeddings, "names": knownNames}

# Save the embeddings and names to a file
with open(embeddingFile, "wb") as f:
    f.write(pickle.dumps(data))

print("Process Completed")
