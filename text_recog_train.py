import numpy as np
import joblib
import os
import shutil
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Function to extract HOG features
def extract_hog_features(images):
    features = []
    for img in images:
        img = img.numpy().reshape(28, 28)  # Convert to 28x28 format
        hog_features = hog(img, orientations=9, pixels_per_cell=(8, 8),
                           cells_per_block=(2, 2), block_norm='L2-Hys')
        features.append(hog_features)
    return np.array(features)

# Load EMNIST dataset (Letters Split)
print("Downloading EMNIST dataset...")
transform = transforms.Compose([transforms.ToTensor()])
emnist_train = datasets.EMNIST(root='./data', split='letters', train=True, download=True, transform=transform)
emnist_test = datasets.EMNIST(root='./data', split='letters', train=False, download=True, transform=transform)

# Convert dataset to NumPy arrays
X_train = emnist_train.data
y_train = emnist_train.targets.numpy()

X_test = emnist_test.data
y_test = emnist_test.targets.numpy()

print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")

# Extract features
print("Extracting HOG features...")
X_train_hog = extract_hog_features(X_train)
X_test_hog = extract_hog_features(X_test)

# Train SVM model
print("Training SVM model...")
svm = SVC(kernel='rbf', C=10, gamma=0.01)
svm.fit(X_train_hog, y_train)

# Evaluate model
y_pred = svm.predict(X_test_hog)
accuracy = accuracy_score(y_test, y_pred)
print(f"SVM Model Accuracy: {accuracy:.2f}")

# Save the model
joblib.dump(svm, "svm_ocr.pkl")
print("Model saved as svm_ocr.pkl")
