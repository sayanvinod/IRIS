import os
from skimage.io import imread
from skimage.transform import resize
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from concurrent.futures import ProcessPoolExecutor
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA

# PREPARE DATA
def load_image(file_path, target_size=(15, 15)):
    try:
        img = imread(str(file_path))
        img = resize(img, target_size)
        return img.flatten()
    except Exception as e:
        print(f"Error loading {file_path.name}: {e}")
        return None

def process_data():
    input_dir = Path("C:/Users/sayan/.cache/kagglehub/datasets/kellysanderson/myopia-image-dataset/versions/1/IMAGES")
    categories = ["Myopia_images", "Normal_images"]

    data = []
    labels = []
    all_file_paths = []  # Track ALL file paths
    
    for category_idx, category in enumerate(categories):
        category_path = input_dir / category
        print(f"Processing category: {category}")
        
        # Get list of image files
        image_files = list(category_path.glob("*.png"))
        print(f"Found {len(image_files)} images in {category}")
        
        # Limit to first 100 images for testing
        image_files = image_files[500:1000]
        print(f"Processing first {len(image_files)} images for testing")
        
        # Use ProcessPoolExecutor to load images
        with ProcessPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(load_image, image_files))
            valid_results = [r for r in results if r is not None]
            data.extend(valid_results)
            labels.extend([category_idx] * len(valid_results))
            
            # Track file paths for valid results
            valid_file_paths = [str(f) for f, r in zip(image_files, results) if r is not None]
            all_file_paths.extend(valid_file_paths)
        
        print(f"Successfully loaded {len(valid_results)} images from {category}")

    data = np.asarray(data)
    labels = np.asarray(labels)
    
    print(f"Final data shape: {data.shape}")
    print(f"Final labels shape: {labels.shape}")
    print(f"Label distribution: {np.bincount(labels)}")
    
    # Check if file paths contain class information (DATA LEAKAGE CHECK)
    myopia_files = [f for f in all_file_paths if "Myopia" in f]
    normal_files = [f for f in all_file_paths if "Normal" in f]
    print(f"Files with 'Myopia' in path: {len(myopia_files)}")
    print(f"Files with 'Normal' in path: {len(normal_files)}")
    
    # Check for identical samples
    unique_samples = np.unique(data, axis=0)
    print(f"Total samples: {len(data)}")
    print(f"Unique samples: {len(unique_samples)}")
    print(f"Duplicate rate: {(len(data) - len(unique_samples)) / len(data) * 100:.2f}%")
    
    return data, labels

if __name__ == '__main__':
    data, labels = process_data()
    
    # TRAIN / TEST SPLIT
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size = 0.2, shuffle = True, stratify = labels)

    # TRAIN CLASSIFIER
    classifier = SVC()

    parameters = [{'gamma': [0.01, 0.001, 0.0001], 'C': [1, 10, 100, 1000]}]

    grid_search = GridSearchCV(classifier, parameters)

    grid_search.fit(X_train, y_train)

    # EVALUATE CLASSIFIER
    best_estimator = grid_search.best_estimator_

    y_pred = best_estimator.predict(X_test)

    score = accuracy_score(y_pred, y_test)

    print(f'{score*100:.2f}% of samples were correctly classified')
    print(f'Best parameters: {grid_search.best_params_}')

    # Cross-Validation
    scores = cross_val_score(classifier, data, labels, cv=5)
    print(f"CV scores: {scores}")
    print(f"CV mean: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")

    # Quick diagnostic: Test with Logistic Regression
    print("\n=== Testing with Logistic Regression ===")
    lr_classifier = LogisticRegression(random_state=42, max_iter=1000)
    lr_classifier.fit(X_train, y_train)
    lr_pred = lr_classifier.predict(X_test)
    lr_score = accuracy_score(lr_pred, y_test)
    print(f'Logistic Regression Accuracy: {lr_score*100:.2f}%')

    # Quick diagnostic: Check data separability with PCA
    print("\n=== PCA Analysis ===")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(data)
    
    # Calculate class separability
    myopia_data = data[labels == 0]
    normal_data = data[labels == 1]
    myopia_mean = myopia_data.mean(axis=1).mean()
    normal_mean = normal_data.mean(axis=1).mean()
    print(f"Myopia mean pixel value: {myopia_mean:.3f}")
    print(f"Normal mean pixel value: {normal_mean:.3f}")
    print(f"Difference: {abs(myopia_mean - normal_mean):.3f}")
    
    if abs(myopia_mean - normal_mean) > 0.1:
        print("⚠️  Large difference in mean pixel values - classes might be trivially separable")
    else:
        print("✅ Mean pixel values are similar")