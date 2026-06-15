import os
from skimage.io import imread
from skimage.transform import resize
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.decomposition import PCA
from concurrent.futures import ProcessPoolExecutor
import matplotlib.pyplot as plt

def load_image_small(file_path, target_size=(15, 15)):
    """Load image at small resolution (original)"""
    try:
        img = imread(str(file_path))
        img = resize(img, target_size)
        return img.flatten()
    except Exception as e:
        print(f"Error loading {file_path.name}: {e}")
        return None

def load_image_large(file_path, target_size=(64, 64)):
    """Load image at larger resolution"""
    try:
        img = imread(str(file_path))
        img = resize(img, target_size)
        return img.flatten()
    except Exception as e:
        print(f"Error loading {file_path.name}: {e}")
        return None

def process_data(image_range=(500, 1000), resolution='small'):
    """Process data with configurable range and resolution"""
    input_dir = Path("C:/Users/sayan/.cache/kagglehub/datasets/kellysanderson/myopia-image-dataset/versions/1/IMAGES")
    categories = ["Myopia_images", "Normal_images"]

    data = []
    labels = []
    all_file_paths = []
    
    # Choose resolution
    if resolution == 'small':
        load_func = load_image_small
        target_size = (15, 15)
    else:
        load_func = load_image_large
        target_size = (64, 64)
    
    for category_idx, category in enumerate(categories):
        category_path = input_dir / category
        print(f"Processing category: {category} with {resolution} resolution")
        
        # Get list of image files
        image_files = list(category_path.glob("*.png"))
        print(f"Found {len(image_files)} images in {category}")
        
        # Use specified range
        start_idx, end_idx = image_range
        image_files = image_files[start_idx:end_idx]
        print(f"Processing images {start_idx}-{end_idx} ({len(image_files)} images)")
        
        # Use ProcessPoolExecutor to load images
        with ProcessPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(load_func, image_files))
            valid_results = [r for r in results if r is not None]
            data.extend(valid_results)
            labels.extend([category_idx] * len(valid_results))
            
            valid_file_paths = [str(f) for f, r in zip(image_files, results) if r is not None]
            all_file_paths.extend(valid_file_paths)
        
        print(f"Successfully loaded {len(valid_results)} images from {category}")

    data = np.asarray(data)
    labels = np.asarray(labels)
    
    print(f"Final data shape: {data.shape}")
    print(f"Final labels shape: {labels.shape}")
    print(f"Label distribution: {np.bincount(labels)}")
    
    return data, labels, all_file_paths

def test_model_performance(data, labels, model_name, model):
    """Test a specific model and return performance metrics"""
    print(f"\n=== Testing {model_name} ===")
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        data, labels, test_size=0.2, shuffle=True, stratify=labels, random_state=42
    )
    
    # Train model
    if model_name == "SVM":
        # Use GridSearch for SVM
        parameters = [{'gamma': [0.01, 0.001, 0.0001], 'C': [1, 10, 100, 1000]}]
        grid_search = GridSearchCV(model, parameters, cv=3)
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        print(f"Best parameters: {grid_search.best_params_}")
    else:
        # Simple fit for other models
        model.fit(X_train, y_train)
        best_model = model
    
    # Predict and evaluate
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Test Accuracy: {accuracy*100:.2f}%")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Myopia', 'Normal']))
    
    # Cross-validation
    cv_scores = cross_val_score(best_model, data, labels, cv=5)
    print(f"CV scores: {cv_scores}")
    print(f"CV mean: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
    
    return accuracy, cv_scores.mean()

def analyze_data_separability(data, labels):
    """Analyze how well the classes can be separated"""
    print("\n=== Data Separability Analysis ===")
    
    # PCA for visualization
    pca = PCA(n_components=2)
    data_pca = pca.fit_transform(data)
    
    # Plot PCA results
    plt.figure(figsize=(10, 8))
    
    plt.subplot(2, 2, 1)
    plt.scatter(data_pca[labels == 0, 0], data_pca[labels == 0, 1], 
                c='red', label='Myopia', alpha=0.6)
    plt.scatter(data_pca[labels == 1, 0], data_pca[labels == 1, 1], 
                c='blue', label='Normal', alpha=0.6)
    plt.title('PCA Visualization (2 components)')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.legend()
    
    # Check class separability metrics
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    lda = LinearDiscriminantAnalysis()
    data_lda = lda.fit_transform(data, labels)
    
    plt.subplot(2, 2, 2)
    plt.scatter(data_lda[labels == 0], np.zeros_like(data_lda[labels == 0]), 
                c='red', label='Myopia', alpha=0.6)
    plt.scatter(data_lda[labels == 1], np.zeros_like(data_lda[labels == 1]), 
                c='blue', label='Normal', alpha=0.6)
    plt.title('LDA Projection')
    plt.xlabel('LDA Component')
    plt.ylabel('')
    plt.legend()
    
    # Data statistics
    plt.subplot(2, 2, 3)
    myopia_data = data[labels == 0]
    normal_data = data[labels == 1]
    
    plt.hist(myopia_data.mean(axis=1), bins=30, alpha=0.7, label='Myopia', color='red')
    plt.hist(normal_data.mean(axis=1), bins=30, alpha=0.7, label='Normal', color='blue')
    plt.title('Distribution of Mean Pixel Values')
    plt.xlabel('Mean Pixel Value')
    plt.ylabel('Frequency')
    plt.legend()
    
    # Feature importance (if using Random Forest)
    plt.subplot(2, 2, 4)
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(data, labels)
    
    # Show top 20 features
    feature_importance = rf.feature_importances_
    top_features = np.argsort(feature_importance)[-20:]
    plt.barh(range(20), feature_importance[top_features])
    plt.title('Top 20 Feature Importances')
    plt.xlabel('Importance')
    
    plt.tight_layout()
    plt.show()
    
    # Print separability metrics
    print(f"PCA explained variance ratio: {pca.explained_variance_ratio_}")
    print(f"LDA explained variance ratio: {lda.explained_variance_ratio_}")
    
    # Check if classes are trivially separable
    myopia_mean = myopia_data.mean(axis=1).mean()
    normal_mean = normal_data.mean(axis=1).mean()
    print(f"Myopia mean pixel value: {myopia_mean:.3f}")
    print(f"Normal mean pixel value: {normal_mean:.3f}")
    print(f"Difference: {abs(myopia_mean - normal_mean):.3f}")

def run_diagnostic_tests():
    """Run comprehensive diagnostic tests"""
    print("=== COMPREHENSIVE DIAGNOSTIC TESTS ===")
    
    # Test 1: Original setup (small resolution, range 500-1000)
    print("\n" + "="*50)
    print("TEST 1: Original Setup (15x15, range 500-1000)")
    print("="*50)
    data1, labels1, paths1 = process_data(image_range=(500, 1000), resolution='small')
    
    # Test different models
    models = {
        "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "SVM": SVC(random_state=42)
    }
    
    results = {}
    for name, model in models.items():
        acc, cv_acc = test_model_performance(data1, labels1, name, model)
        results[name] = (acc, cv_acc)
    
    # Test 2: Different data range
    print("\n" + "="*50)
    print("TEST 2: Different Data Range (15x15, range 1000-1500)")
    print("="*50)
    data2, labels2, paths2 = process_data(image_range=(1000, 1500), resolution='small')
    
    for name, model in models.items():
        acc, cv_acc = test_model_performance(data2, labels2, name, model)
        results[f"{name} (Range 2)"] = (acc, cv_acc)
    
    # Test 3: Higher resolution
    print("\n" + "="*50)
    print("TEST 3: Higher Resolution (64x64, range 500-1000)")
    print("="*50)
    data3, labels3, paths3 = process_data(image_range=(500, 1000), resolution='large')
    
    for name, model in models.items():
        acc, cv_acc = test_model_performance(data3, labels3, name, model)
        results[f"{name} (64x64)"] = (acc, cv_acc)
    
    # Analyze data separability
    print("\n" + "="*50)
    print("DATA SEPARABILITY ANALYSIS")
    print("="*50)
    analyze_data_separability(data1, labels1)
    
    # Summary
    print("\n" + "="*50)
    print("SUMMARY OF RESULTS")
    print("="*50)
    for name, (acc, cv_acc) in results.items():
        print(f"{name:25s}: Test={acc*100:5.1f}%, CV={cv_acc*100:5.1f}%")
    
    # Conclusions
    print("\n" + "="*50)
    print("CONCLUSIONS")
    print("="*50)
    
    # Check if accuracy is suspiciously high
    suspicious_count = sum(1 for acc, cv_acc in results.values() if acc > 0.95)
    total_tests = len(results)
    
    if suspicious_count == total_tests:
        print("🚨 ALL TESTS SHOW >95% ACCURACY - This is suspicious!")
        print("Possible causes:")
        print("1. Dataset is too simple/trivial")
        print("2. Resolution is too low for meaningful features")
        print("3. Classes are trivially separable")
        print("4. Data leakage (though we checked for this)")
    elif suspicious_count > total_tests // 2:
        print("⚠️  Most tests show high accuracy - investigate further")
    else:
        print("✅ Accuracy seems reasonable across different conditions")

if __name__ == '__main__':
    run_diagnostic_tests() 