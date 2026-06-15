import os
from skimage.io import imread
from skimage.transform import resize
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from concurrent.futures import ProcessPoolExecutor
from sklearn.linear_model import LogisticRegression

def load_image_with_resolution(args):
    """Load image with specified resolution - needed for ProcessPoolExecutor"""
    file_path, target_size = args
    try:
        img = imread(str(file_path))
        img = resize(img, target_size)
        return img.flatten()
    except Exception as e:
        print(f"Error loading {file_path.name}: {e}")
        return None

def test_resolution(resolution, sample_size=100):
    """Test classification accuracy at different resolutions"""
    print(f"\n{'='*50}")
    print(f"Testing {resolution[0]}x{resolution[1]} resolution")
    print(f"{'='*50}")
    
    input_dir = Path("C:/Users/sayan/.cache/kagglehub/datasets/kellysanderson/myopia-image-dataset/versions/1/IMAGES")
    categories = ["Myopia_images", "Normal_images"]

    data = []
    labels = []
    
    for category_idx, category in enumerate(categories):
        category_path = input_dir / category
        print(f"Processing {category}...")
        
        # Get list of image files
        image_files = list(category_path.glob("*.png"))
        
        # Use smaller sample for higher resolutions to avoid memory issues
        image_files = image_files[500:500+sample_size]
        
        # Prepare arguments for ProcessPoolExecutor
        args_list = [(f, resolution) for f in image_files]
        
        # Use ProcessPoolExecutor to load images
        with ProcessPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(load_image_with_resolution, args_list))
            valid_results = [r for r in results if r is not None]
            data.extend(valid_results)
            labels.extend([category_idx] * len(valid_results))
        
        print(f"Loaded {len(valid_results)} images from {category}")

    data = np.asarray(data)
    labels = np.asarray(labels)
    
    print(f"Data shape: {data.shape}")
    print(f"Label distribution: {np.bincount(labels)}")
    
    # Calculate class separability
    myopia_data = data[labels == 0]
    normal_data = data[labels == 1]
    myopia_mean = myopia_data.mean(axis=1).mean()
    normal_mean = normal_data.mean(axis=1).mean()
    difference = abs(myopia_mean - normal_mean)
    
    print(f"Myopia mean: {myopia_mean:.3f}")
    print(f"Normal mean: {normal_mean:.3f}")
    print(f"Difference: {difference:.3f}")
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        data, labels, test_size=0.2, shuffle=True, stratify=labels, random_state=42
    )
    
    # Test SVM
    svm = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
    svm.fit(X_train, y_train)
    svm_pred = svm.predict(X_test)
    svm_accuracy = accuracy_score(y_test, svm_pred)
    
    # Test Logistic Regression
    lr = LogisticRegression(random_state=42, max_iter=1000)
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_test)
    lr_accuracy = accuracy_score(y_test, lr_pred)
    
    print(f"SVM Accuracy: {svm_accuracy*100:.2f}%")
    print(f"Logistic Regression Accuracy: {lr_accuracy*100:.2f}%")
    
    return {
        'resolution': resolution,
        'svm_accuracy': svm_accuracy,
        'lr_accuracy': lr_accuracy,
        'mean_difference': difference,
        'data_shape': data.shape
    }

if __name__ == '__main__':
    # Test different resolutions
    resolutions = [
        (15, 15),    # Original (too small)
        (32, 32),    # 2x larger
        (64, 64),    # 4x larger
        (128, 128),  # 8x larger
    ]
    
    results = []
    
    for resolution in resolutions:
        try:
            result = test_resolution(resolution)
            results.append(result)
        except Exception as e:
            print(f"Error with resolution {resolution}: {e}")
            continue
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY OF RESULTS")
    print(f"{'='*60}")
    print(f"{'Resolution':<12} {'SVM Acc':<10} {'LR Acc':<10} {'Mean Diff':<10} {'Features':<10}")
    print("-" * 60)
    
    for result in results:
        print(f"{result['resolution'][0]}x{result['resolution'][1]:<8} "
              f"{result['svm_accuracy']*100:>7.1f}% "
              f"{result['lr_accuracy']*100:>7.1f}% "
              f"{result['mean_difference']:>8.3f} "
              f"{result['data_shape'][1]:>8}")
    
    print(f"\n{'='*60}")
    print("INTERPRETATION:")
    print(f"{'='*60}")
    
    # Find the most realistic accuracy
    realistic_accuracies = [r['svm_accuracy'] for r in results if r['svm_accuracy'] < 0.95]
    
    if realistic_accuracies:
        print("✅ Found realistic accuracy levels!")
        print("   Higher resolution images show more realistic performance.")
        print("   This suggests the original 15x15 resolution was too small.")
    else:
        print("🚨 All resolutions show suspiciously high accuracy.")
        print("   This might indicate the dataset itself is too simple.")
    
    # Check if mean difference decreases with resolution
    mean_diffs = [r['mean_difference'] for r in results]
    if len(mean_diffs) > 1 and mean_diffs[0] > mean_diffs[-1]:
        print("✅ Mean pixel difference decreases with higher resolution.")
        print("   This confirms that 15x15 was oversimplifying the problem.")
    else:
        print("⚠️  Mean pixel difference doesn't decrease with resolution.")
        print("   The dataset might have systematic differences between classes.") 