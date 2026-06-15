import os
from skimage.io import imread
from skimage.transform import resize
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def load_sample_images(sample_size=10):
    """Load a few sample images from each class for visual inspection"""
    input_dir = Path("C:/Users/sayan/.cache/kagglehub/datasets/kellysanderson/myopia-image-dataset/versions/1/IMAGES")
    categories = ["Myopia_images", "Normal_images"]
    
    sample_images = {}
    
    for category in categories:
        category_path = input_dir / category
        image_files = list(category_path.glob("*.png"))
        
        # Load first few images
        images = []
        for i in range(min(sample_size, len(image_files))):
            try:
                img = imread(str(image_files[i]))
                images.append(img)
            except Exception as e:
                print(f"Error loading {image_files[i].name}: {e}")
        
        sample_images[category] = images
    
    return sample_images

def analyze_image_statistics():
    """Analyze statistical differences between classes"""
    input_dir = Path("C:/Users/sayan/.cache/kagglehub/datasets/kellysanderson/myopia-image-dataset/versions/1/IMAGES")
    categories = ["Myopia_images", "Normal_images"]
    
    stats = {}
    
    for category in categories:
        category_path = input_dir / category
        image_files = list(category_path.glob("*.png"))
        
        # Sample 50 images for analysis
        sample_files = image_files[500:550]
        
        means = []
        stds = []
        mins = []
        maxs = []
        
        for file_path in sample_files:
            try:
                img = imread(str(file_path))
                means.append(np.mean(img))
                stds.append(np.std(img))
                mins.append(np.min(img))
                maxs.append(np.max(img))
            except Exception as e:
                continue
        
        stats[category] = {
            'mean': np.array(means),
            'std': np.array(stds),
            'min': np.array(mins),
            'max': np.array(maxs)
        }
    
    return stats

def test_simple_classifier():
    """Test if a simple threshold classifier can achieve high accuracy"""
    input_dir = Path("C:/Users/sayan/.cache/kagglehub/datasets/kellysanderson/myopia-image-dataset/versions/1/IMAGES")
    categories = ["Myopia_images", "Normal_images"]
    
    data = []
    labels = []
    
    for category_idx, category in enumerate(categories):
        category_path = input_dir / category
        image_files = list(category_path.glob("*.png"))
        
        # Sample 100 images
        sample_files = image_files[500:600]
        
        for file_path in sample_files:
            try:
                img = imread(str(file_path))
                # Use simple mean brightness as feature
                mean_brightness = np.mean(img)
                data.append([mean_brightness])
                labels.append(category_idx)
            except Exception as e:
                continue
    
    data = np.array(data)
    labels = np.array(labels)
    
    # Simple threshold classifier
    myopia_means = data[labels == 0]
    normal_means = data[labels == 1]
    
    threshold = (np.mean(myopia_means) + np.mean(normal_means)) / 2
    
    # Predict based on threshold
    predictions = (data.flatten() > threshold).astype(int)
    
    # Calculate accuracy
    accuracy = np.mean(predictions == labels)
    
    return {
        'threshold': threshold,
        'accuracy': accuracy,
        'myopia_mean': np.mean(myopia_means),
        'normal_mean': np.mean(normal_means),
        'myopia_std': np.std(myopia_means),
        'normal_std': np.std(normal_means)
    }

if __name__ == '__main__':
    print("=" * 60)
    print("DATASET INVESTIGATION")
    print("=" * 60)
    
    # 1. Test simple threshold classifier
    print("\n1. Testing Simple Threshold Classifier:")
    print("-" * 40)
    threshold_results = test_simple_classifier()
    
    print(f"Threshold: {threshold_results['threshold']:.3f}")
    print(f"Accuracy: {threshold_results['accuracy']*100:.2f}%")
    print(f"Myopia mean: {threshold_results['myopia_mean']:.3f} ± {threshold_results['myopia_std']:.3f}")
    print(f"Normal mean: {threshold_results['normal_mean']:.3f} ± {threshold_results['normal_std']:.3f}")
    
    if threshold_results['accuracy'] > 0.95:
        print("🚨 Simple brightness threshold achieves >95% accuracy!")
        print("   This confirms the dataset is trivially separable.")
    
    # 2. Analyze image statistics
    print("\n2. Image Statistics Analysis:")
    print("-" * 40)
    stats = analyze_image_statistics()
    
    for category in stats:
        print(f"\n{category}:")
        print(f"  Mean brightness: {np.mean(stats[category]['mean']):.3f} ± {np.std(stats[category]['mean']):.3f}")
        print(f"  Std brightness: {np.mean(stats[category]['std']):.3f} ± {np.std(stats[category]['std']):.3f}")
        print(f"  Min brightness: {np.mean(stats[category]['min']):.3f} ± {np.std(stats[category]['min']):.3f}")
        print(f"  Max brightness: {np.mean(stats[category]['max']):.3f} ± {np.std(stats[category]['max']):.3f}")
    
    # 3. Calculate overlap between classes
    print("\n3. Class Separability Analysis:")
    print("-" * 40)
    
    myopia_means = stats['Myopia_images']['mean']
    normal_means = stats['Normal_images']['mean']
    
    # Calculate overlap
    myopia_min, myopia_max = np.min(myopia_means), np.max(myopia_means)
    normal_min, normal_max = np.min(normal_means), np.max(normal_means)
    
    overlap_min = max(myopia_min, normal_min)
    overlap_max = min(myopia_max, normal_max)
    
    if overlap_max > overlap_min:
        overlap_range = overlap_max - overlap_min
        myopia_range = myopia_max - myopia_min
        normal_range = normal_max - normal_min
        total_range = max(myopia_max, normal_max) - min(myopia_min, normal_min)
        
        overlap_percentage = overlap_range / total_range * 100
        
        print(f"Overlap range: {overlap_range:.3f}")
        print(f"Overlap percentage: {overlap_percentage:.1f}%")
        
        if overlap_percentage < 10:
            print("🚨 Very little overlap between classes!")
            print("   Classes are almost perfectly separable.")
    else:
        print("🚨 No overlap between classes!")
        print("   Classes are perfectly separable.")
    
    # 4. Recommendations
    print("\n4. CONCLUSIONS & RECOMMENDATIONS:")
    print("-" * 40)
    
    if threshold_results['accuracy'] > 0.95:
        print("🚨 PROBLEM IDENTIFIED:")
        print("   The dataset has systematic brightness differences between classes.")
        print("   This makes the classification task trivial and unrealistic.")
        print("\n   POSSIBLE CAUSES:")
        print("   - Different image preprocessing between classes")
        print("   - Different lighting conditions during capture")
        print("   - Different image sources or acquisition methods")
        print("   - Artificial dataset construction with biases")
        print("\n   RECOMMENDATIONS:")
        print("   - Find a different, more realistic medical image dataset")
        print("   - Use a dataset with proper preprocessing and normalization")
        print("   - Look for datasets specifically designed for medical ML")
        print("   - Consider using synthetic data with controlled variations")
    else:
        print("✅ Dataset appears to have realistic class overlap.")
        print("   The classification task is genuinely challenging.") 