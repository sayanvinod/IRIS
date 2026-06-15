import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from ultralytics import YOLO
import os
from pathlib import Path

def load_trained_model(model_path=None):
    """
    Load the trained YOLO model - automatically finds the most recent best.pt
    """
    if model_path is None:
        # Look for the most recent best.pt file
        model_files = []
        for root, dirs, files in os.walk('runs'):
            for file in files:
                if file == 'best.pt':
                    full_path = os.path.join(root, file)
                    model_files.append(full_path)
        
        if model_files:
            # Sort by modification time to get the most recent
            model_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            model_path = model_files[0]
            print(f"Found {len(model_files)} trained models:")
            for i, path in enumerate(model_files, 1):
                mtime = os.path.getmtime(path)
                print(f"  {i}. {path} (modified: {mtime})")
            print(f"Using most recent model: {model_path}")
        else:
            print("No trained models found in runs/ directory")
            return None
    
    if os.path.exists(model_path):
        model = YOLO(model_path)
        print(f"Loaded trained model from {model_path}")
        return model
    else:
        print(f"Trained model not found at {model_path}")
        print("Available model files:")
        # List available model files
        for root, dirs, files in os.walk('runs'):
            for file in files:
                if file.endswith('.pt'):
                    print(f"  {os.path.join(root, file)}")
        return None

def get_predictions_and_labels(model, test_data_path):
    """
    Get predictions and true labels from test dataset
    """
    predictions = []
    true_labels = []
    
    # Get all image files from test directory
    test_dir = Path(test_data_path)
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    
    for img_path in test_dir.rglob('*'):
        if img_path.suffix.lower() in image_extensions:
            # Get true label from directory name
            true_label = img_path.parent.name
            
            # Make prediction
            results = model.predict(str(img_path), verbose=False)
            
            if results and len(results) > 0:
                # Get predicted class
                pred_class = results[0].probs.top1 if hasattr(results[0], 'probs') else results[0].names[0]
                pred_label = results[0].names[pred_class] if hasattr(results[0], 'names') else str(pred_class)
                
                predictions.append(pred_label)
                true_labels.append(true_label)
    
    return predictions, true_labels

def create_confusion_matrix(y_true, y_pred, class_names=None):
    """
    Create and display confusion matrix
    """
    # Create confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=class_names)
    
    # Create figure and axis
    plt.figure(figsize=(10, 8))
    
    # Create heatmap using seaborn
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    # Add text annotations for better readability
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            plt.text(j+0.5, i+0.5, str(cm[i, j]),
                    ha='center', va='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return cm

def print_classification_report(y_true, y_pred, class_names=None):
    """
    Print detailed classification report
    """
    print("\n" + "="*50)
    print("CLASSIFICATION REPORT")
    print("="*50)
    print(classification_report(y_true, y_pred, target_names=class_names))
    print("="*50)

def main():
    """
    Main function to generate confusion matrix
    """
    print("Loading trained model...")
    model = load_trained_model()
    
    if model is None:
        print("Could not load model. Please ensure the model has been trained.")
        return
    
    # Define validation data path (using the Validation directory from your dataset)
    test_data_path = r"C:\Users\sayan\.cache\kagglehub\datasets\yolo_dataset\myopia-image-dataset\versions\1\Validation"
    
    if not os.path.exists(test_data_path):
        print(f"Test data path not found: {test_data_path}")
        print("Please update the test_data_path variable to point to your test dataset.")
        return
    
    print("Getting predictions and labels...")
    predictions, true_labels = get_predictions_and_labels(model, test_data_path)
    
    if not predictions:
        print("No predictions generated. Please check your test dataset path.")
        return
    
    # Get unique class names
    class_names = sorted(list(set(true_labels + predictions)))
    
    print(f"Found {len(predictions)} predictions")
    print(f"Classes: {class_names}")
    
    # Create confusion matrix
    print("Creating confusion matrix...")
    cm = create_confusion_matrix(true_labels, predictions, class_names)
    
    # Print classification report
    print_classification_report(true_labels, predictions, class_names)
    
    # Calculate and print accuracy
    accuracy = np.sum(np.array(true_labels) == np.array(predictions)) / len(true_labels)
    print(f"\nOverall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    print(f"\nConfusion matrix saved as 'confusion_matrix.png'")

if __name__ == "__main__":
    main() 