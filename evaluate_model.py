import os
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

def load_model_and_classes(model_path, class_names_path=None):
    """
    Load a trained model and class names
    """
    # Load the model
    model = tf.keras.models.load_model(model_path)
    
    # Load class names if provided, otherwise try to find them in the model directory
    if class_names_path is None:
        model_dir = os.path.dirname(model_path)
        class_names_path = os.path.join(model_dir, "class_names.json")
    
    if os.path.exists(class_names_path):
        with open(class_names_path, "r") as f:
            class_names = json.load(f)
    else:
        print("Warning: Class names file not found. Using numbered classes instead.")
        # Assume the number of classes from the model's output layer
        num_classes = model.output_shape[-1]
        class_names = [f"Class_{i}" for i in range(num_classes)]
    
    return model, class_names

def evaluate_model(model, test_data_dir, class_names, batch_size=32, img_size=224):
    """
    Evaluate the model on test data and generate evaluation metrics
    """
    # Create a data generator for the test data
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    test_generator = test_datagen.flow_from_directory(
        test_data_dir,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )
    
    # Get the mapping between generator indices and class names
    class_indices = test_generator.class_indices
    idx_to_class = {v: k for k, v in class_indices.items()}
    
    # Evaluate the model
    print("Evaluating model on test data...")
    test_loss, test_acc = model.evaluate(test_generator)
    print(f"Test accuracy: {test_acc:.4f}")
    print(f"Test loss: {test_loss:.4f}")
    
    # Get predictions
    print("Generating predictions for detailed metrics...")
    predictions = model.predict(test_generator)
    y_pred = np.argmax(predictions, axis=1)
    
    # Get true labels
    y_true = test_generator.classes
    
    # Generate classification report
    print("\nClassification Report:")
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    print(report)
    
    # Generate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    
    # Save the confusion matrix
    output_dir = os.path.dirname(os.path.abspath(model_path))
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), bbox_inches='tight')
    plt.close()
    
    # Plot some example predictions
    plot_example_predictions(model, test_generator, class_names, output_dir)
    
    return test_acc, test_loss, report, cm

def plot_example_predictions(model, test_generator, class_names, output_dir, num_examples=5):
    """
    Plot some example predictions from the test set
    """
    # Reset the generator
    test_generator.reset()
    
    # Get a batch of test images
    batch_x, batch_y = next(test_generator)
    
    # Make predictions
    predictions = model.predict(batch_x)
    
    # Plot the examples
    fig, axes = plt.subplots(num_examples, 2, figsize=(12, 4 * num_examples))
    
    for i in range(min(num_examples, len(batch_x))):
        # Get the true and predicted classes
        true_class_idx = np.argmax(batch_y[i])
        pred_class_idx = np.argmax(predictions[i])
        
        true_class = class_names[true_class_idx]
        pred_class = class_names[pred_class_idx]
        confidence = predictions[i][pred_class_idx] * 100
        
        # Display the image
        axes[i, 0].imshow(batch_x[i])
        axes[i, 0].set_title(f"True: {true_class}")
        axes[i, 0].axis('off')
        
        # Display the prediction confidence
        bars = axes[i, 1].barh(
            range(len(class_names)),
            predictions[i] * 100,
            color=['green' if j == pred_class_idx else 'blue' for j in range(len(class_names))]
        )
        axes[i, 1].set_yticks(range(len(class_names)))
        axes[i, 1].set_yticklabels(class_names)
        axes[i, 1].set_xlabel('Confidence (%)')
        axes[i, 1].set_title(f"Prediction: {pred_class} ({confidence:.2f}%)")
        
        # Highlight the correct class
        if true_class_idx != pred_class_idx:
            axes[i, 1].get_yticklabels()[true_class_idx].set_color('red')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'example_predictions.png'), bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Evaluate a trained tree species classification model')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to the trained model directory or file')
    parser.add_argument('--test-data', type=str, required=True,
                        help='Path to the test data directory')
    parser.add_argument('--class-names', type=str, default=None,
                        help='Path to the class names JSON file (optional)')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for evaluation')
    parser.add_argument('--img-size', type=int, default=224,
                        help='Image size for evaluation')
    
    args = parser.parse_args()
    
    # Check if the model path exists
    if not os.path.exists(args.model):
        print(f"Error: Model path '{args.model}' does not exist.")
        return
    
    # Check if the test data directory exists
    if not os.path.exists(args.test_data):
        print(f"Error: Test data directory '{args.test_data}' does not exist.")
        return
    
    # Load the model and class names
    model, class_names = load_model_and_classes(args.model, args.class_names)
    
    # Print model summary
    print("\nModel Summary:")
    model.summary()
    
    # Print class names
    print("\nClass Names:")
    for i, name in enumerate(class_names):
        print(f"  {i}: {name}")
    
    # Evaluate the model
    print("\nEvaluating model...")
    test_acc, test_loss, report, cm = evaluate_model(
        model, args.test_data, class_names, args.batch_size, args.img_size
    )
    
    print("\nEvaluation complete!")
    print(f"Results saved to {os.path.dirname(os.path.abspath(args.model))}")

if __name__ == "__main__":
    main()