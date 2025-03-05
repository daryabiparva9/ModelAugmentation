import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import os
from PIL import Image
import glob
from sklearn.model_selection import train_test_split

# Function to load and preprocess images
def load_and_preprocess_images(image_paths, target_size=(299, 299)):
    images = []
    for img_path in image_paths:
        img = Image.open(img_path).convert('RGB')
        img = img.resize(target_size)
        img_array = np.array(img)
        images.append(img_array)
    
    images = np.array(images)
    # Preprocess for Inception model
    preprocessed_images = preprocess_input(images)
    return preprocessed_images

# Function to extract features using Inception model
def extract_features(preprocessed_images):
    # Load Inception model (without top layers)
    print(f'we have come extract features, len:{len(preprocessed_images)}\n \n')
    base_model = InceptionV3(weights='imagenet', include_top=False)
    
    # Get the output of the first max pooling layer
    feature_extractor = tf.keras.Model(
        inputs=base_model.input,
        outputs=base_model.get_layer('max_pooling2d').output
    )
    
    # Extract features
    features = feature_extractor.predict(preprocessed_images)
    
    # Flatten the features for use with logistic regression
    flattened_features = features.reshape(features.shape[0], -1)
    return flattened_features

# Function to train the biased classifier
def train_biased_classifier(wolf_snow_paths, husky_no_snow_paths):
    # Combine image paths and create labels
    print(f'wolf snow path is {wolf_snow_paths} \n \n \n')
    image_paths = wolf_snow_paths + husky_no_snow_paths
    labels = np.array([1] * len(wolf_snow_paths) + [0] * len(husky_no_snow_paths))
    
    # Load and preprocess images
    preprocessed_images = load_and_preprocess_images(image_paths)
    
    # Extract features
    features = extract_features(preprocessed_images)
    
    # Train logistic regression
    model = LogisticRegression(random_state=42)
    model.fit(features, labels)
    
    return model, features

# Function to evaluate the model
def evaluate_model(model, image_paths, true_labels):
    preprocessed_images = load_and_preprocess_images(image_paths)
    features = extract_features(preprocessed_images)
    
    predictions = model.predict(features)
    accuracy = accuracy_score(true_labels, predictions)
    report = classification_report(true_labels, predictions, target_names=['Husky', 'Wolf'])
    
    return predictions, accuracy, report

# Function to visualize predictions with LIME
def visualize_with_lime(model, image_paths, predictions, true_labels):
    try:
        import lime
        from lime import lime_image
        from skimage.segmentation import mark_boundaries
        
        explainer = lime_image.LimeImageExplainer()
        
        for i, img_path in enumerate(image_paths):
            img = Image.open(img_path).convert('RGB')
            img = img.resize((299, 299))
            img_array = np.array(img)
            
            # Define prediction function for LIME
            def predict_fn(images):
                preprocessed = preprocess_input(images)
                features = extract_features(preprocessed)
                return model.predict_proba(features)
            
            explanation = explainer.explain_instance(
                img_array, 
                predict_fn,
                top_labels=2, 
                hide_color=0, 
                num_samples=1000
            )
            
            # Get the explanation for the predicted class
            pred_class = int(predictions[i])
            
            # Plot the explanation
            plt.figure(figsize=(10, 5))
            
            plt.subplot(1, 2, 1)
            plt.imshow(img_array)
            class_name = "Wolf" if pred_class == 1 else "Husky"
            true_class = "Wolf" if true_labels[i] == 1 else "Husky"
            plt.title(f"Prediction: {class_name}, True: {true_class}")
            
            plt.subplot(1, 2, 2)
            temp, mask = explanation.get_image_and_mask(
                pred_class, 
                positive_only=True, 
                num_features=5, 
                hide_rest=False
            )
            plt.imshow(mark_boundaries(temp / 255.0, mask))
            plt.title("LIME explanation")
            
            plt.tight_layout()
            plt.show()
            
    except ImportError:
        print("LIME package not installed. Run 'pip install lime' to use this function.")

# Main function to run the experiment
def run_wolf_husky_experiment(data_dir):
    """
    Expected directory structure:
    data_dir/
        train/
            wolves_snow/     # Wolves in snow
            huskies_no_snow/ # Huskies without snow
        test/
            wolves_snow/     # Wolves in snow
            wolves_no_snow/  # Wolves without snow
            huskies_snow/    # Huskies in snow
            huskies_no_snow/ # Huskies without snow
    """
    # /home/bipar001/anaconda3/python_test/huskyvswolf/ModelAugmentation/src/data_dir/train/wolves_snow/im3.jpg
    # Load training data (biased dataset)
    wolf_snow_path2 = [f for f in os.listdir(f'{data_train}/wolves_snow') if os.path.isfile(os.path.join(f'{data_train}/wolves_snow',f))]
    print(f'\n new wolf path: {wolf_snow_path2} \n \n')
    wolf_snow_paths = glob.glob(os.path.join(data_dir, 'wolves_snow', '*.[jJ][pP]*'))
    print(f'blah{glob.glob(os.path.join(data_dir, "wolves_snow", "*"))} \n \n')
    husky_no_snow_paths = glob.glob(os.path.join(data_dir, 'huskies_no_snow/*'))
    
    print(f"Training with {len(wolf_snow_paths)} wolves (with snow) and {len(husky_no_snow_paths)} huskies (without snow)\n \n")
    
    # Train biased model
    model, train_features = train_biased_classifier(wolf_snow_paths, husky_no_snow_paths)
    
    # Prepare test set
    wolf_snow_test_paths = glob.glob(os.path.join(data_dir, 'test/wolves_snow/*'))
    wolf_no_snow_test_paths = glob.glob(os.path.join(data_dir, 'test/wolves_no_snow/*'))
    husky_snow_test_paths = glob.glob(os.path.join(data_dir, 'test/huskies_snow/*'))
    husky_no_snow_test_paths = glob.glob(os.path.join(data_dir, 'test/huskies_no_snow/*'))
    
    # Create balanced test set as described in the paper
    balanced_test_paths = (wolf_snow_test_paths[:4] + wolf_no_snow_test_paths[:1] + 
                          husky_snow_test_paths[:1] + husky_no_snow_test_paths[:4])
    
    # Create true labels for balanced test set
    # 1 for wolf, 0 for husky
    balanced_test_labels = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    
    # Evaluate on balanced test set
    predictions, accuracy, report = evaluate_model(model, balanced_test_paths, balanced_test_labels)
    
    # Print results
    print(f"Test Accuracy: {accuracy:.2f}")
    print("Classification Report:")
    print(report)
    
    # Print model coefficients
    print("Model Coefficients:")
    print(f"Intercept: {model.intercept_}")
    print(f"Top 5 positive weights: {model.coef_[0].argsort()[-5:]}")
    print(f"Top 5 negative weights: {model.coef_[0].argsort()[:5]}")
    
    # Show model predictions without explanations
    print("\nTest Predictions without Explanations:")
    for i, path in enumerate(balanced_test_paths):
        img_name = os.path.basename(path)
        pred_class = "Wolf" if predictions[i] == 1 else "Husky"
        true_class = "Wolf" if balanced_test_labels[i] == 1 else "Husky"
        print(f"Image: {img_name}, Prediction: {pred_class}, True: {true_class}")
    
    # Visualize with LIME
    print("\nGenerating LIME explanations...")
    visualize_with_lime(model, balanced_test_paths, predictions, balanced_test_labels)
    
    return model, balanced_test_paths, predictions, balanced_test_labels

# Example usage
if __name__ == "__main__":
    # You'll need to create the directory structure and place the images as described above
    data_train = "./data_dir/train"
    data_test = "/data_dir/test"
    model, test_paths, predictions, true_labels = run_wolf_husky_experiment(data_train)