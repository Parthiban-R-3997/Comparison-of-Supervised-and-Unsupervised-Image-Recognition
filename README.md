Here's the content properly aligned and formatted in Markdown:

# Comparison of Supervised and Unsupervised Image Recognition

## Image Classification and Denoising with CIFAR-10

This document summarizes the processes, results, and analysis of applying different machine learning models on the CIFAR-10 dataset for image classification and image denoising tasks.

## 1. Image Classification

Two different approaches were implemented for image classification:

### 1.1 Support Vector Machine (SVM)

#### Training:
- The CIFAR-10 dataset was loaded using TensorFlow.
- Pixel values were preprocessed by normalizing to a range of 0-1 and flattening the 32x32x3 image arrays into 1D vectors.
- An SVM classifier with an RBF kernel was trained on the training set using all available CPU cores (n_jobs=-1) for faster training. GPU acceleration was attempted but not found to be beneficial for this specific task and model.
- The trained model was saved to svm_model.pkl using pickle.

#### Testing:
- The saved SVM model was loaded.
- Predictions were made on the preprocessed test set.
- Accuracy: 54%
- Classification Report: The classification report revealed that the model performs decently for some classes like 'airplane', 'automobile', and 'ship' but struggles with others like 'cat', 'deer', and 'frog'. This suggests that the features learned by the SVM with an RBF kernel might not be discriminative enough for all classes in the CIFAR-10 dataset.

#### Prediction Examples:
- Predictions were made on a separate set of images stored in the Images/ directory.
- The predicted class labels were displayed alongside the images.
- Predictions were also visualized on a subset of the test set images to further illustrate the model's performance.

### 1.2 Convolutional Neural Network (CNN)

#### Training:
- The CIFAR-10 dataset was loaded and preprocessed similarly to the SVM approach.
- The training set was further split into training and validation sets (80%-20% split).
- A CNN model was defined using the Keras Sequential API. The architecture included:
  - Convolutional layers with ReLU activation
  - Max-pooling layers
  - Dropout for regularization
  - Dense layers with ReLU activation
  - A final dense layer with softmax activation for multi-class classification
- The model was compiled using the Adam optimizer with a learning rate of 0.001 and sparse categorical cross-entropy loss.
- Training was performed for 20 epochs with a batch size of 32.
- Training and validation accuracy and loss were plotted.
- Test Accuracy: 76.28%

#### Model Summary:
- The model.summary() function provided a detailed overview of the CNN architecture, including the output shape and number of parameters for each layer.

#### Saving the Model:
- The trained CNN model was saved to CNNFinal_Model.h5.

#### Evaluation:
- The saved CNN model was loaded.
- Predictions were made on the test set, and a classification report was generated.
- Classification Report: The CNN model significantly outperformed the SVM, achieving an accuracy of 76.28%. This highlights the superiority of CNNs for image classification tasks due to their ability to learn spatial hierarchies of features.

#### Prediction Examples:
- Similar to the SVM, predictions were visualized for images from the Images/ directory and a subset of the test set.

## 2. Image Denoising

An Autoencoder was implemented for image denoising:

### 2.1 Autoencoder

#### Training:
- The CIFAR-10 dataset was loaded, and Gaussian noise was added to the images to create a noisy dataset.
- An autoencoder model was defined using the Keras Sequential API. The architecture consisted of:
  - Encoder: Convolutional and MaxPooling layers to learn a compressed representation of the input.
  - Decoder: Convolutional and UpSampling layers to reconstruct the denoised image from the compressed representation.
- The model was compiled using the Adam optimizer and mean squared error loss.
- Training was performed for a specified number of epochs.
- The trained autoencoder model was saved to Denoising_Model_Final.h5.

#### Testing:
- The saved autoencoder model was loaded.
- The model was evaluated on a noisy test set.
- Test Loss: 22982.15
- Test Accuracy: 33.79%

#### Prediction Examples:
- A visualize_data function was defined to display grids of images.
- The function was used to visualize:
  - Noisy images from the test set.
  - Denoised images predicted by the autoencoder.
  - Original clean images for comparison.
- The autoencoder's denoising capability was further demonstrated on:
  - Images with smaller resolutions.
  - Images with larger resolutions.

## 3. Image Classification using K-means with PCA

#### Training:
- The CIFAR-10 dataset was loaded and preprocessed by normalizing pixel values and flattening the images.
- PCA (Principal Component Analysis) was applied to reduce the dimensionality of the data while retaining 99% of the variance.
- A K-means clustering model was trained on the PCA-transformed training data with 10 clusters, representing the 10 classes in CIFAR-10.
- The trained K-means model was saved to kmeans_model.pkl.

#### Evaluation:
- The saved K-means model was loaded.
- Silhouette Score: 0.0538
- Davies-Bouldin Score: 2.7032
- These scores suggest that the clustering is not very distinct, which is expected given the complexity of image data and the limitations of K-means in high-dimensional spaces.

#### Test Accuracy: 22.11%

#### Prediction Examples:
- Predictions were visualized on a subset of the test set images.
- Predictions were also made and displayed for images from the Images/ directory.

## Conclusion

This project explored different machine learning techniques for image classification and denoising on the CIFAR-10 dataset. The CNN model achieved the highest accuracy (76.28%) for image classification, demonstrating the power of deep learning for this task. The autoencoder showed promising results in denoising images, while the K-means approach, though less accurate, provided insights into clustering image data. Future work could involve exploring more complex CNN architectures, fine-tuning hyperparameters, and experimenting with different autoencoder designs for improved performance.