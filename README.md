# Investigating the Efficacy of <br>K-Nearest Neighbors for <br>Handwritten Digit Recognition

## Introduction

This project aims to explore the feasibility and effectiveness of the **K-Nearest Neighbors (k-NN) algorithm** for the task of **handwritten digit recognition**. The program is implemented using Python and leverages the publicly available scikit-learn handwritten digit dataset. It encompasses sample data loading, preprocessing, evaluation, and application to real-world handwritten samples collected from volunteers. 

## Methodology

- The primary methodology in this project involved leveraging the scikit-learn (sklearn) library in Python, which provides a readily accessible handwritten digit dataset and the implementation of the k-NN classifier,  "KNeighborsClassifier". 
- This dataset comprises 1,797 samples, each representing an 8x8 grayscale image of a digit (0-9) with 64 values (range 0-16), along with corresponding labels. 
- To evaluate the intrinsic effectiveness of the sklearn digit dataset and the k-NN classifier, the dataset was split into distinct training and testing sets using the "train_test_split" function with a ratio of 70% for training and 30% for testing. This resulted in 1,257 training samples and 540 testing samples. 
- The program **bbb** was then implemented to train and evaluate the k-NN classifier across different distance metrics (Euclidean and Manhattan) and a range of k values (from 1 to 49).
- Three volunteers were recruited for the purpose of collecting real-world handwritten digit samples, and each was asked to write digits from 0 to 9, resulting in a total of 30 handwritten digit images.
- Then the program **ccc** was implemented to perform a series of image processing steps: resizing the input image to 8x8 pixels, converting it to grayscale to reduce color information, and optionally applying a binarization threshold to create a binary representation (black and white) of the digit based on option switch. 
- With the real-world handwritten digit images processed into a compatible 8x8 pixel format, the next step was to classify these 30 images using the k-NN classifier trained on the sklearn dataset. Initially, the classifier was used with default parameter settings: "k" set to the square root of the number of training samples (k = √1797 ≈ 42) and distance metric set to Euclidean. 
- Further investigation was conducted for the robustness of the classification and the impact of different parameters. The 30 real-world images from volunteers were classified while varying the image processing method (grayscale vs. binarized), the distance metric (Euclidean vs. Manhattan), and the “k” value (from 1 to 49). 

## Results

- It demonstrated a high level of accuracy, generally ranging between 0.98 and 0.99. Notably, the highest accuracy achieved on the test set was **0.9944**, obtained when the distance metric was set to “Euclidean” and the number of neighbors “k” was 6. 
- The classification of real-world handwritten digits achieved 100% accuracy rate for all 30 volunteer-provided images with the pre-trained classifier and the initial settings (k=42).
- Analysis of various parameter settings showed a high accuracy range of 90% to 100%. Notably, the highest accuracy of 100% was consistently achieved when the input images were processed to grayscale, the Euclidean distance metric was used, and the 'k' value was in the range of 33 to 49 (around the heuristic value of 42).

## Programs

- **get_sample.py**: Python program used to fetch a specified number of samples from sklearn digits dataset and provide a human-friendly visual representation of the variety and quality of the handwritten digits within the dataset.
- **verify_model.py**: Python program used to evaluate the intrinsic effectiveness of the sklearn digit dataset and the k-NN classifier by splitting the dataset into distinct training and testing sets with a ratio of 70% for training and 30% for testing, and evaluate the k-NN classifier across different distance metrics (Euclidean and Manhattan) and a range of k values.
- **knn.py**: Python program used to pipeline the process of loading handwritten images, transform the images into 8x8 pixel values, and classify them using k-NN classifier. 
