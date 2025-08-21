'''
Usage: python verify_model.py

Verify k-NN classifier and data set. 
Using the first 70% as training data and the remaining 30% as testing data.
Find the proper k value by:
  - dist = euclidean, k = 1-49
  - dist = manhattan, k = 1-49

'''

from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

def main():
    # Load digits data set from Scikit-learn
    digits = load_digits()
    x = digits.data
    y = digits.target
    
    # Divide the data set into 30% as testing set, 70% as training set
    x_train, x_test, y_train, y_test = \
        train_test_split(x, y, test_size=0.3, random_state=42)
    print(f"Total = {len(x)}, training = {len(x_train)}, test = {len(x_test)}\n")

    # Train k-NN classifier
    for dist in ["euclidean", "manhattan"]:
        highest_accuracy = 0
        highest_k = 0
        for k in range(1, 50):
            knn = KNeighborsClassifier(n_neighbors = k, metric=dist)
            knn.fit(x_train, y_train)
            
            # Verify model with testing data set
            y_predict = knn.predict(x_test)
            accuracy = accuracy_score(y_test, y_predict)
            if (accuracy > highest_accuracy):
                highest_accuracy = accuracy
                highest_k = k 
            print(f"k = {k:2}, dist = {dist}, ", end = "")
            print(f"model accuracy = {accuracy:.4f}")
        
        print(f"\nHighest accuracy = {highest_accuracy:.4f} at k = {highest_k}\n")
    
if __name__ == "__main__":
    main() 

