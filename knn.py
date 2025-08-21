'''
Usage: python knn.py [dir [threshold [k [dist]]]]

  knn dir                # must specify sub-directory
  knn dir 100            # 100: binary threshold, 0: gray scale (default)
  knn dir 0 42           # 42: number of neighbors k (default=sqrt(1797)=42)
  knn dir 0 0 manhattan  # manhattan: calc distance (default=euclidean)

Pipelined system to:
  1. Train model with all 1,797 samples (with specified k and dist metrics)
  2. Process the image specified by the command-line argument
     - Transfer image to pixel data (gray scale 8x8)
     - Binarized if threshold is specified
  3. Transfer pixel data back to a JPG image (human-friendly purpose)
  4. Predict target with trained k-NN model
  5. Calculate hit count
  
'''

from sklearn.datasets import load_digits
from sklearn.neighbors import KNeighborsClassifier
from PIL import Image
from time import time
import numpy as np
import sys

def image_to_array(image_path, threshold):
    '''
    Transfer a JPG image of handwriting digit to a NumPy array.
    :param image_path: the path to the image file
    :param threshold: reset to white (255) if > threshold, black (0) otherwise
    :return numpy.ndarray (64,) if successful, None otherwise
    '''
    try:
        # Open image
        img = Image.open(image_path)

        # Transfer the color image to gray scale
        img_gray = img.convert("L")

        # Resize image to 8x8 and calculate pixel value with LANCZOS filter
        img_resized = img_gray.resize((8, 8), Image.Resampling.LANCZOS)

        # Get pixel data
        pixel_data = list(img_resized.getdata())

        # Transfer pixel data to flat NumPy array with shape (64,): 1D array
        array_64 = np.array(pixel_data)
        
        if (threshold == 0): # Keep gray scale
            return array_64
        else: # Reset to binary black-and-white
            array_binary = np.where(array_64 > threshold, 255, 0).astype(int)
            return array_binary

    except FileNotFoundError:
        print(f"Cannot open file: {image_path}")
        return None
    
    except Exception as e:
        print(f"Failed when processing image: {e}")
        return None

def array_to_image(data, image_path):
    '''
    Transfer a NumPy array to a JPG image.
    :param data: 8x8 NumPy array
    :param image_path: the path to the image file to be generated
    '''
    # Normalize values to 0-255 and transfer to unsigned integer
    image_data = ((data / 16.0) * 255).astype(np.uint8)
    
    # Transfer NumPy array to image
    img = Image.fromarray(image_data)
    img.save(image_path)

def main(quiet):
    '''
    Process the image specified by the command-line argument.
    :param quiet: specify as True for quiet mode (only prints the results)
    '''    
    # Get specified input image file name
    if len(sys.argv) == 1:
        print(f"No specified image file, please try again.")
        return
    
    image_path = sys.argv[1] # input directory name of image files 
    
    # Get specified threshold value for binary representation
    threshold = 0
    if len(sys.argv) > 2: 
        try:
            threshold = int(sys.argv[2])
        except ValueError:
            print(f"Invalid threshold value. Using default {threshold}.")
    
    # Get specified k = number of neighbors
    k = 0
    if len(sys.argv) > 3: 
        try:
            k = int(sys.argv[3])
        except ValueError:
            print(f"Invalid k value. Using default 42.")
    
    # Get specified distance calculation method
    dist = "euclidean"
    if len(sys.argv) > 4: 
        dist = sys.argv[4]
    
    start_time = time()
    
    # Load digits data set from Scikit-learn
    digits = load_digits()
    x = digits.data
    y = digits.target
    
    # Train k-NN classifier
    if (k == 0):
        k = int(np.sqrt(len(x)))
    knn = KNeighborsClassifier(n_neighbors = k, metric=dist)
    knn.fit(x, y)
    
    train_time = time()
    print(f"bin = {threshold}, k = {k}/{len(x)}, dist = {dist}")
    print(f"Training time: {round(train_time - start_time, 2)} seconds\n")
    
    # Iteratively process image files in directory 'image_path'
    hit = 0
    miss = []
    for i in range(10):
        image_in_file = f"{image_path}\\{i}.jpg"
        image_out_file = f"{image_path}\\{i}_out.jpg"
        
        # Transfer image to pixel data
        digit_flat = image_to_array(image_in_file, threshold)
        
        if digit_flat is None:
            print("Transferring image to array error.")
            return
        
        # Transfer pixel data (64,) to 8x8 NumPy array
        digit_8x8 = digit_flat.reshape(8, 8) 
        if (not quiet):
            print()
            print("#" * 20)
            print(f"Processing file {i}.jpg:\n")
            print("Translated 8x8 array:")
            print(digit_8x8)
            print()
    
        # Normalize values to 0-16 as in the sklearn.load_digits data set
        normal_flat = (digit_flat / 255.0 * 16.0).round().astype(int)
        normal_8x8 = (digit_8x8 / 255.0 * 16.0).round().astype(int)
        if (not quiet):
            print("Normalized 8x8 array:")
            print(normal_8x8)
            print()
        
        # Transfer pixel data back to a JPG image
        array_to_image(digit_8x8, image_out_file)
        if (not quiet):
            print(f"File {i}_out.jpg is generated successfully.\n")
        
        # Reshape data to (1, num_of_features): -1 to auto calculate size
        normal_flat = normal_flat.reshape(1, -1)
        
        # Predict target with trained k-NN model
        predicted_label = knn.predict(normal_flat)[0]
        print(f"Image '{i}.jpg' is classified as: {predicted_label}")
        
        # Hit count
        if (predicted_label == i):
            hit = hit + 1
        else:
            miss = miss + [i]
    
    elapsed_time = time() - start_time
    print(f"\nHit rate = {(hit * 100.0) / (i + 1)}%, miss = {miss}")
    print(f"Total elapsed time: {round(elapsed_time,2)} seconds")
    
if __name__ == "__main__":
    main(True) # True if quiet mode, False if need to print pixel data
