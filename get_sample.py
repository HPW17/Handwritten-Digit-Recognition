'''
Usage: python get_sample.py [n]
       where n is the number of samples, suggest 20-100 (default 50)

digits = load_digits(): digits is a dictionary
  - digits.data: NumPy array (n_samples, 64), each sample is a 
                 flat representation of 64 values (gray level 0-16)
  - digits.target: NumPy array (n_samples,) of label (0-9)
  - digits.images: NumPy array (n_samples, 8, 8), each sample is 8x8 value
  - digits.DESCR: description of the dataset

'''

from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
import sys

# Quiet mode if only show image, no data
quiet = True

# Load digits dataset
digits = load_digits()

# Show n samples, starting from offset
offset = 0

# Get specified number of samples
n_samples = 50
if len(sys.argv) > 1: 
    try:
        n_samples = int(sys.argv[1])
    except ValueError:
        print(f"Invalid number of samples. Using default {n_samples}.")

# Create a new figure window with width 10, height 2 inches
fig = plt.figure(figsize=(10, 10))

for i in range(n_samples):
    # Create subplots within a single figure (rows, columns, index-start from 1) 
    ax = fig.add_subplot(((n_samples - 1) // 10) + 1, 10, i + 1)
    
    # Display image-like 2D array
    #   - digits.image is a NumPy array of 8x8 array
    #   - color map of grayscale, _r indicates reversed
    #   - interpolation controls how the image is displayed in diff resolution
    #   - nearest: each pixel as a single sharp rectangle without smoothing 
    ax.imshow(digits.images[offset + i], cmap=plt.cm.gray, interpolation='nearest')
    
    # Sets the title of current subplot
    ax.set_title(f"Label: {digits.target[offset + i]}")
    
    # Do not display the axis lines marking the data range and ticks
    ax.axis('off')

plt.tight_layout() 
plt.show()

# Print pixel data after closing the image window
if (not quiet):
    print(f"The first {n_samples} pixel data (8x8 array):\n")
    
    for i in range(n_samples):
        print(f"Sample {offset + i}: Label (target) = {digits.target[offset + i]}")
        print(digits.data[offset + i].reshape(8, 8)) # reshape 64 pixel values to 8x8 array
        print("-" * 20)
    
    print(f"\n\nSamples with no reshape:\n")
    
    for i in range(n_samples):
        # prints the string representation of a 1D NumPy array
        print(f"{digits.target[offset + i]}: {digits.data[offset + i]}")
