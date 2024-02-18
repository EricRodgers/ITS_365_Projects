import cv2
import os
import numpy as np
from skimage.feature import local_binary_pattern

# Function to extract color histograms from an image
def extract_color_histogram(image):
    hist = cv2.calcHist([image], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256])
    hist = hist.flatten()
    return hist

# Function to extract LBP texture features from an image
def extract_lbp(image, num_points=8, radius=1, method='default'):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, num_points, radius, method)
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, num_points + 3), range=(0, num_points + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)
    return hist

# Function to extract shape descriptors from an image
def extract_shape_descriptors(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_areas = [cv2.contourArea(contour) for contour in contours]
    return contour_areas

# Function to extract color moments from an image
def extract_color_moments(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    moments = cv2.moments(gray)
    mean = [moments['m00'], moments['m10'], moments['m01']]
    std_dev = [np.sqrt(moments['mu20']), np.sqrt(moments['mu02']), np.sqrt(moments['mu11'])]
    skewness = [
        (moments['mu30'] / np.power(moments['mu20'], 1.5)) if moments['mu20'] > 0 else 0,
        (moments['mu03'] / np.power(moments['mu02'], 1.5)) if moments['mu02'] > 0 else 0
    ]
    return mean + std_dev + skewness

# Function to extract appearance-based features from an image
def extract_appearance_features(image):
    # Example: Detect key points and compute descriptors using ORB
    orb = cv2.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(image, None)
    return descriptors.flatten() if descriptors is not None else []

# Function to extract features from all images in a directory
def extract_features_from_directory(directory):
    features = []
    labels = []

    for filename in os.listdir(directory):
        if filename.endswith(".png"):  # Adjust file extension if necessary
            image_path = os.path.join(directory, filename)
            image = cv2.imread(image_path)
            
            # Extract features from the image
            color_hist = extract_color_histogram(image)
            lbp_hist = extract_lbp(image)
            shape_desc = extract_shape_descriptors(image)
            color_moments = extract_color_moments(image)
            appearance_features = extract_appearance_features(image)

            
            # Concatenate the features into a single feature vector
            combined_features = np.concatenate((color_hist, lbp_hist, shape_desc, color_moments, appearance_features))
            
            # Add features and label to the lists
            features.append(combined_features)
            labels.append(filename.split('_')[0])  # Extract label from filename
    
    return features, labels
    
images_directory = '/Users/Eric/Desktop/Purdue/Junior Year/Spring Semester/ITS365/KNN_Images'
features, labels = extract_features_from_directory(images_directory)

print("Number of images:", len(features))
print("Number of features per image:", len(features[0]))
print("Features for the first image:", features[0])
print("Label for the first image:", labels[0])