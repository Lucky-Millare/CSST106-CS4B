# EXERCISE 3: Advanced Feature Extraction and Image Processing
## IMPORT LIBRARIES
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt
    from google.colab import drive
    
    drive.mount('/content/drive')
# HARRIS CORNER DETECTION
### LOAD IMAGE
    image_path = '/content/drive/MyDrive/image.jpeg'
    image = cv2.imread(image_path)
### CONVERT TO GRAYSCALE
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
### APPLY HARRIS CORNER DETECTION
    corners = cv2.cornerHarris(gray_image, blockSize=2, ksize=3, k=0.04)
    corners_dilated = cv2.dilate(corners, None)
    
    # Thresholding to identify corners
    threshold = 0.01 * corners_dilated.max()
    image[corners_dilated > threshold] = [255, 100, 150]
### DISPLAY THE IMAGE
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Harris Corner Detection')
    plt.axis('off')
    plt.show()
![HARR](https://github.com/user-attachments/assets/2e3febf1-1b67-4fd1-a45c-c0ddba8ee863)

# HOG (Histogram of Oriented Gradients) FEATURE EXTRACTION
### LOAD IMAGE
    image_path = '/content/drive/MyDrive/image.jpeg'
    image = cv2.imread(image_path)
### CONVERT TO GRAYSCALE
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
### APPLY HOG
    # Apply HOG descriptor
    hog_features, hog_image = hog(gray_image,
                                   orientations=9,
                                   pixels_per_cell=(8, 8),
                                   cells_per_block=(2, 2),
                                   visualize=True)
    
    # Rescale the HOG image for better visualization
    hog_image = exposure.rescale_intensity(hog_image, in_range=(0, 10))
### DISPLAY IMAGE
    plt.imshow(hog_image, cmap='gray')
    plt.title('HOG Features')
    plt.axis('off')
    plt.show()
![HOG](https://github.com/user-attachments/assets/68dc52f8-5ec8-4053-a971-0749eceebc6b)

# FAST (Features from Accelerated Segment Test) KEYPOINT DETECTION
### LOAD IMAGE
    image_path = '/content/drive/MyDrive/image.jpeg'
    image = cv2.imread(image_path)
### CONVERT TO GRAYSCALE
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
### APPLY FAST
    fast = cv2.FastFeatureDetector_create()
### KEYPOINTS
    # Detect keypoints
    keypoints = fast.detect(gray_image, None)
    
    # Draw keypoints on the original image
    image_with_keypoints = cv2.drawKeypoints(image, keypoints, None)
### DISPLAY THE IMAGE
    plt.imshow(cv2.cvtColor(image_with_keypoints, cv2.COLOR_BGR2RGB))
    plt.title('FAST Keypoints')
    plt.axis('off')
    plt.show()
![FAST](https://github.com/user-attachments/assets/b7bcd24f-76d7-44d3-884d-3b394e2e057e)

# FEATURE MATCHING USING ORB AND FLANN
### LOAD TWO IMAGE & CONVERT TO GRAYSCALE
    image1_path = '/content/drive/MyDrive/image.jpeg'
    image1 = cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE)
    
    image2_path = '/content/drive/MyDrive/image.jpeg'
    image2 = cv2.imread(image2_path, cv2.IMREAD_GRAYSCALE)
### INITIALIZE ORB DETECTOR
    orb = cv2.ORB_create()
### KEYPOINTS AND DESCRIPTOR
    keypoints1, descriptors1 = orb.detectAndCompute(image1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(image2, None)
### MATCH FEATURE BETWEEN TWO IMAGE USING FLANN MATCHER
    # FLANN parameters for ORB, using LSH index
    FLANN_INDEX_LSH = 6
    index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
    search_params = dict(checks=50)  # Number of times the trees in the index are recursively traversed
    
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    # Ensure descriptors are uint8, which is expected for ORB with LSH
    descriptors1 = descriptors1.astype(np.uint8) # convert to uint8
    descriptors2 = descriptors2.astype(np.uint8) # convert to uint8
    
    
    # Match descriptors using FLANN matcher
    matches = flann.knnMatch(descriptors1, descriptors2, k=2)
    
    # Apply the ratio test to filter good matches (Lowe's ratio test)
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)
    
    # Draw matches
    matched_image = cv2.drawMatches(image1, keypoints1, image2, keypoints2, good_matches, None,
                                    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
### DISPLAY THE IMAGE
    plt.imshow(matched_image)
    plt.title('ORB Features Matching using FLANN')
    plt.axis('off')
    plt.show()
![FALNN2](https://github.com/user-attachments/assets/3d47801d-fd97-40d3-8458-0bbb1d6c4f16)

# IMAGE SEGMENTATION USING WATERSHED ALGORITHM
### LOAD THE IMAGE
    image_path = '/content/drive/MyDrive/image.jpeg'
    image = cv2.imread(image_path)
### CONVERT TO GRAYSCALE
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
### CONVERT THE IMAGE TO BINARY USING THRESHOLDING
    _, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
### FIND SURE BACKGROUND & SURE BACKGROUND USING MORPHOLOGICAL OPERATIONS
    kernel = np.ones((3, 3), np.uint8)
    sure_bg = cv2.dilate(binary_image, kernel, iterations=3)
    dist_transform = cv2.distanceTransform(binary_image, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
### MARKER LABELLING
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
### APPLY WATERSHED ALGORITHM
    markers = cv2.watershed(image, markers)
    
    # Mark boundaries in the original image where markers = -1
    image[markers == -1] = [255, 100, 255]
### DISPLAY THE IMAGE
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Segmented Image with Watershed Algorithm')
    plt.axis('off')
    plt.show()
![WATER](https://github.com/user-attachments/assets/d1351a58-4cb2-48fc-8416-a117e03614ae)

## EXPLANATION
