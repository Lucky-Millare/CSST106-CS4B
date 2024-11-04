## SURF DEPENDENCIES
    # Uninstall existing OpenCV packages
    !pip uninstall -y opencv-python opencv-contrib-python opencv-python-headless
    
    # Install system dependencies
    !apt-get update
    !apt-get install -y build-essential cmake git pkg-config libjpeg-dev libtiff5-dev \
        libjasper-dev libpng-dev libavcodec-dev libavformat-dev libswscale-dev \
        libv4l-dev libxvidcore-dev libx264-dev libgtk-3-dev libatlas-base-dev gfortran \
        python3-dev
    
    # Clone OpenCV and OpenCV contrib
    !git clone https://github.com/opencv/opencv.git
    !git clone https://github.com/opencv/opencv_contrib.git
    
    # Create a build directory
    !mkdir -p opencv/build
    %cd opencv/build
    
    # Configure the build with non-free modules
    !cmake -DOPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules -DOPENCV_ENABLE_NONFREE=ON -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local ..
    
    # Build OpenCV (this may take some time)
    !make -j4
    
    # Install the built OpenCV
    !make install
### IMPORT LIBRARIES
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt
    from skimage.feature import hog
    from skimage import exposure
    from google.colab import drive
    
    drive.mount('/content/drive')
# TASK1: HARRIS CORNER DETECTION
### LOAD THE IMAGE IN GRAYSCALE, CONVERT TO FLOAT
    image_path = '/content/drive/MyDrive/image.jpeg'
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    gray = np.float32(image)
### APPLY HARRIS CORNER DETECTION
    dst = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)
### THRESHOLD FOR AN OPTIMAL VALUE & MARK CORNERS IN RED
    image[dst > 0.01 * dst.max()] = 255
    image_with_corners = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    image_with_corners[dst > 0.01 * dst.max()] = [0, 0, 255]
### DISPLAY THE IMAGE
    plt.figure(figsize=(10,5))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(image_with_corners, cv2.COLOR_BGR2RGB))
    plt.title('Corners Detected')
    plt.axis('off')
    plt.show()
![RED](https://github.com/user-attachments/assets/424c15b3-982b-46b4-b26c-3d250f6ad32f)

# TASK2: HOG FEATURE EXTRACTION
### LOAD THE IMAGE, CONVERT TO GRAYSCALE
    image_path = '/content/drive/MyDrive/image.jpeg'
    image = cv2.imread(image_path)
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
### COMPUTE HOG FEATURES AND VISUALIZATION
    fd, hog_image = hog(gray, orientations=9, pixels_per_cell=(8, 8),
                        cells_per_block=(2, 2), visualize=True,
                        channel_axis=None)
    
    # Rescale histogram for better display
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
### DISPLAY THE IMAGE
    plt.figure(figsize=(10,5))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(hog_image_rescaled, cmap='gray')
    plt.title('HOG Features')
    plt.axis('off')
    
    plt.show()
![HOGM](https://github.com/user-attachments/assets/db95bfab-7291-4951-9108-7d8d1b668cde)

# TASK3: ORB FEATURE EXTRACTION MATCHING
### LOAD THE IMAGE, CONVERT TO GRAYSCALE
    image1_path = '/content/drive/MyDrive/image.jpeg'
    image2_path = '/content/drive/MyDrive/image.jpeg'
    
    image1 = cv2.imread(image1_path)
    image2 = cv2.imread(image2_path)
    
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
### INITIALIZE ORB DETECTOR
    orb = cv2.ORB_create()
### KEYPOINTS AND DESCRIPTOR FOR ORB
    keypoints_orb1, descriptors_orb1 = orb.detectAndCompute(gray1, None)
    keypoints_orb2, descriptors_orb2 = orb.detectAndCompute(gray2, None)
### FLANN PARAMETERS
    FLANN_INDEX_LSH = 6
    index_params = dict(algorithm=FLANN_INDEX_LSH,
                          table_number=6,  # 12
                          key_size=12,     # 20
                          multi_probe_level=1)  # 2
    search_params = dict(checks=50)
### FLANN-BASED MATCH OBJECT
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    # Match descriptors
    matches = flann.knnMatch(descriptors_orb1, descriptors_orb2, k=2)
    
    # Store all the good matches as per Lowe's ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)
    
    # Draw matches
    img_matches = cv2.drawMatches(image1, keypoints_orb1, image2, keypoints_orb2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
### DISPLAY THE IMAGE
    plt.imshow(img_matches)
    plt.axis('off')
    plt.title('ORB Feature Matching with FLANN-based Matcher')
    plt.show()
![ORBB](https://github.com/user-attachments/assets/8566368a-a7e2-4b53-8f5c-8d4d256b9f9b)

# TASK4: SIFT AND SURF FEATURE EXTRACTION
### LOAD THE IMAGE, CONVERT TO GRAYSCALE
    image1_path = '/content/drive/MyDrive/image.jpeg'
    image2_path = '/content/drive/MyDrive/image.jpeg'
    
    image1 = cv2.imread(image1_path)
    image2 = cv2.imread(image2_path)
    
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
### INITIALIZE SIFT & SURF DETECTOR
    sift = cv2.SIFT_create()
    surf = cv2.xfeatures2d.SURF_create()
### KEYPOINTS AND DESCRIPTOR FOR SIFT & SURF
    keypoints_sift1, descriptors_sift1 = sift.detectAndCompute(gray1, None)
    keypoints_sift2, descriptors_sift2 = sift.detectAndCompute(gray2, None)
    
    keypoints_surf1, descriptors_surf1 = surf.detectAndCompute(gray1, None)
    keypoints_surf2, descriptors_surf2 = surf.detectAndCompute(gray2, None)
### DRAAW KEYPOINTS ON THE IMAGE
    image1_sift = cv2.drawKeypoints(image1, keypoints_sift1, None)
    image2_sift = cv2.drawKeypoints(image2, keypoints_sift2, None)
    
    image1_surf = cv2.drawKeypoints(image1, keypoints_surf1, None)
    image2_surf = cv2.drawKeypoints(image2, keypoints_surf2, None)
### DISPLAY THE IMAGE
    plt.subplot(2, 2, 1)
    plt.imshow(image1_sift)
    plt.title('SIFT Keypoints Image 1')
    plt.axis('off')
    
    plt.subplot(2, 2, 2)
    plt.imshow(image2_sift)
    plt.title('SIFT Keypoints Image 2')
    plt.axis('off')
    
    plt.subplot(2, 2, 3)
    plt.imshow(image1_surf)
    plt.title('SURF Keypoints Image 1')
    plt.axis('off')
    
    plt.subplot(2, 2, 4)
    plt.imshow(image2_surf)
    plt.title('SURF Keypoints Image 2')
    plt.axis('off')
    
    plt.show()
![SIFTS](https://github.com/user-attachments/assets/83ab03ae-6545-4d7a-ba55-721557845a18)

# TASK5: FEATURE MATCHING USING BRUTE-FORCE MATCHER
### LOAD THE IMAGE, CONVERT TO GRAYSCALE
    image1_path = '/content/drive/MyDrive/image.jpeg'
    image2_path = '/content/drive/MyDrive/image.jpeg'
    
    image1 = cv2.imread(image1_path)
    image2 = cv2.imread(image2_path)
    
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
### INITIALIZE ORB DETECTOR
    orb = cv2.ORB_create()
### KEYPOINTS AND DESCRIPTOR FOR ORB
    keypoints_orb1, descriptors_orb1 = orb.detectAndCompute(gray1, None)
    keypoints_orb2, descriptors_orb2 = orb.detectAndCompute(gray2, None)
### CREATE BFmatcher OBJECT
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
    # Match descriptors
    matches = bf.match(descriptors_orb1, descriptors_orb2)
    
    # Sort them in the order of their distance
    matches = sorted(matches, key=lambda x: x.distance)
    
    # Draw first 10 matches
    img_matches = cv2.drawMatches(image1, keypoints_orb1, image2, keypoints_orb2, matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
### DISPLAY THE IMAGE
    plt.imshow(img_matches)
    plt.title('Brute-Force Feature Matching')
    plt.axis('off')
    plt.show()
![BFFM](https://github.com/user-attachments/assets/2850f0a7-a59c-41b1-bd23-3ae7f837f9e6)

# TASK6: IMAGE SEGMENTATION USING WATERSHED ALGORITHM
### LOAD THE IMAGE, CONVERT TO GRAYSCALE
    image_path = '/content/drive/MyDrive/image.jpeg'
    image = cv2.imread(image_path)
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
### APPLY THRESHOLD TO GET BINARY IMAGE
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
### NOISE REMOVAL USING MORPHOLOGICAL OPERATION
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
### SURE BACKGROUND AREA, FINDING SURE FOREGROUND AREA, AND FINDING UNKNOWN REGION
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
    
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
### MARKER LABELING
    ret, markers = cv2.connectedComponents(sure_fg)
    
    # Add one to all labels so that sure background is not 0, but 1
    markers = markers + 1
    
    # Now, mark the region of unknown with zero
    markers[unknown == 255] = 0
### APPLY WATERSHED ALGORITHM
    markers = cv2.watershed(image, markers)
    image[markers == -1] = [255, 0, 0]
### DISPLAY THE IMAGE
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Segmented Image')
    plt.axis('off')
    plt.show()
![WATERS](https://github.com/user-attachments/assets/8ca2851c-e515-43a9-a040-1c7d0ea9e8bb)

## EXPLANATION
In this code, I first set up the environment by uninstalling existing OpenCV packages and installing necessary system dependencies before cloning the OpenCV and OpenCV contrib repositories. I then perform several image processing tasks using OpenCV, such as Harris corner detection, HOG feature extraction, and ORB feature matching, to analyze and visualize features in images. Finally, I implement image segmentation using the watershed algorithm, allowing for the identification of distinct regions in an image based on their pixel characteristics.
