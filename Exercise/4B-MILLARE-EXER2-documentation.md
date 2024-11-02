## Dependencies for Exer 2
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
## IMPORT LIBRARIES
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt
    from google.colab.patches import cv2_imshow
    from google.colab import drive
    
    drive.mount('/content/drive')
# SIFT FEATURE EXTRACTION
### LOAD IMAGE & CONVERT TO GRAYSCALE
    image_path = '/content/drive/MyDrive/image.jpeg'
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
### INITIALIZE SIFT DETECTOR
    sift = cv2.SIFT_create()
### DETECT AND DRAW KEYPOINTS FOR SIFT
    keypoints, descriptors = sift.detectAndCompute(gray_image, None)

    image_with_keypoints = cv2.drawKeypoints(image, keypoints, None)
### DISPLAY THE IMAGE
    plt.imshow(cv2.cvtColor(image_with_keypoints, cv2.COLOR_BGR2RGB))
    plt.title('SIFT Keypoints')
    plt.show()
![SIFT](https://github.com/user-attachments/assets/8e13492c-c7be-4625-ae11-5937c90e8e72)

# SURF FEATURE EXTRACTION
### LOAD THE IMAGE & CONVERT TO GRAYSCALE
    image_path = '/content/drive/MyDrive/image.jpeg'
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
### INITIALIZE SURF DETECTOR
    surf = cv2.xfeatures2d.SURF_create()
### DETECT AND DRAW KEYPOINTS FOR SURF
    keypoints, descriptors = surf.detectAndCompute(gray_image, None)
    
    image_with_keypoints = cv2.drawKeypoints(image, keypoints, None)
### DISPLAY THE IMAGE
    plt.imshow(cv2.cvtColor(image_with_keypoints, cv2.COLOR_BGR2RGB))
    plt.title('SURF Keypoints')
    plt.show()
![SURF](https://github.com/user-attachments/assets/8aa9f9b7-a752-4476-b377-320e100452a1)

# ORB FEATURE EXTRACTION
### LOAD THE IMAGE & CONVERT TO GRAYSCALE 
    image_path = '/content/drive/MyDrive/image.jpeg'
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
### INITIALIZE ORB DETECTOR
    orb = cv2.ORB_create()
### DETECT AND DRAW KEYPOINTS FOR ORB
    keypoints, descriptors = orb.detectAndCompute(gray_image, None)
    
    image_with_keypoints = cv2.drawKeypoints(image, keypoints, None)
### DISPLAY THE IMAGE
    plt.imshow(cv2.cvtColor(image_with_keypoints, cv2.COLOR_BGR2RGB))
    plt.title('ORB Keypoints')
    plt.show()
![ORB](https://github.com/user-attachments/assets/815b1d88-a932-4ba6-93c4-c5198a1646a9)

# FEATURE MATCHING USING SIFT
### LOAD TWO IMAGE
    image1_path = '/content/drive/MyDrive/image.jpeg'
    image1 = cv2.imread(image1_path)
    image2_path = '/content/drive/MyDrive/image.jpeg'
    image2 = cv2.imread(image2_path)
### INITIALIZE SIFT DETECTOR KEYPOINTS AND DESCRIPTOR
    keypoints1, descriptors1 = sift.detectAndCompute(image1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(image2, None)
### INITIALIZE THE MATCHER
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck= True)
### MATCH DESCRIPTOR
    matches = bf.match(descriptors1, descriptors2)
### SORT MATCHES BY DISTANCE
    matches = sorted(matches, key=lambda x: x.distance)
### DRAW MATCHES
    image_matches = cv2.drawMatches(image1, keypoints1, image2, keypoints2, matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
### DISPLAY THE IMAGE
    plt.imshow(image_matches)
    plt.title('Feature Matching with SIFT')
    plt.show
![MATCHSIFT](https://github.com/user-attachments/assets/e69bd36d-735e-45bf-85f2-1f0aeb01d1c5)

# Real-World Applications (Image Stitching using Homography)
### LOAD TWO IMAGE
    image1_path = '/content/drive/MyDrive/image.jpeg'
    image1 = cv2.imread(image1_path)
    image2_path = '/content/drive/MyDrive/second.jpg'
    image2 = cv2.imread(image2_path)
### CONVERT TO GRAYSCALE
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
### INITIALIZE ORB DETECTOR, KEYPOINTS AND DESCRIPTORS
    orb = cv2.ORB_create()
    
    keypoints1, descriptors1 = orb.detectAndCompute(gray1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(gray2, None)
### MATCHING FEATURE USING BFmatcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
    matches = bf.match(descriptors1, descriptors2)
    
    matches = sorted(matches, key=lambda x: x.distance)
### APPLY RATIO TEST
    # Draw the matches
    matched_image = cv2.drawMatches(image1, keypoints1, image2, keypoints2, matches[:50], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    
    # Find the homography matrix
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)

    # Get the dimensions of the first image
    h, w = image1.shape[:2]
    
    # Warp image2 to align with image1
    aligned_image = cv2.warpPerspective(image2, H, (w, h))
### DISPLAY THE OUTPUT
    # Show the matched image
    cv2_imshow(matched_image)
    
    # Show the aligned image
    cv2_imshow(aligned_image)
![REAL](https://github.com/user-attachments/assets/55076087-b56d-44f4-a5fd-269655887c90)

# COMBINING SIFT AND ORB
### LOAD TWO IMAGE
    image1_path = '/content/drive/MyDrive/image.jpeg'
    image1 = cv2.imread(image1_path)
    image2_path = '/content/drive/MyDrive/image.jpeg'
    image2 = cv2.imread(image2_path)
### SIFT & ORB DETECTOR, KEYPOINTS AND DESCRIPTOR
    sift = cv2.SIFT_create()
    orb = cv2.ORB_create()
    
    keypoints1_sift, descriptors1_sift = orb.detectAndCompute(image1, None)
    keypoints2_sift, descriptors2_sift = orb.detectAndCompute(image2, None)
    
    keypoints1_orb, descriptors1_orb = orb.detectAndCompute(image1, None)
    keypoints2_orb, descriptors2_orb = orb.detectAndCompute(image2, None)
### MATCH SIFT & ORB DESCRIPTOR
    bf_sift = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches_sift = bf_sift.match(descriptors1_sift, descriptors2_sift)
    matches_sift = sorted(matches_sift, key=lambda x: x.distance)
    
    bf_orb = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches_orb = bf_orb.match(descriptors1_orb, descriptors2_orb)
    matches_orb = sorted(matches_orb, key=lambda x: x.distance)
    
    # Combine matches
    combined_matches = matches_sift + matches_orb
    combined_matches = sorted(combined_matches, key=lambda x: x.distance)
    
    # Draw combined matches
    matched_image = cv2.drawMatches(
        image1, keypoints1_sift + keypoints1_orb,
        image2, keypoints2_sift + keypoints2_orb,
        combined_matches[:50], None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
### DISPLAY COMBINE RESULT
    plt.imshow(matched_image)
    plt.title('Combining SIFT and ORB')
    plt.axis('off')
    plt.show()
![COMBINE](https://github.com/user-attachments/assets/788786d6-ab26-460e-aee6-b9ae711261ec)

# EXPLANATION
  In this exercise, I explored various feature extraction techniques and feature matching methods using OpenCV on Google Colab, including SIFT, SURF, and ORB detectors. building OpenCV with non-free modules was challenging, particularly for enabling the SURF detector in Colab. However, with guidance and additional dependencies, I was able to configure it successfully. Each feature detection and matching method offered unique advantages, making them suitable for different applications depending on speed and accuracy requirements.
