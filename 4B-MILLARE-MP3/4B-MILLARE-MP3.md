### SURF DETECTOR DEPENDENCIES
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
# Machine Problem No. 3: Feature Extraction and Object Detection

### IMPORT LIBRARIES
    import cv2
    import numpy as np
    from matplotlib import pyplot as plt
    from google.colab import drive
    
    drive.mount('/content/drive')
### STEP1: LOAD IMAGE
    image1_path = '/content/drive/MyDrive/image.jpeg'
    image2_path = '/content/drive/MyDrive/image.jpeg'
    
    image1 = cv2.imread(image1_path)
    image2 = cv2.imread(image2_path)
### CONVERT TO GRAYSCLALE
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
### STEP2: INITIALIZE SIFT, SURF, & ORB DETECTOR, KEYPOINTS AND DESCRIPTOR
    # Initialize SIFT
    sift = cv2.SIFT_create()
    
    # Detect keypoints and descriptors using SIFT
    keypoints_sift1, descriptors_sift1 = sift.detectAndCompute(gray1, None)
    keypoints_sift2, descriptors_sift2 = sift.detectAndCompute(gray2, None)
    
    # Initialize SURF
    surf = cv2.xfeatures2d.SURF_create()
    
    # Detect keypoints and descriptors using SURF
    keypoints_surf1, descriptors_surf1 = surf.detectAndCompute(gray1, None)
    keypoints_surf2, descriptors_surf2 = surf.detectAndCompute(gray2, None)
    
    # Initialize ORB
    orb = cv2.ORB_create()
    
    # Detect keypoints and descriptors using ORB
    keypoints_orb1, descriptors_orb1 = orb.detectAndCompute(gray1, None)
    keypoints_orb2, descriptors_orb2 = orb.detectAndCompute(gray2, None)
### STEP3: FEATURE MATCHING WITH BRUTE-FORCE AND FLANN
    # Brute-Force Matcher for SIFT and SURF (use NORM_L2 for these floating-point descriptors)
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches_sift = bf.match(descriptors_sift1, descriptors_sift2)
    matches_sift = sorted(matches_sift, key=lambda x: x.distance)
    
    matches_surf = bf.match(descriptors_surf1, descriptors_surf2)
    matches_surf = sorted(matches_surf, key=lambda x: x.distance)
    
    # Brute-Force Matcher for ORB (use NORM_HAMMING for binary descriptors)
    bf_orb = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches_orb = bf_orb.match(descriptors_orb1, descriptors_orb2)
    matches_orb = sorted(matches_orb, key=lambda x: x.distance)
    
    # FLANN Matcher (for SIFT and SURF)
    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    matches_flann_sift = flann.knnMatch(descriptors_sift1, descriptors_sift2, k=2)
    matches_flann_surf = flann.knnMatch(descriptors_surf1, descriptors_surf2, k=2)
    
    # Apply ratio test for FLANN matcher (Lowe's ratio test)
    good_matches_sift = []
    good_matches_surf = []
    for m, n in matches_flann_sift:
        if m.distance < 0.75 * n.distance:
            good_matches_sift.append(m)
    
    for m, n in matches_flann_surf:
        if m.distance < 0.75 * n.distance:
            good_matches_surf.append(m)
### STEP4: IMAGE ALIGNMENT USING HOMOGRAPHY
    # Use SIFT keypoints to compute homography (or you can change this to SURF or ORB)
    src_pts = np.float32([keypoints_sift1[m.queryIdx].pt for m in good_matches_sift]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints_sift2[m.trainIdx].pt for m in good_matches_sift]).reshape(-1, 1, 2)
    
    # Find the homography matrix
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    
    # Warp image1 to align with image2
    h, w = image1.shape[:2]
    aligned_image = cv2.warpPerspective(image1, M, (w, h))
    
    # Display Matches for each Method
    def display_matches(title, img1, kp1, img2, kp2, matches):
        result_img = cv2.drawMatches(img1, kp1, img2, kp2, matches[:50], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        plt.figure(figsize=(10,5))
        plt.title(title)
        plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
        plt.show()
    
    # Display Matches for SIFT, SURF, and ORB
    display_matches('SIFT + Brute Force', image1, keypoints_sift1, image2, keypoints_sift2, matches_sift)
    display_matches('SURF + Brute Force', image1, keypoints_surf1, image2, keypoints_surf2, matches_surf)
    display_matches('ORB + Brute Force', image1, keypoints_orb1, image2, keypoints_orb2, matches_orb)
    
    # Display FLANN Matches
    display_matches('SIFT + FLANN', image1, keypoints_sift1, image2, keypoints_sift2, good_matches_sift)
    display_matches('SURF + FLANN', image1, keypoints_surf1, image2, keypoints_surf2, good_matches_surf)
    
    # Display and Save the Aligned Image
    plt.figure(figsize=(10,5))
    plt.title('Aligned Image using Homography (SIFT)')
    plt.imshow(cv2.cvtColor(aligned_image, cv2.COLOR_BGR2RGB))
    plt.show()
    
    # Save the aligned image
    cv2.imwrite('aligned_image.jpg', aligned_image)
![SIFTB](https://github.com/user-attachments/assets/35924680-20c2-4d13-baa7-c2c8be676510)    
![MP3](https://github.com/user-attachments/assets/de316161-88b7-4e2c-9ea4-ed00a95249a4)
![MP32](https://github.com/user-attachments/assets/a8de3e1b-1d31-4a62-b7ea-37e12f048013)
![SIFTF](https://github.com/user-attachments/assets/b2708bfe-c572-4e7b-b5ca-176f9df37145)
![SURFF](https://github.com/user-attachments/assets/92ddcf8a-64c8-4f22-a3d5-3b9f81a6ff31)
![orbf](https://github.com/user-attachments/assets/f353acba-fa16-40a8-8bb1-8c3127c9f699)
![al](https://github.com/user-attachments/assets/e19a0e21-c34d-4769-8774-9b1a8ea6e629)

