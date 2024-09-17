# CSST106-MP2
## MILLARE, LUCKY OWELL U.
# Exercise1-Image-Processing-Techniques
## Google Colab Notebook - Link
https://colab.research.google.com/drive/1FmzuXHSPj8iDk9DohsyHHJJOWdimuvH6?usp=sharing
### Hands-On Exploration:
### 1. Install OpenCV
    !pip install opencv-python-headless
### 2. Import Libraries
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt

    def display_image(img,title="Image"):
      plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
      plt.title(title)
      plt.axis('off')
      plt.show()
    
    def display_images(img1,img2,title1="Image1",title2="Image2"):
      plt.subplot(1,2,1)
      plt.imshow(cv2.cvtColor(img1,cv2.COLOR_BGR2RGB))
      plt.title(title1)
      plt.axis('off')
    
      plt.subplot(1,2,1)
      plt.imshow(cv2.cvtColor(img1,cv2.COLOR_BGR2RGB))
      plt.title(title2)
      plt.axis('off')
    
      plt.show()
      
### 3. Load Image
    from google.colab import files
    from io import BytesIO
    from PIL import Image
    
    uploaded = files.upload()
    
    image_path = next(iter(uploaded))
    image = Image.open(BytesIO(uploaded[image_path]))
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    display_image(image, "Original Image")
    
### EXERCISE 1. Scaling and Rotation
    def scale_image(image, scale_factor):
      height, width = image.shape[:2]
      scale_img = cv2.resize(image,(int(width * scale_factor), int(height * scale_factor)), interpolation = cv2.INTER_LINEAR)
      return scale_img
    
    def rotate_image(image, angle):
      height, width = image.shape[:2]
      center = (width//2, height//2)
      matrix = cv2.getRotationMatrix2D(center, angle, 1)
      rotated_image = cv2.warpAffine(image, matrix, (width, height))
      return rotated_image
    
    scaled_image = scale_image(image, 0.5)
    display_image(scaled_image, "Scaled Image (50%)")
    
    rotated_image = rotate_image(image, 45)
    display_image(rotated_image, "Rotated Image (45°)")

### Exercise 2: Blurring Techniques
    gussian_blur = cv2.GaussianBlur(image, (11,11), 0)
    display_image(gussian_blur, "Gussian Blur")
    
    median_blur = cv2.medianBlur(image,17)
    display_image(median_blur, "Median Blur")
    
    bilateral_blur = cv2.bilateralFilter(image, 99, 99, 99)
    display_image(bilateral_blur, "Bilateral Blur")

### **3. Edge Detection using Canny**
    edge = cv2.Canny(image,100 ,150)
    display_image(edge, "Canny Edge Detection")

## Problem Solving Session:
One of the Challenge I encounter in our lab session is the uploading a image in colab notebook this is because of having a BAYER in the code because of TAB too much in the Laboratory session.

![IMAGE](https://github.com/user-attachments/assets/05a9a055-c10a-4ba8-bac2-4aa1a64845ca)

To fix the problem is we have to remove the BAYER in the code so we can upload a image without error

![dsds](https://github.com/user-attachments/assets/6fcc0ace-64a1-483b-9d63-56b7e8221336)

### Scenario-Based Problems: Solve scenarios where you must choose and apply appropriate image processing techniques
In the Exercise 1 Hand on Exploration we following the step of our professor instructed in the TV screen the problem I encounter in the one of the Image processing Techniques that been instructing is in the Scaling and Rotation. The problem I encounter in that Part is I didn’t copy properly the code the professor instructed because I TAB it without knowing the code is not the same so when I run the code there is a error.
![Picture1](https://github.com/user-attachments/assets/d2bdd697-a6ec-4fe2-95f9-4632cd95b4be)

To solve the problem I try to compare my code to my seatmates and learn what is wrong in my code the problem is in the scale_img I do not have this (int(width * scale_factor) that will be used in to calculate the new width of the image after scaling. After I fix it the error is gone.
![Picture2](https://github.com/user-attachments/assets/b334f2ca-2519-47ae-884c-62df15931ce0)
