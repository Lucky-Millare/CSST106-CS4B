# CSST 106 - Exercise 1
## LUCKY OWELL U. MILLARE
## GOOGLE COLAB LINK
https://colab.research.google.com/drive/1FmzuXHSPj8iDk9DohsyHHJJOWdimuvH6?usp=sharing
## PDF FILE
[4B-MILLARE-EXER1.pdf](https://github.com/user-attachments/files/17013910/4B-MILLARE-EXER1.pdf)
## Hands-On Exploration
### 1. Install OpenCV
    !pip install opencv-python-headless
### **2. Import Libraries**
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt
    
    def display_image(img, title="Image"):
      plt.figure(figsize=(6,3))
      plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
      plt.title(title)
      plt.axis("off")
      plt.show()
    
    def display_images(img1, img2, title1="Image 1", title2="Image 2"):
      plt.figure(figsize=(6,3))
      plt.subplot(1,2,1)
      plt.imshow(cv2.cvtColor(img1,cv2.COLOR_BGR2RGB))
      plt.title(title1)
      plt.axis("off")
    
      plt.subplot(1,2,2)
      plt.imshow(cv2.cvtColor(img2,cv2.COLOR_BGR2RGB))
      plt.title(title2)
      plt.axis("off")
      plt.show()
###  3. **Load Image**
    from google.colab import drive
    drive.mount('/content/drive')
    
    image_path = '/content/drive/MyDrive/image.jpeg'  # Replace with your image path in Google Drive
    image = cv2.imread(image_path)
    display_image(image, "Original Image")
    
    '''
    from google.colab import files
    from io import BytesIO
    from PIL import Image
    
    uploaded = files.upload()
    
    image_path = next(iter(uploaded))
    image = Image.open(BytesIO(uploaded[image_path]))
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    display_image(image, "Original Image")
    '''
###  EXERCISE 1. Scaling and Rotation
    def scale_image(image, scale_factor):
      height, width = image.shape[:2]
      scale_img = cv2.resize(image,(int(width * scale_factor), int(height * scale_factor)), interpolation = cv2.INTER_LINEAR)
      return scale_img
    
    def rotate_image(image, angle):
      height, width = image.shape[:2]
      center = (width//2,height//2)
      matrix = cv2.getRotationMatrix2D(center,angle,1)
      rotated_image = cv2.warpAffine(image,matrix,(width,height))
      return rotated_image
    
    scaled_image = scale_image(image, 0.5)
    display_image(scaled_image,"Scaled Image")
    
    rotated_image = rotate_image(image, 45)
    display_image(rotated_image,"Rotated Image")
