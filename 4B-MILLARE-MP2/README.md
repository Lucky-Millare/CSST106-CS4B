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
