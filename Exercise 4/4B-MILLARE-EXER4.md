# Exercise 4: Object Detection and Recognition
### IMPORT LIBRARIES
    import cv2
    import matplotlib.pyplot as plt
    import numpy as np
    import tensorflow as tf
    import tensorflow_hub as hub
    import urllib.request
    import tarfile
    import os
    import time
    from skimage.feature import hog
    from sklearn import svm
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    from google.colab.patches import cv2_imshow
    from google.colab import drive
    
    drive.mount('/content/drive')
# HOG(HISTOGRAM OF ORIENTED GRADIENTS) OBJECT DETECTION
### LOAD THE IMAGE
    image_path = '/content/drive/MyDrive/image.jpeg'
    image = cv2.imread(image_path)
### CONVERT TO GRAYSCALE
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
### APPLY HOG DESCRIPTOR
    features, hog_image = hog(gray_image, orientations=9, pixels_per_cell=(8, 8),
                              cells_per_block=(2, 2), visualize=True)
### Display the Image
    plt.figure(figsize=(8, 8))
    plt.imshow(hog_image, cmap='gray')
    plt.title('HOG Image')
    plt.axis('off')
    plt.show()
![exer41](https://github.com/user-attachments/assets/e646314a-ba94-4a82-9f08-169373008792)

# YOLO(YOU ONLY LOOK ONCE) OBJECT DETECTION
### INSTALL DEPENDENCIES
    !pip install opencv-python-headless
    # Download YOLOv3 weights, config, and class names
    !wget https://pjreddie.com/media/files/yolov3.weights
    !wget https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg?raw=true -O yolov3.cfg
    !wget https://github.com/pjreddie/darknet/blob/master/data/coco.names?raw=true -O coco.names
### LOAD YOLO MODEL AND CONFIGURATION
    net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')
    
    # Load class names
    with open('coco.names', 'r') as f:
        classes = [line.strip() for line in f.readlines()]
### LOAD THE IMAGE
    image_path = '/content/drive/MyDrive/image2.0.jpg'
    image = cv2.imread(image_path)
### PREPARE THE IMAGE FOR YOLO
    height, width, _ = image.shape
    blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
### PROCESS DETECTION
    # Get output layer names
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    
    # Perform detection
    outs = net.forward(output_layers)
    
    # Process detections
    boxes = []
    confidences = []
    class_ids = []
    
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
    
            # Temporarily comment out the confidence check
            # if confidence > 0.5:
    
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)
    
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)
    
            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)
    
    
    # Apply Non-Max Suppression
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    
    # Draw bounding boxes
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(image, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
### DISPLAY THE IMAGE
    plt.figure(figsize=(8, 8))
    plt.imshow(image1)
    plt.title('YOLO Object Detection')
    plt.axis('off')
    plt.show()
![exer42](https://github.com/user-attachments/assets/da4c77b1-501b-4aa9-961c-a15da2386607)

# SSD(SINGLE SHOT MULTIBOX DETECTOR)
### INSTALL DEPENDENCIES
    !pip install tensorflow opencv-python matplotlib
    !wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz
    !tar -xvf ssd_mobilenet_v2_coco_2018_03_29.tar.gz
### LOAD PRE-TRAINED SSD MODEL
    ssd_model = tf.saved_model.load("ssd_mobilenet_v2_coco_2018_03_29/saved_model")
### LOAD THE IMAGE
    image_path = '/content/drive/MyDrive/image2.0.jpg'
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
### RUN THE MODEL
    input_tensor = tf.convert_to_tensor(image_rgb)
    input_tensor = input_tensor[tf.newaxis, ...]
    # Get the inference function from the SavedModel
    detect_fn = ssd_model.signatures['serving_default']
    
    # Now you can call the inference function with the input tensor
    detections = detect_fn(input_tensor)
### VISUALIZE THE BOUNDING BOX
    detection_boxes = detections['detection_boxes'][0].numpy()
    detection_classes = detections['detection_classes'][0].numpy().astype(np.int32)
    detection_scores = detections['detection_scores'][0].numpy()
    
    # Set a threshold score for displaying boxes
    threshold = 0.5
    height, width, _ = image.shape
    for i in range(len(detection_boxes)):
        if detection_scores[i] >= threshold:
            box = detection_boxes[i] * [height, width, height, width]
            y_min, x_min, y_max, x_max = box.astype(int)
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            label = str(detection_classes[i])
            cv2.putText(image, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
### DISPLAY THE IMAGE
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('SSD Object Detection')
    plt.axis('off')
    plt.show()
![exer43](https://github.com/user-attachments/assets/ca8ba8c5-384b-48f3-b520-5f04210f4f81)

# Traditional vs. Deep Learning Object Detection Comparison
### INSTALL DEPENDENCIES
    !pip install tensorflow==2.12.0 tensorflow-hub==0.13.0
### LOAD THE IMAGE
    image_path = '/content/drive/MyDrive/home.jpg'
    image = cv2.imread(image_path)
### RESIZE THE IMAGE
    resized_image = cv2.resize(image, (416, 416))
### CONVERT TO GRAYSCALE
    gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
### HOG-SVM IMPLEMENTATION
    # Extract HOG features
    start_time = time.time()
    # Remove the 'multichannel' argument as it's not needed for grayscale images
    hog_features, hog_image = hog(gray_image, pixels_per_cell=(8, 8),
                                  cells_per_block=(2, 2), visualize=True)
    
    # Use a simple SVM model (This is a placeholder; in practice, a trained SVM model would be used)
    clf = svm.SVC(kernel='linear')
    
    # Measure HOG-SVM processing time
    hog_svm_time = time.time() - start_time
    print(f"HOG-SVM Detection Time: {hog_svm_time:.4f} seconds")
### YOLO MODEL IMPLEMENTATION
    yolo_net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
    with open("coco.names", "r") as f:
        class_names = [line.strip() for line in f.readlines()]
    
    # Prepare the image for YOLO
    blob = cv2.dnn.blobFromImage(resized_image, 1/255, (416, 416), (0, 0, 0), swapRB=True, crop=False)
    yolo_net.setInput(blob)

    # YOLO forward pass
    start_time = time.time()
    layer_names = yolo_net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in yolo_net.getUnconnectedOutLayers()]
    outs = yolo_net.forward(output_layers)
    yolo_time = time.time() - start_time

    # Process YOLO outputs
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * resized_image.shape[1])
                center_y = int(detection[1] * resized_image.shape[0])
                w = int(detection[2] * resized_image.shape[1])
                h = int(detection[3] * resized_image.shape[0])
    
                # Calculate bounding box coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
    
                # Draw bounding box and label
                label = f"{class_names[class_id]}: {confidence:.2f}"
                cv2.rectangle(resized_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(resized_image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
### DISPLAY THE IMAGE, PERFORMANCE COMPARISON
    plt.figure(figsize=(10, 10))
    plt.subplot(1, 2, 1)
    plt.imshow(hog_image)
    plt.title('HOG-SVM Image')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(resized_image)
    plt.title('YOLO Detected Image')
    plt.axis('off')
    
    plt.show()
    
    # Accuracy comparison may require a ground-truth dataset, which is simplified here.
    # For time comparison:
    print(f"HOG-SVM Time: {hog_svm_time:.4f}s vs YOLO Time: {yolo_time:.4f}s")
![exer44](https://github.com/user-attachments/assets/b1494469-3f93-4ca8-8d3b-044243f7c228)

### EXPLANATION
