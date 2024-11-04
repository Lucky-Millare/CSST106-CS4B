# Mid-term Project: Implementing Object Detection on a Dataset
# Group Members: Gajitos, Jude B. , Millare, Lucky Owell U.
## ALGORITHM
### _YOLO (You Only Look Once)_
## DATASET USED:
### _https://drive.google.com/drive/folders/1rFKBceWbuKe4UmPuRxHGO35aqZEll6ND?usp=drive_link_
### LIBRARIES USED:
    import cv2
    import glob
    import os
    import zipfile
    import time
    import torch
    from ultralytics import YOLO
    from matplotlib import pyplot as plt
    from sklearn.metrics import precision_score, recall_score, accuracy_score
    from google.colab import drive
    drive.mount('/content/drive')
### DATA PREPARATION
    from google.colab import drive
    drive.mount('/content/drive')
    
    import zipfile
    #For Unzipping the file from google drive to colab cloud
    zip_ref = zipfile.ZipFile("/content/drive/MyDrive/CSST106/Aquarium Combined.v2-raw-1024.yolov8.zip", "r")
    zip_ref.extractall("/content/dataset")
    zip_ref.close()
### MODEL BUILDING
    !pip install ultralytics
    
    from ultralytics import YOLO
    import os
    
    train_dir = '/content/dataset/train'
    val_dir = '/content/dataset/valid'
    test_dir = '/content/dataset/test'
    
    import cv2
    import glob
    
    def preprocess_images(input_folder, output_folder, img_size=(640, 640)):
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
    
        for img_path in glob.glob(f"{input_folder}/*.jpg"):
            img = cv2.imread(img_path)
            img = cv2.resize(img, img_size)
            normalized_img = img / 255.0  # Normalize pixel values to [0, 1]
            cv2.imwrite(os.path.join(output_folder, os.path.basename(img_path)), normalized_img * 255)
    
    preprocess_images(train_dir, '/content/dataset/train_preprocessed')
    preprocess_images(val_dir, '/content/dataset/val_preprocessed')
    preprocess_images(test_dir, '/content/dataset/test_preprocessed')

### TRAINING THE MODEL
    from ultralytics import YOLO

    # Load a pretrained YOLOv8 model for transfer learning
    model = YOLO('yolov8n.pt')
    
    # Train the model with your dataset
    model.train(data='/content/dataset/data.yaml', epochs=20, batch=16, imgsz=640)
### TESTING
    # Load the trained YOLOv8 model
    model = YOLO("/content/runs/detect/train/weights/best.pt")  # Replace with the actual path to your trained model
    
    # Evaluate the model on the test set
    results = model.val(data='/content/dataset/data.yaml', split='test')
    
    from matplotlib import pyplot as plt
    import cv2
    
    # Add more image paths to the list
    test_images = [
        "/content/dataset/test/images/IMG_2301_jpeg_jpg.rf.2c19ae5efbd1f8611b5578125f001695.jpg",
        "/content/dataset/test/images/IMG_2632_jpeg_jpg.rf.f44037edca490b16cbf06427e28ea946.jpg",
        "/content/dataset/test/images/IMG_2448_jpeg_jpg.rf.28ce79dab47ad525751d5407be09bc3d.jpg",
        "/content/dataset/test/images/IMG_8595_MOV-0_jpg.rf.312ab0b8b9fca18134aee88044f45a06.jpg"
    ]
    
    # Loop through each image path
    for idx, img_path in enumerate(test_images, start=1):
        # Run inference
        results = model(img_path)
    
        # Display the image with predictions
        results[0].show()
    
        # Save each result with a unique filename
        results[0].save(filename=f"/content/dataset/results/IMG_{idx}.jpg")
### EVALUATION
    import time
    import torch
    from sklearn.metrics import precision_score, recall_score, accuracy_score
    from ultralytics import YOLO  # Assuming you're using YOLO from ultralytics
    
    # Load the trained model
    model = YOLO("/content/runs/detect/train/weights/best.pt")  # Replace with your model's path
    
    # Load the dataset (assumes images and labels are in a compatible format)
    test_images = [
        "/content/dataset/test/images/IMG_2301_jpeg_jpg.rf.2c19ae5efbd1f8611b5578125f001695.jpg",
        "/content/dataset/test/images/IMG_2632_jpeg_jpg.rf.f44037edca490b16cbf06427e28ea946.jpg",
        "/content/dataset/test/images/IMG_2448_jpeg_jpg.rf.28ce79dab47ad525751d5407be09bc3d.jpg",
        "/content/dataset/test/images/IMG_8595_MOV-0_jpg.rf.312ab0b8b9fca18134aee88044f45a06.jpg"
    ]  
    # Replace with your dataset paths
    ground_truths = [
        {'boxes': [[100, 150, 200, 250], [300, 50, 400, 150]], 'labels': [0, 1]},
        {'boxes': [[50, 100, 150, 200]], 'labels': [1]},
        {'boxes': [[200, 250, 300, 350]], 'labels': [0]},
        {'boxes': [[10, 20, 100, 150], [250, 180, 350, 280]], 'labels': [1, 0]}  
    ]
    
    # Initialize lists for true and predicted labels
    true_labels = []
    pred_labels = []
    
    start_time = time.time()
    
    for img_path, gt in zip(test_images, ground_truths):
        results = model(img_path)
        detections = results[0].boxes.data if results[0].boxes is not None else torch.empty((0, 6))  # Handle empty results
        
        # Extract labels from detections and convert them to integers
        detected_labels = detections[:, -1].int().tolist() if len(detections) > 0 else []
    
        # Append true labels from ground truth
        true_labels.extend(gt['labels'])
    
        # Handle the case where there are more predictions than ground truth
        if len(detected_labels) > len(gt['labels']):
            # Extend true_labels with a placeholder value (-1) to match the number of detected labels
            true_labels.extend([-1] * (len(detected_labels) - len(gt['labels'])))
        elif len(detected_labels) < len(gt['labels']):
            # Extend detected_labels with a placeholder value (-1) to match the number of ground truth labels
            detected_labels.extend([-1] * (len(gt['labels']) - len(detected_labels)))
    
        # Append detected labels
        pred_labels.extend(detected_labels)
    
    inference_time = time.time() - start_time
    
    # Compute metrics (use zero_division=1 to handle cases without positive predictions)
    precision = precision_score(true_labels, pred_labels, average='weighted', zero_division=1)
    recall = recall_score(true_labels, pred_labels, average='weighted', zero_division=1)
    accuracy = accuracy_score(true_labels, pred_labels)
    
    print(f'Precision: {precision:.2f}')
    print(f'Recall: {recall:.2f}')
    print(f'Accuracy: {accuracy:.2f}')
    print(f'Inference time: {inference_time:.2f} seconds')
### COMPARISON EXPLANATION
One-stage detectors like YOLOv8 have faster processing times than two-stage algorithms and hence have a higher potential for application in real-time uses. More classic is HOG-SVM, much slower because of the feature extraction step involved in the procedure apart from the final separate classification stage, which may seriously limit this detector's use for real-time applications. Nevertheless, for simple object detection tasks, HOG-SVM can also provide decent accuracy. The Single Shot MultiBox Detector, similar to YOLO, is also a one-shot detector; at the same time, the accuracy depends upon the size and scale of the object present within the image. In general, SSD offers a very good trade-off between speed and accuracy, but it cannot achieve the detection accuracy of complex or small objects compared to YOLOv8 because of the advanced architecture in YOLOv8. Empirical tests, where the models' inference speeds would be measured on the same hardware and their mAP calculated on a common dataset, would show tangible insights into their relative performances.
