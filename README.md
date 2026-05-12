# Real-Time-Vision-Based-Traffic-Surveillance-for-Smart-Transportation-Infrastructure
Real-time vehicle detection, tracking, congestion analysis, and speed estimation using YOLOv8 and IoU-based multi-object tracking.

Project Overview

This project implements a real-time intelligent traffic surveillance and analytics framework using deep learning and computer vision techniques. The system performs vehicle detection, multi-object tracking, traffic density analysis, congestion classification, vehicle counting, and speed estimation from traffic video streams.

The framework is developed using Python 3.8, PyTorch, OpenCV, and Ultralytics YOLOv8. Vehicle detection is performed using the YOLOv8n model, while IoU-based multi-object tracking is used for maintaining object identities across frames.

## Key Features

- Real-time vehicle detection using YOLOv8n
- IoU-based multi-object tracking
- Vehicle counting and per-class statistics
- Traffic density and flow-rate computation
- Congestion classification
- Vehicle speed estimation
- Real-time visualization and analytics dashboard

 Project Structure

```text
src/
│── main.py
│── detector.py
│── tracker.py
│── metrics.py
│── visualization.py
│── utils.py
│── config.py

```
Software Requirements
•	Python 3.8
•	PyTorch
•	OpenCV
•	Ultralytics YOLOv8
•	NumPy

Hardware Configuration
•	Intel i7 Processor
•	8 GB RAM
•	CUDA-enabled GPU

Dataset Information
Vehicle detection and congestion analysis were developed using the following Kaggle dataset:
Top-View Vehicle Detection Image Dataset

https://www.kaggle.com/datasets/farzadnekouei/top-view-vehicle-detection-image-dataset

Dataset details:
•	536 training images
•	90 validation images
•	YOLOv8 annotation format
•	Images resized to 640×640
•	Horizontal flip augmentation applied
Speed estimation testing videos were collected from:

https://www.pexels.com/search/vehicles%20in%20lanes/

## Sample Working Videos

- [Sample Video 1 – Google Drive Link](https://drive.google.com/file/d/1mRvmeObUN9UmmCivt-DlCIf2OtUJ9p1L/view?usp=sharing)

- [Sample Video 2 – Google Drive Link](https://drive.google.com/file/d/1fUqmKRAkzxzS4E29boWtVwxl3jdGQhl1/view?usp=sharing)

- [Sample Video 3 – Google Drive Link](https://drive.google.com/file/d/1vPhkFHodsuF1_wCleEvweKww4Pke1vwI/view?usp=sharing)
