# Object Detection for Fresh Produce Quality Control with Interactive Assistant

##Project Overview

This project is an AI-powered solution designed to streamline quality control in the agricultural sector. It leverages a YOLO-based object detection model to automatically identify and classify various vegetables from images, paired with an intelligent chatbot developed using Langflow and RAG (Retrieval-Augmented Generation). 

This chatbot assists users with project-related questions, implementation guidance, and vegetable characteristics, providing a robust and interactive tool for agricultural technology applications.

Features
YOLO-based Object Detection System: Real-time identification and classification of vegetables in images, optimized for accuracy and speed.

Langflow-based Chatbot: An intelligent chatbot capable of answering project-related questions, providing setup guidance, explaining code, and troubleshooting common issues.

User-Friendly Interface: Built on Streamlit, enabling easy interaction for both image-based and real-time webcam detection.

Project Requirements and Criteria

## Part 1: Object Detection System

### 1. Data Source
Dataset Creation: Collect and annotate images of various vegetables.

Minimum Images per Class: 100 images per class.

Classes: At least 5 classes selected from options like bean, bitter gourd, bottle gourd, brinjal, broccoli, cabbage, capsicum, carrot, cauliflower, cucumber, papaya, potato, pumpkin, radish, and tomato.

### 2. Data Split: 70% training, 15% validation, 15% testing.

Example Dataset: Vegetable Image Dataset on Kaggle

### 3. Data Collection and Annotation

Image Collection: Capture diverse lighting conditions and angles.

Annotation Tools: Use tools like Roboflow for labeling.

Annotation Format: Save annotations in YOLO format.

Data Configuration: Create a data.yaml file specifying classes and paths.

### 4. Model Development

Model Architecture: YOLOv8 or YOLOv11 as the base architecture.

Machine Learning Workflow:

Problem Formulation

Data Preparation

Model Development

Model Deployment

Performance Targets:

Mean Average Precision (mAP) > 0.75

Inference time < 100ms per image on CPU

### 5. Training Requirements
   
Transfer Learning: Start from pre-trained COCO weights.

Data Augmentation: Apply techniques to improve robustness.

Overfitting Prevention: Use callbacks and track metrics.

Metrics Logging: Track training with TensorBoard.

Best Weights: Save during training for optimal performance.

## Part 2: Project Assistant Chatbot

### 1. Chatbot Development

Knowledge Base:

Project documentation

Model architecture details

Training procedures

Dataset information

Vegetable characteristics

### 2. RAG (Retrieval-Augmented Generation) Implementation
   
Langflow Pipeline Components:

Document loader for project documentation

Text splitter (chunk size: 500, overlap: 50)

Embedding generation

Vector store (Chroma or FAISS)

Language model integration (ChatGPT or equivalent)

Custom prompt templates

### 3. Chatbot Capabilities
   
User Assistance:

Provide information on project implementation details, model architecture, training process, dataset, and vegetable identification.

Offer code explanations and troubleshooting support.

Guide users through the setup process.

### 4. Langflow Requirements
   
Diagram and Memory: Create and export a Langflow diagram with conversation memory.

Error Handling: Ensure robust error handling.

Source Citations: Include citations in chatbot responses.

Prompt Management: Save and document all custom prompt templates.
