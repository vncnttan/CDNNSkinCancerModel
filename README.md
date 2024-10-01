# CDNN Skin Cancer Detection Model

Convolutional Deep Neural Network (CDNN) designed for the detection of skin cancer from dermoscopic images. The goal of the project is to develop an AI model that can classify different types of skin lesions, aiding in early diagnosis and treatment.

## Features
- Image Classification: Classifies dermoscopic images into various categories (e.g., benign, malignant).
- Imbalanced Datra Handling: Implements techniques to manage data imbalance, improving model performance.
- High Accuracy: Optimized for real-world scenarios with a focus on precision and recall.

## Learning Experiences:
- <b>Handling Imbalanced Data:</b> The skin cancer dataset had a significant class imbalance, where benign cases outnumbered malignant ones. I applied techniques such as class weighting, oversampling the minority class using SMOTE Technique to create synthethic images to ensure the model did not become biased.
- <b>Transfer Learning:</b> Working on the model training from scratch made me realized the importance of task understanding and the benefit for transfer learning, which become important for future project rather than making the model from scratch again.
- <b>Defining DCNN Architecture from scratch:</b> Creating model architectures from scratch (RESNET, EfficientNet, and GoogleNet) introduces me to their approach and tackling problems in Image Classification and how they handle them.

## Tech Stack
- Framework: Tensorflow/Keras
- Model Architectures tried: ResNet, EfficientNet, GoogleNet
- Dataset: HAM1000 Skin Cancer Dataset
