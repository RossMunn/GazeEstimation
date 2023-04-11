Gaze Estimation using Convolutional Neural Networks

Introduction
This Python project uses deep learning techniques to estimate the gaze direction of a person's eyes from a live webcam feed. The project is divided into three main folders:

Data Processing: This folder contains code that takes a dataset and extracts eye patches for both the left and right eye. It then splits this data into train and test datasets.

Model Training: This folder contains code that trains two Convolutional Neural Networks (CNNs) to estimate the gaze direction of a person's eyes. One CNN is trained on the left eye patches and the other on the right eye patches.

Gaze Estimation: This folder contains code that imports the trained models and runs them on a live webcam feed. It uses a feature extractor and prediction averaging code to estimate the gaze direction.

Each folder has its own Anaconda environment to run the code.
(will be updated)


Data Processing

Activate the data processing environment.
Navigate to the data_processing folder.
Place your dataset in the data folder.
Run process_data.py to extract the eye patches and split the data into train and test datasets.

Model Training

Activate the model training environment.
Navigate to the model_training folder.
Place the train and test datasets in the data folder.
Run train_model.py to train the CNNs on the eye patches.

Gaze Estimation

Activate the gaze estimation environment.
Navigate to the gaze_estimation folder.
Run gaze_estimator.py to run the trained models on a live webcam feed.

Conclusion
This project demonstrates how deep learning techniques can be used to estimate the gaze direction of a person's eyes. By using CNNs and a live webcam feed, this technology can have practical applications in fields such as user interface design, virtual reality, and even medical diagnosis.
