# Directional Eye Gaze Estimation System

The following code is for my final year dissertation project. This code is separated into 3 parts with 3 corresponding anaconda environments that need to be configured for execution of the code.

Before these environments are configured, the Eye-Chimera Dataset needs to have its terms and conditions agreed to before a dataset download can be done. Upon signing the form, the dataset can be downloaded from the link below the form. Both the form and dataset can be found on this website- http://imag.pub.ro/common/staff/cflorea/cf_research.html. Once the dataset is installed, the following environments can be configured to run the system.

### Preprocessing & data collection environment

Using the 'process.yaml' file the anaconda environment to run the dataset preprocessing files of 'eyeExtract.py', 'datasetSplit.py' and 'dataCollection.py'can be run. Of which, 'eyeExtract.py' extracts all the left and right eyes from the Eye-Chimera Dataset. 'datasetSplit.py' splits the extracted eye data from eyeExtrcat into training and testing folders. 'dataCollection.py' collects a user's eye images for system calibration into sorted folders.

To configure the process environment, the following steps can be followed:
1. Open your terminal/command prompt. In Windows, you can use the Anaconda Prompt. On MacOS and Linux, any terminal should be fine.
2. Navigate to the directory where your YAML file is stored by using the '/cd' 
3. Once you have the YAML file, you can create the new environment from it using the following command: 'conda env create -f process.yaml'
4. Once the environment is created, you can activate it using the following command: 'conda activate process'

With this the process environment should be installed and all necessary dataset preprocessing can be done for training.



### CNN model training environment

This environment is for the training of the left and right eye CNNs for GPUs, however if you do not have a GPU or chose not to use one a regular TensorFlow environment can be made with Python version 3.9. However, if you wish to use a GPU the 'train.yaml' file has the required dependencies. The train environment will allow you to run the 'leftEye.py' & 'rightEye.py' training files to train the CNNs using the preprocessed dataset.

To configure the train environment, the following steps can be followed:
1. Open your terminal/command prompt. In Windows, you can use the Anaconda Prompt. On MacOS and Linux, any terminal should be fine.
2. Navigate to the directory where your YAML file is stored by using the '/cd' 
3. Once you have the YAML file, you can create the new environment from it using the following command: 'conda env create -f train.yaml'
4. Once the environment is created, you can activate it using the following command: 'conda activate train'


### Live gaze estimations environment

Using the 'estimate.yaml' file, the anaconda environment to run the live gaze estimation and prototype GUI files of 'foundational.py', 'calibration.py' and 'prototype.py' can be run. Of which 'foundational.py', 'calibration.py' will both produce live gaze estimation using your webcam with the calibration system using calibration data to do this. The 'prototype.py' file will run will calibration data requirements and output the prototype navigational GUI with the user webcam.

To configure the estimate environment, the following steps can be followed:
1. Open your terminal/command prompt. On Windows, you can use the Anaconda Prompt. On MacOS and Linux, any terminal should be fine.
2. Navigate to the directory where your YAML file is stored by using the '/cd' 
3. Once you have the YAML file, you can create the new environment from it using the following command: 'conda env create -f estimate.yaml'
4. Once the environment is created, you can activate it using the following command: 'conda activate estimate'


With all three of these environments configured the dataset can be processed, calibration data can be recorded, models can be trained and live gaze estimations with the prototype can be run. However, there is a testing folder which can be run using either the 'process' or 'estimate' environments but have been excluded as a custom testing dataset is required to use these.

