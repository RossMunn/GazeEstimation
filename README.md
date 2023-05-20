# Directional Eye Gaze Estimation System

The following code is for my final year dissertation project. This code is seperated into 3 parts with 3 corropsonding anaconda environments that need to be configured for execution of the code.

Before these environments are configured the Eye-Chimera Dataset needs to have its terms and conditions agreed to before a dataset download can be done. Upon signing the form the dataset can be downloaded fromthe link below the form. Both the form and dataset can be found on this website- http://imag.pub.ro/common/staff/cflorea/cf_research.html. Once the dataset is installed the following enviroments can be configured to run the system.

### Preprocessing & data collection environment

Using the 'process.yaml' file the anaconda environment to run the dataset preprocessing files of eyeExtract, datasetSplit and dataCollection can be run. Of which eyeExtract extracts all the left and right eyes from the Eye-Chimera Dataset. datasetSplit splits the extracted eye data from eyeExtrcat into training and testing folders. dataCollection collects a users eye images for system calibration into sorted folders.

To configure the process environment the following steps can be followed:
1. Open your terminal/command prompt. In Windows, you can use the Anaconda Prompt. On MacOS and Linux, any terminal should be fine.
2. Navigate to the directory where your YAML file is stored by using the '/cd' 
3. Once you have the YAML file, you can create the new environment from it using the following command: 'conda env create -f process.yaml'
4. Once the environment is created, you can activate it using the following command: 'conda activate env_name'

With this the process environment should be installed and all nessacary dataset preprocessing can be done for training.



### CNN model training environment

This environment is for the training of the left and right eye CNNs for GPUs however if you do not have a GPU or chse not to use one a regular TensorFlow environment can be made with Python version 3.9. However if you wish to use a GPU the 'train.yaml' file has the required dependancies. The train environment will allow you to run the leftEye & rightEye training files to train the CNNs using the preprocessed dataset.

To configure the train environment the following steps can be followed:
1. Open your terminal/command prompt. In Windows, you can use the Anaconda Prompt. On MacOS and Linux, any terminal should be fine.
2. Navigate to the directory where your YAML file is stored by using the '/cd' 
3. Once you have the YAML file, you can create the new environment from it using the following command: 'conda env create -f train.yaml'
4. Once the environment is created, you can activate it using the following command: 'conda activate env_name'


### Live gaze estimations environment

This environment

