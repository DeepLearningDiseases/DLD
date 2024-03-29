# DeepLearningDiseases: Installation Guide

This installation guide provides comprehensive instructions for building the **DLD** project from scratch and running the model on preprocessed data. By following the steps outlined below, you will be able to set up the project environment and successfully execute the model.

## Prerequisites

Before proceeding with the installation, ensure that you have the following prerequisites:

-   **Operating System:** The project supports the following operating systems:
    
    -   Windows 10 or later
    -   macOS Mojave (10.14) or later
    -   Ubuntu 18.04 or later
    
-   **Python:** The project requires Python 3.7 or higher. If you haven't installed Python yet, follow the official Python installation guide for your operating system.
    
-   **Libraries:** The project relies on the following libraries and dependencies. Make sure you have them installed before proceeding:

	-   **NumPy** (1.18.0 or higher): A library for numerical computing in Python.
    -   **TensorFlow** (2.5.0 or higher): An open-source machine learning framework.
    -   **Pandas** (1.0.0 or higher): A powerful data manipulation library.
    -   **PIL** (Python Imaging Library): A library for opening, manipulating, and saving many different image file formats.
    -   **Joblib**: A library for parallel computing and caching in Python.
    -   **Glob**: A library for finding files and directories using wildcard patterns.
    -   **scikit-learn**: A machine learning library for classification, regression, and clustering tasks.
    -   **SimpleITK**: An image analysis library with a focus on medical imaging.
    -   **tflearn**: A modular and transparent deep learning library built on top of TensorFlow.
    -   **h5py**: A library for working with HDF5 files in Python.

You can install the required libraries by running the following command in your command-line interface:

    pip install numpy tensorflow pandas matplotlib pillow joblib glob scikit-learn SimpleITK tflearn h5py`

- **Google Colab:** Google Colab provides a cloud-based Jupyter notebook environment with pre-installed libraries and the ability to leverage powerful GPUs and TPUs for accelerated computation.

## Building the Project from Scratch

Building the project from scratch involves the following steps (make sure you have all the required files mentioned below downloaded from GitHub under a single folder):

1.  **Generate Preprocessing Data:** Preprocess the raw data by performing data cleaning, normalization, or augmentation as needed.
	- Under a single directory, create separate folders namely, *data*, *hdf5_data*, and *preprocessed_data*.
		- *data*: Will contain cropped images for training, testing, and validation. These are the images used by the model.
		- *preprocessed_data* : Will contains the split data (reference to the CT files) for training, testing, and validation. 
		- *hdf5_data* : Will contain the hdf5 dataset(s) and labels.
	- Download the following folders from GitHub: *metadata*, *mhd_data*, *src*.
		- *metadata*: Contains the metadata for the *mhd_data*.
		- *mhd_data*: Contains the raw CT Scan files (.mhd format).
		- *src* : Contains the preprocessing scripts.
	- Run the *generate_preprocessing_data.py* script from within the *src* folder in order to read in the .mhd CT files, generate preprocessed data, crop the images, create labels, and generate Training-Test-Validation Split.
	
  3.  **Generate Augmented Images:** Generate more images for positive cases by rotating the already existing positive case images.
	  - Run the *augment_training_data.py* script from within the *src*  folder to augment and create more training data (done by rotating the images).
  
2.  **Build HDF5 Dataset:** Convert the preprocessed data into an efficient HDF5 file format for storage and retrieval.
	 -	Run *hdf5_dataset.py* from within the *src* directory to create hdf5 datasets for training, testing, and validation.

Steps to run the model:
-  *Please refer to the steps of the following section ***Running the Model on Preprocessed Data*** in order to run the model.*
    

## Running the Model on Preprocessed Data

Running the model on preprocessed data involves the following steps (make sure you have all the required files generated by building the model from scratch):


To run the model on preprocessed data, follow these instructions:

1.  **Upload Preprocessed Folders to Google Drive:** Ensure that your preprocessed data is organized in specific folders on your local machine. Next, under a single directory, upload these folders and files (from GitHub) to your Google Drive account. These folders can also be found on our GitHub repo Required preprocessed folders:
	- *data*
	- *hdf5_data*
	- *preprocessed_data*
    
3.  **Upload the Model to Google Drive:** Similarly, upload the model file to your Google Drive account. The model .ipynb file can be found on the GitHub repo.
    
4.  **Change the Input Folder Mount Point in the Model:** Open the model file with Google Colab and locate the code section where the *INPUT_FOLDER* variable folder is specified. Replace the local path with the appropriate mount point of the Google Drive folder where you uploaded the preprocessed data.
    
5.  **Run the Model File:** After making the necessary changes to the input folder mount point in the model, save the file and execute it. This will train the model and show all the outputs and metrics of the model.