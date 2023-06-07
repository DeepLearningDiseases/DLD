"""
    Generates preprocessing metadata and training, testing, validation images
"""

from joblib import Parallel, delayed
import numpy as np
import pandas as pd
import glob
from PIL import Image
from sklearn.model_selection import train_test_split
import SimpleITK as sitk

mhd_path = '../mhd_data/'               # path for referring to CT images

class CT(object):

    # file needs to be a mhd file
    # coords are the annotated coordinates that would get cropped
    # ds    :   CT header data
    # dir   :   directory that contains the .mhd files
    def __init__(self, file = None, coords = None, dir = None):
        self.file = file
        self.coords = coords
        self.metadata = None
        self.image = None
        self.dir = dir

    def reset_coords(self, coords):
        self.coords = coords

    # Read the mhd image
    def read_mhd_image(self):
        path = glob.glob(self.dir + self.file + '.mhd')
        self.metadata = sitk.ReadImage(path[0])
        self.image = sitk.GetArrayFromImage(self.metadata)

    # cartesian to voxel coordinates
    def get_voxel_coords(self):
        origin = self.metadata.GetOrigin()
        resolution = self.metadata.GetSpacing()
        # formula from online sources
        voxel_coords = [np.absolute(self.coords[j]-origin[j])/resolution[j] for j in range(len(self.coords))]
        return tuple(voxel_coords)
    
    def get_image(self):
        return self.image
    
    # Crops the self.image around the coords
    # and returns the cropped image
    def get_subimage(self, width):
        self.read_mhd_image()
        x, y, z = self.get_voxel_coords()
        subImage = self.image[int(z), int(y-width/2):int(y+width/2), int(x-width/2):int(x+width/2)]
        return subImage   
    
    # Convert HU (Hounds units) to grayscale
    # From:  SITK Documentation
    def normalize_planes(self, npzarray):
        maxHU = 400.
        minHU = -1000.
        npzarray = (npzarray - minHU) / (maxHU - minHU)
        npzarray[npzarray>1] = 1.
        npzarray[npzarray<0] = 0.
        return npzarray
    
    # Save the cropped image
    def save_cropped(self, filename, width):
        image = self.get_subimage(width)
        image = self.normalize_planes(image)
        Image.fromarray((image * 255).astype(np.uint8), mode='L').save(filename)


# Generate cropped images for a CT image (referred by index)
def create_data(index, out_dir, X_data,  width = 50):
    file = np.asarray(X_data.loc[index])[0]
    coords = np.asarray(X_data.loc[index])[1:]
    scan = CT(file, coords, mhd_path)
    outfile = out_dir + str(index) + '.jpg'
    scan.save_cropped(outfile, width)


# generate training and testing files (5:1 negatives to postives)
def generate_test_train_val(filename):
    candidates = pd.read_csv(filename)

    positive_cases = candidates[candidates['class']==1].index  
    negative_cases = candidates[candidates['class']==0].index

    np.random.seed(42)
    negative_indexes = np.random.choice(negative_cases, len(positive_cases)*5, replace = False)

    candidates_dataframe = candidates.iloc[list(positive_cases)+list(negative_indexes)]

    X = candidates_dataframe.iloc[:,:-1]
    y = candidates_dataframe.iloc[:,-1]
    
    # Training-Testing split (80-20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 42)
    
    # Training-Validating split (80-20)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.20, random_state = 42)

    # Save to pickle format
    folder = '../preprocessed_data/'
    X_train.to_pickle(folder+'training_data')
    y_train.to_pickle(folder+'training_labels')
    X_test.to_pickle(folder+'testing_data')
    y_test.to_pickle(folder+'testing_labels')
    X_val.to_pickle(folder+'val_data')
    y_val.to_pickle(folder+'val_labels')


if __name__ == '__main__':
    candidates_file = '../metadata/candidates.csv'

    # Generate train, test, val data
    generate_test_train_val(candidates_file)

    output_directory = '../data/'
    num_processes = 4
    
    types = {'test': 'testing', 'train': 'training', 'val': 'val'}
    
    # generate cropped images for testing, training, and val
    for k, v in types.items():
        input_file = f'../preprocessed_data/{v}_data'
        X_data = pd.read_pickle(input_file)
        Parallel(n_jobs = num_processes)(delayed(create_data)(index, output_directory+f'{k}/image_', X_data) for index in X_data.index)
