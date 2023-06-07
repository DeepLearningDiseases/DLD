"""
  We have low number of positive cases, so we
  augment the data here to have more training
  data available for the positive class.
"""

from PIL import Image
import pandas as pd
from joblib import Parallel, delayed

# For new images
random_num1 = 1000000
random_num2 = 2000000

# a function that takes in an index, and returns two rotated images
# one by 90 degree and one by 180 degree
def augment(image_index):
    image_path = f'../data/train/image_{image_index}.jpg'
    img = Image.open(image_path)
    
    img90 = img.rotate(90)
    img90_path = f'../data/train/image_{image_index+random_num1}.jpg'
    img90.convert('L').save(img90_path)

    img180 = img.rotate(180)
    img180_path = f'../data/train/image_{image_index+random_num2}.jpg'
    img180.convert('L').save(img180_path)

    return img90_path, img180_path

# Main
if __name__ == '__main__':
    X_train = pd.read_pickle('../preprocessed_data/training_data')
    y_train = pd.read_pickle('../preprocessed_data/training_labels')

    # Get the indexes of positive cases to augment
    augIndexes = X_train[y_train == 1].index

    # Parallize the function augment, uses 4 processes
    results = Parallel(n_jobs=4)(delayed(augment)(idx) for idx in augIndexes)

    for img90_path, img180_path in results:
        print(f'Done saving images: {img90_path}, {img180_path}')
