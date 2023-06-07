## DLD Deep Learning Diseases

### Abstract

DLD Deep Learning Diseases aims to find cancerous tumors in CT scans. We hope to have a greater accuracy than human identification (30% in a 50% cancer frequency) through a deep learning image processing approach.

### Data

Our dataset comes from the Cancer Imaging Archive. Since the data is imbalanced towards no cancerous nodules we augment our data to balance occurances of both instances.

### Approach
We will find cancerous tumors through an image processing approach. We will use the Watershed algorithm for image segmentation to identify non native nodules in the lungs. 

The advantage of the Watershed algorithm is that it works well in grayscale imaging. This is advantageous because the format of our data are CT scans which are grayscale. 

We have chosen CT scans because we are able to identify nodules through the usage of Houndsfields units that signify the presence of nodules. 

We process the CT scans into modified images with their constrast and zoom enhanced to ease the task for the model. We create a 10 layer deep neural network to find potential tumors. 

### Results

We report a 82% accuracy on a 50% cancer frequency dataset, meaning we miss 18% in a 50% cancer frequency set. Our goal to be more accurate than human practitioners has been achieved as we are almost twice as accurate as humans.