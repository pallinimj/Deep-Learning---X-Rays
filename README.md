# NU_489_capstone

Deep Learning Approach to Thoracic Cavity Disease Feature Identification and Classification

### Project Scope

The project design is to utilize deep learning and computer vision techniques to perform detection, classification, and segmentation of various types of lung diseases  

### Desired Outcome

High accuracy and low loss in classification of selected lung diseases

### Dataset

Quality datasets incorporating medical imaging technology along with labels are required in order to perform supervised learning. Very few datasets exist. Time does not allow creation and curation a specific dataset. The project will use the following (ordered in ease of use and access)

  1. ChestX-ray8, source: https://www.kaggle.com/nih-chest-xrays/data
  2. MIMIC-CXR (special access required), source: https://physionet.org/content/mimic-cxr/2.0.0/
  3. CheXpert (special access required), source: https://stanfordmlgroup.github.io/competitions/chexpert/ 
      
      3a. If downloaded, CheXpert can be a tf.dataset allowing increased functionality in the pipeline similiar to MNIST, source: https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/image_classification/chexpert.py


### Small NIH Dataset for Image Segmentation

credit: https://github.com/rsummers11/CADLab/tree/master/Lung_Segmentation_XLSor

download at: https://nihcc.app.box.com/s/r8kf5xcthjvvvf6r7l1an99e1nj4080m

