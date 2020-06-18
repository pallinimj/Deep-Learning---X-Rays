# Northwestern Capstone

# Deep Learning Approach to Thoracic Cavity Disease Feature Identification and Classification

### Project Scope

The project design is to utilize deep learning and computer vision techniques to perform detection, classification, and segmentation of various types of lung diseases  

### Desired Outcome

High accuracy and low loss in classification of selected lung diseases

### Dataset

Quality datasets incorporating medical imaging technology along with labels are required in order to perform supervised learning. Very few datasets exist. Time does not allow creation and curation a specific dataset. The project will use the NIH data made available through Kaggle (source: https://www.kaggle.com/nih-chest-xrays/data).  This data consists of 112,000 thoracic cavity x-rays that show 15 different outcomes, 14 medical conditions as well as no conditions.  The data is heavily imbalanced, with 56.8% of the images showing no condition.  

### Baseline Model

The model uses a convolutional neural network (CNN) architecture to achieve its results. It uses three separate dropout layers to reduce overfitting. It also utilizes several layers of Conv2D and MaxPooling2D to pick up important features in each image to differentiate the conditions from clean scans (Figure 2). The initial baseline model uses a sample of 5,600 out of the 112,000 total images, making it easier to update and train the model. The model uses a train/test split of 80% for training and 20% for testing. Additionally, the model trained through 15 epochs, each with a batch size of 44. This produced a training accuracy of 63.5% and a validation accuracy of 60.5%. Also, using this model to make predictions on the test data produced a ROC score of 0.578.

### Transfer Learning

The secondary model is a Bucket of Models method of ensemble learning. By utilizing a model selection algorithm, we are directly applying the concept of transfer learning and attempting to discover the best network architecture for the problem. The aforementioned challenges and early mitigation efforts remain the same, however the Bucket of Models provided a rapid prototype capability to the team. The rapid prototyping illustrated early signs of improved performance with the VGG16 architecture achieving higher rates of accuracy. We allowed certain hyperparameters to either remain as default or modified with the exception of the following. These remained static during all training and testing runs.

Model | Precision | Accuracy | Sensitivity | Specificity
----- | --------- | -------- | ----------- | -----------
VGG16 | 0.94 | 0.97 | 0.97 | 0.97
VGG19 | 0.96 | 0.93 | 0.84 | 0.98
MobileNet | 1 | 0.86 | 0.59 | 1
InceptionV3 | 1 | 0.66 | 0.03 | 1
ResNet50 | 0.36 | 0.36 | 1 | 0
Xception | 1 | 0.88 | 0.66 | 1




  


