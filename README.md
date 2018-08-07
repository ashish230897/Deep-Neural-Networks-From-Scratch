# Designing Your Own Deep Neural Network
This repository will walk you through the process of creating a **Deep Neural Network** without using any of the deep learning *libraries* like *Tensorflow*, *Keras*, *Caffe* etc.  
This method of training a neural network isn't obviously the most efficient one and it is the *most inefficient* way of deploying a deep learning model.  
**BUT** while implementing your first model in this way, you will be familiarized with every minute step that has to be taken care of while dealing with models! What's more exciting than knowing the nook and corner of your own *model*!  
                                                                                               
Here I have trained the model on about 1000 images of dogs and cats for image classification task. It is just a *demonstration* of how a hand made deep learning model can be trained on a dataset. The best practices involved in training a good model is not followed as it is not the purpose of this repository.
                                                                                                          
                                                                                                   
## Getting Started
These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. 
                                                             
                                                                       
### Prerequisites
The libraries required for the above project are : matplotlib, numpy, PIL.  
                                                                          
                                                             
### Installing 
```
pip install numpy  
pip install matplotlib  
pip install Pillow
```  
Create a folder named 'Data'. The data required to train the model has to be kept in the Data directory.  
Inside the Data folder, create two new folders named 'Train' and 'Test'.
The Train folder there should contain the training images. The images can be found [here](https://www.kaggle.com/c/dogs-vs-cats)
                                                                   
                                                   
## Training the Model
The project above follows the following pattern while training the model:  
* Load the data
* Initialize parameters
* Forward Propagation
* Compute Cost
* Backward Propagation
* Update Parameters  
                           
The file *main.py* is to be run to start the training.  
The files *forward_pass_utils.py*, *backward_pass_utils.py* and *utils.py* contains the functions required by the main code.  
In the file *main.py*, the list **layer_dims** can be altered, although the first parameter(12288) is set considering the images to be of size (64x64) and the last parameter is 1 since it is a binary classification problem. The number of hidden layers and the number of hidden units can be tweaked.  
The parameters are saved after each epoch. To load a model from last saved checkpoint, pass the load_pretrained parameter as True else pass it as False.  
                                                             
**You can also view the provided *jupyter notebook* file with a notebook viewer to follow the code step by step on a notebook.**  


## Acknowledgements
* Inspired by Andrew Ng!
* Got the code support from deeplearning.ai first course.

**In case of any *doubts/confusions* do shoot a mail at : ashish.agrawal2123@gmail.com**


