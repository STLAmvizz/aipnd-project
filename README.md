# AI Programming with Python Project

Project code for Udacity's AI Programming with Python Nanodegree program. In this project, students first develop code for an image classifier built with PyTorch, then convert it into a command line application.

## Housekeeping Notes:

Data is located at: assets/flower_MFV

"models" directory is where I would keep all stored models. So for example I would save a model as "models/VGG16_BestModel_072823.pth"

"checkpoints" directory is where I would keep all stored checkpoints. So for example I would save a checkpoint as "checkpoints/VGG16_Checkpoint_072823.pth"

Notice that "models" and "checkpoints" do not contain anything on Github. In order to populate/create these directories you will need to change the neccessary variables in the Jupyter Notebook -OR- follow the example instructions in the "Scripting Notes" section.

## Scripting Notes:

"helper.py" is a script that cannot be run standalone. It houses some useful functions that were developed in "Image_Classifier_Project.ipynb" and were ported over to a *.py script in order to make "train.py" and "predict.py" function.

"train.py" MUST be run prior to "predict.py"

train.py: python train.py <data_dir> <model_choice> <learning_rate> <training_epochs> <gpu> <model_dir> <checkpoint_dir>
    <data_dir> : The directory in which your data is stored "assets/flowers_MFV"
    <model_choice> : "v" means that you would like to train a VGG16 model. "r" means that you would like to train a Resnet50 model.
    <learning_rate> : The rate at which gradient descent moves for backpropogation. The Adam optimizer is used for training either of these models so a low number like "0.001" is recommended
    <training_epochs> : The number of iterations through the training data you would like to perform "5"
    <gpu> : "y" means that if the GPU is available, you would like to utilize it. Anything other than "y" means that training will happen on the cpu.
    <model_dir> : The location for where the trainied model will be stored "models/<model_name>_date.pth" would be a good example of what to put here
    <checkpoint_dir> : The location for where the checkpoint for the model will be stored "checkpoints/<model_name>_Checkpoint_data.pth" would be a good example of what to put here.

So here is an example of training, validating, and saving a checkpoint of a resnet50 model with a learning rate of 0.001 on 5 epochs utilizing the gpu:
python train.py assets/flowers_MFV r 0.001 5 y models/RESNET50_BestModel_080123.pth checkpoints/RESNET50_Checkpoint_080123.pth

![train.py Example Output](screenshots/train_screenshot_output.jpg?raw=true "train.py Example Output")