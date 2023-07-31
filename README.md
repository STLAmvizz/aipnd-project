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

train.py: python train.py