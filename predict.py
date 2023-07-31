# This file contains the main prediction functionality
# The goal of this file is to be able to take in an image to predict and have some output convey what that input is
# CLI: python predict.py <prediction_image_path> <checkpoint_dir>
# Where <prediction_image_path> is the directory to the image path for a single image
# Where <checkpoint_dir> is the directory where the checkpoint was saved from 'train.py'

# Imports for scripting
import argparse
import helper

# Imports for PyTorch
import torch
from torch import nn, optim
from torchvision import models
from collections import OrderedDict




if __name__ == "__main__":

    # User created arguments
    ########################################################################################################################
    parser = argparse.ArgumentParser()
    parser.add_argument('prediction_image_path', help="assets/flowers_MFV/test/1/image_06743.jpg")
    parser.add_argument('checkpoint_dir', help="checkpoints/VGG16_Checkpoint_072823.pth")
    parser.add_argument('model_choice', help="v --> VGG16; r --> Resnet50")
    parser.add_argument('learning_rate', help="learning rate that was used in train.py")
    parser.add_argument('topk', help="number of classes to show")
    parser.add_argument('file', help="cat_to_name.json")
    args = parser.parse_args()
    ########################################################################################################################

    # Reinstantiate the model that we use
    ########################################################################################################################
    lr = float(args.learning_rate)
    # Set up the original VGG16 Model
    # Use the default weights for everything at first
    weightsvgg16 = models.VGG16_Weights.DEFAULT
    vgg16 = models.vgg16(weights=weightsvgg16)

    # Freeze Parameters so we don't backprop through them
    for param in vgg16.parameters():
        param.requires_grad = False

    # Remember that each of these names in the OrderedDict CANNOT be the same as any other name... so name them uniquely!
    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(25088, 2500)),
        ('relu1', nn.ReLU()),
        ('dropout1', nn.Dropout(0.2)),
        ('fc2', nn.Linear(2500, 1024)),
        ('relu2', nn.ReLU()),
        ('dropout2', nn.Dropout(0.2)),
        ('fc3', nn.Linear(1024,256)),
        ('relu3', nn.ReLU()),
        ('dropout3', nn.Dropout(0.2)),
        ('fc4', nn.Linear(256,102)),
        ('output1', nn.LogSoftmax(dim=1))
    ]))

    vgg16.classifier = classifier
    vgg16.name = 'vgg16'

    optimizervgg16 = optim.Adam(vgg16.classifier.parameters(), lr=lr)

    # Set up the original Resnet50 Model
    # Use the default weights for everything at first
    weightsrn50 = models.ResNet50_Weights.DEFAULT
    resnet50 = models.resnet50(weights=weightsrn50)

    # Freeze Parameters so we don't backprop through them
    for param in resnet50.parameters():
        param.requires_grad = False

    fc = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(2048, 500)),
        ('relu1', nn.ReLU()),
        ('dropout1', nn.Dropout(0.2)),
        ('fc2', nn.Linear(500, 255)),
        ('relu2', nn.ReLU()),
        ('dropout2', nn.Dropout(0.2)),
        ('fc3', nn.Linear(255,102)),
        ('output1', nn.LogSoftmax(dim=1))
    ]))

    resnet50.fc = fc
    resnet50.name = 'resnet50'

    optimizerresnet50 = optim.Adam(resnet50.fc.parameters(), lr=lr)

    # Let the user choose which of the models they would like to train
    if args.model_choice == 'v':
        model = vgg16
        optimizer = optimizervgg16
    elif args.model_choice == 'r':
        model = resnet50
        optimizer = optimizerresnet50
    ########################################################################################################################

    # Load the Checkpoint that was created in "train.py"
    ########################################################################################################################
    model_vgg16, optimizer_vgg16, num_epoch, \
    train_losses_loaded, valid_losses_loaded, \
    valid_accuracy_loaded, best_valid_loss_loaded, \
    class_to_idx = helper.load_checkpoint(filepath_=args.checkpoint_dir,
                                          model_name=model,
                                          optimizer_name=optimizer)
    ########################################################################################################################

    # Show the user the Predicted Flower as well as the Probability of that Prediction
    ########################################################################################################################
    topk = int(args.topk)
    file = args.file
    dict = helper.final_output(args.prediction_image_path, model_vgg16, class_to_idx, topk, file)
    print(dict)
    ########################################################################################################################