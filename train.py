# This file contains the main training functionality

# The goal of this file is to be able to run this command line
# CLI: python train.py <data_dir> <model_dir> <checkpoint_dir>
# Where <data_dir> is where all of the data is held
# Where <model_dir> is where the trained model is saved
# Where <checkpoint_dir> is where the trained model and other attributes is saved

# Imports for scripting
import argparse
import helper

# Imports for PyTorch
import torch
from torchvision import models
from torch import nn, optim
from collections import OrderedDict


if __name__ == "__main__":

    # User created arguments
    ########################################################################################################################
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', help="Directory in which data is held")
    parser.add_argument('model_choice', help="v --> VGG16; r --> Resnet50")
    parser.add_argument('learning_rate', help="0.001")
    parser.add_argument('training_epochs', help="10")
    parser.add_argument('gpu', help="y")
    parser.add_argument('model_dir', help="Directory in which the model will be saved")
    parser.add_argument('checkpoint_dir', help="Directory in which the checkpoint will be saved")
    args = parser.parse_args()
    ########################################################################################################################

    # Test all of the inputs and their types
    ########################################################################################################################
    # print("model choice: {}".format(args.model_choice))
    # print("type model choice: {}".format(type(args.model_choice)))
    # print("learning rate: {}".format(float(args.learning_rate)))
    # print("type learning rate: {}".format(type(float(args.learning_rate))))
    # print("training epochs: {}".format(int(args.training_epochs)))
    # print("type training epochs: {}".format(type(int(args.training_epochs))))
    # print("gpu: {}".format(args.gpu))
    # print("type gpu: {}".format(type(args.gpu)))

    # quit()
    ########################################################################################################################

    # Instantiate the Model(s)
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

    # Train the instantiated model
    ########################################################################################################################
    path = args.model_dir
    dataloaders_train = helper.transform_normalize_load_batch(args.data_dir)[0]
    dataloaders_valid = helper.transform_normalize_load_batch(args.data_dir)[1]
    epochs = int(args.training_epochs)
    print_every = 10

    # Optimizer & criterion
    criterion = nn.NLLLoss()

    # Move the model over to the appropriate device
    if args.gpu == 'y':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = 'cpu'
    
    model.to(device)

    # Some initalizations required for training/validating
    best_valid_loss = float('inf')
    steps = 0
    running_loss_train = 0

    # Let's also add some lists so that we can track losses and accuracies
    train_losses = []
    valid_losses = []
    valid_accuracy = []

    # An epoch is defined as a single run through of the entire (training) set of data
    # 6552 images/labels
    for epoch in range(epochs):

        # Put the model into training
        model.train()

        # dataloaders_train contains 103 'passes' which accounts for ALL images/labels
        for images_train, labels_train in dataloaders_train:

            # Increment the steps for nice readouts
            steps += 1

            # Move things over to the GPU!
            images_train =  images_train.to(device)
            labels_train =  labels_train.to(device)

            # The classic pattern
            optimizer.zero_grad()
            logps = model(images_train)
            loss = criterion(logps, labels_train)
            loss.backward()
            optimizer.step()
            running_loss_train += loss.item()

            if steps % print_every == 0:

                # Put the model into eval mode
                model.eval()

                with torch.no_grad():

                    running_loss_valid = 0
                    accuracy_valid = 0

                    # dataloaders_valid contains 818 images
                    # so for a batch size of 64 we get 13 'passes' which accounts for ALL images/labels
                    for images_valid, labels_valid in dataloaders_valid:

                        # Move things over the GPU!
                        images_valid = images_valid.to(device)
                        labels_valid = labels_valid.to(device)

                        logps      = model(images_valid)
                        valid_loss = criterion(logps, labels_valid)
                        running_loss_valid += valid_loss.item()

                        # Calculate Accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equal = top_class == labels_valid.view(*top_class.shape)
                        accuracy_valid += torch.mean(equal.type(torch.FloatTensor)).item()

                rlt = running_loss_train/print_every
                rlv = running_loss_valid/len(dataloaders_valid)
                acc = accuracy_valid/len(dataloaders_valid)
                
                print(f"Epoch {epoch+1}/{epochs}.. "
                        f"Train loss: {rlt:.3f}.. "
                        f"Valid loss: {rlv:.3f}.. "
                        f"Valid accuracy: {acc:.3f}")
                
                train_losses.append(rlt)
                valid_losses.append(rlv)
                valid_accuracy.append(acc)
                
                if valid_loss < best_valid_loss:
                    best_valid_loss = valid_loss

                    torch.save(model.state_dict(), path)
                
                # Reinitialize the running_loss to be 0 for the new epoch
                running_loss_train = 0
    ########################################################################################################################

    # Test the network post training
    ########################################################################################################################
    state_dict = torch.load(args.model_dir)
    model = model
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)

    dataloaders_test = helper.transform_normalize_load_batch(args.data_dir)[2]

    with torch.no_grad():
        running_test_loss = 0
        test_accuracy = 0
        for images_test, labels_test in dataloaders_test:

            # Move things over the GPU!
            images_test = images_test.to(device)
            labels_test = labels_test.to(device)

            logps = model(images_test)
            test_loss = criterion(logps, labels_test)
            running_test_loss += test_loss.item()

            # Calculate Accuracy
            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equal = top_class == labels_test.view(*top_class.shape)
            test_accuracy += torch.mean(equal.type(torch.FloatTensor)).item()

    print(f"Test accuracy: {test_accuracy/len(dataloaders_test):.3f}")
    ########################################################################################################################

    # Create the checkpoint
    ########################################################################################################################
    image_datasets_train = helper.transform_normalize_load_batch(args.data_dir)[3]
    model.class_to_idx = image_datasets_train.class_to_idx

    checkpoint = {'model': models.vgg16(),
                  'state_dict': model.state_dict(),
                  'optimizer': optimizer,
                  'optimizer_state': optimizer.state_dict(),
                  'Num_Epochs': epochs,
                  'train_losses_list': train_losses,
                  'valid_losses_list': valid_losses,
                  'valid_accuracy_list': valid_accuracy,
                  'valid_loss': best_valid_loss,
                  'class_to_idx': model.class_to_idx
                  }
    
    torch.save(checkpoint, args.checkpoint_dir)
    ########################################################################################################################