# This file contains helper functions that train.py and predict.py may utilize

# Imports
import torch
from torchvision import datasets, transforms, models
import json
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Functions
def check_gpu():
    '''
        :None
        This function only serves to help the user understand if their machine is capable of running on the gpu
    '''

    # Check if GPU/CUDA capability is even possible with the downloaded version of torch

    # Verify that the GPU is available
    print("Torch version:",torch.__version__)
    print("Is CUDA enabled?",torch.cuda.is_available())

    return None

def transform_normalize_load_batch(data_path, batch_size=64):
    '''
        :data_path - the path of the flowers data for this project
        :batch_size - the requested batch_size for the training, default is 64

        This function will transform, normalize, load, and batch
        
        :return - 3 dataloaders (train, valid, and test)
    '''

    # These paths are unique to how the data is stored!
    data_dir = data_path
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    data_transforms_train = transforms.Compose([transforms.RandomRotation(30),
                                                transforms.RandomResizedCrop(224),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406],
                                                                     [0.229, 0.224, 0.225])])

    data_transforms_valid = transforms.Compose([transforms.Resize(255),
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406],
                                                                     [0.229, 0.224, 0.225])])

    data_transforms_test = transforms.Compose([transforms.Resize(255),
                                               transforms.CenterCrop(224),
                                               transforms.ToTensor(),
                                               transforms.Normalize([0.485, 0.456, 0.406],
                                                                    [0.229, 0.224, 0.225])])

    image_datasets_train = datasets.ImageFolder(train_dir, transform=data_transforms_train)
    image_datasets_valid = datasets.ImageFolder(valid_dir, transform=data_transforms_valid)
    image_datasets_test  = datasets.ImageFolder(test_dir,  transform=data_transforms_test)

    dataloaders_train = torch.utils.data.DataLoader(image_datasets_train, batch_size=batch_size, shuffle=True)
    dataloaders_valid = torch.utils.data.DataLoader(image_datasets_valid, batch_size=batch_size)
    dataloaders_test  = torch.utils.data.DataLoader(image_datasets_test,  batch_size=batch_size)

    return (dataloaders_train, dataloaders_valid, dataloaders_test, image_datasets_train)

def label_mapping(file):
    '''
        :No inputs

        This function was provided to us.
        It simply reads a json file and returns the contents of that json file

        :return - cat_to_name (json file)
    '''

    with open(file, 'r') as f:
        cat_to_name = json.load(f)

    return cat_to_name

def load_checkpoint(filepath_, model_name, optimizer_name):
    checkpoint = torch.load(filepath_)

    model_name.load_state_dict(checkpoint['state_dict']) # Return the model!
    optimizer_name.load_state_dict(checkpoint['optimizer_state']) # Return the optimizer!
    num_epoch = checkpoint['Num_Epochs']
    train_losses = checkpoint['train_losses_list']
    valid_losses = checkpoint['valid_losses_list'],
    valid_accuracy_list = checkpoint['valid_accuracy_list']
    best_valid_loss = checkpoint['valid_loss']
    class_to_idx = checkpoint['class_to_idx']
    
    return model_name, optimizer_name, num_epoch, train_losses, valid_losses, \
           valid_accuracy_list, best_valid_loss, class_to_idx

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''

    # Open the image using the 'Image' toolkit and get the original dimensions
    _256 = int(256)
    _224 = int(224)
    ogWidth, ogHeight = image.size
    ar = ogWidth/ogHeight

    # Determine which is longer
    if ogWidth < ogHeight:
        newWidth = _256
        newHeight = int(_256/ar)
    else:
        newHeight = _256
        newWidth = int(_256*ar)
    
    # This resizes the image where the shortest side is 256 and the aspect ratio is maintained
    imResize = image.resize((newWidth, newHeight))

    # This crops the newly resized image so that it is 224x224
    # This performs a center crop
    left   = int((newWidth - _224) / 2)
    top    = int((newHeight - _224) / 2)
    right  = int((newWidth + _224) / 2)
    bottom = int((newHeight + _224) / 2)
    imCrop = imResize.crop((left, top, right, bottom))

    # Now we maniuplate the image
    imNp = np.array(imCrop)
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])

    # Transform the np.array via the mean and the standard deviation
    imNp = imNp / 255
    imNp = imNp - mean
    imNp = imNp / std

    # Reorder so that PyTorch is happy
    imNp = imNp.transpose((2,0,1))

    return imNp

def predict(image_path, model, class_to_idx, topk=5, gpu='y'):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # Open the image via the path and process the image per our previous function
    image = Image.open(image_path)
    image = process_image(image)

    if gpu == 'y':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = 'cpu'

    # imTen = torch.from_numpy(image)
    imTen = torch.tensor(image).float()

    # Need to add "1" to the beginning to dummy the batch size
    # unsqueeze_ is the only method I could find that does exactly that
    imTen = imTen.unsqueeze(0)

    # Move items to device
    im = imTen.to(device)
    model.to(device)

    # Make sure that the model is in eval mode
    model.eval()

    # Now we run through our model for the single image we have

    with torch.no_grad():

        # Move the model forward as we normally would
        # But this time convert the log prob to normal prob
        logps = model(im)
        prob = torch.exp(logps)

        # Utilize topk method
        probs, indices = torch.topk(prob, topk)

        # Need to move classes to the cpu in order for numpy to work
        indice = indices.to('cpu')
        indice = indice.numpy()[0]

        # Same can be said about the probabilities
        probs = probs.to('cpu')
        probs = probs.numpy()[0].tolist()

        # Use dictionary comprehension to make value, key pairs
        idx_to_class = {val: key for key, val in class_to_idx.items()}

        # Spit out the correct class labels
        classes = [idx_to_class[index] for index in indice]
        

    return probs, classes

def final_output(image_path, model, class_to_idx, topk, file, gpu):

    '''
        :image_path - this is the path of the image you want to check
            It is exactly the same path that we tested 'predict' on
        :model - this is the model that you want to use
            It is a model that we saved from before using a checkpoint
    '''

    # Getting the picture of the image that you are investigating
    image = Image.open(image_path)

    probs, classes = predict(image_path, model, class_to_idx, topk, gpu)

    cat_to_name = label_mapping(file)

    # matplotlib plays nicer with numpy arrays
    probs = np.array(probs)
    classes = np.array(classes)

    # Now we have the names of the flowers and not just their assigned numbers...
    names = [cat_to_name[num] for num in classes]

    # Now we just need to graph everything!
    plt.figure(figsize=(12,4))

    ax1 = plt.subplot(2,1,1)
    ax1.imshow(image)
    ax1.axis('off')
    ax1.set_title(names[0])

    ax2 = plt.subplot(2,1,2)
    ax2.barh(names, probs)

    ax2.invert_yaxis()
    ax2.set_xlabel('Probability')

    image_ = image_path

    plt.savefig('prediction.png')
    print("Prediction image can be found at: prediction.png")

    return {"Name of Predicted Flower": names[0],
            "Probability of Prediction": round(probs[0]*100, 2)}