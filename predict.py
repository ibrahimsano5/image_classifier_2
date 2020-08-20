import torch
from torch import optim, nn, tensor
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, transforms, models
import json
from collections import OrderedDict
import PIL
from PIL import Image
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import copy
import argparse
import utility

parser = argparse.ArgumentParser()
parser.add_argument("--image", default="image_path", type=str)
parser.add_argument("--checkpoint", default="save_checkpoint.pth",type=str)
parser.add_argument("--top_k", default=5, type=int)
parser.add_argument("--category_names", default='cat_to_name.json', action="store")
parser.add_argument("--gpu", default=False, action="store_true")
results = parser.parse_args()

#All saved data were included in utility.py and in checkpoint, but there's a problem when calling previous functions by importing datas, so
#I copied and pasted them back here, but the program without these is the right one. Copied data (copied from train.py) start from "#copied"
#to "#pasted"


#Copied --->

data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

data_transforms = { 
    'train': transforms.Compose([
        transforms.RandomResizedCrop(size=224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],
                             [0.229,0.224,0.225])
    ]),
    'valid': transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],
                             [0.229,0.224,0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],
                             [0.229,0.224,0.225])
    ])
}
image_datasets = {
    'train': datasets.ImageFolder(train_dir, transform=data_transforms['train']),
    'valid': datasets.ImageFolder(valid_dir, transform=data_transforms['valid']),
    'test': datasets.ImageFolder(test_dir, transform=data_transforms['test'])
}
dataloaders = {
    'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=64, shuffle=True),
    'valid': torch.utils.data.DataLoader(image_datasets['valid'], batch_size=64, shuffle=True),
    'test': torch.utils.data.DataLoader(image_datasets['test'], batch_size=64, shuffle=True)
}

model = models.vgg16(pretrained=True)
model
for param in model.parameters():
    param.requires_grad = False
classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(25088, 512)),
        ('ReLu1', nn.ReLU()),
        ('Dropout1', nn.Dropout(0.05)),
        ('fc3', nn.Linear(512, 102)),
        ('output', nn.LogSoftmax(dim=1))]))
print('Utility imported')
#<--- Pasted

model.classifier = classifier
model.class_to_idx = image_datasets['train'].class_to_idx

def load_checkpoint(path="save_checkpoint.pth"):
    checkpoint = torch.save(save_checkpoint, path)
    
    model = checkpoint['model']
    
    model.class_to_idx = checkpoint["class_to_idx"]
    model.classifier = classifier
    model.epochs = checkpoint['epochs']
    model.criterion = checkpoint['criterion']
    model.load_state_dict(checkpoint["state_dic"])
    optimizer.load_state_dict(checkpoint['optimizer_state_dic'])
    
    return model

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    process_image = transforms.Compose([ transforms.Resize(255), transforms.CenterCrop(224), 
                                         transforms.ToTensor(), 
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    
    picture = process_image(Image.open(image))
    
    return picture

def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    model.to(device)
    
    picture = process_image(image_path).to(device)
    image = picture.unsqueeze_(0)
    
    model.eval()
    with torch.no_grad():
        log_probs = model.forward(image)
    
    ps = torch.exp(log_probs)
    top_k, classes_index = ps.topk(topk, dim=1)
    top_k, classes_index = np.array(top_k.to('cpu')[0]), np.array(classes_index.to('cpu')[0])
    idx_to_class = {x: y for y, x in model.class_to_idx.items()}
    
    classes = []
    for idx in classes_index:
        classes.append(idx_to_class[idx])
    
    return list(top_k), list(classes)

image_path = "flowers/test/33/image_06486.jpg"
probs, classes = predict(image_path, model)
print(probs)
print(classes)
print("Prediction made, end of Part 2.")