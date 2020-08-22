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
import copy
import argparse
import numpy as np
import matplotlib.pyplot as plt
' 

parser = argparse.ArgumentParser()
parser.add_argument("--image_path", default="image_path", type=str)
parser.add_argument("--checkpoint", default="checkpoint.pth",type=str)
parser.add_argument("--top_k", default=5, type=int)
parser.add_argument("--filepath", default='cat_to_name.json', action="store")
parser.add_argument("--gpu", default=False, action="store_true")
results = parser.parse_args()

image_path= results.image_path
checkpoint= results.checkpoint
top_k= results.top_k
filepath= results.filepath
device = torch.device('cuda' if gpu and torch.cuda.is_available() else 'cpu')

import json
with open(filepath, 'r') as f:
    cat_to_name = json.load(f)
print(cat_to_name)


def load_checkpoint(path):
    checkpoint = torch.load(path)
    
    model = checkpoint['model']
    model.class_to_idx = checkpoint["class_to_idx"]
    model.classifier = classifier
    model.epochs = checkpoint['epochs']
    model.criterion = checkpoint['criterion']
    model.load_state_dict(checkpoint["state_dic"])
    optimizer.load_state_dict(checkpoint['optimizer_state_dic'])
    
    return model
    print(model)
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
    top_k, classes_index = np.array(top_k.to(power)[0]), np.array(classes_index.to(power)[0])
    idx_to_class = {x: y for y, x in model.class_to_idx.items()}
    
    classes = []
    for idx in classes_index:
        classes.append(idx_to_class[idx])
    
    return list(top_k), list(classes)

model = load_checkpoint(checkpoint)
probs, classes = predict(image_path, model, top_k)
print(probs)
print(classes)
print("Prediction made, end of Part 2.") #Good luck in life.