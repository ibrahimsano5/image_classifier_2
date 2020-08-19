import numpy as np
import argparse
import json
import torch
from PIL import Image
from torchvision import datasets, transforms, models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser = argparse.ArgumentParser()
parser.add_argument("image_path", action="store", type = str)
parser.add_argument("save_checkpoint", action="store", type = str)
parser.add_argument("--category_names", default="cat_to_name.json", action="store")
parser.add_argument("--gpu", default=False, action="store_true")
parser.add_argument("--top_k", default=5, type=int)
return parser.parse_args()

def load_checkpoint(path="save_checkpoint.pth"):
    
    load_checkpoint = torch.load("save_checkpoint.pth")
    model = models.vgg16(pretrained=True)
    for param in model.parameters():
        param.requires_grad=False
    
    model.class_to_idx = load_checkpoint["class_to_idx"]
    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(25088, 512)),
        ('ReLu1', nn.ReLU()),
        ('Dropout1', nn.Dropout(0.05)),
        ('fc3', nn.Linear(512, 102)),
        ('output', nn.LogSoftmax(dim=1))]))
    model.classifier = classifier
    model.class_to_idx = load_checkpoint["class_to_idx"]
    model.load_state_dict(load_checkpoint["state_dic"])
    optimizer.load_state_dict(checkpoint['optimizer_state_dic'])
    
    return model

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