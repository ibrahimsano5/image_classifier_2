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
results = parser.parse_args()

def load_checkpoint(path="save_checkpoint.pth"):
    checkpoint = torch.load(filepath)
    
    model = checkpoint['model']
    
    model.class_to_idx = checkpoint["class_to_idx"]
    model.classifier = classifier
    model.epochs = checkpoint['epochs']
    model.criterion = checkpoint['criterion']
    model.load_state_dict(checkpoint["state_dic"])
    optimizer.load_state_dict(checkpoint['optimizer_state_dic'])
    
    return model
    
    
    
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