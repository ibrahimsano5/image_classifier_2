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

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", default="./flowers", action="store")
parser.add_argument("--save_dir", default="./save_checkpoint.pth", action="store")
parser.add_argument("--learning_rate", default=0.01, type=float, action="store")
parser.add_argument("--arch", default="vgg16", type = str, action="store")
parser.add_argument("--epochs", default=3, type=int, action="store")
parser.add_argument("--gpu", default=False, action="store_true")
parser.add_argument("--hidden_layers", default=512, type=int, action="store")
results = parser.parse_args()

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

print(dataloaders['train'])
print(dataloaders['valid'])
print(dataloaders['test'])

model = models.vgg16(pretrained=True)
for param in model.parameters():
    param.requires_grad = False
classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(25088, 512)),
        ('ReLu1', nn.ReLU()),
        ('Dropout1', nn.Dropout(0.05)),
        ('fc3', nn.Linear(512, 102)),
        ('output', nn.LogSoftmax(dim=1))]))

model.classifier = classifier

criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)

cuda = torch.cuda.is_available()

if cuda:
    model.cuda()
else:
    model.cpu()
    
epochs = 3
print_every = 5
steps = 0

for epoch in range(epochs):
    model.train()
    training_loss = 0
    training_accuracy = 0
    
    for images, labels in iter(dataloaders['train']):
        steps += 1
        inputs, labels = Variable(images), Variable(labels)
        optimizer.zero_grad()
        
        if cuda:
            inputs, labels = inputs.cuda(), labels.cuda()
        output = model.forward(inputs)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        training_loss += loss.item()
        ps_train = torch.exp(output).data
        equality_train = (labels.data == ps_train.max(1)[1])
        training_accuracy += equality_train.type_as(torch.FloatTensor()).mean()
        if steps % print_every == 0:
            model.eval()
            validation_accuracy = 0
            validation_loss = 0
            for images, labels in dataloaders['valid']:
                with torch.no_grad():
                    inputs = Variable(images)
                    labels = Variable(labels)

                    if cuda:
                        inputs, labels = inputs.cuda(), labels.cuda()

                    output = model.forward(inputs)

                    validation_loss += criterion(output, labels).item()

                    ps = torch.exp(output).data
                    equality = (labels.data == ps.max(1)[1])

                    validation_accuracy += equality.type_as(torch.FloatTensor()).mean()
                
            print("Epoch {}/{} \n ".format(epoch+1, epochs),
                  "Training Loss: {:.3f} ->".format(training_loss/print_every),
                  "Validation Loss: {:.3f} ->".format(validation_loss/len(dataloaders['valid'])),
                  "Training Accuracy: {:.3f} ->".format(training_accuracy/len(dataloaders['train'])),
                  "Validation Accuracy: {:.3f}  ".format(validation_accuracy/len(dataloaders['valid'])))
            
            training_loss = 0
            model.train()
model.class_to_idx = image_datasets['train'].class_to_idx
save_checkpoint = {"class_to_idx" : model.class_to_idx,
                   "criterion": nn.NLLLoss(),
                   "model": model,
                   "arch": "vgg16",
                   "classifier": classifier,
                   "means" : [0.485, 0.456, 0.406],
                   "stdev" : [0.229, 0.224, 0.225],
                   "state_dic" : model.state_dict,
                   "optimizer_state_dic" : optimizer.state_dict(),
                   "epochs" : 3
}
torch.save(save_checkpoint, 'save_checkpoint.pth')
print("Model has been trained and saved.")